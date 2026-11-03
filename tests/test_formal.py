from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import rilab.formal as formal
from rilab.artifacts import write_json
from rilab.config import load_arena_config
from rilab.evaluation import evaluate_run, paired_bootstrap
from rilab.strength import register_protocol
from test_evaluation import synthetic_run
from test_strength import deployed

ROOT = Path(__file__).parents[1]


@pytest.fixture
def registered(tmp_path, monkeypatch):
    config_path = ROOT / "configs/arena_strength.yaml"
    plan_path = ROOT / "configs/strength_protocol.yaml"
    config, metadata = deployed(load_arena_config(config_path))
    for name, b in config["backends"].items():
        archive = tmp_path / f"{name}.zip"
        archive.write_bytes(b"unit-test archive")
        b["deployment"]["archive_sha256"] = formal.sha(archive)
    register_protocol(plan_path, config_path, config, metadata, tmp_path)
    batch = tmp_path / "batch"
    batch.mkdir()
    source = "rilab/agent.py"
    manifest = {
        "config": config,
        "models": metadata,
        "rilab_dirty": False,
        "rilab_commit": "fixture_commit",
        "dependencies": {},
        "runtime_source_sha256": {source: formal.sha(ROOT / source)},
    }
    monkeypatch.setattr(
        formal, "committed_bytes", lambda commit, name: (ROOT / name).read_bytes()
    )
    write_json(batch / "summary.json", {"replay_digest": "fixture_digest"})
    write_json(
        tmp_path / "verification.json",
        {
            "valid": True,
            "runs": 1,
            "games_per_run": 1792,
            "digests": ["fixture_digest"],
            "deployment": {
                name: {
                    "revision": b["revision"],
                    "verified_weight_sha256": b["model_identity"],
                    "source_dirty": False,
                }
                for name, b in config["backends"].items()
            },
        },
    )
    return batch, manifest


def test_sealed_formal_registration_passes(registered):
    batch, manifest = registered
    result = formal.validate_completed_registration(batch, manifest)
    assert result["eligible"] and all(result["checks"].values())
    assert (
        result["registration"]["plan"]["decision"]["minimum_meaningful_rank_gain"]
        == 0.05
    )


@pytest.mark.parametrize(
    "fault",
    ["registration", "config", "model", "dirty", "source", "archive", "completion"],
)
def test_reject_invalid_formal_eligibility(registered, fault):
    batch, manifest = registered
    if fault == "registration":
        with (batch.parent / "protocol.registered.json").open("a") as f:
            f.write("\n")
    elif fault == "config":
        manifest["config"]["backends"]["v1_2_0"]["response"]["pass_bias"] = 0
    elif fault == "model":
        manifest["models"]["v1_2_0"]["response"]["default_pass_bias"] = 0
    elif fault == "dirty":
        manifest["rilab_dirty"] = True
    elif fault == "source":
        manifest["runtime_source_sha256"]["rilab/agent.py"] = "0" * 64
    elif fault == "archive":
        (batch.parent / "v1_2_0.zip").write_bytes(b"changed")
    else:
        p = batch.parent / "verification.json"
        x = json.loads(p.read_text())
        x["runs"] = 2
        write_json(p, x)
    with pytest.raises(ValueError):
        formal.validate_completed_registration(batch, manifest)


@pytest.mark.parametrize(
    "rank_ci,last_ci,expected",
    [
        ([-0.2, -0.1], [-0.04, 0.01], "improvement"),
        ([-0.2, -0.1], [-0.01, 0.03], "insufficient_evidence"),
        ([0.06, 0.2], [-0.04, 0.03], "regression"),
        ([-0.04, -0.01], [-0.04, 0.01], "insufficient_evidence"),
    ],
)
def test_formal_decision_uses_sealed_thresholds_and_guardrail(
    tmp_path, monkeypatch, rank_ci, last_ci, expected
):
    import rilab.evaluation as evaluation

    run, _, _ = synthetic_run(tmp_path, monkeypatch)
    decision = yaml.safe_load((ROOT / "configs/strength_protocol.yaml").read_text())[
        "decision"
    ]
    monkeypatch.setattr(
        formal,
        "validate_completed_registration",
        lambda *args: {
            "eligible": True,
            "registration": {"plan": {"decision": decision}},
            "checks": {"test": True},
        },
    )
    monkeypatch.setattr(
        formal, "replay_recorded_games", lambda *args: {"valid": True, "model_calls": 0}
    )
    clusters = [
        {"candidate": {"games": 4, "hands": 4}, "control": {"games": 4, "hands": 4}}
    ] * 2
    metrics = paired_bootstrap(clusters, iterations=100)
    metrics["mean_rank"]["ci95"] = rank_ci
    metrics["last_place_rate"]["ci95"] = last_ci
    calls = []

    def fixed(*args, **kwargs):
        calls.append(kwargs)
        return deepcopy(metrics)

    monkeypatch.setattr(evaluation, "paired_bootstrap", fixed)
    report = evaluate_run(
        run, tmp_path / "report", formal=True, iterations=100, rng_seed=0
    )
    assert report["analysis_kind"] == "formal_preregistered"
    assert report["strength_verdict"] == expected
    assert all(
        c["iterations"] == decision["bootstrap_iterations"]
        and c["rng_seed"] == decision["bootstrap_rng_seed"]
        for c in calls
    )


class ReplayBackend:
    def __init__(self, name, config):
        self.backend_id, self.profile = name, config["adapter_profile"]
        self.metadata = {task: {"task": task} for task in ("turn", "response")}
        self.requests = 0

    def predict(self, task, state, candidates):
        self.requests += 1
        order = ["tsumo", "ron", "chankan", "discard", "pass"]
        selected = min(
            candidates,
            key=lambda c: order.index(c["kind"]) if c["kind"] in order else 99,
        )
        return {"task": task, "selected_action": selected, "model": self.metadata[task]}

    def check_unchanged(self):
        pass


def test_offline_replay_reconstructs_sparse_inputs_and_rejects_bad_terminal_score(
    tmp_path,
):
    from rilab.batch import _run_game
    from rilab.backends.http_backend import BackendRegistry
    from rilab.scheduling import expand_schedule

    config = load_arena_config(ROOT / "configs/arena_selfplay.yaml")
    for name in ("games", "results", "decisions"):
        (tmp_path / name).mkdir()
    registry = BackendRegistry(config["backends"], factory=ReplayBackend)
    manifest = {
        "config": config,
        "models": {name: b.metadata for name, b in registry.backends.items()},
    }
    result = _run_game(
        config,
        registry,
        expand_schedule(config)[0],
        tmp_path,
        100,
        full=False,
        budget=100_000_000,
    )
    verified = formal.replay_recorded_games(tmp_path, manifest, [result], workers=1)
    assert verified["all_full_projected_inputs_and_action_bindings_verified"]
    assert verified["decisions"] == result["decisions"] and verified["model_calls"] == 0
    changed = deepcopy(result)
    changed["scores"][0] += 100
    with pytest.raises(ValueError, match="replay output/digest"):
        formal.replay_recorded_games(tmp_path, manifest, [changed], workers=1)
