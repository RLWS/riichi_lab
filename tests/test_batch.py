from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from rilab.artifacts import audit_batch
from rilab.batch import run_batch
from rilab.config import load_arena_config
from rilab.decision import AdapterError
from rilab.scheduling import expand_schedule

ROOT = Path(__file__).parents[1]


@pytest.fixture
def config():
    value = load_arena_config(ROOT / "configs/arena_selfplay.yaml")
    value["engine"]["seeds"] = [1]
    return value


def test_schedule_preserves_pairs_controls_and_six_combinations(config):
    schedule = expand_schedule(config)
    assert len(schedule) == 14
    candidate, control, two = schedule[:4], schedule[4:8], schedule[8:]
    assert [g["focal_seat"] for g in control] == list(range(4))
    for a, b in zip(candidate, control):
        assert a["paired_key"] == b["paired_key"]
        assert a["focal_seat"] == b["focal_seat"]
        assert a["seats"][a["focal_seat"]] == "v1_3_0"
        assert b["seats"] == ["v1_2_0"] * 4
    assert len({tuple(g["seats"]) for g in two}) == 6
    assert all(g["seats"].count("v1_3_0") == 2 for g in two)
    assert schedule == expand_schedule(deepcopy(config))


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [-1], [2**32], None])
def test_schedule_rejects_invalid_seeds(config, seeds):
    config["engine"]["seeds"] = seeds
    with pytest.raises(ValueError, match="seeds"):
        expand_schedule(config)


@pytest.mark.parametrize(
    "defect", ["missing_control", "focal", "opponents", "protocol"]
)
def test_schedule_rejects_incomplete_pairs(config, defect):
    if defect == "missing_control":
        config["matchups"].pop(1)
    elif defect == "focal":
        config["matchups"][1]["focal_seat"] = 1
    elif defect == "opponents":
        config["matchups"][1]["seats"][2] = "v1_3_0"
    else:
        config["evaluation"].pop("cluster")
    with pytest.raises(ValueError):
        expand_schedule(config)


class Backend:
    instances = []
    failure = None

    def __init__(self, name, config):
        self.backend_id = name
        self.profile = config["adapter_profile"]
        self.metadata = {"test": True}
        self.requests = 0
        self.closed = False
        self.instances.append(self)

    def predict(self, task, state, candidates):
        self.requests += 1
        if self.failure == "inference" and self.requests > 3:
            raise AdapterError("injected inference failure")
        order = ["tsumo", "ron", "chankan", "discard", "pass"]
        selected = min(
            candidates,
            key=lambda c: order.index(c["kind"]) if c["kind"] in order else 99,
        )
        return {"selected_action": selected}

    def check_unchanged(self):
        if self.failure == "metadata":
            raise AdapterError("model metadata changed during run")

    def close(self):
        self.closed = True


@pytest.fixture
def one_game(config, monkeypatch):
    import rilab.batch as batch

    schedule = expand_schedule(config)[:1]
    monkeypatch.setattr(batch, "expand_schedule", lambda _: schedule)
    Backend.instances = []
    Backend.failure = None
    yield config
    Backend.failure = None


@pytest.mark.parametrize("full", [False, True])
def test_batch_real_engine_replay_and_artifact_audit(one_game, tmp_path, full):
    one_game["artifacts"]["full_snapshots_first_run"] = full
    first = run_batch(one_game, tmp_path / "first", backend_factory=Backend)
    second = run_batch(one_game, tmp_path / "second", backend_factory=Backend)
    assert first["valid"] and second["valid"]
    assert first["replay_digest"] == second["replay_digest"]
    assert first["games"] == second["games"]
    assert all(b.closed for b in Backend.instances)
    path = tmp_path / "first"
    assert not list((path / "decisions").glob("*.full.jsonl.gz"))
    with gzip.open(path / "decisions/g00000.jsonl.gz", "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert all("bindings" in r for r in records)
    assert all("state" in r for r in records[::100])
    assert all("state" in r for r in records) == full
    assert audit_batch(path) == first["games"]
    with (path / "games/g00000.mjai.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="event digest"):
        audit_batch(path)
    (path / "results/g00000.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        audit_batch(path)


@pytest.mark.parametrize("failure", ["inference", "metadata", "capacity", "disk"])
def test_failure_keeps_invalid_run_and_closes_backends(
    one_game, tmp_path, monkeypatch, failure
):
    import rilab.batch as batch

    Backend.failure = failure
    if failure == "capacity":
        one_game["artifacts"]["max_bytes"] = 1
    if failure == "disk":
        original = batch.write_json

        def fail_game_result(path, value):
            if path.parent.name == "results":
                raise OSError("injected disk failure")
            original(path, value)

        monkeypatch.setattr(batch, "write_json", fail_game_result)
    out = tmp_path / "failed"
    with pytest.raises((AdapterError, OSError)):
        run_batch(one_game, out, backend_factory=Backend)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["valid"] is False
    assert summary["completed_games"] == 0
    assert all(b.closed for b in Backend.instances)
    assert list((out / "decisions").glob("*.full.jsonl.gz"))
    if failure != "disk":
        result = json.loads((out / "results/g00000.json").read_text())
        assert result["valid"] is False


def test_failure_stops_new_games_and_preserves_completed(
    one_game, tmp_path, monkeypatch
):
    import rilab.batch as batch

    schedule = expand_schedule(one_game)[:3]
    monkeypatch.setattr(batch, "expand_schedule", lambda _: schedule)
    original = batch._run_game
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[2]["game_id"])
        if len(calls) == 2:
            Backend.failure = "inference"
        return original(*args, **kwargs)

    monkeypatch.setattr(batch, "_run_game", fail_second)
    out = tmp_path / "partial"
    with pytest.raises(AdapterError, match="inference"):
        run_batch(one_game, out, backend_factory=Backend)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["completed_games"] == 1 and summary["valid"] is False
    assert calls == ["g00000", "g00001"]
    assert json.loads((out / "results/g00000.json").read_text())["valid"] is True
    assert json.loads((out / "results/g00001.json").read_text())["valid"] is False
    assert (out / "decisions/g00001.full.jsonl.gz").is_file()
