from collections import Counter
import gzip
import json
from pathlib import Path

import httpx
import pytest
from riichienv import Observation, RiichiEnv, GameRule, calculate_shanten

from rilab.adapters import create_adapter
from rilab.agent import MjaiBotAgent
from rilab.config import load_arena_config
from rilab.decision import AdapterError
from rilab.mapping.action_mapping import bind_candidates, ensure_legal
from rilab.tiles import from_physical
from rilab.tracker.state_tracker import StateTracker
from rilab.backends.http_backend import HttpPredictBackend
from scripts.capture_directed import setup_events

ROOT = Path(__file__).parents[1]


def load_game(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def test_tracker_replays_all_engine_fixtures():
    for path in sorted((ROOT / "tests/fixtures/engine/games").glob("*.gz")):
        trackers = [StateTracker(i) for i in range(4)]
        for decision in load_game(path)["decisions"]:
            tracker = trackers[decision["player_id"]]
            tracker.consume(decision["new_events"])
            if "observation" in decision:
                obs = Observation.deserialize_from_base64(decision["observation"])
                tracker.check_observation(obs)
                state = tracker.snapshot(tracker.context(obs.legal_actions()))
                assert state["hand"] == dict(
                    Counter(from_physical(t) for t in obs.hand)
                )
                bindings = bind_candidates(
                    obs, tracker.context(obs.legal_actions()), state
                )
                for binding in bindings:
                    ensure_legal(binding.action, obs.legal_actions())
            if (
                any(a["action_type"] == "Ron" for a in decision["legal_actions"])
                and decision["chosen"]["action_type"] != "Ron"
            ):
                tracker.confirm_missed_ron()


@pytest.mark.parametrize("profile", ["legacy_v1_2", "full_state_v2"])
def test_profiles_are_distinct_and_do_not_mutate_tracker(profile):
    source = json.loads(
        (ROOT / "tests/fixtures/http_contract/engine_source.json").read_text()
    )
    obs = Observation.deserialize_from_base64(source["turn_observation"])
    tracker = StateTracker(0)
    tracker.consume(obs.new_events())
    before = tracker.snapshot()
    state = create_adapter(profile).project(before)
    assert tracker.snapshot() == before
    if profile == "legacy_v1_2":
        assert state["scores"] == [250] * 4
        assert state["remaining_draws_estimate"] == 17
        assert state["dora_base_indices"] == [10]
        assert "feature_schema" not in state and "public_events" not in state
    else:
        expected = json.loads(
            (ROOT / "tests/fixtures/http_contract/golden_turn_request.json").read_text()
        )["state"]
        assert state == expected


def test_legacy_profile_preserves_historical_counter_semantics():
    request = json.loads(
        (ROOT / "tests/fixtures/http_contract/golden_response_request.json").read_text()
    )
    state = request["state"]
    state["all_discards"][1] = ["9p", "1z"]
    state["turn_number"] = 1
    state["riichi_declared_by_player"][0] = True
    state["riichi_sticks"] = 3
    projected = create_adapter("legacy_v1_2").project(state)
    assert projected["turn_number"] == 2
    assert projected["remaining_draws_estimate"] == 16
    assert projected["riichi_sticks"] == 2


def test_chankan_snapshot_precedes_meld_upgrade():
    tracker = StateTracker(1)
    for event in setup_events("chankan"):
        if event["type"] == "start_kyoku":
            event = dict(event)
        tracker.apply(event)
    pre = tracker.snapshot()
    tracker.apply({"type": "kakan", "actor": 0, "pai": "4m", "consumed": ["4m"] * 3})
    data = json.loads(
        (ROOT / "tests/fixtures/engine/directed/chankan.json").read_text()
    )
    obs = Observation.deserialize_from_base64(data["responses"][0]["observation"])
    context = tracker.context(obs.legal_actions())
    state = tracker.snapshot(context)
    assert context.chankan
    assert state["all_melds"][0][0]["kind"] == "pon"
    assert tracker.snapshot()["all_melds"][0][0]["kind"] == "kakan"
    assert state["public_events"] == pre["public_events"]
    assert state["response_target"] == {"source": 0, "tile": "4m"}


class GreedyBackend:
    def __init__(self, name, profile):
        self.backend_id, self.profile = name, profile
        self.calls = []

    def predict(self, task, state, candidates):
        self.calls.append((task, state, candidates))
        # Prefer special actions so the test traverses calls and two-stage riichi.
        order = [
            "tsumo",
            "ron",
            "chankan",
            "riichi_discard",
            "ankan",
            "kakan",
            "daiminkan",
            "pon",
            "chi",
            "discard",
            "pass",
        ]

        def key(candidate):
            rank = order.index(candidate["kind"]) if candidate["kind"] in order else 99
            if task == "response" and state["actor"] != 3:
                rank = 3 if candidate["kind"] == "pass" else rank
            shanten = 0
            if candidate["kind"] == "discard":
                tiles = []
                for tile, count in state["hand"].items():
                    count -= tile == candidate["tile"]
                    tiles.extend(
                        [i for i in range(136) if from_physical(i) == tile][:count]
                    )
                shanten = calculate_shanten(tiles)
            return rank, shanten

        selected = min(candidates, key=key)
        return {"selected_action": selected}


def test_full_game_with_injected_backends_and_isolated_trackers():
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=1)
    backends = [
        GreedyBackend("old", "legacy_v1_2"),
        GreedyBackend("new", "full_state_v2"),
    ]
    agents = [
        MjaiBotAgent(p, backends[p % 2], create_adapter(backends[p % 2].profile))
        for p in range(4)
    ]
    observations = env.reset()
    forced = 0
    while not env.done():
        actions = {p: agents[p].act(o) for p, o in sorted(observations.items())}
        observations = env.step(actions)
        for p in actions:
            forced += agents[p].last_decision["forced_riichi_discard"]
            agents[p].confirm_action(accepted=True)
    assert len({id(a.tracker) for a in agents}) == 4
    assert all(b.calls for b in backends)
    assert forced > 0
    assert {e["type"] for e in env.mjai_log} >= {"reach", "reach_accepted", "chi"}
    for backend in backends:
        for _, state, _ in backend.calls:
            assert state["actor"] % 2 == (backend.backend_id == "new")
            assert not any(key in state for key in ("hands", "wall", "tehais"))


def test_http_backend_checks_contract_routes_and_null_zero():
    config = load_arena_config(ROOT / "configs/arena_selfplay.yaml")["backends"][
        "v1_3_0"
    ]
    info = json.loads(
        (ROOT / "tests/fixtures/http_contract/baseline_models.json").read_text()
    )["models"]
    requests = []

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=info)
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "task": "response",
                "selected_action": payload["legal_actions"][0],
                "model": info["models"]["response"],
            },
        )

    client = httpx.Client(base_url="http://new", transport=httpx.MockTransport(handle))
    backend = HttpPredictBackend("new", config, client=client)
    backend.predict("response", {}, [{"kind": "pass"}])
    backend.config["response"]["pass_bias"] = 0
    backend.predict("response", {}, [{"kind": "pass"}])
    assert [r["pass_bias"] for r in requests] == [None, 0]
    with pytest.raises(AdapterError, match="candidate limit"):
        backend.predict("response", {}, [{"kind": "pass"}] * 17)
    backend.close()


def test_bad_http_contract_closes_client():
    config = load_arena_config(ROOT / "configs/arena_selfplay.yaml")["backends"][
        "v1_3_0"
    ]
    info = json.loads(
        (ROOT / "tests/fixtures/http_contract/baseline_models.json").read_text()
    )["models"]
    info["models"]["turn"]["feature_schema"] = "unknown"
    client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=info)),
    )
    with pytest.raises(AdapterError, match="incompatible"):
        HttpPredictBackend("bad", config, client=client)
    assert client.is_closed


@pytest.mark.parametrize("accepted", [False, True])
def test_action_confirmation_controls_missed_ron(accepted):
    data = json.loads(
        (ROOT / "tests/fixtures/engine/directed/multi_ron.json").read_text()
    )
    obs = Observation.deserialize_from_base64(data["responses"][0]["observation"])
    backend = GreedyBackend("test", "full_state_v2")
    backend.predict = lambda task, state, candidates: {
        "selected_action": {"kind": "pass"}
    }
    agent = MjaiBotAgent(1, backend, create_adapter(backend.profile))
    agent.act(obs)
    assert not agent.tracker.temporary_furiten
    agent.confirm_action(accepted=accepted)
    assert agent.tracker.temporary_furiten is accepted


def test_failed_run_preserves_invalid_summary(tmp_path):
    from rilab.selfplay import run_single_game

    config = load_arena_config(ROOT / "configs/arena_selfplay.yaml")

    def fail(name, backend):
        raise AdapterError("test failure")

    out = tmp_path / "failed"
    with pytest.raises(AdapterError, match="test failure"):
        run_single_game(
            config, seats=["v1_3_0"] * 4, seed=1, output_dir=out, backend_factory=fail
        )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["valid"] is False
    assert "test failure" in summary["error"]
