"""B0 uses real local WebSocket sockets and locked engine observations, no platform account."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from riichienv import Observation
from websockets.asyncio.server import serve

from rilab.adapters import create_adapter
from rilab.backends.http_backend import AsyncHttpPredictBackend
from rilab.online.client import connect_session
from rilab.online.policy import OnlinePolicy, compatible, wire_action
from rilab.online.session import Budget, OnlineSession
from rilab.tracker.state_tracker import StateTracker

FIXTURES = Path(__file__).parent / "fixtures/http_contract"


def request(task="turn", rid=1, grace=500, bank=500):
    source = json.loads((FIXTURES / "engine_source.json").read_text())
    encoded = source[f"{task}_observation"]
    obs = Observation.deserialize_from_base64(encoded)
    tracker = StateTracker(obs.player_id)
    tracker.consume(obs.new_events())
    context = tracker.context(obs.legal_actions())
    possible = [wire_action(a, obs, context) for a in obs.legal_actions()]
    for a in possible:
        for key in ("actor", "target", "tsumogiri"):
            a.pop(key, None)  # Server may omit optional fields.
    return {
        "type": "request_action",
        "request_id": rid,
        "observation": encoded,
        "possible_actions": possible,
        "time": {"grace_ms": grace, "bank_ms": bank, "deadline_ms": grace + bank},
    }


class Backend:
    def __init__(self, delay=0, fail=False, resistant=False):
        self.calls = 0
        self.cancelled = False
        self.delay, self.fail, self.resistant = delay, fail, resistant

    async def predict(self, task, state, candidates):
        self.calls += 1
        state.clear()  # Mutating HTTP input must not mutate the owner's snapshot.
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            if not self.resistant:
                raise
            await asyncio.sleep(0.01)
        if self.fail:
            raise RuntimeError("backend secret must not enter logs")
        return {"selected_action": candidates[0]}


async def exchange(scenario, backend=None, seat=0, **options):
    policy = OnlinePolicy(create_adapter("full_state_v2"))
    session = OnlineSession(
        policy,
        backend or Backend(),
        send_reserve_seconds=0.002,
        inbound_estimate_seconds=0.002,
        finish_timeout_seconds=0.05,
        **options,
    )

    async def handler(ws):
        assert ws.request.headers["Authorization"] == "Bearer test-platform-secret"
        await ws.send(json.dumps({"type": "start_game", "id": seat}))
        await scenario(ws, session)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        summary = await connect_session(
            f"ws://127.0.0.1:{port}", "test-platform-secret", session
        )
    return session, summary


async def send(ws, message):
    await ws.send(json.dumps(message))


async def finish(ws):
    await send(ws, {"type": "end_game"})
    await send(ws, {"type": "validation_result", "passed": True})


@pytest.mark.parametrize("task", ["turn", "response"])
def test_request_duplicate_ack_and_validation(task):
    async def scenario(ws, session):
        message = request(task)
        await send(ws, {"type": "notice", "detail": "test-platform-secret"})
        await send(ws, {"type": "tsumo", "actor": 0, "pai": "1m"})  # Trace only.
        await send(ws, message)
        await send(ws, message)
        reply = json.loads(await ws.recv())
        assert reply["request_id"] == 1
        assert any(compatible(reply, a) for a in message["possible_actions"])
        assert not any(r["kind"] == "execution" for r in session.records)
        await send(ws, {"type": "action_ack", "request_id": 1, "status": "accepted"})
        await send(ws, message)  # Duplicate after confirmation must not replay events.
        await finish(ws)

    seat = Observation.deserialize_from_base64(request(task)["observation"]).player_id
    session, summary = asyncio.run(exchange(scenario, seat=seat))
    assert summary == {
        "state": "Closed",
        "completed": True,
        "faults": [],
        "validation_result": {"type": "validation_result", "passed": True},
        "requests": 1,
        "replies": 1,
    }
    assert session.backend.calls == 1
    assert "test-platform-secret" not in json.dumps(session.records)
    assert sum(r["kind"] == "duplicate_request" for r in session.records) == 2
    assert sum(r["kind"] == "execution" for r in session.records) == 1


@pytest.mark.parametrize("failure", ["slow", "error"])
@pytest.mark.parametrize("task", ["turn", "response"])
def test_fallback_once_and_late_inference_ignored(failure, task):
    backend = Backend(
        delay=0.08 if failure == "slow" else 0, fail=failure == "error", resistant=True
    )

    async def scenario(ws, session):
        message = request(task, grace=30)
        await send(ws, message)
        reply = json.loads(await ws.recv())
        if task == "turn":
            assert reply["type"] == "dahai" and reply["tsumogiri"] is True
        else:
            assert reply["type"] == "none"
        await asyncio.sleep(
            0.04
        )  # Even cancellation-resistant inference cannot reply twice.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ws.recv(), 0.02)
        await send(ws, {"type": "action_ack", "request_id": 1, "status": "accepted"})
        await finish(ws)

    seat = Observation.deserialize_from_base64(request(task)["observation"]).player_id
    session, summary = asyncio.run(exchange(scenario, backend, seat=seat))
    assert summary["replies"] == 1 and not summary["faults"]
    assert next(r for r in session.records if r["kind"] == "reply")["fallback"]
    if failure == "slow":
        assert backend.cancelled


@pytest.mark.parametrize("status", ["rejected", "unparseable", "stale", "defaulted"])
def test_all_ack_statuses_cancel_work_and_record_actual_execution(status):
    backend = Backend(delay=0.1, resistant=True)

    async def scenario(ws, session):
        await send(ws, request())
        await asyncio.sleep(0.01)
        await send(
            ws,
            {
                "type": "action_ack",
                "request_id": 1,
                "status": status,
                "action": {"type": "none"} if status == "defaulted" else None,
            },
        )
        await asyncio.sleep(0.03)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ws.recv(), 0.02)
        await finish(ws)

    session, summary = asyncio.run(exchange(scenario, backend))
    assert summary["replies"] == 0
    execution = next(r for r in session.records if r["kind"] == "execution")
    assert execution["status"] == status
    assert execution["actual_action"] == (
        {"type": "none"} if status == "defaulted" else None
    )
    assert bool(summary["faults"]) == (status != "defaulted")


def test_expired_budget_yields_server_default():
    async def scenario(ws, session):
        await send(ws, request(grace=0, bank=0))
        await asyncio.sleep(0.01)
        await send(
            ws,
            {
                "type": "action_ack",
                "request_id": 1,
                "status": "defaulted",
                "action": {"type": "dahai", "pai": "9s"},
            },
        )
        await finish(ws)

    session, summary = asyncio.run(exchange(scenario))
    assert summary["replies"] == 0 and session.backend.calls == 0
    assert any(r["kind"] == "server_default_required" for r in session.records)


def test_disconnect_cancels_inference_no_reconnect():
    backend = Backend(delay=1)

    async def scenario(ws, session):
        await send(ws, request())
        await asyncio.sleep(0.01)
        await ws.close()

    session, summary = asyncio.run(exchange(scenario, backend))
    assert summary["state"] == "Disconnected" and not summary["completed"]
    assert summary["replies"] == 0 and backend.cancelled
    assert session.policy.pending_riichi is None


@pytest.mark.parametrize("ending", ["timeout", "close", "late_inference"])
def test_end_game_bounded_wait_and_cancel(ending):
    async def scenario(ws, session):
        if ending == "late_inference":
            await send(ws, request())
            await asyncio.sleep(0.01)
        await send(ws, {"type": "end_game"})
        if ending == "close":
            return
        await asyncio.sleep(0.09)

    session, summary = asyncio.run(
        exchange(scenario, Backend(delay=0.1, resistant=True))
    )
    assert summary["state"] == "Closed" and summary["replies"] == 0
    assert summary["completed"] == (ending == "close")
    if ending != "close":
        assert "validation_result_timeout" in summary["faults"]


@pytest.mark.parametrize("conflict", ["tile", "consumed", "state_message"])
def test_compatibility_fault_never_guesses(conflict):
    async def scenario(ws, session):
        message = request()
        if conflict == "tile":
            message["possible_actions"][0]["pai"] = "5mr"
        elif conflict == "consumed":
            message["possible_actions"].append(
                {"type": "pon", "pai": "1m", "consumed": ["1m", "1m"]}
            )
        elif conflict == "state_message":
            message = {"type": "new_rule_event", "affects_state": True}
        await send(ws, message)
        await asyncio.sleep(0.02)

    session, summary = asyncio.run(exchange(scenario))
    assert summary["faults"] and summary["replies"] == 0
    assert session.backend.calls == 0


def test_budget_dynamic_fields_and_network_allowances():
    assert Budget.read(
        {"deadline_ms": 1000, "grace_ms": 300, "bank_ms": 700},
        10,
        inbound=0.05,
        reserve=0.1,
    ) == Budget(10.15, 10.85)
    assert Budget.read(
        {"deadline_ms": 18000, "grace_ms": 100, "bank_ms": 0}, 10, inbound=0, reserve=0
    ) == Budget(10.1, 10.1)
    with pytest.raises(Exception):
        Budget.read(
            {"deadline_ms": float("nan"), "grace_ms": 1, "bank_ms": 0},
            0,
            inbound=0,
            reserve=0,
        )


def test_semantics_consumed_order_red_tiles_and_optional_fields():
    local = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "6m",
        "consumed": ["5mr", "4m"],
    }
    assert compatible(
        local, {"type": "chi", "pai": "6m", "consumed": ["4m", "5mr"], "extra": 1}
    )
    assert not compatible(local, {"type": "chi", "pai": "6m", "consumed": ["4m", "5m"]})
    assert not compatible(
        local, {"type": "chi", "pai": "6m", "consumed": ["4m", "5mr"], "target": 2}
    )
    assert not compatible(local, {"type": "chi", "pai": "6m"})


def test_async_http_contract_and_real_cancellation():
    async def check():
        models = json.loads((FIXTURES / "baseline_models.json").read_text())["models"]
        cancelled = asyncio.Event()
        payloads = []

        async def handler(req):
            if req.url.path == "/v1/models":
                return httpx.Response(200, json=models)
            payloads.append(json.loads(req.content))
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        client = httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        backend = AsyncHttpPredictBackend(
            "online",
            {
                "adapter_profile": "full_state_v2",
                "api_base": "http://test",
                "timeout_seconds": 1,
                "response": {"pass_bias": 0},
            },
            client=client,
        )
        try:
            await backend.warmup()
            task = asyncio.create_task(
                backend.predict("response", {}, [{"kind": "pass"}])
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set() and payloads[0]["pass_bias"] == 0
        finally:
            await backend.close()

    asyncio.run(check())


def fixture_riichi_pair():
    import gzip
    from copy import deepcopy
    from rilab.tiles import from_physical
    from riichienv import GameRule, RiichiEnv

    decisions = json.load(
        gzip.open(FIXTURES.parent / "engine/games/game_0003.json.gz", "rt")
    )["decisions"]
    index = next(
        i for i, d in enumerate(decisions) if d["chosen"]["action_type"] == "Riichi"
    )
    actor = decisions[index]["player_id"]
    tracker = StateTracker(actor, match_rounds=4)
    for d in decisions[:index]:
        if d["player_id"] == actor:
            tracker.consume(d["new_events"])
    second = next(d for d in decisions[index + 1 :] if d["player_id"] == actor)
    tile = from_physical(second["chosen"]["engine"]["tile"])
    # Replay recorded physical actions to obtain the unsampled second Observation.
    # This keeps both requests engine-generated instead of inventing serialized state.
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=3)
    observations = env.reset()
    for step in range(second["step"]):
        actions = {}
        for pid, obs in sorted(observations.items()):
            obs.new_events()
            recorded = next(
                d for d in decisions if d["step"] == step and d["player_id"] == pid
            )
            actions[pid] = next(
                a
                for a in obs.legal_actions()
                if a.to_dict() == recorded["chosen"]["engine"]
            )
        observations = env.step(actions)
    second = dict(second, observation=observations[actor].serialize_to_base64())

    def message(decision, tracker, rid):
        obs = Observation.deserialize_from_base64(decision["observation"])
        tracker = deepcopy(tracker)
        tracker.consume(obs.new_events())
        context = tracker.context(obs.legal_actions())
        return {
            "type": "request_action",
            "request_id": rid,
            "observation": decision["observation"],
            "possible_actions": [
                wire_action(a, obs, context) for a in obs.legal_actions()
            ],
            "time": {"grace_ms": 500, "bank_ms": 500, "deadline_ms": 1000},
        }, tracker

    first_msg, next_tracker = message(decisions[index], tracker, 1)
    second_msg, _ = message(second, next_tracker, 2)
    return actor, tracker, first_msg, second_msg, tile


def test_riichi_two_requests_one_reply_each_and_ack_gates_plan():
    actor, tracker, first, second, tile = fixture_riichi_pair()

    class RiichiBackend(Backend):
        async def predict(self, task, state, candidates):
            self.calls += 1
            return {"selected_action": {"kind": "riichi_discard", "tile": tile}}

    async def scenario(ws, session):
        session.policy.tracker = tracker
        await send(ws, first)
        reply = json.loads(await ws.recv())
        assert reply["type"] == "reach" and reply["request_id"] == 1
        assert session.policy.pending_riichi is None
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ws.recv(), 0.01)
        await send(ws, {"type": "action_ack", "request_id": 1, "status": "accepted"})
        await send(ws, second)
        reply = json.loads(await ws.recv())
        assert reply["type"] == "dahai" and reply["request_id"] == 2
        assert any(compatible(reply, action) for action in second["possible_actions"])
        await send(ws, {"type": "action_ack", "request_id": 2, "status": "accepted"})
        await finish(ws)

    session, summary = asyncio.run(exchange(scenario, RiichiBackend(), seat=actor))
    assert summary["replies"] == 2 and not summary["faults"]
    assert session.backend.calls == 1
    assert session.policy.pending_riichi is None


def test_state_event_inside_observation_fails_closed(monkeypatch):
    import rilab.online.policy as module

    message = request()
    obs = Observation.deserialize_from_base64(message["observation"])

    class Proxy:
        def __getattr__(self, name):
            return getattr(obs, name)

        def new_events(self):
            return [*obs.new_events(), json.dumps({"type": "unknown_state"})]

    class Decoder:
        @staticmethod
        def deserialize_from_base64(value):
            return Proxy()

    monkeypatch.setattr(module, "Observation", Decoder)

    async def scenario(ws, session):
        await send(ws, message)
        await asyncio.sleep(0.01)

    session, summary = asyncio.run(exchange(scenario))
    assert summary["replies"] == 0 and summary["faults"]
    assert session.backend.calls == 0


def test_log_capacity_stops_session_and_still_produces_summary():
    async def scenario(ws, session):
        await send(ws, request())
        await asyncio.sleep(0.01)

    _, summary = asyncio.run(exchange(scenario, log_limit=1))
    assert summary["faults"] and summary["replies"] == 0


def test_redaction_handles_quotes_and_nested_credentials():
    session = OnlineSession(
        OnlinePolicy(create_adapter("full_state_v2")), Backend(), secrets=('a"b\nc',)
    )
    session.record(
        "test",
        nested={"authorization": "Bearer random", "detail": 'prefix a"b\nc suffix'},
    )
    assert session.records[0]["nested"] == {
        "authorization": "[REDACTED]",
        "detail": "prefix [REDACTED] suffix",
    }


def test_fallback_win_and_post_call_discard_from_locked_engine():
    from copy import deepcopy
    from riichienv import ActionType, GameRule, RiichiEnv
    from scripts.capture_fixtures import choose_action

    def live_decisions():
        env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=3)
        observations = env.reset()
        while not env.done():
            actions = {}
            for pid, obs in sorted(observations.items()):
                events = list(obs.new_events())
                actions[pid] = choose_action(obs, pid)
                yield {
                    "player_id": pid,
                    "new_events": events,
                    "observation": obs.serialize_to_base64(),
                }
            observations = env.step(actions)

    trackers = [StateTracker(i, match_rounds=4) for i in range(4)]
    covered = set()
    for d in live_decisions():
        tracker = trackers[d["player_id"]]
        before = deepcopy(tracker)
        tracker.consume(d["new_events"])
        if "observation" not in d:
            continue
        obs = Observation.deserialize_from_base64(d["observation"])
        legal = obs.legal_actions()
        context = tracker.context(legal)
        last = json.loads(d["new_events"][-1])
        is_post_call = (
            last["type"] in ("chi", "pon") and last["actor"] == d["player_id"]
        )
        wins = [a for a in legal if a.action_type in (ActionType.Ron, ActionType.Tsumo)]
        if not wins and not is_post_call:
            continue
        policy = OnlinePolicy(create_adapter("full_state_v2"))
        policy.tracker = before
        prepared = policy.prepare(
            {
                "observation": d["observation"],
                "possible_actions": [wire_action(a, obs, context) for a in legal],
            }
        )
        fallback = prepared.wires[prepared.fallback_index]
        if wins:
            assert fallback["type"] == "hora"
            assert fallback["pai"] and fallback["target"] is not None
            covered.add("win")
        if is_post_call:
            assert obs.drawn_tile is None
            assert fallback["type"] == "dahai" and fallback["tsumogiri"] is False
            covered.add("post_call")
        if covered == {"win", "post_call"}:
            break
    assert covered == {"win", "post_call"}


def test_validation_client_warmup_independent_keys_and_persisted_logs(
    tmp_path, monkeypatch
):
    import os
    import yaml
    import rilab.online.client as module

    monkeypatch.setenv("TEST_PLATFORM_TOKEN", "platform-secret")
    monkeypatch.setenv("TEST_HTTP_KEY", "http-secret")
    models = json.loads((FIXTURES / "baseline_models.json").read_text())["models"]
    warmed = []

    async def handler(req):
        assert req.headers["X-API-Key"] == "http-secret"
        assert "platform-secret" not in str(req.headers)
        if req.url.path == "/v1/models":
            warmed.append(True)
            return httpx.Response(200, json=models)
        payload = json.loads(req.content)
        task = req.url.path.rsplit("/", 1)[-1]
        return httpx.Response(
            200,
            json={
                "task": task,
                "model": models["models"][task],
                "selected_action": payload["legal_actions"][0],
            },
        )

    def factory(name, config):
        return AsyncHttpPredictBackend(
            name,
            config,
            client=httpx.AsyncClient(
                base_url="http://test",
                headers={"X-API-Key": os.environ[config["api_key_env"]]},
                transport=httpx.MockTransport(handler),
            ),
        )

    monkeypatch.setattr(module, "AsyncHttpPredictBackend", factory)

    async def check():
        async def server_handler(ws):
            assert warmed
            assert ws.request.headers["Authorization"] == "Bearer platform-secret"
            await send(ws, {"type": "start_game", "id": 0})
            await send(ws, request())
            reply = json.loads(await ws.recv())
            assert reply["request_id"] == 1
            await send(ws, {"type": "notice", "details": "platform-secret http-secret"})
            await send(
                ws, {"type": "action_ack", "request_id": 1, "status": "accepted"}
            )
            await finish(ws)

        async with serve(server_handler, "127.0.0.1", 0) as server:
            config = {
                "online": {
                    "endpoint_validate": f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                    "token_env": "TEST_PLATFORM_TOKEN",
                    "send_reserve_seconds": 0.002,
                    "inbound_estimate_seconds": 0.002,
                    "finish_timeout_seconds": 0.1,
                },
                "backend": {
                    "adapter_profile": "full_state_v2",
                    "api_base": "http://test",
                    "api_key_env": "TEST_HTTP_KEY",
                    "timeout_seconds": 1,
                },
            }
            path = tmp_path / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            summary = await module.run_validation(path, tmp_path / "output")
            assert summary["completed"] and not summary["faults"]
            logs = (tmp_path / "output/session.jsonl").read_text(encoding="utf-8")
            assert "platform-secret" not in logs and "http-secret" not in logs
            assert (
                json.loads((tmp_path / "output/summary.json").read_text())["replies"]
                == 1
            )
            config["backend"]["api_key_env"] = "TEST_PLATFORM_TOKEN"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with pytest.raises(ValueError, match="independent"):
                await module.run_validation(path, tmp_path / "rejected")

    asyncio.run(check())


def test_hand_end_invalidates_pending_inference():
    async def scenario(ws, session):
        await send(ws, request())
        await asyncio.sleep(0.01)
        await send(ws, {"type": "end_kyoku"})
        await asyncio.sleep(0.03)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ws.recv(), 0.02)
        await finish(ws)

    _, summary = asyncio.run(exchange(scenario, Backend(delay=0.1, resistant=True)))
    assert summary["replies"] == 0
