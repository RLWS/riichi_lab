"""B2 finite batches over local WebSockets; never use a real ranked account."""

import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.server import serve
import yaml

from rilab.cli import main
from rilab.online.ranked import load_ranked_config, run_ranked
from test_online import Backend, request, send

ROOT = Path(__file__).resolve().parents[1]


class RankedBackend(Backend):
    profile = "full_state_v2"
    metadata = {"test": "immutable model"}

    def __init__(self, *, change_at=None, warmup_delay=0, **kwargs):
        super().__init__(**kwargs)
        self.checks = 0
        self.change_at = change_at
        self.warmup_delay = warmup_delay
        self.closed = False

    async def warmup(self):
        await asyncio.sleep(self.warmup_delay)

    async def check_unchanged(self):
        self.checks += 1
        if self.checks == self.change_at:
            raise RuntimeError("model metadata changed")

    async def close(self):
        self.closed = True


def config_file(tmp_path, endpoint, *, change=None):
    config = yaml.safe_load(
        (ROOT / "configs/arena_ranked.yaml").read_text(encoding="utf-8")
    )
    config["online"].update(
        endpoint_ranked=endpoint,
        token_env="TEST_RANKED_TOKEN",
        send_reserve_seconds=0.002,
        inbound_estimate_seconds=0.002,
    )
    config["backend"].pop("api_key_env")
    config["ranked"].update(
        max_games=3,
        max_total_seconds=10,
        session_timeout_seconds=2,
        matchmaking_timeout_seconds=1,
    )
    config["ranked"]["stop"].update(min_bank_ms=100, min_deadline_margin_ms=100)
    if change:
        change(config)
    path = tmp_path / "ranked.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


async def game(
    ws, *, status="accepted", elapsed=20, bank=500, grace=500, missing_timing=False
):
    await send(ws, {"type": "start_game", "id": 0})
    await send(ws, request(grace=grace))
    if status != "defaulted":
        reply = json.loads(await ws.recv())
        assert reply["request_id"] == 1
    ack = {
        "type": "action_ack",
        "request_id": 1,
        "status": status,
        "elapsed_ms": elapsed,
        "bank_ms": bank,
        "bank_consumed_ms": 0,
    }
    if missing_timing:
        ack.pop("elapsed_ms")
    if status == "defaulted":
        ack["action"] = {"type": "dahai", "pai": "9s"}
    await send(ws, ack)
    await send(ws, {"type": "end_game", "scores": [25000] * 4})
    await ws.wait_closed()


async def batch(tmp_path, scenario, backend=None, change=None):
    connections = []
    backend = backend or RankedBackend()

    async def handler(ws):
        assert ws.request.path == "/ws/ranked"
        assert ws.request.headers["Authorization"] == "Bearer ranked-test-secret"
        connections.append(len(connections) + 1)
        await scenario(ws)

    async with serve(handler, "127.0.0.1", 0) as server:
        path = config_file(
            tmp_path,
            f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws/ranked",
            change=change,
        )
        result = await run_ranked(
            path, tmp_path / "batch", backend_factory=lambda *args: backend
        )
    return result, connections, backend


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("TEST_RANKED_TOKEN", "ranked-test-secret")


def test_fixed_budget_and_isolated_games_with_complete_logs(tmp_path):
    result, connections, backend = asyncio.run(batch(tmp_path, game))
    assert result["completed"] and result["status"] == "completed_budget"
    assert (
        result["attempted"] == result["games_started"] == result["games_completed"] == 3
    )
    assert len(connections) == 3 and backend.checks == 6 and backend.closed
    manifest = json.loads((tmp_path / "batch/manifest.json").read_text())
    assert manifest["config"]["ranked"]["max_games"] == 3
    for index in range(1, 4):
        directory = tmp_path / "batch" / f"game_{index:04d}"
        logs = (directory / "session.jsonl").read_text(encoding="utf-8")
        assert "ranked-test-secret" not in logs
        records = [json.loads(line) for line in logs.splitlines()]
        prepared = next(r for r in records if r["kind"] == "prepared")
        assert prepared["bindings"] and prepared["state"] and prepared["candidates"]
        assert prepared["task"] == "turn" and prepared["new_events"]
        assert prepared["hand_id"] == 1
        report = json.loads((directory / "review.json").read_text())
        assert report["protocol_passed"] and not report["b1_protocol_passed"]
        assert report["mode"] == "ranked" and report["validation_result"] is None
        assert report["requests"] == report["replies"] == 1


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("reject", "session_or_protocol_failure"),
        ("default", "defaulted_limit"),
        ("latency", "server_latency_limit"),
        ("bank", "bank_limit"),
        ("margin", "deadline_margin_limit"),
        ("timing", "missing_or_invalid_ack_timing"),
        ("disconnect", "session_or_protocol_failure"),
        ("compatibility", "session_or_protocol_failure"),
    ],
)
def test_fault_or_guardrail_prevents_next_connection(tmp_path, failure, reason):
    async def scenario(ws):
        if failure == "disconnect":
            await send(ws, {"type": "start_game", "id": 0})
            await ws.close()
        elif failure == "compatibility":
            await send(ws, {"type": "start_game", "id": 0})
            msg = request()
            msg["possible_actions"] = [{"type": "none"}]
            await send(ws, msg)
            await ws.wait_closed()
        else:
            await game(
                ws,
                status={"reject": "rejected", "default": "defaulted"}.get(
                    failure, "accepted"
                ),
                elapsed=3000
                if failure == "latency"
                else 950
                if failure == "margin"
                else 20,
                bank=50 if failure == "bank" else 500,
                missing_timing=failure == "timing",
            )

    result, connections, _ = asyncio.run(batch(tmp_path, scenario))
    assert not result["completed"] and len(connections) == result["attempted"] == 1
    assert reason in result["stop_reasons"]
    assert (tmp_path / "batch/game_0001/session.jsonl").is_file()


def test_inference_failure_stops_even_if_fallback_allowance_is_nonzero(tmp_path):
    result, connections, _ = asyncio.run(
        batch(
            tmp_path,
            game,
            RankedBackend(fail=True),
            change=lambda c: c["ranked"]["stop"].update(max_fallbacks=10),
        )
    )
    assert len(connections) == 1 and "inference_failure" in result["stop_reasons"]
    assert result["fallbacks"] == 1


def test_fallback_counts_accumulate_across_games(tmp_path):
    async def scenario(ws):
        await game(ws, grace=30)

    result, connections, _ = asyncio.run(
        batch(
            tmp_path,
            scenario,
            RankedBackend(delay=0.1),
            change=lambda c: c["ranked"]["stop"].update(max_fallbacks=1),
        )
    )
    assert len(connections) == 2 and result["fallbacks"] == 2
    assert result["stop_reasons"] == ["fallback_limit"]


@pytest.mark.parametrize("change_at", [1, 2, 3])
def test_model_change_stops_without_another_game(tmp_path, change_at):
    result, connections, backend = asyncio.run(
        batch(tmp_path, game, RankedBackend(change_at=change_at))
    )
    assert result["status"] == "failed" and not result["completed"]
    assert len(connections) == (0 if change_at == 1 else 1)
    assert backend.closed


def test_disk_failure_stops_before_next_game(tmp_path, monkeypatch):
    import rilab.online.ranked as module

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(module, "persist_session", fail)
    result, connections, _ = asyncio.run(batch(tmp_path, game))
    assert result["status"] == "failed" and len(connections) == 1
    assert result["games_started"] == 1
    assert result["stop_reasons"] == ["OSError"]
    assert (
        json.loads((tmp_path / "batch/summary.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize("limit", ["matchmaking", "session", "total"])
def test_time_limits_prevent_infinite_wait_and_new_games(tmp_path, limit):
    async def scenario(ws):
        if limit == "session":
            await send(ws, {"type": "start_game", "id": 0})
        else:
            await send(ws, {"type": "queue_status", "position": 1})
        await ws.wait_closed()

    def change(c):
        c["ranked"][
            {
                "matchmaking": "matchmaking_timeout_seconds",
                "session": "session_timeout_seconds",
                "total": "max_total_seconds",
            }[limit]
        ] = 0.03

    result, connections, _ = asyncio.run(batch(tmp_path, scenario, change=change))
    assert len(connections) <= 1 and not result["completed"]
    if not connections:
        assert limit == "total" and result["attempted"] == 0
        return
    log = (tmp_path / "batch/game_0001/session.jsonl").read_text()
    assert (
        "matchmaking_timeout" if limit == "matchmaking" else "session_time_limit"
    ) in log


def test_cancel_preserves_current_session_and_never_reconnects(tmp_path):
    async def check():
        ready = asyncio.Event()
        backend = RankedBackend()
        count = []

        async def handler(ws):
            count.append(1)
            await send(ws, {"type": "start_game", "id": 0})
            ready.set()
            await ws.wait_closed()

        async with serve(handler, "127.0.0.1", 0) as server:
            path = config_file(
                tmp_path,
                f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws/ranked",
            )
            task = asyncio.create_task(
                run_ranked(
                    path, tmp_path / "batch", backend_factory=lambda *args: backend
                )
            )
            await ready.wait()
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(count) == 1 and backend.closed
        summary = json.loads((tmp_path / "batch/summary.json").read_text())
        assert summary["status"] == "cancelled" and not summary["completed"]
        assert (tmp_path / "batch/game_0001/session.jsonl").is_file()

    asyncio.run(check())


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_games", 0),
        ("max_games", True),
        ("max_total_seconds", float("inf")),
        ("session_timeout_seconds", -1),
    ],
)
def test_invalid_limits_rejected_before_network(tmp_path, field, value):
    path = config_file(
        tmp_path,
        "ws://localhost/ws/ranked",
        change=lambda c: c["ranked"].update({field: value}),
    )
    with pytest.raises(ValueError):
        load_ranked_config(path)


def test_explicit_thresholds_and_independent_credentials(tmp_path):
    path = config_file(
        tmp_path,
        "ws://localhost/ws/ranked",
        change=lambda c: c["ranked"]["stop"].pop("min_bank_ms"),
    )
    with pytest.raises(ValueError, match="complete"):
        load_ranked_config(path)
    path = config_file(
        tmp_path,
        "ws://localhost/ws/ranked",
        change=lambda c: c["backend"].update(api_key_env="TEST_RANKED_TOKEN"),
    )
    with pytest.raises(ValueError, match="independent"):
        load_ranked_config(path)


def test_dry_run_needs_no_token_service_or_output(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RIICHI_DEV_BOT_TOKEN", raising=False)
    assert (
        main(
            [
                "arena-online",
                "--mode",
                "ranked",
                "--config",
                str(ROOT / "configs/arena_ranked.yaml"),
                "--dry-run",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["connected"] is False and result["ranked"]["max_games"] == 1
    assert list(tmp_path.iterdir()) == []


def test_guardrail_boundaries_are_inclusive_and_budget_is_not_extended(tmp_path):
    async def scenario(ws):
        await game(ws, elapsed=20, bank=500)

    def change(c):
        c["ranked"].update(max_games=1)
        c["ranked"]["stop"].update(
            max_server_elapsed_ms=20, min_bank_ms=500, min_deadline_margin_ms=980
        )

    result, connections, _ = asyncio.run(batch(tmp_path, scenario, change=change))
    assert result["completed"] and len(connections) == 1


def test_total_budget_includes_model_startup_and_never_connects_if_expired(tmp_path):
    result, connections, backend = asyncio.run(
        batch(
            tmp_path,
            game,
            RankedBackend(warmup_delay=0.5),
            change=lambda c: c["ranked"].update(max_total_seconds=0.03),
        )
    )
    assert result["status"] == "failed" and result["attempted"] == 0
    assert not connections and backend.closed


def test_existing_output_is_not_overwritten_or_resumed(tmp_path):
    path = config_file(tmp_path, "ws://localhost/ws/ranked")
    output = tmp_path / "batch"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("preserved")

    async def check():
        with pytest.raises(FileExistsError):
            await run_ranked(path, output)

    asyncio.run(check())
    assert marker.read_text() == "preserved"
