from argparse import Namespace
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rilab.cli import cmd_probe, main


def model_response():
    return {
        "models": {
            task: {
                "task": task,
                "checkpoint": f"{task}.pt",
                "state_dim": 8173,
                "action_dim": 89,
                "candidate_limit": 16,
            }
            for task in ("turn", "response")
        }
    }


@pytest.mark.parametrize(
    "status,payload",
    [
        (401, {"error": "unauthorized"}),
        (503, {"error": "unavailable"}),
        (200, {"models": {}}),
        (200, {"models": {"turn": {"task": "response"}}}),
        (200, {"models": {task: {"task": task} for task in ("turn", "response")}}),
        (200, "not a models object"),
    ],
)
def test_probe_rejects_failed_or_invalid_models(status, payload, capsys):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(status, json=payload)

    with patch(
        "rilab.cli.httpx.Client",
        return_value=httpx.Client(transport=httpx.MockTransport(handler)),
    ):
        assert cmd_probe(Namespace(api_base="http://test", timeout=1)) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "failed" in output.err


@pytest.mark.parametrize("health", [{"status": "error"}, [], {"error": "not ready"}])
def test_probe_rejects_unhealthy_service(health):
    with patch(
        "rilab.cli.httpx.Client",
        return_value=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=health)
            )
        ),
    ):
        assert cmd_probe(Namespace(api_base="http://test", timeout=1)) == 1


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_probe_network_and_decode_failures(failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, text="not json")

    with patch(
        "rilab.cli.httpx.Client",
        return_value=httpx.Client(transport=httpx.MockTransport(handler)),
    ):
        assert cmd_probe(Namespace(api_base="http://test", timeout=1)) == 1


def test_probe_accepts_legacy_metadata_without_feature_schema(capsys):
    def handler(request):
        return httpx.Response(
            200,
            json={"status": "ok"}
            if request.url.path == "/health"
            else model_response(),
        )

    with patch(
        "rilab.cli.httpx.Client",
        return_value=httpx.Client(transport=httpx.MockTransport(handler)),
    ):
        assert cmd_probe(Namespace(api_base="http://test", timeout=1)) == 0
    assert '"models"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode,config_name,prefix",
    [
        ("ranked", "arena_ranked_v120.yaml", "ranked_v1.2.0_legacy_v1_2_r8_g100_pbdefault"),
        ("ranked", "arena_ranked_v130.yaml", "ranked_v1.3.0_full_state_v2_r8_g100_pbdefault"),
        ("validate", "arena_online.yaml", "validate_full_state_v2_r4_pbdefault"),
    ],
)
def test_online_generates_output_directory(
    mode, config_name, prefix, tmp_path, monkeypatch, capsys
):
    config = Path(__file__).resolve().parents[1] / "configs" / config_name
    monkeypatch.chdir(tmp_path)
    runner = (
        "rilab.online.ranked.run_ranked"
        if mode == "ranked"
        else "rilab.online.client.run_validation"
    )
    result = {"completed": True, "faults": [], "validation_result": {"passed": True}}
    with (
        patch(runner, new_callable=AsyncMock, return_value=result) as run,
        patch("rilab.cli.datetime") as clock,
    ):
        clock.now.return_value = datetime(2026, 11, 4, 12, 34, 56, 123456)
        assert main(["arena-online", "--mode", mode, "--config", str(config)]) == 0
    expected = str(Path("outputs") / f"{prefix}_20261104_123456_123456")
    run.assert_awaited_once_with(str(config), expected)
    output = capsys.readouterr()
    assert json.loads(output.out)["output_dir"] == expected
    assert expected in output.err
    # Only the real runner creates artifacts, after its credential checks.
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("mode", ["ranked", "validate"])
def test_online_preserves_explicit_output_directory(mode, capsys):
    runner = (
        "rilab.online.ranked.run_ranked"
        if mode == "ranked"
        else "rilab.online.client.run_validation"
    )
    result = {"completed": True, "faults": [], "validation_result": {"passed": True}}
    with patch(runner, new_callable=AsyncMock, return_value=result) as run:
        assert main(
            [
                "arena-online", "--mode", mode, "--config", "custom.yaml",
                "--output-dir", "my-run",
            ]
        ) == 0
    run.assert_awaited_once_with("custom.yaml", "my-run")
    assert json.loads(capsys.readouterr().out)["output_dir"] == "my-run"
