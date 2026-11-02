from argparse import Namespace
from unittest.mock import patch

import httpx
import pytest

from rilab.cli import cmd_probe


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
