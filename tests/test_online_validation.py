import json
from pathlib import Path

import pytest

from rilab.online.validation import distribution, review_session


def make_trace(tmp_path, *, status="accepted", fallback=False, duplicate=False):
    source = json.loads(
        (
            Path(__file__).parent / "fixtures/http_contract/engine_source.json"
        ).read_text()
    )
    records = [
        {
            "kind": "received",
            "message": {
                "type": "request_action",
                "request_id": 1,
                "time": {"grace_ms": 3000},
            },
        },
        {
            "kind": "reply",
            "request_id": 1,
            "fallback": fallback,
            "elapsed_seconds": 0.1,
        },
        {
            "kind": "received",
            "message": {
                "type": "action_ack",
                "request_id": 1,
                "status": status,
                "elapsed_ms": 250,
                "bank_consumed_ms": 0,
            },
        },
    ]
    if duplicate:
        records.append(records[1])
    records[0]["message"]["observation"] = source["turn_observation"]
    records.extend(
        [
            {"kind": "received", "message": {"type": "end_game"}},
            {
                "kind": "received",
                "message": {"type": "validation_result", "passed": True},
            },
        ]
    )
    summary = {
        "requests": 1,
        "replies": 1,
        "state": "Closed",
        "completed": True,
        "faults": [],
        "validation_result": {"type": "validation_result", "passed": True},
    }
    (tmp_path / "session.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_review_real_receipts_and_latency(tmp_path):
    make_trace(tmp_path)
    report = review_session(tmp_path)
    assert report["b1_protocol_passed"]
    assert report["observation_roundtrips_equal"] == 1
    assert report["ack_statuses"] == {"accepted": 1}
    assert report["local_reply_ms"]["p95"] == 100
    assert report["server_elapsed_ms"]["max"] == 250
    assert report["server_minus_local_ms"]["p50"] == 150
    assert not report["fallback_request_ids"] and not report["defaulted_request_ids"]
    assert report["strength_claim"] is None


@pytest.mark.parametrize("status", ["rejected", "stale", "unparseable"])
def test_successful_platform_result_cannot_hide_bad_ack(tmp_path, status):
    make_trace(tmp_path, status=status)
    report = review_session(tmp_path)
    assert not report["b1_protocol_passed"]
    assert "unsuccessful_ack" in report["audit_errors"]


def test_default_and_fallback_are_exposed_even_if_platform_passes(tmp_path):
    make_trace(tmp_path, status="defaulted", fallback=True)
    report = review_session(tmp_path)
    assert report["b1_protocol_passed"]
    assert report["fallback_request_ids"] == report["defaulted_request_ids"] == [1]


def test_duplicate_reply_invalidates_review(tmp_path):
    make_trace(tmp_path, duplicate=True)
    assert "duplicate_reply:1" in review_session(tmp_path)["audit_errors"]


def test_distribution_interpolates_without_claiming_independent_samples():
    assert distribution([]) is None
    assert distribution([100, 0])["p95"] == 95


@pytest.mark.parametrize("damage", ["missing_terminal", "invalid_observation"])
def test_missing_platform_evidence_or_serialization_failure_invalidates_review(
    tmp_path, damage
):
    make_trace(tmp_path)
    path = tmp_path / "session.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if damage == "missing_terminal":
        records = records[:-2]
    else:
        records[0]["message"]["observation"] = "invalid"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert not review_session(tmp_path)["b1_protocol_passed"]
