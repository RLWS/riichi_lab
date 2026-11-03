"""Review a completed B1 trace without confusing platform activation with strength."""

from collections import Counter
import base64
import hashlib
import json
from pathlib import Path

from riichienv import Observation


def distribution(values):
    if not values:
        return None
    values = sorted(values)

    def quantile(p):
        position = (len(values) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {
        "count": len(values),
        "min": values[0],
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": values[-1],
    }


def review_session(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    path = directory / "session.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    requests, replies, acks = {}, {}, {}
    terminal = []
    errors = []
    for record in records:
        if record["kind"] == "received":
            msg = record["message"]
            if msg["type"] in {"end_game", "validation_result"}:
                terminal.append(msg)
            if msg["type"] == "request_action":
                rid = msg["request_id"]
                requests.setdefault(rid, msg)
            elif msg["type"] == "action_ack":
                rid = msg["request_id"]
                if rid in acks and acks[rid] != msg:
                    errors.append(f"conflicting_ack:{rid}")
                acks[rid] = msg
        elif record["kind"] == "reply":
            rid = record["request_id"]
            if rid in replies:
                errors.append(f"duplicate_reply:{rid}")
            replies[rid] = record
    if set(requests) != set(acks):
        errors.append("request_ack_ids_differ")
    if not set(replies).issubset(requests):
        errors.append("reply_without_request")
    if summary["requests"] != len(requests) or summary["replies"] != len(replies):
        errors.append("summary_count_mismatch")
    statuses = Counter(a["status"] for a in acks.values())
    if set(statuses) - {"accepted", "defaulted"}:
        errors.append("unsuccessful_ack")
    for rid, ack in acks.items():
        if ack["status"] == "accepted" and rid not in replies:
            errors.append(f"accepted_without_reply:{rid}")
    overhead = [
        acks[rid]["elapsed_ms"] - reply["elapsed_seconds"] * 1000
        for rid, reply in replies.items()
        if rid in acks and "elapsed_ms" in acks[rid]
    ]
    result = summary.get("validation_result")
    if (
        not terminal
        or terminal[-1] != result
        or not any(m["type"] == "end_game" for m in terminal[:-1])
    ):
        errors.append("missing_or_conflicting_terminal_evidence")
    roundtrips = 0
    for rid, request in requests.items():
        try:
            original = json.loads(base64.b64decode(request["observation"]))
            obs = Observation.deserialize_from_base64(request["observation"])
            restored = json.loads(base64.b64decode(obs.serialize_to_base64()))
            if original != restored:
                errors.append(f"observation_roundtrip_difference:{rid}")
            else:
                roundtrips += 1
        except Exception:
            errors.append(f"observation_decode_error:{rid}")
    accepted = bool(
        result
        and result.get("passed") is True
        and summary["completed"]
        and not summary["faults"]
        and not errors
        and requests
    )
    return {
        "b1_protocol_passed": accepted,
        "validation_result": result,
        "session_state": summary["state"],
        "faults": summary["faults"],
        "audit_errors": errors,
        "requests": len(requests),
        "replies": len(replies),
        "observation_roundtrips_equal": roundtrips,
        "ack_statuses": dict(statuses),
        "fallback_request_ids": sorted(
            rid for rid, r in replies.items() if r["fallback"]
        ),
        "defaulted_request_ids": sorted(
            rid for rid, a in acks.items() if a["status"] == "defaulted"
        ),
        "local_reply_ms": distribution(
            [r["elapsed_seconds"] * 1000 for r in replies.values()]
        ),
        "server_elapsed_ms": distribution(
            [a["elapsed_ms"] for a in acks.values() if "elapsed_ms" in a]
        ),
        "server_minus_local_ms": distribution(overhead),
        "server_exceeded_grace": sum(
            a.get("elapsed_ms", 0) > requests[rid]["time"]["grace_ms"]
            for rid, a in acks.items()
            if rid in requests
        ),
        "bank_consumed_ms": sum(a.get("bank_consumed_ms", 0) for a in acks.values()),
        "slowest_requests": [
            {
                "request_id": rid,
                "server_elapsed_ms": ack["elapsed_ms"],
                "local_reply_ms": replies[rid]["elapsed_seconds"] * 1000
                if rid in replies
                else None,
                "time_budget": requests[rid]["time"],
                "bank_consumed_ms": ack.get("bank_consumed_ms"),
                "bank_ms": ack.get("bank_ms"),
            }
            for rid, ack in sorted(
                acks.items(),
                key=lambda item: item[1].get("elapsed_ms", -1),
                reverse=True,
            )[:5]
            if rid in requests and "elapsed_ms" in ack
        ],
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(
            (directory / "summary.json").read_bytes()
        ).hexdigest(),
        "latency_note": "server-minus-local includes transit, server scheduling and clock measurement differences; not one-way latency",
        "strength_claim": None,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = review_session(args.session_dir)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "b1_protocol_passed": report["b1_protocol_passed"],
                "requests": report["requests"],
                "ack_statuses": report["ack_statuses"],
            }
        )
    )
    return 0 if report["b1_protocol_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
