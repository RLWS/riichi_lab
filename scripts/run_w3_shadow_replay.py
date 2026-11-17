"""Run W3b same-state shadow replay on sampled W2 decisions.

Examples:
  D:\\Anaconda3\\python.exe scripts\\run_w3_shadow_replay.py --output-dir outputs\\w3_shadow_v140_20261117
  D:\\Anaconda3\\python.exe scripts\\run_w3_shadow_replay.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.backends.http_backend import HttpPredictBackend  # noqa: E402
from rilab.w2 import load_plan  # noqa: E402
from rilab.w3_shadow import (  # noqa: E402
    aggregate,
    render_report,
    replay_records,
    sample_records,
)
from scripts.run_v140_validation import start_service  # noqa: E402


def _backend_config(service: dict, api_base: str) -> dict:
    return {
        "adapter_profile": service["adapter_profile"],
        "api_base": api_base.rstrip("/"),
        "timeout_seconds": service.get("timeout_seconds", 30.0),
        "response": service["response"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs/v1_4_0_validation/w2_plan.yaml",
    )
    parser.add_argument(
        "--w2-dir", type=Path, default=ROOT / "outputs/w2_v140_20261118/batch"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/w3_shadow_v140_20261117",
    )
    parser.add_argument("--games-per-arm", type=int, default=32)
    parser.add_argument("--max-turn-per-arm", type=int, default=200)
    parser.add_argument("--max-response-per-arm", type=int, default=200)
    parser.add_argument("--v130-api-base")
    parser.add_argument("--v140-api-base")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("games_per_arm", "max_turn_per_arm", "max_response_per_arm"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    plan = load_plan(args.plan)
    records = sample_records(
        args.w2_dir,
        games_per_arm=args.games_per_arm,
        max_turn_per_arm=args.max_turn_per_arm,
        max_response_per_arm=args.max_response_per_arm,
    )
    counts = {}
    for record in records:
        key = f"{record['arm']}/{record['task']}"
        counts[key] = counts.get(key, 0) + 1
    if args.dry_run:
        print(
            json.dumps(
                {
                    "valid": True,
                    "records": len(records),
                    "counts": counts,
                    "plan": str(args.plan),
                    "w2_dir": str(args.w2_dir),
                },
                indent=2,
            )
        )
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "results.jsonl").exists():
        parser.error(f"output already contains results.jsonl: {output_dir}")
    if bool(args.v130_api_base) != bool(args.v140_api_base):
        parser.error("provide both API bases, or neither")
    services = []
    v130 = None
    v140 = None
    try:
        if args.v130_api_base and args.v140_api_base:
            api_bases = {
                "v130": args.v130_api_base,
                "v140": args.v140_api_base,
            }
        else:
            services = []
            services.append(
                start_service("v130", plan["services"]["v130"], output_dir)
            )
            services.append(
                start_service("v140", plan["services"]["v140"], output_dir)
            )
            api_bases = {service.service_id: service.api_base for service in services}
        v130 = HttpPredictBackend(
            "v130",
            _backend_config(plan["services"]["v130"], api_bases["v130"]),
        )
        v140 = HttpPredictBackend(
            "v140",
            _backend_config(plan["services"]["v140"], api_bases["v140"]),
        )
        limits = {
            task: min(
                v130.metadata[task]["candidate_limit"],
                v140.metadata[task]["candidate_limit"],
            )
            for task in ("turn", "response")
        }
        common = []
        excluded = {"turn": 0, "response": 0}
        for record in records:
            if len(record["candidates"]) <= limits[record["task"]]:
                common.append(record)
            else:
                excluded[record["task"]] += 1
        rows = replay_records(common, v130, v140)
        v130.check_unchanged()
        v140.check_unchanged()
        with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        result = aggregate(rows)
        result["sample_counts"] = counts
        result["common_support_limits"] = limits
        result["excluded_common_support"] = excluded
        result["services"] = {
            "v130": {"api_base": api_bases["v130"], "requests": v130.requests},
            "v140": {"api_base": api_bases["v140"], "requests": v140.requests},
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "report.md").write_text(
            render_report(result), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "records": len(rows),
                    "errors": result["errors"],
                    "common_support_limits": result["common_support_limits"],
                    "excluded_common_support": result["excluded_common_support"],
                    "by_arm_task": {
                        key: {
                            "n": value.get("n", 0),
                            "agree_rate": value.get("agree_rate"),
                            "source_match_rate": value.get("source_match_rate"),
                            "v130_call_rate": value.get("v130_call_rate"),
                            "v140_call_rate": value.get("v140_call_rate"),
                            "v130_call_only_rate": value.get("v130_call_only_rate"),
                            "v140_call_only_rate": value.get("v140_call_only_rate"),
                            "v130_riichi_rate": value.get("v130_riichi_rate"),
                            "v140_riichi_rate": value.get("v140_riichi_rate"),
                        }
                        for key, value in result["by_arm_task"].items()
                    },
                    "report": str(output_dir / "report.md"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        if v130 is not None:
            v130.close()
        if v140 is not None:
            v140.close()
        for service in services:
            service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
