"""Read-only W3 diagnostics for W2/A3 local logs.

Examples:
  D:\\Anaconda3\\python.exe scripts\\run_w3_diagnostics.py --output-dir outputs\\w3_v140_20261117
  D:\\Anaconda3\\python.exe scripts\\run_w3_diagnostics.py --decision-stride 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.w3 import (  # noqa: E402
    render_report,
    seed_sensitivity,
    summarize_a3,
    summarize_batch,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w2-dir", type=Path, default=ROOT / "outputs/w2_v140_20261118/batch"
    )
    parser.add_argument(
        "--w2-smoke-dir", type=Path, default=ROOT / "outputs/w2_v140_smoke/smoke"
    )
    parser.add_argument(
        "--a3-dir",
        type=Path,
        default=ROOT / "outputs/v140_engineering_8seeds_20261112_113340/batch",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/w3_v140_20261117"
    )
    parser.add_argument("--decision-stride", type=int, default=4)
    args = parser.parse_args(argv)
    if args.decision_stride <= 0:
        parser.error("--decision-stride must be positive")
    for path in (args.w2_dir, args.w2_smoke_dir, args.a3_dir):
        if not (path / "summary.json").is_file():
            parser.error(f"missing summary.json: {path}")

    w2_formal = summarize_batch(args.w2_dir, decision_stride=args.decision_stride)
    w2_smoke = summarize_batch(args.w2_smoke_dir, decision_stride=1)
    a3 = summarize_a3(args.a3_dir, decision_stride=1)
    result = {
        "schema_version": 1,
        "w2_formal": w2_formal,
        "w2_smoke": w2_smoke,
        "a3": a3,
        "seed_sensitivity": seed_sensitivity(w2_formal, w2_smoke, a3),
        "notes": [
            "Read-only diagnostics; no model calls.",
            "Decision metrics are sampled by --decision-stride.",
            "Online 200 is excluded until a sanitized request-state export exists.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(result), encoding="utf-8"
    )
    compact = {
        "w2_arms": {
            arm: {
                "games": item["games"],
                "mean_rank": item["mean_rank"],
                "fourth_rate": item["fourth_rate"],
                "win_rate": item["hands"]["win_rate"],
                "deal_rate": item["hands"]["deal_rate"],
                "riichi_rate": (
                    item["hands"]["riichi_hands"] / item["hands"]["hands"]
                    if item["hands"]["hands"]
                    else None
                ),
                "riichi_deal_rate": item["hands"]["riichi_deal_rate"],
                "open_win_rate": item["hands"]["open_win_rate"],
                "open_deal_rate": item["hands"]["open_deal_rate"],
                "turn_decisions": item["decisions"]["turn"].get("decisions", 0),
                "riichi_selected": item["decisions"]["turn"].get(
                    "selected_riichi_discard", 0
                ),
                "riichi_pass": item["decisions"]["turn"].get("riichi_pass", 0),
                "response_decisions": item["decisions"]["response"].get(
                    "decisions", 0
                ),
                "call_rate": item["decisions"]["response"].get("call_rate"),
                "pass_when_call_legal_rate": item["decisions"]["response"].get(
                    "pass_when_call_legal_rate"
                ),
            }
            for arm, item in w2_formal.items()
        },
        "a3": {
            name: {
                "games": item["games"],
                "mean_rank": item["mean_rank"],
                "fourth_rate": item["fourth_rate"],
                "win_rate": item["hands"]["win_rate"],
                "deal_rate": item["hands"]["deal_rate"],
                "riichi_rate": (
                    item["hands"]["riichi_hands"] / item["hands"]["hands"]
                    if item["hands"]["hands"]
                    else None
                ),
                "riichi_deal_rate": item["hands"]["riichi_deal_rate"],
            }
            for name, item in a3.items()
        },
        "seed_sensitivity": result["seed_sensitivity"],
        "output": str(args.output_dir),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
