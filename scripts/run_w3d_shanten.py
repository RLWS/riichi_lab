"""W3d: add shanten/waits/value context to W3c outcome rows.

Examples:
  D:\\Anaconda3\\python.exe scripts\\run_w3d_shanten.py --output-dir outputs\\w3d_shanten_20261117
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.w3d import aggregate, load_rows, render_report  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w3c-dir", type=Path, default=ROOT / "outputs/w3c_outcomes_20261117"
    )
    parser.add_argument(
        "--w2-dir", type=Path, default=ROOT / "outputs/w2_v140_20261118/batch"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/w3d_shanten_20261117",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not (args.w3c_dir / "outcomes.jsonl").is_file():
        parser.error(f"missing outcomes.jsonl: {args.w3c_dir}")
    if not (args.w2_dir / "summary.json").is_file():
        parser.error(f"missing summary.json: {args.w2_dir}")
    if args.dry_run:
        rows = [
            json.loads(line)
            for line in (args.w3c_dir / "outcomes.jsonl").open(encoding="utf-8")
            if line.strip()
        ]
        print(
            json.dumps(
                {
                    "valid": True,
                    "w3c_rows": len(rows),
                    "w2_dir": str(args.w2_dir),
                    "w3c_dir": str(args.w3c_dir),
                },
                indent=2,
            )
        )
        return 0
    rows = load_rows(args.w3c_dir, args.w2_dir)
    result = aggregate(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "shanten.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
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
                "rows": result["rows"],
                "shaped": result["shaped"],
                "by_source": result["by_source"],
                "by_source_other": result["by_source_other"],
                "report": str(output_dir / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
