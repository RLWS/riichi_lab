"""One-revision worker for complete Tenhou fixture state/candidate parity."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--fixtures-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--encode-stride", default=1, type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.append(str(repo / ".venv/Lib/site-packages"))
    from rilab.parity import check_file, check_red_chankan

    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    reports = []
    for path in sorted(args.fixtures_root.glob("*.xml")):
        result = check_file(path, args.profile, encode_stride=args.encode_stride)
        counts.update(result["counts"])
        reports.append(
            {k: v for k, v in result.items() if k not in {"record", "golden"}}
        )
        with gzip.open(
            args.output_dir / (path.stem + ".json.gz"), "wt", encoding="utf-8"
        ) as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
        print(path.name, result["counts"], flush=True)
    boundary = check_red_chankan(
        args.profile, repo / "tests/fixtures/engine/directed/red_chankan.json"
    )
    xml_valid = (
        bool(reports)
        and counts["unexplained_differences"] == 0
        and counts["known_differences"] == 0
        and counts["feature_pairs"] == counts["snapshots"]
    )
    report = {
        "profile": args.profile,
        "files": len(reports),
        "counts": dict(counts),
        "xml_valid": xml_valid,
        "valid": xml_valid and boundary["valid"],
        "encode_stride": args.encode_stride,
        "reports": reports,
        "boundary": boundary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "reports"}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
