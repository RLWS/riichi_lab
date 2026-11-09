"""Run one validated screening batch in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.batch import run_batch  # noqa: E402
from rilab.config import load_arena_config  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest-extra",
        help="JSON object merged into the batch manifest before the first game",
    )
    parser.add_argument(
        "--protocol-registration",
        type=Path,
        help="sealed protocol.registered.json; its hash is anchored in manifest",
    )
    args = parser.parse_args(argv)
    config = load_arena_config(args.config)
    extra = {}
    if args.manifest_extra:
        extra = json.loads(args.manifest_extra)
        if not isinstance(extra, dict):
            raise ValueError("--manifest-extra must be a JSON object")
    if args.protocol_registration:
        path = args.protocol_registration.resolve()
        extra["protocol_registration"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    summary = run_batch(config, args.output_dir, manifest_extra=extra)
    print(
        json.dumps(
            {
                "valid": summary["valid"],
                "completed_games": summary["completed_games"],
                "expected_games": summary["expected_games"],
                "replay_digest": summary.get("replay_digest"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        ),
        flush=True,
    )
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
