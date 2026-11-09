"""Offline replay audit for one screening group; never calls a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.artifacts import audit_batch, write_json  # noqa: E402
from rilab.formal import replay_recorded_games  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    games = audit_batch(run)
    result = replay_recorded_games(run, manifest, games, workers=args.workers)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    return 0 if result.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
