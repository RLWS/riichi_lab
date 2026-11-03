"""Manually launch one fixed-budget strength batch; never start training."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rilab.config import load_arena_config  # noqa: E402
from rilab.strength import validate_plan  # noqa: E402
from scripts.run_a1_integration import main as run_integration  # noqa: E402
import yaml  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mj-root", type=Path, default=ROOT.parent / "train_mj")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/arena_strength.yaml"
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/strength_protocol.yaml"
    )
    parser.add_argument(
        "--timeout", type=int, default=86400, help="whole batch timeout, seconds"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate plan only; no services or games",
    )
    args = parser.parse_args(argv)
    config = load_arena_config(args.config)
    plan = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    schedule = validate_plan(plan, config)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    print(
        f"Fixed budget: {len(config['engine']['seeds'])} seed clusters, {len(schedule)} half-games, one run.",
        flush=True,
    )
    print(f"Artifacts: {args.output_dir.resolve()}", flush=True)
    if args.dry_run:
        print("Plan valid. No services, games or output directories created.")
        return
    run_integration(
        [
            "--train-mj-root",
            str(args.train_mj_root),
            "--output-dir",
            str(args.output_dir),
            "--config",
            str(args.config),
            "--batch",
            "--batch-runs",
            "1",
            "--batch-timeout",
            str(args.timeout),
            "--strength-protocol",
            str(args.protocol),
        ]
    )


if __name__ == "__main__":
    main()
