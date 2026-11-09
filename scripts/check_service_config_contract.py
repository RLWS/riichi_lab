"""Verify that a frozen source accepts or rejects a given service configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--service-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expect-error",
        default=None,
        help="require an exception containing this text; otherwise require success",
    )
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    weights = args.weights_root.resolve()
    sys.path.insert(0, str(source))
    import yaml
    from mjai.serving.predictor import build_predictor_from_config
    from mjai.utils.config import load_resolved_yaml

    config = load_resolved_yaml(args.service_config.resolve())
    config["device"] = "cpu"
    for task in ("turn", "response"):
        checkpoint = Path(str(config["models"][task]["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = weights / checkpoint
        config["models"][task]["checkpoint"] = str(checkpoint.resolve())
    result = {"source_root": str(source), "service_config": str(args.service_config)}
    try:
        predictor = build_predictor_from_config(config, base_dir=weights)
        result["ok"] = True
        result["error"] = None
        result["models"] = predictor.model_metadata()
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["models"] = None
    expected = args.expect_error
    passed = (
        (expected in (result["error"] or ""))
        if expected
        else result["ok"] is True
    )
    result["passed"] = passed
    result["expected_error"] = expected
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "error": result["error"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
