"""Isolated development server for a single exported train_mj revision."""

import argparse
from pathlib import Path
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--service-config",
        type=Path,
        help="external service YAML; default is source-root/configs/api_local.yaml",
    )
    parser.add_argument(
        "--resolved-config-out",
        type=Path,
        help="write the exact resolved service configuration used by this process",
    )
    args = parser.parse_args()
    source = args.source_root.resolve()
    weights = args.weights_root.resolve()
    sys.path.insert(0, str(source))
    import torch
    import uvicorn
    from mjai.api.app import create_app
    from mjai.utils.config import load_resolved_yaml

    torch.set_num_threads(1)
    config_path = (
        args.service_config.resolve()
        if args.service_config
        else source / "configs/api_local.yaml"
    )
    config = load_resolved_yaml(config_path)
    config["device"] = "cpu"
    config.setdefault("api", {})["host"] = "127.0.0.1"
    config["api"]["port"] = args.port
    for task in ("turn", "response"):
        checkpoint = Path(str(config["models"][task]["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = weights / checkpoint
        config["models"][task]["checkpoint"] = str(checkpoint.resolve())
    config["api"].update(debug_log_requests=False, debug_log_responses=False)
    config["security"] = {"allow_without_api_key": True}
    if args.resolved_config_out:
        target = args.resolved_config_out.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    app = create_app(config=config)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
