"""Isolated development server for a single exported train_mj revision."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import torch
    import uvicorn
    from mjai.api.app import create_app
    from mjai.utils.config import load_resolved_yaml

    torch.set_num_threads(1)
    config = load_resolved_yaml(args.source_root / "configs/api_local.yaml")
    config["device"] = "cpu"
    for task in ("turn", "response"):
        config["models"][task]["checkpoint"] = str(
            args.weights_root / config["models"][task]["checkpoint"]
        )
    config["api"].update(debug_log_requests=False, debug_log_responses=False)
    config["security"] = {"allow_without_api_key": True}
    app = create_app(config=config)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
