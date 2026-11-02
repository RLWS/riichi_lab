"""rilab CLI 入口（方案 §10）：probe / parity-check / arena-selfplay / arena-online。

A0 只注册命令并实现 probe（契约探测）；其余命令随里程碑推进实现。
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def cmd_probe(args: argparse.Namespace) -> int:
    """契约探测（A0-4）：读取 /health 与 /v1/models，输出基线报告。

    hidden_size 不作为兼容性条件（方案 §3），报告按原样记录全部元信息。
    """
    base = args.api_base.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base}/health").json()
        models = client.get(f"{base}/v1/models").json()
    report = {"api_base": base, "health": health, "models": models}
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _not_implemented(name: str) -> int:
    print(f"rilab {name}: not implemented yet (see docs/development-plan.md)", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rilab", description="RiichiLab adapter CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="契约探测：/health 与 /v1/models 基线")
    p_probe.add_argument("--api-base", default="http://127.0.0.1:8000", help="train_mj serve-api 地址")
    p_probe.add_argument("--timeout", type=float, default=10.0)
    p_probe.set_defaults(func=cmd_probe)

    p_parity = sub.add_parser("parity-check", help="状态/特征对拍（A2）")
    p_parity.add_argument("--config", default="configs/parity.yaml")
    p_parity.set_defaults(func=lambda a: _not_implemented("parity-check"))

    p_selfplay = sub.add_parser("arena-selfplay", help="批量自对弈（A3）")
    p_selfplay.add_argument("--config", default="configs/arena_selfplay.yaml")
    p_selfplay.set_defaults(func=lambda a: _not_implemented("arena-selfplay"))

    p_online = sub.add_parser("arena-online", help="riichi.dev 在线对局（B1）")
    p_online.add_argument("--config", default="configs/arena_online.yaml")
    p_online.set_defaults(func=lambda a: _not_implemented("arena-online"))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
