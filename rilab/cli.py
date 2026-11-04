"""rilab CLI 入口（方案 §10）：probe / parity-check / arena-selfplay / arena-online。

A1 提供注入后端的手工四席单局；自动编排与其他命令随里程碑推进。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import httpx


def cmd_probe(args: argparse.Namespace) -> int:
    """契约探测（A0-4）：读取 /health 与 /v1/models，输出基线报告。

    hidden_size 不作为兼容性条件（方案 §3），报告按原样记录全部元信息。
    """
    base = args.api_base.rstrip("/")
    try:
        with httpx.Client(timeout=args.timeout) as client:
            health_response = client.get(f"{base}/health")
            health_response.raise_for_status()
            health = health_response.json()
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise ValueError("/health must return status=ok")
            models_response = client.get(f"{base}/v1/models")
            models_response.raise_for_status()
            models = models_response.json()
            if not isinstance(models, dict) or not isinstance(
                models.get("models"), dict
            ):
                raise ValueError("/v1/models must contain a models object")
            for task in ("turn", "response"):
                info = models["models"].get(task)
                if not isinstance(info, dict) or info.get("task") != task:
                    raise ValueError(f"missing or invalid model task: {task}")
                for field in ("state_dim", "action_dim", "candidate_limit"):
                    if type(info.get(field)) is not int or info[field] <= 0:
                        raise ValueError(f"invalid {task}.{field}")
                if (
                    not isinstance(info.get("checkpoint"), str)
                    or not info["checkpoint"]
                ):
                    raise ValueError(f"missing {task}.checkpoint")
    except (httpx.HTTPError, ValueError) as exc:
        print(f"rilab probe failed: {exc}", file=sys.stderr)
        return 1
    report = {"api_base": base, "health": health, "models": models}
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _not_implemented(name: str) -> int:
    print(
        f"rilab {name}: not implemented yet (see docs/development-plan.md)",
        file=sys.stderr,
    )
    return 2


def cmd_selfplay(args: argparse.Namespace) -> int:
    from rilab.config import load_arena_config
    from rilab.selfplay import run_single_game
    from rilab.batch import run_batch

    if not args.output_dir or ((args.seed is None) != (args.seats is None)):
        print(
            "requires --output-dir; single game additionally requires both --seed and --seats",
            file=sys.stderr,
        )
        return 2
    config = load_arena_config(args.config)
    if args.seed is None:
        summary = run_batch(config, args.output_dir)
    else:
        summary = run_single_game(
            config,
            seed=args.seed,
            seats=args.seats.split(","),
            output_dir=args.output_dir,
        )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    from rilab.parity_runner import run_parity

    if not args.output_dir:
        print("parity-check requires --output-dir (new directory)", file=sys.stderr)
        return 2
    report = run_parity(args.config, args.output_dir)
    print(json.dumps({"valid": report["valid"], "output_dir": args.output_dir}))
    return 0 if report["valid"] else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from rilab.evaluation import evaluate_run

    report = evaluate_run(
        args.run_dir,
        args.output_dir,
        iterations=args.iterations,
        rng_seed=args.bootstrap_seed,
        formal=args.formal,
    )
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "analysis_kind": report["analysis_kind"],
                "strength_verdict": report["strength_verdict"],
                "output_dir": args.output_dir,
            }
        )
    )
    return 0


def _online_output_dir(config_path: str, mode: str) -> str:
    """Build a readable, filesystem-safe directory from non-secret run parameters."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    backend = config["backend"]
    bias = backend.get("response", {}).get("pass_bias")
    parts = [mode]
    if tag := backend.get("revision", {}).get("tag"):
        parts.append(tag)
    parts.extend(
        [backend["adapter_profile"], f"r{config['online'].get('match_rounds', 4)}"]
    )
    if mode == "ranked":
        parts.append(f"g{config['ranked']['max_games']}")
    parts.append(f"pb{bias if bias is not None else 'default'}")
    # Keep even custom config values within a practical Windows path length.
    slug = "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", str(p))[:24] for p in parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(Path("outputs") / f"{slug}_{timestamp}")


def cmd_online(args: argparse.Namespace) -> int:
    import asyncio
    from rilab.online.client import run_validation

    config_path = args.config or (
        "configs/arena_ranked.yaml"
        if args.mode == "ranked"
        else "configs/arena_online.yaml"
    )

    if args.mode == "ranked":
        from rilab.online.ranked import load_ranked_config, run_ranked

        if args.dry_run:
            config = load_ranked_config(config_path)
            print(
                json.dumps(
                    {"mode": "ranked", "connected": False, "ranked": config["ranked"]},
                    ensure_ascii=False,
                )
            )
            return 0
        output_dir = args.output_dir or _online_output_dir(config_path, args.mode)
        print(f"Output directory: {output_dir}", file=sys.stderr)
        summary = asyncio.run(run_ranked(config_path, output_dir))
        summary = {**summary, "output_dir": output_dir}
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if summary["completed"] else 1
    if args.dry_run:
        raise ValueError("validate does not support --dry-run")

    output_dir = args.output_dir or _online_output_dir(config_path, args.mode)
    print(f"Output directory: {output_dir}", file=sys.stderr)
    summary = asyncio.run(run_validation(config_path, output_dir))
    summary = {**summary, "output_dir": output_dir}
    print(json.dumps(summary, ensure_ascii=False))
    return (
        0
        if (
            summary["completed"]
            and not summary["faults"]
            and summary["validation_result"]
            and summary["validation_result"]["passed"]
        )
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rilab", description="RiichiLab adapter CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="契约探测：/health 与 /v1/models 基线")
    p_probe.add_argument(
        "--api-base", default="http://127.0.0.1:8000", help="train_mj serve-api 地址"
    )
    p_probe.add_argument("--timeout", type=float, default=10.0)
    p_probe.set_defaults(func=cmd_probe)

    p_parity = sub.add_parser("parity-check", help="状态/特征对拍（A2）")
    p_parity.add_argument("--config", default="configs/parity.yaml")
    p_parity.add_argument("--output-dir", help="new parity report directory")
    p_parity.set_defaults(func=cmd_parity)

    p_selfplay = sub.add_parser(
        "arena-selfplay", help="配置日程批量运行，或显式手工四席单局"
    )
    p_selfplay.add_argument("--config", default="configs/arena_selfplay.yaml")
    p_selfplay.add_argument("--seed", type=int)
    p_selfplay.add_argument("--seats", help="four comma-separated backend IDs")
    p_selfplay.add_argument("--output-dir", help="new artifact directory")
    p_selfplay.set_defaults(func=cmd_selfplay)

    p_evaluate = sub.add_parser(
        "arena-evaluate", help="A4 配对种子簇统计与预注册正式验收"
    )
    p_evaluate.add_argument("--run-dir", required=True)
    p_evaluate.add_argument("--output-dir", required=True, help="new report directory")
    p_evaluate.add_argument("--iterations", type=int, default=10000)
    p_evaluate.add_argument("--bootstrap-seed", type=int, default=20261103)
    p_evaluate.add_argument(
        "--formal",
        action="store_true",
        help="verify preregistration, offline replay and apply sealed decision rules",
    )
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_online = sub.add_parser("arena-online", help="单次平台验证或 B2 有界排位")
    p_online.add_argument("--mode", choices=("validate", "ranked"), default="validate")
    p_online.add_argument(
        "--dry-run",
        action="store_true",
        help="ranked config check only; no network or credentials",
    )
    p_online.add_argument(
        "--config",
        help="defaults to arena_online.yaml for validate, arena_ranked.yaml for ranked",
    )
    p_online.add_argument(
        "--output-dir",
        help="new artifact directory (default: outputs/<core config parameters>_<local timestamp>)",
    )
    p_online.set_defaults(func=cmd_online)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
