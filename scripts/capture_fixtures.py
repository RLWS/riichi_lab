"""A0-3 黄金夹具采集（方案 §2、开发计划 A0-3）。

一次性脚本：用定向策略（贪心鸣牌/立直/和牌）驱动锁定的 RiichiEnv，
在每次决策点固化：每席 new_events 增量、合法动作（MJAI 形式）、实际执行动作、
Observation base64 序列化往返校验；终局后保存整局 mjai_log 与结果。

不做的事：不用随机运行代替罕见边界覆盖——采集后的 index.json 如实记录
各类边界的命中情况，未命中的边界登记为 uncovered，留待定向构造（A0 补充）。

用法：
    uv run python scripts/capture_fixtures.py --seeds 1-40 --out tests/fixtures/engine
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import riichienv
from riichienv import Action, ActionType, GameRule, RiichiEnv

HONOR_TILES = {"1z", "2z", "3z", "4z", "5z", "6z", "7z"}


_ACTION_TYPE_NAMES = (
    "Discard", "Chi", "Pon", "Daiminkan", "Ankan", "Kakan",
    "Riichi", "Ron", "Tsumo", "Pass", "Kita", "KyushuKyuhai",
)
# PyO3 枚举无 .name，str() 为大写变体（如 "DISCARD"）；映射回 CamelCase 供策略分支使用
_VARIANT_TO_NAME = {
    str(getattr(ActionType, n)).split(".")[-1]: n for n in _ACTION_TYPE_NAMES
}


def type_name(action: Action) -> str:
    variant = str(action.action_type).split(".")[-1]
    return _VARIANT_TO_NAME.get(variant, variant)


def tile_str(action: Action) -> str | None:
    """Action -> mjai 牌字符串；tile 为 None 时返回 None。"""
    mjai = action.to_mjai()
    if isinstance(mjai, dict):
        return mjai.get("pai") or mjai.get("tile")
    return getattr(mjai, "pai", None) or getattr(mjai, "tile", None)


def is_terminal_or_honor(tile: str | None) -> bool:
    if tile is None:
        return False
    if tile in HONOR_TILES:
        return True
    return tile[0] in "19" and tile[1] in "mps"


def choose_action(obs, player_id: int) -> Action:
    """定向策略（按席位分工，最大化边界事件命中，不追求棋力）。

    - 0 号位"立直追求者"：只和牌/立直/向听数贪心弃牌，不鸣牌 -> 命中 reach/reach_accepted/hora。
    - 其余"贪心鸣牌位"：和牌 > 杠 > 碰/吃 > 幺九处理 -> 命中 chi/pon/kakan/daiminkan。
    """
    legal = obs.legal_actions()
    by_type: dict[str, list[Action]] = {}
    for a in legal:
        by_type.setdefault(type_name(a), []).append(a)

    # 和牌（自摸/荣和）所有席位最优先
    for name in ("Ron", "Tsumo"):
        if by_type.get(name):
            return by_type[name][0]

    if player_id in (0, 1, 2):
        # 立直追求者（0-2 号位）：不鸣牌，杠只暗杠，一旦可立直立即宣言
        if by_type.get("Riichi"):
            return by_type["Riichi"][0]
        if by_type.get("Ankan"):
            return by_type["Ankan"][0]
        discards = by_type.get("Discard", [])
        if discards:
            return _shanten_greedy_discard(obs, discards)
    else:
        # 贪心鸣牌位
        for name in ("Kakan", "Daiminkan", "Ankan"):
            if by_type.get(name):
                return by_type[name][0]
        for name in ("Pon", "Chi"):
            if by_type.get(name):
                return by_type[name][0]
        discards = by_type.get("Discard", [])
        if discards:
            drawn = obs.drawn_tile
            for a in discards:
                if drawn is not None and a.tile == drawn and is_terminal_or_honor(tile_str(a)):
                    return a
            isolated = [a for a in discards if is_terminal_or_honor(tile_str(a))]
            return isolated[0] if isolated else discards[0]

    if by_type.get("Pass"):
        return by_type["Pass"][0]
    return legal[0]


def _shanten_greedy_discard(obs, discards: list[Action]) -> Action:
    """向听数贪心弃牌：最小化弃牌后向听数；平手时优先摸切（自然产出两种 tsumogiri 口径）。"""
    from riichienv import calculate_shanten

    hand = list(obs.hand)
    drawn = obs.drawn_tile
    best: tuple[int, int, Action] | None = None  # (shanten, tsumogiri_rank, action)
    for a in discards:
        if a.tile is None:
            continue
        remaining = list(hand)
        if a.tile in remaining:
            remaining.remove(a.tile)
        try:
            s = calculate_shanten(remaining)
        except Exception:  # noqa: BLE001 - 非常规手牌（如国士）回退到首个合法弃牌
            s = 8
        tsumogiri_rank = 0 if (drawn is not None and a.tile == drawn) else 1
        key = (s, tsumogiri_rank, a.tile if a.tile is not None else 0)
        if best is None or key < (best[0], best[1], best[2].tile if best[2].tile is not None else 0):
            best = (s, tsumogiri_rank, a)
    return best[2] if best else discards[0]


def action_to_json(action: Action) -> dict:
    mjai = action.to_mjai()
    if isinstance(mjai, dict):
        out = dict(mjai)
    else:
        out = {}
    out.setdefault("type", None)
    out["action_type"] = type_name(action)
    return out


def capture_game(seed: int) -> dict:
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=seed)
    obs_dict = env.reset()
    decisions: list[dict] = []
    step = 0
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            new_events = list(obs.new_events())
            legal = obs.legal_actions()
            chosen = choose_action(obs, pid)
            # 序列化往返：合法动作数量必须一致（方案 §2 夹具要求）
            b64 = obs.serialize_to_base64()
            from riichienv import Observation

            obs2 = Observation.deserialize_from_base64(b64)
            roundtrip_ok = len(obs2.legal_actions()) == len(legal)
            decisions.append(
                {
                    "step": step,
                    "player_id": pid,
                    "new_events": new_events,
                    "legal_actions": [action_to_json(a) for a in legal],
                    "chosen": action_to_json(chosen),
                    "serialize_roundtrip_legal_count_ok": roundtrip_ok,
                }
            )
            actions[pid] = chosen
        obs_dict = env.step(actions)
        step += 1
        if step > 10000:  # 安全阀：异常循环保护
            raise RuntimeError(f"seed={seed} exceeded 10000 steps")
    log = env.mjai_log
    return {
        "seed": seed,
        "decisions": decisions,
        "mjai_log": [e if isinstance(e, str) else json.dumps(e) for e in log],
        "scores": list(env.scores()),
        "ranks": list(env.ranks()),
    }


def analyze_events(mjai_log: list[str]) -> dict:
    stats: dict[str, int] = {}
    dahai_windows: list[list[dict]] = []
    current_window: list[dict] = []
    events: list[dict] = []
    for raw in mjai_log:
        try:
            ev = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        kind = ev.get("type", "?")
        stats[f"count_{kind}"] = stats.get(f"count_{kind}", 0) + 1
        events.append(ev)
        if kind == "dahai":
            if current_window:
                dahai_windows.append(current_window)
            current_window = [ev]
        elif kind == "hora":
            current_window.append(ev)
    if current_window:
        dahai_windows.append(current_window)

    tsumogiri_tiles: dict[str, set[str]] = {}
    for raw in mjai_log:
        try:
            ev = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if ev.get("type") == "dahai":
            pai = ev.get("pai", "?")
            flag = "tsumogiri" if ev.get("tsumogiri") else "tedashi"
            tsumogiri_tiles.setdefault(pai, set()).add(flag)

    # 吃碰大明杠后同 actor 立即弃牌（post_call_discard）
    post_call_discard = 0
    for prev, cur in zip(events, events[1:]):
        if prev.get("type") in ("chi", "pon", "daiminkan") and cur.get("type") == "dahai":
            if prev.get("actor") is not None and prev.get("actor") == cur.get("actor"):
                post_call_discard += 1

    return {
        "event_counts": stats,
        "multi_ron_windows": sum(1 for w in dahai_windows if sum(1 for e in w if e.get("type") == "hora") >= 2),
        "tiles_with_both_tsumogiri_and_tedashi": sorted(
            p for p, flags in tsumogiri_tiles.items() if flags == {"tsumogiri", "tedashi"}
        ),
        "post_call_discard_count": post_call_discard,
    }


def game_categories(record: dict, analysis: dict) -> list[str]:
    counts = analysis["event_counts"]
    cats: list[str] = []
    if counts.get("count_start_kyoku", 0) > 1:
        cats.append("cross_kyoku")
    if counts.get("count_ryukyoku", 0) > 0:
        cats.append("ryukyoku_end")
    if counts.get("count_hora", 0) > 0:
        cats.append("hora_end")
    if counts.get("count_reach", 0) > 0:
        cats.append("riichi_declare")
    if counts.get("count_reach_accepted", 0) > 0:
        cats.append("riichi_accepted")
    if counts.get("count_chi", 0) > 0:
        cats.append("chi")
    if counts.get("count_pon", 0) > 0:
        cats.append("pon")
    if counts.get("count_daiminkan", 0) > 0:
        cats.append("daiminkan")
    if counts.get("count_ankan", 0) > 0:
        cats.append("ankan")
    if counts.get("count_kakan", 0) > 0:
        cats.append("kakan")
    if counts.get("count_dora", 0) > 0:
        cats.append("dora_reveal")
    if analysis["multi_ron_windows"] > 0:
        cats.append("multi_ron")
    if analysis.get("post_call_discard_count", 0) > 0:
        cats.append("post_call_discard")
    if analysis["tiles_with_both_tsumogiri_and_tedashi"]:
        cats.append("same_tile_tsumogiri_and_tedashi")
    return cats


REQUIRED_CATEGORIES = [
    "cross_kyoku",
    "ryukyoku_end",
    "hora_end",
    "riichi_declare",
    "riichi_accepted",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_reveal",
    "multi_ron",
    "same_tile_tsumogiri_and_tedashi",
    "chankan",  # 抢杠：事件侧无法从 mjai_log 直接区分，先登记待定向构造
    "post_call_discard",  # 吃碰后弃牌：由决策序列分析（A0-3 补充判定器）
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1-40", help="种子范围，如 1-40 或 3,7,11")
    parser.add_argument("--out", default="tests/fixtures/engine")
    parser.add_argument("--keep-going", action="store_true", help="单局失败不中断（默认失败即终止）")
    args = parser.parse_args()

    if "-" in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-", 1))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]

    out_dir = Path(args.out)
    games_dir = out_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "engine": {
            "package": "riichienv",
            "version": riichienv.__version__ if hasattr(riichienv, "__version__") else "0.4.11",
            "game_mode": "4p-red-half",
            "rule": "default_tenhou",
            "python": platform.python_version(),
        },
        "games": [],
        "coverage": {},
    }

    for seed in seeds:
        try:
            record = capture_game(seed)
        except Exception as exc:  # noqa: BLE001 - 夹具采集失败必须显式记录
            if not args.keep_going:
                raise
            index["games"].append({"seed": seed, "error": repr(exc)})
            continue
        analysis = analyze_events(record["mjai_log"])
        cats = game_categories(record, analysis)
        path = games_dir / f"game_{seed:04d}.json.gz"
        payload = {k: v for k, v in record.items() if k != "seed"}
        payload["analysis"] = analysis
        payload["categories"] = cats
        payload["engine"] = index["engine"]
        # 紧凑 JSON + gzip：黄金夹具入库，避免仓库膨胀（方案 §7 日志治理原则）
        import gzip

        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        roundtrip_ok = all(d["serialize_roundtrip_legal_count_ok"] for d in record["decisions"])
        index["games"].append(
            {"seed": seed, "file": path.name, "categories": cats, "decisions": len(record["decisions"]), "serialize_roundtrip_ok": roundtrip_ok}
        )

    covered: dict[str, list[int]] = {}
    for g in index["games"]:
        for c in g.get("categories", []):
            covered.setdefault(c, []).append(g["seed"])
    index["coverage"] = {
        "required": REQUIRED_CATEGORIES,
        "covered": {c: covered.get(c, []) for c in REQUIRED_CATEGORIES},
        "uncovered": [c for c in REQUIRED_CATEGORIES if not covered.get(c)],
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    print(json.dumps(index["coverage"], ensure_ascii=False, indent=1))
    print(f"games={len(index['games'])} index -> {out_dir / 'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
