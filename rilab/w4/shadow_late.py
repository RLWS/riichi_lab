"""Offline Turn comparison on the linked, late opponent-riichi opportunities."""

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

from .public import CONFIG, read, save


def compare(links, traces, cfg, root):
    import torch
    import yaml
    from rilab.w3_shadow import normalize_state

    training = Path("../train_mj").resolve()
    sys.path.insert(0, str(training))
    from mjai.serving.checkpoint import load_policy_checkpoint
    from mjai.serving.predictor import MahjongPredictor

    torch.set_num_threads(1)
    predictors = {}
    for bot, version in ((493, "130"), (521, "140")):
        models = yaml.safe_load(
            (CONFIG.parent / f"service_v{version}.yaml").read_text()
        )["models"]
        policies = {}
        for task in ("turn", "response"):
            path = training / models[task]["checkpoint"]
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != cfg["expected_model_identity"][str(bot)][task + "_sha256"]:
                raise ValueError("local checkpoint differs from frozen manifest")
            policies[task] = load_policy_checkpoint(path, task=task, device="cpu")
        predictors[bot] = MahjongPredictor(policies["turn"], policies["response"])

    rows = []
    counts = defaultdict(Counter)
    groups = defaultdict(Counter)
    for s in links:
        bot = s["bot"]
        if (
            not s.get("recent")
            or s["status"] != "unique"
            or not s.get("decision_shard")
            or s["model_identity"] != cfg["expected_model_identity"][str(bot)]
        ):
            continue
        hands = traces[s["trace_sha256"]]["hand_results"]
        with gzip.open(root / s["decision_shard"], "rt", encoding="utf8") as trace:
            for r in map(json.loads, trace):
                if (
                    r["task"] != "turn"
                    or not r["opponent_accepted"]
                    or r["state"]["turn_number"] <= 12
                    or not any(c["kind"] == "riichi_discard" for c in r["candidates"])
                ):
                    continue
                selected = {
                    b: model.predict_turn(
                        normalize_state(r["state"], model.turn_policy.feature_schema),
                        r["candidates"],
                        top_k=1,
                    ).selected_action
                    for b, model in predictors.items()
                }
                observed = r.get("prediction", {}).get("selected_action")
                consistent = selected[bot] == observed
                counts[bot]["opportunities"] += 1
                counts[bot]["prediction_reproduced"] += consistent
                ex = r.get("execution", {})
                executed = (
                    ex.get("status") == "accepted"
                    and (ex.get("actual_action") or {}).get("type") == "reach"
                )
                other = selected[521 if bot == 493 else 493]["kind"]
                counts[bot]["disagree_on_reach"] += (
                    selected[493]["kind"] == "riichi_discard"
                ) != (selected[521]["kind"] == "riichi_discard")
                rows.append(
                    dict(
                        bot=bot,
                        game_id=s["game_id"],
                        request_id=r["request_id"],
                        hand_id=r["hand_id"],
                        observed=observed,
                        predictions=selected,
                        logged_prediction_reproduced=consistent,
                        executed_reach=executed,
                    )
                )
                if executed and consistent and observed["kind"] == "riichi_discard":
                    group = groups[f"{bot}/{other}"]
                    result = hands[r["hand_index"]]
                    group["hands"] += 1
                    for key in ("deal", "win", "loss", "income"):
                        group[key] += result[key]
    return dict(counts=counts, executed_by_other_action=groups, requests=rows)


if __name__ == "__main__":
    cfg = read(CONFIG)
    root = Path(cfg["output"])
    traces = {s["trace_sha256"]: s for s in read(root / "local_sessions.json")}
    if (root / "cloud_traces.json").exists():
        traces.update(read(root / "cloud_traces.json"))
    save(
        root / "shadow_late.json",
        compare(read(root / "linked_games.json"), traces, cfg, root),
    )
