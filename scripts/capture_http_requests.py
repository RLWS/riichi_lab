"""Build A0 full_state_v2 requests from the first discard window only.

This limited fixture builder is not the A1 incremental tracker.
Run: python -m scripts.capture_http_requests
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from riichienv import Action, ActionType, GameRule, RiichiEnv


def tile_name(tile: str) -> str:
    honors = "ESWNPFC"
    if tile in honors:
        return f"{honors.index(tile) + 1}z"
    return "0" + tile[1] if tile.endswith("r") else tile


def physical_name(tile: int) -> str:
    return tile_name(json.loads(Action(ActionType.Discard, tile=tile).to_mjai())["pai"])


def initial_request(obs, *, response: bool) -> dict:
    events = [json.loads(e) for e in obs.new_events()]
    assert all(
        e["type"] in {"start_game", "start_kyoku", "tsumo", "dahai"} for e in events
    )
    start = next(e for e in events if e["type"] == "start_kyoku")
    discards = [e for e in events if e["type"] == "dahai"]
    actor = obs.player_id
    state = dict(
        feature_schema="full_state_v2",
        actor=actor,
        dealer=start["oya"],
        seat_wind=(actor - start["oya"]) % 4,
        round_wind="ESWN".index(start["bakaze"]),
        round_number=start["kyoku"] - 1,
        match_rounds=8,
        remaining_tiles=70 - sum(e["type"] == "tsumo" for e in events),
        honba=start["honba"],
        riichi_sticks=start["kyotaku"],
        turn_number=sum(e["type"] == "tsumo" and e["actor"] == actor for e in events),
        scores=list(obs.scores),
        hand=dict(Counter(physical_name(t) for t in obs.hand)),
        discard_history=[
            dict(
                actor=e["actor"],
                tile=tile_name(e["pai"]),
                tsumogiri=e["tsumogiri"],
                riichi_discard=False,
            )
            for e in discards
        ],
        all_discards=[
            [tile_name(e["pai"]) for e in discards if e["actor"] == seat]
            for seat in range(4)
        ],
        public_events=[],
        dora_indicators=[tile_name(start["dora_marker"])],
        all_melds=[[], [], [], []],
        own_melds=[],
        riichi_declared=False,
        riichi_declared_by_player=[False] * 4,
        riichi_turn_by_player=[None] * 4,
        temporary_furiten=False,
        riichi_furiten=False,
        last_discard_actor=discards[-1]["actor"] if discards else None,
        last_discard_tile=tile_name(discards[-1]["pai"]) if discards else None,
    )
    candidates = []
    if response:
        state["response_target"] = {
            "source": discards[-1]["actor"],
            "tile": tile_name(discards[-1]["pai"]),
        }
    for action in obs.legal_actions():
        raw = json.loads(action.to_mjai())
        kind = {"dahai": "discard", "none": "pass", "hora": "ron"}.get(
            raw["type"], raw["type"]
        )
        assert kind in {"discard", "pass", "chi", "pon", "daiminkan", "ron"}
        candidate = {"kind": kind}
        if "pai" in raw:
            candidate["tile"] = tile_name(raw["pai"])
        if kind not in ("discard", "pass"):
            candidate.update(
                source=discards[-1]["actor"], tile=tile_name(discards[-1]["pai"])
            )
        if "consumed" in raw:
            candidate["consumed_tiles"] = [tile_name(t) for t in raw["consumed"]]
        if candidate not in candidates:
            candidates.append(candidate)
    request = {
        "state": state,
        "legal_actions": candidates,
        "top_k": min(5, len(candidates)),
    }
    if response:
        request["pass_bias"] = None
    return request


def main() -> None:
    out = Path("tests/fixtures/http_contract")
    for seed in range(1, 50):
        env = RiichiEnv(
            game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=seed
        )
        initial = env.reset()[0]
        for discard in initial.legal_actions():
            if discard.action_type != ActionType.Discard:
                continue
            env = RiichiEnv(
                game_mode="4p-red-half", rule=GameRule.default_tenhou(), seed=seed
            )
            turn = env.reset()[0]
            responses = env.step({0: discard})
            for seat, response in sorted(responses.items()):
                if seat == 0 or not any(
                    a.action_type in (ActionType.Chi, ActionType.Pon)
                    for a in response.legal_actions()
                ):
                    continue
                turn_request = initial_request(turn, response=False)
                response_request = initial_request(response, response=True)
                for task, request in (
                    ("turn", turn_request),
                    ("response", response_request),
                ):
                    (out / f"golden_{task}_request.json").write_text(
                        json.dumps(request, indent=2), encoding="utf-8"
                    )
                source = {
                    "engine": "riichienv==0.4.11",
                    "rule": "default_tenhou",
                    "seed": seed,
                    "discard": discard.to_dict(),
                    "turn_observation": turn.serialize_to_base64(),
                    "response_observation": response.serialize_to_base64(),
                }
                (out / "engine_source.json").write_text(
                    json.dumps(source, indent=2), encoding="utf-8"
                )
                print("captured engine golden requests", seed, seat)
                return
    raise RuntimeError("No initial call window found")


if __name__ == "__main__":
    main()
