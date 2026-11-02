from copy import deepcopy

from rilab.decision import AdapterError
from rilab.tiles import dora_base_index


class LegacyV12Adapter:
    profile = "legacy_v1_2"

    def project(self, state: dict) -> dict:
        # v1.2.0 _player_view_snapshot uses hundred-point units and an
        # approximation from river length, not the new actual draw counter.
        state = deepcopy(state)
        actor = state["actor"]
        own_draws = len(state["all_discards"][actor]) + int(
            sum(state["hand"].values()) % 3 == 2
        )
        if any(score % 100 for score in state["scores"]):
            raise AdapterError(
                "legacy score cannot be expressed in hundred-point units"
            )
        allowed = {
            "actor",
            "hand",
            "scores",
            "dealer",
            "round_wind",
            "seat_wind",
            "honba",
            "riichi_sticks",
            "dora_indicators",
            "all_discards",
            "all_melds",
            "own_melds",
            "riichi_declared",
            "riichi_declared_by_player",
            "riichi_turn_by_player",
            "temporary_furiten",
            "riichi_furiten",
            "last_discard_actor",
            "last_discard_tile",
            "response_target",
        }
        result = {k: v for k, v in state.items() if k in allowed}
        result["scores"] = [score // 100 for score in state["scores"]]
        result["turn_number"] = own_draws
        result["remaining_draws_estimate"] = max(0, 18 - own_draws)
        result["dora_base_indices"] = sorted(
            {dora_base_index(t) for t in state["dora_indicators"]}
        )
        # Historical v1.2.0 GameState.apply_reach deducts score but never adds
        # kyotaku. Match that trained input, rather than silently modernizing it.
        result["riichi_sticks"] -= sum(state["riichi_declared_by_player"])
        result["discard_history"] = [
            {"actor": e["actor"], "tile": e["tile"]} for e in state["discard_history"]
        ]
        return result
