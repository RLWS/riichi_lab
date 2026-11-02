"""Map serving candidates to original engine objects, never to guessed moves."""

from __future__ import annotations

import json

from riichienv import ActionType, check_riichi_candidates

from rilab.decision import AdapterError, CandidateBinding, action_key
from rilab.tiles import from_physical, canonical, base_index


def bind_candidates(obs, context, state: dict) -> list[CandidateBinding]:
    legal = obs.legal_actions()
    bindings: dict[str, CandidateBinding] = {}

    def add(candidate, action, riichi_tile=None):
        key = json.dumps(candidate, sort_keys=True)
        # Prefer tsumogiri for otherwise equivalent ordinary physical copies.
        if key not in bindings or action.tile == obs.drawn_tile:
            bindings[key] = CandidateBinding(candidate, action, riichi_tile)

    riichi = next((a for a in legal if a.action_type == ActionType.Riichi), None)
    riichi_tiles = set(check_riichi_candidates(obs.hand)) if riichi else set()
    for action in legal:
        kind = action.action_type
        if kind == ActionType.Riichi:
            continue
        candidate = {}
        if kind == ActionType.Discard:
            tile = from_physical(action.tile)
            candidate = {"kind": "discard", "tile": tile}
            if action.tile in riichi_tiles:
                add({"kind": "riichi_discard", "tile": tile}, riichi, tile)
        elif kind in (ActionType.Chi, ActionType.Pon, ActionType.Daiminkan):
            candidate = {
                "kind": {
                    ActionType.Chi: "chi",
                    ActionType.Pon: "pon",
                    ActionType.Daiminkan: "daiminkan",
                }[kind],
                "tile": from_physical(action.tile),
                "source": context.source,
                "consumed_tiles": canonical(
                    from_physical(t) for t in action.consume_tiles
                ),
            }
        elif kind == ActionType.Ankan:
            consumed = canonical(from_physical(t) for t in action.consume_tiles)
            candidate = {
                "kind": "ankan",
                "tile": consumed[0],
                "source": context.actor,
                "consumed_tiles": consumed,
            }
        elif kind == ActionType.Kakan:
            added = from_physical(action.tile)
            pon = next(
                (
                    m
                    for m in state["own_melds"]
                    if m["kind"] == "pon"
                    and base_index(m["called_tile"]) == base_index(added)
                ),
                None,
            )
            if pon is None:
                raise AdapterError("legal kakan lacks a matching public pon")
            candidate = {
                "kind": "kakan",
                "tile": pon["called_tile"],
                "source": pon["source"],
                "consumed_tiles": canonical([*pon["consumed_tiles"], added]),
            }
        elif kind == ActionType.Tsumo:
            candidate = {"kind": "tsumo"}
        elif kind == ActionType.Ron:
            candidate = {
                "kind": "chankan" if context.chankan else "ron",
                "source": context.source,
                "tile": context.tile,
            }
        elif kind == ActionType.Pass:
            candidate = {"kind": "pass"}
        elif kind == ActionType.KyushuKyuhai:
            candidate = {"kind": "kyuushu_kyuuhai"}
        else:
            raise AdapterError(f"unsupported four-player action: {kind}")
        add(candidate, action)
    if riichi and not any(b.riichi_tile for b in bindings.values()):
        raise AdapterError("riichi offered but no discard candidate can be bound")
    if not bindings:
        raise AdapterError("empty decision candidates")
    return list(bindings.values())


def resolve_selection(
    selected: dict, bindings: list[CandidateBinding]
) -> CandidateBinding:
    matches = [b for b in bindings if b.candidate == selected]
    if len(matches) != 1:
        raise AdapterError("model selected_action is not a unique submitted candidate")
    return matches[0]


def ensure_legal(action, legal) -> None:
    if action_key(action) not in {action_key(a) for a in legal}:
        raise AdapterError("bound engine action is no longer legal")
