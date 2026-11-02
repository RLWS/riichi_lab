"""Per-seat visible state. Hidden hands and future events are never stored."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

from riichienv import ActionType

from rilab.decision import AdapterError, DecisionContext
from rilab.tiles import from_mjai, from_physical, base_index, canonical


class StateTracker:
    def __init__(self, player_id: int, *, match_rounds: int = 8):
        if player_id not in range(4):
            raise AdapterError("only four-player seats are supported")
        self.player_id = player_id
        self.match_rounds = match_rounds
        self.hand_id = 0
        self.initialized = False
        self.last_event = None
        self.pre_kan = None
        self.kan_target = None

    def consume(self, events) -> None:
        for raw in events:
            self.apply(json.loads(raw) if isinstance(raw, str) else raw)

    def _remove(self, tile: str) -> None:
        if self.hand[tile] <= 0:
            raise AdapterError(f"seat {self.player_id}: consumed tile missing: {tile}")
        self.hand[tile] -= 1
        if self.hand[tile] == 0:
            del self.hand[tile]

    def _public(self, kind: str, **fields) -> None:
        self.public_events.append(
            {"kind": kind, "discard_count": len(self.discard_history), **fields}
        )

    def apply(self, event: dict) -> None:
        kind = event["type"]
        actor = event.get("actor")
        if kind == "start_game":
            if "id" in event and event["id"] != self.player_id:
                raise AdapterError("start_game seat mismatch")
            return
        if kind == "start_kyoku":
            self.hand_id += 1
            self.initialized = True
            self.hand = Counter(from_mjai(t) for t in event["tehais"][self.player_id])
            self.dealer = event["oya"]
            self.round_wind = "ESWN".index(event["bakaze"])
            self.round_number = event["kyoku"] - 1
            self.scores = list(event["scores"])
            self.honba = event["honba"]
            self.riichi_sticks = event["kyotaku"]
            self.remaining_tiles = 70
            self.turn_number = 0
            self.all_discards = [[], [], [], []]
            self.all_melds = [[], [], [], []]
            self.discard_history = []
            self.public_events = []
            self.dora_indicators = [from_mjai(event["dora_marker"])]
            self.riichi = [False] * 4
            self.pending = [False] * 4
            self.riichi_turns = [None] * 4
            self.last_discard_actor = self.last_discard_tile = None
            self.temporary_furiten = self.riichi_furiten = False
            self.pre_kan = self.kan_target = None
        elif not self.initialized:
            raise AdapterError(f"event before start_kyoku: {kind}")
        elif kind == "tsumo":
            self.remaining_tiles -= 1
            if self.remaining_tiles < 0:
                raise AdapterError("draw count below zero")
            if actor == self.player_id:
                self.hand[from_mjai(event["pai"])] += 1
                self.turn_number += 1
                self.temporary_furiten = False
        elif kind == "dahai":
            tile = from_mjai(event["pai"])
            if actor == self.player_id:
                self._remove(tile)
            self.all_discards[actor].append(tile)
            self.discard_history.append(
                {
                    "actor": actor,
                    "tile": tile,
                    "tsumogiri": event.get("tsumogiri"),
                    "riichi_discard": self.pending[actor],
                }
            )
            self.last_discard_actor, self.last_discard_tile = actor, tile
        elif kind == "reach":
            self.pending[actor] = True
            self._public("reach1", actor=actor)
        elif kind == "reach_accepted":
            if not self.pending[actor] or self.riichi[actor]:
                raise AdapterError("unexpected reach acceptance")
            self._public("reach2", actor=actor)
            self.pending[actor] = False
            self.riichi[actor] = True
            self.riichi_turns[actor] = len(self.all_discards[actor])
            self.scores[actor] -= 1000
            self.riichi_sticks += 1
        elif kind in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            consumed = canonical(from_mjai(t) for t in event["consumed"])
            if kind == "kakan":
                self.pre_kan = self.snapshot()
                added = from_mjai(event["pai"])
                self.kan_target = {"source": actor, "tile": added}
                pon = next(
                    (
                        m
                        for m in self.all_melds[actor]
                        if m["kind"] == "pon"
                        and base_index(m["called_tile"]) == base_index(added)
                    ),
                    None,
                )
                if pon is None:
                    raise AdapterError("kakan without a public pon")
                if actor == self.player_id:
                    self._remove(added)
                self._public(
                    kind, actor=actor, source=pon["source"], tile=pon["called_tile"]
                )
                pon["kind"] = "kakan"
                pon["consumed_tiles"] = canonical([*pon["consumed_tiles"], added])
            else:
                tile = consumed[0] if kind == "ankan" else from_mjai(event["pai"])
                source = actor if kind == "ankan" else event["target"]
                if kind != "ankan" and (
                    self.last_discard_actor,
                    self.last_discard_tile,
                ) != (source, tile):
                    raise AdapterError("meld does not match last discard")
                if actor == self.player_id:
                    for t in consumed:
                        self._remove(t)
                    if kind != "ankan":
                        self.temporary_furiten = False
                self._public(kind, actor=actor, source=source, tile=tile)
                self.all_melds[actor].append(
                    {
                        "kind": kind,
                        "source": source,
                        "called_tile": tile,
                        "consumed_tiles": consumed,
                    }
                )
                if kind != "ankan":
                    self.last_discard_actor = self.last_discard_tile = None
        elif kind == "dora":
            tile = from_mjai(event["dora_marker"])
            self.dora_indicators.append(tile)
            self._public("dora", tile=tile)
        elif kind == "hora":
            if "deltas" in event:
                self.scores = [
                    s + d for s, d in zip(self.scores, event["deltas"], strict=True)
                ]
        elif kind == "ryukyoku":
            if "deltas" in event:
                self.scores = [
                    s + d for s, d in zip(self.scores, event["deltas"], strict=True)
                ]
        elif kind not in {"end_kyoku", "end_game"}:
            raise AdapterError(f"unsupported state event: {kind}")
        self.last_event = deepcopy(event)

    def check_observation(self, obs) -> None:
        if obs.player_id != self.player_id:
            raise AdapterError("observation seat mismatch")
        if self.hand != Counter(from_physical(t) for t in obs.hand):
            raise AdapterError(f"seat {self.player_id}: hand diverged from engine")
        if self.scores != list(obs.scores) or self.dora_indicators != [
            from_physical(t) for t in obs.dora_indicators
        ]:
            raise AdapterError("scores or dora diverged from engine")

    def context(self, legal) -> DecisionContext:
        turn_types = {
            ActionType.Discard,
            ActionType.Riichi,
            ActionType.Tsumo,
            ActionType.Ankan,
            ActionType.Kakan,
            ActionType.KyushuKyuhai,
        }
        is_turn = any(a.action_type in turn_types for a in legal)
        if is_turn:
            return DecisionContext("turn", self.hand_id, self.player_id)
        chankan = self.last_event and self.last_event["type"] in ("kakan", "ankan")
        target = (
            self.kan_target
            if chankan
            else {"source": self.last_discard_actor, "tile": self.last_discard_tile}
        )
        if not target or target["source"] is None or target["tile"] is None:
            raise AdapterError("response without a visible target")
        return DecisionContext(
            "response",
            self.hand_id,
            self.player_id,
            target["source"],
            target["tile"],
            bool(chankan),
        )

    def snapshot(self, context: DecisionContext | None = None) -> dict:
        if not self.initialized:
            raise AdapterError("snapshot before initialization")
        if context and context.chankan:
            if self.pre_kan is None:
                raise AdapterError("missing pre-kan snapshot")
            state = deepcopy(self.pre_kan)
        else:
            state = dict(
                actor=self.player_id,
                hand=dict(self.hand),
                dealer=self.dealer,
                round_wind=self.round_wind,
                round_number=self.round_number,
                match_rounds=self.match_rounds,
                seat_wind=(self.player_id - self.dealer) % 4,
                scores=list(self.scores),
                honba=self.honba,
                riichi_sticks=self.riichi_sticks,
                remaining_tiles=self.remaining_tiles,
                turn_number=self.turn_number,
                all_discards=deepcopy(self.all_discards),
                discard_history=deepcopy(self.discard_history),
                all_melds=deepcopy(self.all_melds),
                own_melds=deepcopy(self.all_melds[self.player_id]),
                dora_indicators=list(self.dora_indicators),
                public_events=deepcopy(self.public_events),
                riichi_declared=self.riichi[self.player_id],
                riichi_declared_by_player=list(self.riichi),
                riichi_turn_by_player=list(self.riichi_turns),
                temporary_furiten=self.temporary_furiten,
                riichi_furiten=self.riichi_furiten,
                last_discard_actor=self.last_discard_actor,
                last_discard_tile=self.last_discard_tile,
            )
        if context and context.task == "response":
            state["response_target"] = {"source": context.source, "tile": context.tile}
        return state

    def confirm_missed_ron(self) -> None:
        self.temporary_furiten = True
        if self.riichi[self.player_id]:
            self.riichi_furiten = True
