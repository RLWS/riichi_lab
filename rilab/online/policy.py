"""Observation preparation and exact MJAI action compatibility for online sessions."""

from __future__ import annotations

from dataclasses import dataclass
import json

from riichienv import ActionType, Observation

from rilab.decision import AdapterError, PendingRiichiPlan
from rilab.mapping.action_mapping import bind_candidates
from rilab.tiles import from_physical
from rilab.tracker.state_tracker import StateTracker


def wire_action(action, obs, context):
    result = json.loads(action.to_mjai())
    result["actor"] = obs.player_id
    if action.action_type in (
        ActionType.Chi,
        ActionType.Pon,
        ActionType.Daiminkan,
        ActionType.Ron,
    ):
        result["target"] = context.source
    if action.action_type == ActionType.Tsumo:
        result["target"] = obs.player_id
    if action.action_type in (ActionType.Tsumo, ActionType.Ron):
        result["pai"] = from_physical(action.tile)
    if action.action_type == ActionType.Discard:
        # Exact physical identity proves draw/held origin, including ordinary copies.
        result["tsumogiri"] = action.tile == obs.drawn_tile
    return result


def compatible(local, remote):
    """Compare required semantics; optional fields constrain only when supplied."""
    kind = remote.get("type")
    required = {
        "dahai": ("pai",),
        "chi": ("pai", "consumed"),
        "pon": ("pai", "consumed"),
        "daiminkan": ("pai", "consumed"),
        "ankan": ("consumed",),
        "kakan": ("pai", "consumed"),
        "reach": (),
        "hora": (),
        "ryukyoku": (),
        "none": (),
    }
    if kind not in required or local["type"] != kind:
        return False
    for key in required[kind]:
        if key not in remote:
            return False
    for key in ("pai", "consumed", "actor", "target", "tsumogiri"):
        if key not in remote:
            continue
        if key == "consumed":
            if not isinstance(remote[key], list) or sorted(remote[key]) != sorted(
                local.get(key, [])
            ):
                return False
        elif remote[key] != local.get(key):
            return False
    return True


@dataclass(frozen=True)
class Prepared:
    task: str
    state_json: str
    candidates_json: str
    bindings: tuple
    wires: tuple
    hand_id: int
    ron_available: bool
    forced_index: int | None
    fallback_index: int | None
    events_json: str


class OnlinePolicy:
    """Only the session owner calls prepare/confirm; inference receives JSON snapshots."""

    def __init__(self, adapter, *, match_rounds=4):
        self.adapter = adapter
        self.match_rounds = match_rounds
        self.tracker = None
        self.pending_riichi = None
        self.player_id = None

    def clear(self):
        self.pending_riichi = None

    def prepare(self, message):
        obs = Observation.deserialize_from_base64(message["observation"])
        if self.player_id is not None and obs.player_id != self.player_id:
            raise AdapterError("observation does not match assigned online seat")
        if self.tracker is None:
            self.tracker = StateTracker(obs.player_id, match_rounds=self.match_rounds)
        if obs.player_id != self.tracker.player_id:
            raise AdapterError("online observation seat changed")
        events = list(obs.new_events())
        self.tracker.consume(events)
        self.tracker.check_observation(obs)
        legal = obs.legal_actions()
        context = self.tracker.context(legal)
        snapshot = self.tracker.snapshot(context)
        bindings = tuple(bind_candidates(obs, context, snapshot))
        wires = tuple(wire_action(b.action, obs, context) for b in bindings)
        # Validate the complete raw engine legal set, including physical copies
        # merged by candidate binding, before using any model or fallback action.
        raw_wires = [wire_action(a, obs, context) for a in legal]
        remote = message["possible_actions"]
        if (
            not isinstance(remote, list)
            or not remote
            or not all(isinstance(a, dict) for a in remote)
        ):
            raise AdapterError("invalid possible_actions")
        if any(not any(compatible(w, r) for r in remote) for w in raw_wires) or any(
            not any(compatible(w, r) for w in raw_wires) for r in remote
        ):
            raise AdapterError("local/online legal action compatibility conflict")
        forced = None
        if self.pending_riichi:
            plan = self.pending_riichi
            if plan.hand_id == context.hand_id and self.tracker.pending[obs.player_id]:
                forced = next(
                    (
                        i
                        for i, b in enumerate(bindings)
                        if b.candidate == {"kind": "discard", "tile": plan.tile}
                    ),
                    None,
                )
            if forced is None:
                self.clear()

        # Stable priority: win, pass, legal tsumogiri, deterministic legal discard.
        def priority(i):
            a = bindings[i].action
            if a.action_type in (ActionType.Tsumo, ActionType.Ron):
                return (0, i)
            if a.action_type == ActionType.Pass:
                return (1, i)
            if a.action_type == ActionType.Discard:
                return (2 if a.tile == obs.drawn_tile else 3, a.tile)
            return (4, i)

        safe = [
            i
            for i, b in enumerate(bindings)
            if b.action.action_type
            in (ActionType.Tsumo, ActionType.Ron, ActionType.Pass, ActionType.Discard)
        ]
        fallback = min(safe, key=priority) if safe else None
        return Prepared(
            context.task,
            json.dumps(self.adapter.project(snapshot)),
            json.dumps([b.candidate for b in bindings]),
            bindings,
            wires,
            context.hand_id,
            any(a.action_type == ActionType.Ron for a in legal),
            forced,
            fallback,
            json.dumps(events),
        )

    def confirm(self, prepared, index, status, actual=None):
        # Never commit an intended action before an authoritative receipt.
        if status != "accepted":
            self.clear()
            if (
                status == "defaulted"
                and actual
                and prepared.ron_available
                and actual.get("type") != "hora"
            ):
                self.tracker.confirm_missed_ron()
            return
        if index is None:
            raise AdapterError("accepted acknowledgement without a submitted action")
        binding = prepared.bindings[index]
        if prepared.ron_available and binding.action.action_type != ActionType.Ron:
            self.tracker.confirm_missed_ron()
        self.pending_riichi = (
            PendingRiichiPlan(prepared.hand_id, binding.riichi_tile)
            if binding.riichi_tile
            else None
        )
