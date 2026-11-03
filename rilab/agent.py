"""Seat-isolated agent with injected backend and explicit action confirmation."""

from __future__ import annotations

from riichienv import ActionType

from rilab.decision import AdapterError, PendingRiichiPlan
from rilab.mapping.action_mapping import (
    bind_candidates,
    ensure_legal,
    resolve_selection,
)
from rilab.tiles import from_physical
from rilab.tracker.state_tracker import StateTracker


class MjaiBotAgent:
    def __init__(self, player_id: int, backend, adapter, *, match_rounds: int = 8):
        if backend.profile != adapter.profile:
            raise AdapterError("backend/profile injection mismatch")
        self.player_id = player_id
        self.backend = backend
        self.adapter = adapter
        self.tracker = StateTracker(player_id, match_rounds=match_rounds)
        self.pending_riichi = None
        self.unconfirmed = None
        self.last_decision = None

    def act(self, obs):
        if self.unconfirmed is not None:
            raise AdapterError("previous action must be confirmed before next decision")
        events = list(obs.new_events())
        self.tracker.consume(events)
        self.tracker.check_observation(obs)
        legal = obs.legal_actions()
        context = self.tracker.context(legal)
        if self.pending_riichi and self.pending_riichi.hand_id != context.hand_id:
            self.pending_riichi = None
        state = self.adapter.project(self.tracker.snapshot(context))
        if self.pending_riichi:
            plan = self.pending_riichi
            if not self.tracker.pending[self.player_id]:
                raise AdapterError(
                    "riichi plan no longer agrees with declaration state"
                )
            action = next(
                (
                    a
                    for a in legal
                    if a.action_type == ActionType.Discard
                    and from_physical(a.tile) == plan.tile
                ),
                None,
            )
            if action is None:
                raise AdapterError("planned riichi discard is not legal")
            selected = {"kind": "discard", "tile": plan.tile}
            result = None
            bindings = []
            self.pending_riichi = None
        else:
            bindings = bind_candidates(obs, context, self.tracker.snapshot(context))
            result = self.backend.predict(
                context.task, state, [b.candidate for b in bindings]
            )
            binding = resolve_selection(result["selected_action"], bindings)
            action, selected = binding.action, binding.candidate
            if binding.riichi_tile:
                self.pending_riichi = PendingRiichiPlan(
                    context.hand_id, binding.riichi_tile
                )
        ensure_legal(action, legal)
        missed_ron = (
            any(a.action_type == ActionType.Ron for a in legal)
            and action.action_type != ActionType.Ron
        )
        self.unconfirmed = missed_ron
        self.last_decision = {
            "player_id": self.player_id,
            "backend_id": self.backend.backend_id,
            "profile": self.adapter.profile,
            "hand_id": context.hand_id,
            "task": context.task,
            "new_events": events,
            "state": state,
            "candidates": [b.candidate for b in bindings],
            "bindings": [
                {
                    "candidate": b.candidate,
                    "engine_action": b.action.to_dict(),
                    "riichi_tile": b.riichi_tile,
                }
                for b in bindings
            ],
            "selected_action": selected,
            "engine_action": action.to_dict(),
            "prediction": result,
            "forced_riichi_discard": result is None,
        }
        return action

    def confirm_action(self, *, accepted: bool):
        if self.unconfirmed is None:
            raise AdapterError("no action awaiting confirmation")
        if accepted:
            if self.unconfirmed:
                self.tracker.confirm_missed_ron()
        else:
            self.pending_riichi = None
        self.unconfirmed = None
