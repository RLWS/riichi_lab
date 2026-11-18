"""Serving-time veto for late extra Riichi under opponent Riichi threat.

The wrapper keeps the v1.4.0 Turn model everywhere except one narrow case:

* the state has at least one opponent Riichi (``opp_riichi_count > 0``),
* ``turn_number > 12``, and
* v1.4.0 selects ``riichi_discard``.

It then asks the old v1.3.0 Turn model on a v2-normalized copy of the same
state.  If the old model selects ``discard``, that discard replaces the Riichi
declaration.  Otherwise the v1.4.0 prediction is returned unchanged.

This is W5's treatment arm.  It never changes candidates, calibration, model
weights, or the Response model.
"""

from __future__ import annotations

from copy import deepcopy

from rilab.adapters.task_routed import TaskRoutedAdapter
from rilab.backends.http_backend import HttpPredictBackend
from rilab.decision import AdapterError
from rilab.w3_shadow import normalize_state

RIICHI_KIND = "riichi_discard"
DISCARD_KIND = "discard"
DEFAULT_VETO_POLICY = {
    "opp_riichi_count_min": 1,
    "turn_number_min_exclusive": 12,
}


def opp_riichi_count(state: dict) -> int:
    """Opponents with accepted Riichi, excluding the acting seat.

    Mirrors ``rilab.w3.state_features`` so offline diagnostics and the serving
    veto use the same definition.
    """
    flags = state.get("riichi_declared_by_player") or []
    actor = state.get("actor")
    riichi = [bool(item) for item in flags]
    own = bool(riichi[actor]) if actor is not None and 0 <= actor < len(riichi) else False
    return max(0, sum(riichi) - (1 if own else 0))


def is_late(turn_number) -> bool:
    return isinstance(turn_number, (int, float)) and turn_number > 12


class VetoTurnBackend:
    profile = "veto_turn"

    def __init__(self, backend_id: str, config: dict):
        self.backend_id = backend_id
        self.config = deepcopy(config)
        self.turn_backend = None
        self.veto_backend = None
        self.response_backend = None
        try:
            policy = dict(DEFAULT_VETO_POLICY)
            policy.update(deepcopy(config.get("veto_policy") or {}))
            self.veto_policy = policy
            turn_config = deepcopy(config["turn_backend"])
            veto_config = deepcopy(config["veto_backend"])
            response_config = deepcopy(config["response_backend"])
            turn_config.setdefault("timeout_seconds", 30.0)
            veto_config.setdefault("timeout_seconds", 30.0)
            response_config.setdefault("timeout_seconds", 30.0)
            self.turn_backend = HttpPredictBackend(
                f"{backend_id}:turn", turn_config
            )
            self.veto_backend = HttpPredictBackend(
                f"{backend_id}:veto", veto_config
            )
            self.response_backend = HttpPredictBackend(
                f"{backend_id}:response", response_config
            )
            self.metadata = {
                "turn": self.turn_backend.metadata["turn"],
                "response": self.response_backend.metadata["response"],
                "turn_service": {
                    "backend_id": self.turn_backend.backend_id,
                    "profile": self.turn_backend.profile,
                    "metadata": self.turn_backend.metadata,
                    "calibration": self.turn_backend.calibration,
                },
                "veto_service": {
                    "backend_id": self.veto_backend.backend_id,
                    "profile": self.veto_backend.profile,
                    "metadata": self.veto_backend.metadata,
                    "calibration": self.veto_backend.calibration,
                },
                "response_service": {
                    "backend_id": self.response_backend.backend_id,
                    "profile": self.response_backend.profile,
                    "metadata": self.response_backend.metadata,
                    "calibration": self.response_backend.calibration,
                },
                "veto_policy": deepcopy(self.veto_policy),
            }
            self.calibration = {
                "turn": self.turn_backend.calibration,
                "veto": self.veto_backend.calibration,
                "response": self.response_backend.calibration,
            }
        except BaseException:
            self.close()
            raise

    @property
    def requests(self) -> int:
        return (
            (self.turn_backend.requests if self.turn_backend else 0)
            + (self.veto_backend.requests if self.veto_backend else 0)
            + (self.response_backend.requests if self.response_backend else 0)
        )

    def profile_for_task(self, task: str) -> str:
        if task == "turn" and self.turn_backend is not None:
            return self.turn_backend.profile
        if task == "response" and self.response_backend is not None:
            return self.response_backend.profile
        raise AdapterError(f"unsupported task for backend: {task}")

    def create_adapter(self):
        return TaskRoutedAdapter(
            self.profile_for_task("turn"),
            self.profile_for_task("response"),
        )

    def validate_adapter(self, adapter) -> None:
        if not isinstance(adapter, TaskRoutedAdapter):
            raise AdapterError("veto turn backend requires TaskRoutedAdapter")
        expected = {
            "turn": self.profile_for_task("turn"),
            "response": self.profile_for_task("response"),
        }
        if adapter.profiles != expected:
            raise AdapterError(
                f"veto turn adapter profiles differ from backend: "
                f"{adapter.profiles} != {expected}"
            )

    def _in_veto_scope(self, state: dict) -> bool:
        minimum = int(self.veto_policy["opp_riichi_count_min"])
        threshold = self.veto_policy["turn_number_min_exclusive"]
        turn_number = state.get("turn_number")
        return (
            opp_riichi_count(state) >= minimum
            and isinstance(turn_number, (int, float))
            and turn_number > threshold
        )

    def predict(self, task: str, state: dict, candidates: list[dict]) -> dict:
        if task == "response" and self.response_backend is not None:
            return self.response_backend.predict(task, state, candidates)
        if task != "turn" or self.turn_backend is None:
            raise AdapterError(f"unsupported task for backend: {task}")
        result = self.turn_backend.predict(task, state, candidates)
        selected = result.get("selected_action")
        if not isinstance(selected, dict) or selected.get("kind") != RIICHI_KIND:
            return result
        if not self._in_veto_scope(state):
            return result
        old_result = self.veto_backend.predict(
            "turn", normalize_state(state, "full_state_v2"), candidates
        )
        old_selected = old_result.get("selected_action")
        if not isinstance(old_selected, dict) or old_selected.get("kind") != DISCARD_KIND:
            return result
        return {
            **result,
            "selected_action": old_selected,
            "veto": {
                "applied": True,
                "reason": "opp_riichi_late_riichi_to_discard",
                "new_selected": selected,
                "old_selected": old_selected,
                "old_model": self.veto_backend.metadata["turn"],
                "old_predictions": old_result,
            },
        }

    def check_unchanged(self) -> None:
        if self.turn_backend is not None:
            self.turn_backend.check_unchanged()
        if self.veto_backend is not None:
            self.veto_backend.check_unchanged()
        if self.response_backend is not None:
            self.response_backend.check_unchanged()

    def close(self) -> None:
        if self.turn_backend is not None:
            self.turn_backend.close()
        if self.veto_backend is not None:
            self.veto_backend.close()
        if self.response_backend is not None:
            self.response_backend.close()
