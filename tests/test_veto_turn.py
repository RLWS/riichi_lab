from __future__ import annotations

from copy import deepcopy

import pytest

from rilab.adapters.task_routed import TaskRoutedAdapter
from rilab.backends.veto_turn import VetoTurnBackend, opp_riichi_count
from rilab.decision import AdapterError

RIICHI = {"kind": "riichi_discard", "tile": "3p"}
DISCARD = {"kind": "discard", "tile": "1m"}


def _result(task, selected):
    return {
        "task": task,
        "model": {"task": task, "feature_schema": "fake"},
        "selected_action": deepcopy(selected),
        "raw_selected_action": deepcopy(selected),
        "selection_mode": "calibrated",
        "top_k": [deepcopy(selected)],
    }


class FakeBackend:
    results = {}
    calls = []

    def __init__(self, backend_id, config):
        self.backend_id = backend_id
        self.profile = config["adapter_profile"]
        self.config = config
        self.metadata = {
            "turn": {"task": "turn", "feature_schema": self.profile},
            "response": {"task": "response", "feature_schema": self.profile},
        }
        self.calibration = {"fake": True}
        self.requests = 0
        self.closed = False

    def predict(self, task, state, candidates):
        self.calls.append(
            {
                "backend_id": self.backend_id,
                "task": task,
                "state": deepcopy(state),
                "candidates": deepcopy(candidates),
            }
        )
        self.requests += 1
        return deepcopy(self.results[self.backend_id])

    def check_unchanged(self):
        return None

    def close(self):
        self.closed = True


@pytest.fixture
def fake_backend(monkeypatch):
    FakeBackend.results = {}
    FakeBackend.calls = []
    monkeypatch.setattr("rilab.backends.veto_turn.HttpPredictBackend", FakeBackend)
    return FakeBackend


CONFIG = {
    "adapter_profile": "veto_turn",
    "turn_backend": {
        "adapter_profile": "full_state_v3",
        "api_base": "http://127.0.0.1:1",
        "response": {},
    },
    "veto_backend": {
        "adapter_profile": "full_state_v2",
        "api_base": "http://127.0.0.1:2",
        "response": {},
    },
    "response_backend": {
        "adapter_profile": "full_state_v2",
        "api_base": "http://127.0.0.1:3",
        "response": {},
    },
}


def _state(*, opponents: int, turn_number: int, own_riichi: bool = False):
    flags = [False, False, False, False]
    flags[0] = own_riichi
    for seat in range(1, 1 + opponents):
        flags[seat] = True
    return {
        "actor": 0,
        "riichi_declared_by_player": flags,
        "turn_number": turn_number,
        "own_discards": ["1m"],
        "all_discards": [["1m"], [], [], []],
    }


def _backend(turn_action, veto_action, response_action=None):
    FakeBackend.results = {
        "T:turn": _result("turn", turn_action),
        "T:veto": _result("turn", veto_action),
        "T:response": _result("response", response_action or {"kind": "pass"}),
    }
    return VetoTurnBackend("T", deepcopy(CONFIG))


def _veto_calls():
    return [call for call in FakeBackend.calls if call["backend_id"] == "T:veto"]


def test_opp_riichi_count_excludes_own_declaration():
    assert opp_riichi_count(_state(opponents=1, turn_number=5)) == 1
    assert (
        opp_riichi_count(_state(opponents=1, turn_number=5, own_riichi=True)) == 1
    )
    assert opp_riichi_count({"actor": 0, "riichi_declared_by_player": []}) == 0


def test_no_veto_without_opponent_riichi(fake_backend):
    backend = _backend(RIICHI, DISCARD)
    result = backend.predict("turn", _state(opponents=0, turn_number=15), [RIICHI, DISCARD])
    assert result["selected_action"] == RIICHI
    assert "veto" not in result
    assert _veto_calls() == []


def test_no_veto_when_not_late(fake_backend):
    backend = _backend(RIICHI, DISCARD)
    result = backend.predict("turn", _state(opponents=2, turn_number=12), [RIICHI, DISCARD])
    assert result["selected_action"] == RIICHI
    assert "veto" not in result
    assert _veto_calls() == []


def test_no_veto_when_v140_selects_discard(fake_backend):
    backend = _backend(DISCARD, DISCARD)
    result = backend.predict("turn", _state(opponents=2, turn_number=15), [RIICHI, DISCARD])
    assert result["selected_action"] == DISCARD
    assert "veto" not in result
    assert _veto_calls() == []


def test_veto_replaces_late_riichi_with_old_discard(fake_backend):
    backend = _backend(RIICHI, DISCARD)
    state = _state(opponents=2, turn_number=13)
    candidates = [RIICHI, DISCARD]
    result = backend.predict("turn", state, candidates)
    assert result["selected_action"] == DISCARD
    veto = result["veto"]
    assert veto["applied"] is True
    assert veto["reason"] == "opp_riichi_late_riichi_to_discard"
    assert veto["new_selected"] == RIICHI
    assert veto["old_selected"] == DISCARD
    assert veto["old_model"]["task"] == "turn"
    assert veto["old_predictions"]["selected_action"] == DISCARD
    calls = _veto_calls()
    assert len(calls) == 1
    assert calls[0]["state"]["feature_schema"] == "full_state_v2"
    assert "own_discards" not in calls[0]["state"]
    assert calls[0]["candidates"] == candidates


def test_own_riichi_does_not_count_as_opponent_threat(fake_backend):
    backend = _backend(RIICHI, DISCARD)
    result = backend.predict(
        "turn",
        _state(opponents=0, turn_number=15, own_riichi=True),
        [RIICHI, DISCARD],
    )
    assert result["selected_action"] == RIICHI
    assert "veto" not in result


def test_no_veto_when_old_model_also_riichis(fake_backend):
    backend = _backend(RIICHI, RIICHI)
    result = backend.predict("turn", _state(opponents=1, turn_number=15), [RIICHI, DISCARD])
    assert result["selected_action"] == RIICHI
    assert "veto" not in result
    assert len(_veto_calls()) == 1


def test_response_routes_without_veto(fake_backend):
    backend = _backend(RIICHI, DISCARD, response_action={"kind": "pass"})
    result = backend.predict("response", _state(opponents=2, turn_number=15), [{"kind": "pass"}])
    assert result["selected_action"] == {"kind": "pass"}
    assert "veto" not in result
    assert _veto_calls() == []


def test_backend_contract_metadata_adapter_and_close(fake_backend):
    backend = _backend(RIICHI, DISCARD)
    assert backend.profile_for_task("turn") == "full_state_v3"
    assert backend.profile_for_task("response") == "full_state_v2"
    assert backend.metadata["turn"]["feature_schema"] == "full_state_v3"
    assert backend.metadata["veto_service"]["profile"] == "full_state_v2"
    assert backend.metadata["veto_policy"]["turn_number_min_exclusive"] == 12
    assert backend.calibration == {
        "turn": {"fake": True},
        "veto": {"fake": True},
        "response": {"fake": True},
    }
    adapter = backend.create_adapter()
    assert isinstance(adapter, TaskRoutedAdapter)
    assert adapter.profiles == {
        "turn": "full_state_v3",
        "response": "full_state_v2",
    }
    backend.validate_adapter(adapter)
    with pytest.raises(AdapterError):
        backend.validate_adapter(object())
    backend.predict("turn", _state(opponents=1, turn_number=15), [RIICHI, DISCARD])
    backend.predict("response", _state(opponents=1, turn_number=15), [RIICHI])
    assert backend.requests == 3
    backend.check_unchanged()
    backend.close()
    assert backend.turn_backend.closed
    assert backend.veto_backend.closed
    assert backend.response_backend.closed
