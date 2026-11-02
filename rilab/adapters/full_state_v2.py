from copy import deepcopy

from rilab.decision import AdapterError


class FullStateV2Adapter:
    profile = "full_state_v2"

    def project(self, state: dict) -> dict:
        state = deepcopy(state)
        if len(state["public_events"]) > 32 or len(state["discard_history"]) > 192:
            raise AdapterError("state exceeds full_state_v2 history capacity")
        state["feature_schema"] = "full_state_v2"
        return state
