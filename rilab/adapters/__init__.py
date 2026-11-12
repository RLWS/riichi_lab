from rilab.adapters.full_state_v2 import FullStateV2Adapter
from rilab.adapters.full_state_v3 import FullStateV3Adapter
from rilab.adapters.full_state_v2_action_biases import (
    FullStateV2ActionBiasesAdapter,
)
from rilab.adapters.legacy_v12 import LegacyV12Adapter
from rilab.decision import AdapterError


def create_adapter(profile: str):
    if profile == "full_state_v3":
        return FullStateV3Adapter()
    if profile == "full_state_v2":
        return FullStateV2Adapter()
    if profile == "full_state_v2_action_biases":
        return FullStateV2ActionBiasesAdapter()
    if profile == "legacy_v1_2":
        return LegacyV12Adapter()
    raise AdapterError(f"unsupported adapter profile: {profile}")
