"""Same full_state_v2 tensor layout with the v1.4.0 calibration contract."""

from rilab.adapters.full_state_v2 import FullStateV2Adapter


class FullStateV2ActionBiasesAdapter(FullStateV2Adapter):
    profile = "full_state_v2_action_biases"
