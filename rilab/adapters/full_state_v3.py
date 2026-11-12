"""v3 model inputs retain the full_state_v2 public-state wire contract.

Candidate features are encoded by the pinned train_mj v3 serving process.
"""

from rilab.adapters.full_state_v2 import FullStateV2Adapter


class FullStateV3Adapter(FullStateV2Adapter):
    profile = "full_state_v3"

    def project(self, state: dict) -> dict:
        projected = super().project(state)
        # v3 wait-quality features read this alias, as does the training builder.
        projected["own_discards"] = list(projected["all_discards"][projected["actor"]])
        return projected
