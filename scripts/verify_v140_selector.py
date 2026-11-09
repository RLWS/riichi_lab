"""Deterministic logits-level checks for the v1.4.0 response selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    import torch
    from mjai.serving.action_adapter import (
        select_response_action,
        _apply_no_yaku_call_penalty,
    )

    biases_b = {"ron": 0.0, "kan": 0.4, "pon": 1.4, "chi": 1.6}
    cases = []

    def candidate(kind, **extra):
        return {"kind": kind, **extra}

    def run_case(name, logits, actions, expected_kind, *, biases=None, penalty=0.0,
                 no_yaku_indices=()):
        features = torch.zeros((len(actions), 87), dtype=torch.float32)
        for index in no_yaku_indices:
            features[index, 86] = 1.0
        selection = select_response_action(
            torch.tensor(logits, dtype=torch.float32),
            actions,
            action_biases=biases or biases_b,
            no_yaku_call_penalty=penalty,
            candidate_features=features,
        )
        actual = selection.selected_action
        passed = actual.get("kind") == expected_kind
        cases.append(
            {
                "name": name,
                "passed": passed,
                "expected_kind": expected_kind,
                "selected_kind": actual.get("kind"),
                "selected_index": selection.selected_index,
                "raw_index": selection.raw_index,
                "raw_kind": selection.raw_action.get("kind"),
                "logits": logits,
                "actions": actions,
                "action_biases": selection.action_biases,
                "no_yaku_call_penalty": selection.no_yaku_call_penalty,
                "no_yaku_indices": list(no_yaku_indices),
            }
        )
        return selection

    # The calibrated best non-pass (ron) must win even though chi has the raw max.
    run_case(
        "calibrate_then_select_B_ron_beats_raw_chi",
        [0.0, 1.5, 0.2],
        [candidate("pass"), candidate("chi", tile="3p"), candidate("ron", tile="3p")],
        "ron",
    )

    # ron/chankan share the ron bias; kan only applies to daiminkan.
    run_case(
        "chankan_uses_ron_bias",
        [0.0, 1.5, 0.2],
        [
            candidate("pass"),
            candidate("chi", tile="3p"),
            candidate("chankan", tile="3p"),
        ],
        "chankan",
    )
    run_case(
        "daiminkan_uses_kan_bias",
        [0.0, 1.5, 0.5],
        [
            candidate("pass"),
            candidate("chi", tile="3p"),
            candidate("daiminkan", tile="3p"),
        ],
        "daiminkan",
    )

    # Exact calibrated tie prefers pass; other exact ties keep input order.
    run_case(
        "exact_tie_prefers_pass",
        [0.0, 0.0],
        [candidate("pass"), candidate("ron", tile="3p")],
        "pass",
    )
    run_case(
        "exact_non_pass_tie_keeps_input_order",
        [1.0, 1.0],
        [candidate("chi", tile="2p"), candidate("chi", tile="5p")],
        "chi",
    )

    # Only pass and no pass are both defined boundaries.
    run_case("only_pass", [0.0], [candidate("pass")], "pass")
    run_case(
        "no_pass_uses_calibrated_scores",
        [1.5, 0.2],
        [candidate("chi", tile="3p"), candidate("ron", tile="3p")],
        "ron",
    )

    # ron=0 is not a forced win: a higher pass still wins.
    run_case(
        "ron_zero_does_not_force_ron",
        [0.5, 0.2],
        [candidate("pass"), candidate("ron", tile="3p")],
        "pass",
    )

    # No-yaku penalty applies only to chi/pon/daiminkan, never ron/chankan.
    penalty_actions = [
        candidate("pass"),
        candidate("chi", tile="3p"),
        candidate("ron", tile="3p"),
    ]
    run_case(
        "no_yaku_penalty_does_not_penalize_ron",
        [0.0, 1.5, 0.2],
        penalty_actions,
        "ron",
        penalty=0.3,
        no_yaku_indices=(1, 2),
    )
    features = torch.zeros((3, 87), dtype=torch.float32)
    features[1, 86] = 1.0
    features[2, 86] = 1.0
    adjusted = _apply_no_yaku_call_penalty(
        torch.tensor([0.0, 1.5, 0.2], dtype=torch.float32),
        penalty_actions,
        no_yaku_call_penalty=0.3,
        candidate_features=features,
    )
    cases.append(
        {
            "name": "no_yaku_penalty_values",
            "passed": abs(float(adjusted[1]) - 1.2) < 1e-6
            and abs(float(adjusted[2]) - 0.2) < 1e-6
            and abs(float(adjusted[0]) - 0.0) < 1e-6,
            "adjusted": adjusted.tolist(),
            "expected": [0.0, 1.2, 0.2],
        }
    )

    run_case(
        "no_yaku_penalty_does_not_penalize_chankan",
        [0.0, 1.5, 0.2],
        [
            candidate("pass"),
            candidate("pon", tile="3p"),
            candidate("chankan", tile="3p"),
        ],
        "chankan",
        penalty=0.3,
        no_yaku_indices=(1, 2),
    )
    run_case(
        "no_yaku_penalty_can_force_pass_on_daiminkan",
        [0.0, 0.6],
        [candidate("pass"), candidate("daiminkan", tile="3p")],
        "pass",
        penalty=0.3,
        no_yaku_indices=(1,),
    )

    # pass itself is not reduced by the penalty, even when it is a no-yaku shape.
    run_case(
        "no_yaku_penalty_does_not_penalize_pass",
        [0.0, 1.5],
        [candidate("pass"), candidate("chi", tile="3p")],
        "pass",
        penalty=10.0,
        no_yaku_indices=(0, 1),
    )

    payload = {
        "valid": all(case["passed"] for case in cases),
        "source_root": str(source),
        "cases": cases,
    }
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
