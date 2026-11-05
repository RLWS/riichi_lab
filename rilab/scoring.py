"""Separate observed version benefit, uncertainty and a deployment gate."""

import math


def rank_score(mean_rank):
    """Linear four-player placement utility: first=100, fourth=0; not Elo."""
    if (
        type(mean_rank) not in (int, float)
        or not math.isfinite(mean_rank)
        or not 1 <= mean_rank <= 4
    ):
        raise ValueError("four-player mean rank must be in [1, 4]")
    return 100 * (4 - mean_rank) / 3


def rank_scorecard(
    rank, last_place=None, *, minimum_gain=None, guardrail=None, scope="unspecified"
):
    if minimum_gain is not None and (
        type(minimum_gain) not in (int, float)
        or not math.isfinite(minimum_gain)
        or minimum_gain <= 0
    ):
        raise ValueError("minimum gain must be positive")
    if guardrail is not None and (
        type(guardrail) not in (int, float)
        or not math.isfinite(guardrail)
        or not 0 <= guardrail <= 1
    ):
        raise ValueError("invalid last-place guardrail")
    candidate, control = rank["candidate"]["value"], rank["control"]["value"]
    gain = control - candidate
    lo, hi = rank["ci95"]
    if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
        raise ValueError("invalid rank confidence interval")
    result = {
        "scope": scope,
        "observed_direction": "improvement"
        if gain > 0
        else "regression"
        if gain < 0
        else "same",
        "observed_rank_gain": gain,
        "rank_gain_ci95": [-hi, -lo],
        "direction_evidence": "improvement_supported"
        if hi < 0
        else "regression_supported"
        if lo > 0
        else "uncertain",
        "linear_rank_score": {
            "candidate": rank_score(candidate),
            "control": rank_score(control),
            "difference": gain * 100 / 3,
            "difference_ci95": [-hi * 100 / 3, -lo * 100 / 3],
            "definition": "100*(4-mean_rank)/3; first=100, second=66.67, third=33.33, fourth=0",
            "interpretation": "linear presentation of the same rank data, not platform rating/Elo or additional evidence",
        },
        "practical_gain": {
            "minimum_registered_gain": minimum_gain,
            "supported": hi <= -minimum_gain if minimum_gain is not None else None,
        },
        "last_place_guardrail": {
            "registered_maximum_increase": guardrail,
            "passed": last_place["ci95"][1] <= guardrail
            if last_place is not None and guardrail is not None
            else None,
        },
        "interpretation": "An inconclusive gate does not erase an observed benefit or demonstrate equality. Evidence applies only to the stated opponent population and deployment.",
    }
    return result
