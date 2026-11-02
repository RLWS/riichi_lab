"""Offline XML-event conversion and state/candidate feature parity.

The training parser/encoders are development dependencies, not production
inference dependencies. Future labels are used only as post-decision feedback.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from rilab.adapters import create_adapter
from rilab.decision import DecisionContext, AdapterError
from rilab.tracker.state_tracker import StateTracker


def to_mjai(tile: str) -> str:
    if tile[1] == "z":
        return "ESWNPFC"[int(tile[0]) - 1]
    return "5" + tile[1] + "r" if tile[0] == "0" else tile


def convert_event(event, seat: int) -> dict | None:
    """Convert one normalized Tenhou event, masking other players immediately."""
    kind, actor, payload = event.event_type, event.actor, event.payload
    if kind == "init":
        seed = [int(p) for p in payload["seed"].split(",")]
        return {
            "type": "start_kyoku",
            "bakaze": "ESWN"[payload["round_wind"]],
            "kyoku": seed[0] % 4 + 1,
            "honba": seed[1],
            "kyotaku": seed[2],
            "oya": payload["oya"],
            "scores": [int(v) * 100 for v in payload["ten"].split(",")],
            "dora_marker": to_mjai(payload["dora_indicator"]),
            "tehais": [
                [to_mjai(t) for t in payload[f"hai{p}"]] if p == seat else ["?"] * 13
                for p in range(4)
            ],
        }
    if kind == "draw":
        return {
            "type": "tsumo",
            "actor": actor,
            "pai": to_mjai(payload["tile"]) if actor == seat else "?",
        }
    if kind == "discard":
        return {
            "type": "dahai",
            "actor": actor,
            "pai": to_mjai(payload["tile"]),
            "tsumogiri": payload["tsumogiri"],
        }
    if kind == "reach":
        return {
            "type": "reach" if payload["step"] == 1 else "reach_accepted",
            "actor": actor,
        }
    if kind == "dora":
        return {"type": "dora", "dora_marker": to_mjai(payload["tile"])}
    if kind == "meld":
        meld_kind = payload["kind"]
        if meld_kind not in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            raise AdapterError(f"unsupported Tenhou meld: {meld_kind}")
        consumed = list(payload["consumed_tiles"])
        result = {"type": meld_kind, "actor": actor}
        if meld_kind == "kakan":
            # Tenhou stores the original called tile separately; MJAI pai is
            # the actual added tile and consumed is the old complete pon.
            result.update(
                pai=to_mjai(consumed[-1]),
                consumed=[
                    to_mjai(payload["called_tile"]),
                    *map(to_mjai, consumed[:-1]),
                ],
            )
        elif meld_kind == "ankan":
            result["consumed"] = [to_mjai(t) for t in consumed]
        else:
            result.update(
                target=payload["source"],
                pai=to_mjai(payload["called_tile"]),
                consumed=[to_mjai(t) for t in consumed],
            )
        return result
    if kind == "agari":
        return {
            "type": "hora",
            "actor": int(payload["who"]),
            "target": int(payload["fromWho"]),
        }
    if kind == "ryuukyoku":
        return {"type": "ryukyoku"}
    if kind in {"go", "un", "taikyoku"}:
        return None
    raise AdapterError(f"unsupported Tenhou event: {kind}")


def feature_digest(values) -> str:
    flat = []
    for value in values:
        flat.extend(value if isinstance(value, list) else [value])
    return hashlib.sha256(struct.pack(f"<{len(flat)}f", *flat)).hexdigest()


def check_file(path: Path, profile: str, *, encode_stride: int = 1) -> dict:
    # Imported only inside a parity worker with one pinned train_mj revision.
    from mjai.data.tenhou.parser import TenhouParser
    from mjai.data.builders.sample_builder import (
        TurnSampleBuilder,
        ResponseSampleBuilder,
    )
    from mjai.features.state_features import encode_turn_state, encode_response_state
    from mjai.features.action_features import (
        encode_turn_candidate_actions,
        encode_response_candidate_actions,
    )
    from mjai.serving.validation import validate_and_normalize_state

    events = TenhouParser().parse_file(path)
    samples = [
        *TurnSampleBuilder().build_samples(events),
        *ResponseSampleBuilder().build_samples(events),
    ]
    by_event = {}
    for sample in samples:
        by_event.setdefault((sample.kyoku_index, sample.step_index), []).append(sample)
    trackers = [StateTracker(p) for p in range(4)]
    adapter = create_adapter(profile)
    pre_reach = {}
    counts = Counter()
    differences = []
    golden = []
    record = []

    def compare(sample, *, pre=False):
        tracker = trackers[sample.actor]
        chankan = sample.decision_type == "response" and current.event_type == "meld"
        target = sample.state.get("response_target", {})
        actual_tile = (
            current.payload["consumed_tiles"][-1] if chankan else target.get("tile")
        )
        context = DecisionContext(
            sample.decision_type,
            tracker.hand_id,
            sample.actor,
            target.get("source"),
            actual_tile,
            chankan,
        )
        if pre and sample.label_action["kind"] == "riichi_discard":
            canonical_state = pre_reach[sample.actor]
        else:
            canonical_state = tracker.snapshot(context)
        actual = adapter.project(canonical_state)
        expected = sample.state
        key = f"{path.name}:{sample.kyoku_index}:{sample.step_index}:{sample.actor}:{sample.decision_type}"
        mismatch = {
            field: {"actual": actual[field], "expected": expected[field]}
            for field in set(actual) & set(expected)
            if actual[field] != expected[field]
        }
        entry = {
            "key": key,
            "task": sample.decision_type,
            "actor": sample.actor,
            "label": sample.label_action,
            "state": actual,
        }
        counts["snapshots"] += 1
        counts[f"label_{sample.decision_type}_{sample.label_action['kind']}"] += 1
        if mismatch:
            defect = (
                "DEF-001"
                if (
                    chankan
                    and set(mismatch) == {"response_target"}
                    and actual_tile != target.get("tile")
                )
                else None
            )
            differences.append({"key": key, "defect": defect, "fields": mismatch})
            counts["known_differences" if defect else "unexplained_differences"] += 1
        elif counts["snapshots"] % encode_stride == 0:
            normalized = (
                validate_and_normalize_state(
                    sample.decision_type, actual, require_full_state=True
                )
                if profile == "full_state_v2"
                else validate_and_normalize_state(sample.decision_type, actual)
            )
            state_encoder = (
                encode_turn_state
                if sample.decision_type == "turn"
                else encode_response_state
            )
            candidate_encoder = (
                encode_turn_candidate_actions
                if sample.decision_type == "turn"
                else encode_response_candidate_actions
            )
            state_features = state_encoder(normalized)
            candidate_features = candidate_encoder(sample.legal_actions, normalized)
            if state_features != state_encoder(
                expected
            ) or candidate_features != candidate_encoder(
                sample.legal_actions, expected
            ):
                differences.append(
                    {
                        "key": key,
                        "defect": None,
                        "fields": {"encoded_features": "mismatch"},
                    }
                )
                counts["unexplained_differences"] += 1
            counts["feature_pairs"] += 1
            entry.update(
                state_feature_sha256=feature_digest(state_features),
                candidate_feature_sha256=feature_digest(candidate_features),
            )
        if not any(
            g["task"] == entry["task"] and g["label"]["kind"] == entry["label"]["kind"]
            for g in golden
        ):
            golden.append(entry)
        record.append(
            {
                "decision": {
                    "actor": sample.actor,
                    "task": sample.decision_type,
                    "label": sample.label_action,
                    "state": actual,
                }
            }
        )

    for current in events:
        related = by_event.get((current.kyoku_index, current.step_index), [])
        if current.event_type == "go":
            if current.payload["type"] & 16:
                raise AdapterError("sanma is outside the parity contract")
            for tracker in trackers:
                tracker.match_rounds = 8 if current.payload["type"] & 8 else 4
            record.append({"match_rounds": trackers[0].match_rounds})
        if current.event_type == "init":
            pre_reach.clear()
        if current.event_type == "reach" and current.payload["step"] == 1:
            pre_reach[current.actor] = trackers[current.actor].snapshot()
        for sample in related:
            if sample.decision_type == "turn":
                compare(sample, pre=True)
        for seat, tracker in enumerate(trackers):
            converted = convert_event(current, seat)
            if converted:
                tracker.apply(converted)
                record.append({"event": converted, "seat": seat})
        for sample in related:
            if sample.decision_type == "response":
                compare(sample)
        # Offline outcome feedback follows, and never enters the prediction
        # snapshot. Candidates come from the independent training oracle here.
        for sample in related:
            if (
                sample.decision_type == "response"
                and sample.label_action["kind"] == "pass"
            ):
                if any(a["kind"] in {"ron", "chankan"} for a in sample.legal_actions):
                    trackers[sample.actor].confirm_missed_ron()
                    record.append({"confirmed_missed_ron": sample.actor})
    return {
        "file": path.name,
        "profile": profile,
        "counts": dict(counts),
        "differences": differences,
        "golden": golden,
        "record": record,
    }


def check_red_chankan(profile: str, fixture: Path) -> dict:
    """Reproduce DEF-001 through the actual training sample builder.

    This is a partial directed setup, not a complete Tenhou game. Only public
    prerequisites and the responder's full hand matter to this comparison.
    """
    from mjai.data.schema.events import Event
    from mjai.data.builders.sample_builder import ResponseSampleBuilder
    from mjai.features.state_features import encode_response_state
    from mjai.serving.validation import validate_and_normalize_state
    from riichienv import Observation
    from rilab.mapping.action_mapping import bind_candidates
    from rilab.tiles import from_mjai

    case = json.loads(fixture.read_text(encoding="utf-8"))
    converted = []
    tracker = StateTracker(1)
    for step, event in enumerate(case["setup_events"]):
        tracker.apply(event)
        kind = event["type"]
        if kind == "start_kyoku":
            payload = {
                f"hai{p}": (
                    [from_mjai(t) for t in event["tehais"][p]]
                    if p in (0, 1)
                    else (["5m"] if p == 2 else [])
                )
                for p in range(4)
            }
            payload.update(
                seed="0,0,0",
                ten="250,250,250,250",
                oya=0,
                round_wind=0,
                round_number=0,
                match_rounds=8,
                dora_indicator="9s",
            )
            normalized_kind = "init"
        elif kind == "pon":
            payload = {
                "kind": "pon",
                "source": event["target"],
                "called_tile": from_mjai(event["pai"]),
                "consumed_tiles": [from_mjai(t) for t in event["consumed"]],
            }
            normalized_kind = "meld"
        else:
            payload = {
                "tile": from_mjai(event["pai"]),
                "tsumogiri": event.get("tsumogiri", False),
            }
            normalized_kind = "draw" if kind == "tsumo" else "discard"
        converted.append(
            Event(
                "red_chankan",
                0,
                step,
                normalized_kind,
                event.get("actor"),
                payload=payload,
            )
        )
    step = len(converted)
    converted.append(
        Event(
            "red_chankan",
            0,
            step,
            "meld",
            0,
            payload={
                "kind": "kakan",
                "source": 2,
                "called_tile": "5m",
                "consumed_tiles": ["5m", "5m", "0m"],
            },
        )
    )
    converted.append(
        Event(
            "red_chankan", 0, step + 1, "agari", 1, payload={"who": "1", "fromWho": "0"}
        )
    )
    samples = ResponseSampleBuilder().build_samples(converted)
    expected = next(s.state for s in samples if s.step_index == step and s.actor == 1)
    tracker.apply(case["trigger"])
    obs = Observation.deserialize_from_base64(case["responses"][0]["observation"])
    context = tracker.context(obs.legal_actions())
    actual = create_adapter(profile).project(tracker.snapshot(context))
    binding = next(
        b
        for b in bind_candidates(obs, context, tracker.snapshot(context))
        if b.candidate["kind"] == "chankan"
    )
    normalized = (
        validate_and_normalize_state("response", actual, require_full_state=True)
        if profile == "full_state_v2"
        else validate_and_normalize_state("response", actual)
    )
    a, b = encode_response_state(normalized), encode_response_state(expected)
    differences = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    return {
        "defect": "DEF-001",
        "profile": profile,
        "actual_response_target": actual["response_target"],
        "training_response_target": expected["response_target"],
        "legal_candidate": binding.candidate,
        "different_feature_indices": differences,
        "valid": actual["response_target"] == expected["response_target"]
        and not differences,
    }
