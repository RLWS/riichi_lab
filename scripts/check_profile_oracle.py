"""A1 profile smoke parity against a fixed train_mj snapshot builder.

Uses full engine logs only as an offline reference. No full-information state
is sent to a production backend. This is NOT A2 XML/candidate/furiten parity.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--profile", required=True, choices=["legacy_v1_2", "full_state_v2"]
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(repo))
    # Run in the train_mj dev Python; the engine is an optional test dependency.
    sys.path.append(str(repo / ".venv/Lib/site-packages"))
    from mjai.rules.state import GameState
    from mjai.data.builders.sample_builder import _player_view_snapshot
    from mjai.features.state_features import encode_turn_state
    from mjai.serving.validation import validate_and_normalize_state
    from rilab.adapters import create_adapter
    from rilab.tiles import from_mjai, canonical
    from rilab.tracker.state_tracker import StateTracker

    adapter = create_adapter(args.profile)
    checked = encoded = 0
    for path in sorted((repo / "tests/fixtures/engine/games").glob("*.gz"))[
        : args.games
    ]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            events = [json.loads(e) for e in json.load(f)["mjai_log"]]
        trackers = [StateTracker(p) for p in range(4)]
        state = None
        for index, event in enumerate(events):
            kind, actor = event["type"], event.get("actor")
            if kind in {"hora", "ryukyoku", "end_kyoku", "end_game"}:
                continue
            if kind == "start_kyoku":
                payload = {
                    f"hai{p}": [from_mjai(t) for t in hand]
                    for p, hand in enumerate(event["tehais"])
                }
                payload.update(
                    ten=",".join(str(s // 100) for s in event["scores"]),
                    oya=event["oya"],
                    round_wind="ESWN".index(event["bakaze"]),
                    round_number=event["kyoku"] - 1,
                    match_rounds=8,
                    dora_indicator=from_mjai(event["dora_marker"]),
                    seed=f"0,{event['honba']},{event['kyotaku']}",
                )
                state = GameState.from_init_payload(0, payload)
            elif kind == "tsumo":
                state.apply_draw(actor, from_mjai(event["pai"]))
            elif kind == "dahai":
                state.apply_discard(
                    actor, from_mjai(event["pai"]), event.get("tsumogiri")
                ) if args.profile == "full_state_v2" else state.apply_discard(
                    actor, from_mjai(event["pai"])
                )
            elif kind in {"reach", "reach_accepted"}:
                state.apply_reach(actor, 1 if kind == "reach" else 2)
            elif kind == "dora":
                state.add_dora_indicator(from_mjai(event["dora_marker"]))
            elif kind in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
                consumed = canonical(from_mjai(t) for t in event["consumed"])
                called = consumed[0] if kind == "ankan" else from_mjai(event["pai"])
                source = actor if kind == "ankan" else event.get("target", actor)
                if kind == "kakan":
                    from mjai.rules.tiles import tile_to_base_index

                    pon = next(
                        m
                        for m in state.melds[actor]
                        if m.kind == "pon"
                        and m.base_index == tile_to_base_index(called)
                    )
                    consumed = [*pon.consumed_tiles, called]
                    called, source = pon.called_tile, pon.source
                state.apply_meld(actor, kind, source, called, consumed)
            for tracker in trackers:
                tracker.apply(event)
            if state is None or kind not in {
                "tsumo",
                "dahai",
                "reach_accepted",
                "chi",
                "pon",
                "ankan",
                "kakan",
                "dora",
            }:
                continue
            for seat, tracker in enumerate(trackers):
                actual = adapter.project(tracker.snapshot())
                expected = _player_view_snapshot(state, seat)
                # Extra derived/legacy convenience fields are compared through
                # their encoder as well. No unknown field differences are waived.
                for key in set(actual) & set(expected):
                    if actual[key] != expected[key]:
                        raise AssertionError(
                            f"{path.name}:{index}:seat={seat}:{key}: {actual[key]!r} != {expected[key]!r}"
                        )
                checked += 1
                if index % 29 == 0:
                    normalized = (
                        validate_and_normalize_state(
                            "turn", actual, require_full_state=True
                        )
                        if args.profile == "full_state_v2"
                        else validate_and_normalize_state("turn", actual)
                    )
                    if encode_turn_state(normalized) != encode_turn_state(expected):
                        raise AssertionError(
                            f"encoded state differs: {path.name}:{index}:{seat}"
                        )
                    encoded += 1
    result = {
        "profile": args.profile,
        "games": args.games,
        "snapshot_comparisons": checked,
        "encoded_state_comparisons": encoded,
        "valid": True,
        "scope": "A1 engine-event profile smoke; excludes XML, candidate parity and missed-win resolution",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
