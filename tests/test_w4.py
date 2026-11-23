from io import BytesIO
import gzip
import json

import pytest

from rilab.w4.analysis import behavior, contrast, link, utc
from rilab.w4.public import collect, read
from rilab.w4.traces import export


def test_cutoff_pagination_and_id_join(monkeypatch, tmp_path):
    rows = [
        dict(
            game_id=i,
            played_at="2026-11-18 10:00:0" + str(i),
            bot=7,
            game_type="ranked",
            player_count=4,
            seat=1,
            rank=2,
            score=25000,
            rating_before=1700,
            mu_before=20,
            rating_delta=1,
        )
        for i in (1, 2, 3)
    ]
    for row in rows:
        row["game_id"] = str(row["game_id"])

    def get(api, path, **params):
        if path == "bots/7/games":
            if params["cursor"].startswith("2026-11-18 10:00:03"):
                return dict(games=rows[::-1][:2], has_more=True)
            return dict(games=rows[:2][::-1], has_more=False)
        if path.startswith("games/"):
            g = rows[int(path[-1]) - 1]
            return dict(
                id=g["game_id"],
                players=[
                    dict(
                        seat=i,
                        bot_id=b,
                        rank=i + 1,
                        score=25000,
                        bot_name=str(b),
                        rating_delta=1,
                        is_disconnected=False,
                        is_penalized=False,
                    )
                    for i, b in reversed(list(enumerate([11, 7, 12, 13])))
                ],
            )
        bot = int(path.split("/")[1])
        seat = [11, 7, 12, 13].index(bot)
        if bot == 13:
            return dict(games=[])
        return dict(games=[dict(g, seat=seat, rank=seat + 1) for g in rows])

    monkeypatch.setattr("rilab.w4.public.PublicAPI.get", get)
    collect(
        dict(
            output=tmp_path,
            request_interval_seconds=0,
            accounts=[7],
            cutoff_utc=rows[1]["played_at"],
            window=1,
        )
    )
    games = read(tmp_path / "tables.json")
    assert [g["game_id"] for g in games] == ["2", "1"]
    assert games[0]["recent"] and not games[1]["recent"]
    assert [p["rating_before"] for p in games[0]["opponents"]] == [1700, 1700, None]


def test_link_requires_vector_time_and_one_to_one():
    g = dict(
        bot=521,
        game_id="a",
        seat=0,
        scores=[4, 3, 2, 1],
        played_at="2026-11-13 11:00:00",
    )
    s = dict(
        bot=521,
        trace_sha256="x",
        source_id="one",
        seat=0,
        scores=g["scores"],
        trace_mtime_utc=utc(g["played_at"]),
    )
    assert link([g], [s, dict(s, source_id="copy")], 1)[0]["aliases"] == ["copy"]
    for bad in [
        dict(s, scores=[4, 2, 3, 1]),
        dict(s, bot=493),
        dict(s, seat=1),
        dict(s, trace_mtime_utc=s["trace_mtime_utc"] + 2),
    ]:
        assert link([g], [bad], 1)[0]["status"] == "unmatched"
    assert link([g, dict(g, game_id="b")], [s], 1)[0]["status"] == "ambiguous"
    assert {r["status"] for r in link([g], [s, dict(s, trace_sha256="y")], 1)} == {
        "ambiguous"
    }


def test_first_late_opportunity_and_privacy(tmp_path):
    def event(kind, **rest):
        return dict(kind="received", message=dict(type=kind, **rest))

    def decision(i, actual):
        return [
            dict(
                kind="prepared",
                request_id=i,
                task="turn",
                hand_id=1,
                state=dict(turn_number=13, authorization="secret"),
                candidates=[dict(kind="riichi_discard")],
                bindings=[],
            ),
            dict(
                kind="execution",
                request_id=i,
                status="accepted",
                actual_action=dict(type=actual),
            ),
        ]

    records = [
        event("start_game", id=0),
        event("start_kyoku"),
        event("reach", actor=2),
        *decision(1, "dahai"),
        event("reach_accepted", actor=2),
        *decision(2, "dahai"),
        *decision(3, "reach"),
        event("pon", actor=2),
        event("hora", actor=2, target=0, deltas=[-4000, 0, 4000, 0]),
        event("end_kyoku"),
        event("end_game", scores=[21000, 25000, 29000, 25000]),
    ]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in records)
    trace = export(BytesIO(raw), tmp_path / "decisions")
    with pytest.raises(ValueError, match="checksum"):
        export(BytesIO(raw), tmp_path, "wrong")
    shard = next((tmp_path / "decisions").glob("*.gz"))
    assert "secret" not in gzip.decompress(shard.read_bytes()).decode()
    result = behavior([({}, trace, trace)], tmp_path)["counts"]
    assert (result["late_opp"], result["late_reach"], result["late_other_hands"]) == (
        2,
        1,
        1,
    )
    assert result["late_other_deal"] == result["closed_deal"] == 1
    assert result["missing_prediction"] == 3


def test_only_shared_cells_contribute():
    rows = [
        dict(bot=b, seat=s, rank=r, score=100 * r, date="d")
        for b, s, r in [(493, 0, 1), (493, 0, 3), (521, 0, 4), (521, 3, 1)]
    ]
    result = contrast(rows, ["seat"])
    assert result["delta"] == [2, 1, 200]
    assert result["represented"] == {493: 2, 521: 1}


def test_same_coarse_band_can_hide_absent_continuous_support():
    from rilab.w4.analysis import comparison

    games = [
        dict(
            bot=b,
            seat=0,
            rank=1,
            score=40000,
            rating_before=rating,
            mu_before=20,
            played_at="2026-11-13 01:01:00",
            opponents=[dict(rating_before=1750, mu_before=20)] * 3,
        )
        for b, rating in [(493, 1814), (521, 1802)]
    ]
    result = comparison(games)
    assert result["date+seat+opp_band+self_band"]["weight"] == 1
    assert result["within_date_range"]["2026-11-13"]["delta"] is None
