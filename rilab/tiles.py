"""Lossless MJAI/engine-to-serving tile notation (no rule engine)."""

from __future__ import annotations

HONORS = "ESWNPFC"


def from_mjai(tile: str) -> str:
    if len(tile) == 1 and tile in HONORS:
        return f"{HONORS.index(tile) + 1}z"
    if tile.endswith("r") and tile[:1] == "5" and tile[1:2] in "mps":
        return "0" + tile[1]
    if len(tile) == 2 and tile[1] in "mps" and tile[0] in "123456789":
        return tile
    raise ValueError(f"invalid visible MJAI tile: {tile!r}")


def from_physical(tile: int) -> str:
    if type(tile) is not int or not 0 <= tile < 136:
        raise ValueError(f"invalid physical tile: {tile}")
    base = tile // 4
    if base >= 27:
        return f"{base - 26}z"
    rank = base % 9 + 1
    return ("0" if rank == 5 and tile % 4 == 0 else str(rank)) + "mps"[base // 9]


def base_index(tile: str) -> int:
    suit = tile[1]
    return (
        (27 + int(tile[0]) - 1)
        if suit == "z"
        else "mps".index(suit) * 9 + (5 if tile[0] == "0" else int(tile[0])) - 1
    )


def canonical(tiles) -> list[str]:
    return sorted(tiles, key=lambda t: (base_index(t), t[0] != "0", t))


def dora_base_index(indicator: str) -> int:
    base = base_index(indicator)
    if base < 27:
        return (base // 9) * 9 + (base % 9 + 1) % 9
    if base < 31:
        return 27 + (base - 27 + 1) % 4
    return 31 + (base - 31 + 1) % 3
