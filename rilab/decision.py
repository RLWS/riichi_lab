"""Decision identity and lossless links back to the current legal actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from riichienv import Action


class AdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionContext:
    task: str
    hand_id: int
    actor: int
    source: int | None = None
    tile: str | None = None
    chankan: bool = False


@dataclass(frozen=True)
class CandidateBinding:
    candidate: dict[str, Any]
    action: Action
    riichi_tile: str | None = None


@dataclass(frozen=True)
class PendingRiichiPlan:
    hand_id: int
    tile: str


def action_key(action: Action) -> tuple:
    return (
        int(action.action_type),
        action.tile,
        tuple(action.consume_tiles),
        action.actor,
    )
