"""Small typed document shapes used at the backend boundaries."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class PlayerDoc(TypedDict, total=False):
    _id: str
    username: str
    email: str
    elo: int
    games_played: int
    is_test: bool
    created_at: datetime


class EloChange(TypedDict):
    delta: int
    before: int
    after: int
    username: str | None


class GamePlayerRecord(TypedDict, total=False):
    player_id: str | None
    display_name: str
    is_bot: bool
    seat: int
    finish_pos: int
    team_result: str
    elo_before: int | None
    elo_after: int | None


class GameRecord(TypedDict, total=False):
    _id: str
    mode: str
    difficulty: str
    duration_seconds: int
    played_at: datetime
    result_type: str
    forfeiter_seat: int
    trick_history: list[dict[str, Any]]
    players: list[GamePlayerRecord]
