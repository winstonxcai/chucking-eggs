"""Game result persistence and Elo side effects."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import db
from .elo import BOT_ELOS, compute_elo_delta, compute_forfeit_elo
from .types import EloChange, GameRecord

if TYPE_CHECKING:
    from .game_room import GameRoom

logger = logging.getLogger(__name__)


def _partner_for(seat: int) -> int:
    return 2 if seat == 0 else (0 if seat == 2 else (3 if seat == 1 else 1))


async def compute_elo_changes(room: GameRoom, rewards: dict[int, int]) -> dict[int, EloChange]:
    """Fetch player docs and compute Elo changes. Returns {} on any DB error."""
    try:
        human_docs: dict[int, dict] = {}
        for seat in room.human_seats:
            pid = room.player_ids.get(seat)
            if pid:
                doc = await asyncio.wait_for(db.get_player_by_id(pid), timeout=5.0)
                if doc:
                    human_docs[seat] = doc

        def _seat_elo(seat: int) -> int:
            if seat in human_docs:
                return human_docs[seat].get("elo", 1200)
            return room.player_infos[seat].get("elo") or BOT_ELOS.get(room.difficulty, 1500)

        finish_order = room.env.finish_order
        elo_changes: dict[int, EloChange] = {}
        for seat in room.human_seats:
            if seat not in human_docs:
                continue
            partner = _partner_for(seat)
            opps = [s for s in range(4) if s != seat and s != partner]
            before = human_docs[seat].get("elo", 1200)
            delta = compute_elo_delta(
                player_elo=before,
                partner_elo=_seat_elo(partner),
                opp1_elo=_seat_elo(opps[0]),
                opp2_elo=_seat_elo(opps[1]),
                player_games=human_docs[seat].get("games_played", 0),
                finish_pos=finish_order.index(seat),
                reward=rewards[seat],
            )
            elo_changes[seat] = {
                "delta": delta,
                "before": before,
                "after": before + delta,
                "username": human_docs[seat].get("username"),
            }
        return elo_changes
    except Exception:
        logger.exception("Elo computation failed for game %s", room.game_id)
        return {}


async def compute_forfeit_elo_change(room: GameRoom, forfeiter_seat: int) -> dict[int, EloChange]:
    """Compute the forfeit Elo penalty for one human seat."""
    try:
        pid = room.player_ids.get(forfeiter_seat)
        if not pid:
            return {}
        doc = await asyncio.wait_for(db.get_player_by_id(pid), timeout=5.0)
        if not doc:
            return {}

        partner = _partner_for(forfeiter_seat)
        opps = [s for s in range(4) if s != forfeiter_seat and s != partner]

        def _seat_elo(seat: int) -> int:
            return room.player_infos[seat].get("elo") or BOT_ELOS.get(room.difficulty, 1500)

        before = doc.get("elo", 1200)
        delta = compute_forfeit_elo(
            player_elo=before,
            partner_elo=_seat_elo(partner),
            opp1_elo=_seat_elo(opps[0]),
            opp2_elo=_seat_elo(opps[1]),
            player_games=doc.get("games_played", 0),
        )
        return {
            forfeiter_seat: {
                "delta": delta,
                "before": before,
                "after": before + delta,
                "username": doc.get("username"),
            },
        }
    except Exception:
        logger.exception("Forfeit Elo computation failed for game %s", room.game_id)
        return {}


def build_completed_game_record(
    room: GameRoom,
    rewards: dict[int, int],
    elo_changes: dict[int, EloChange],
) -> GameRecord:
    finish_order = room.env.finish_order
    return {
        "_id": room.game_id,
        "mode": room.mode,
        "difficulty": room.difficulty,
        "duration_seconds": int(time.time() - room.start_time),
        "played_at": datetime.now(timezone.utc),
        "trick_history": room.trick_history,
        "players": [
            {
                "player_id": room.player_ids.get(seat) if seat in room.human_seats else None,
                "display_name": (
                    elo_changes[seat]["username"]
                    if seat in elo_changes and elo_changes[seat].get("username")
                    else room.player_infos[seat]["name"]
                ),
                "is_bot": seat not in room.human_seats,
                "seat": seat,
                "finish_pos": finish_order.index(seat) + 1 if seat in finish_order else -1,
                "team_result": (
                    "win" if rewards[seat] > 0 else ("loss" if rewards[seat] < 0 else "neutral")
                ),
                "elo_before": elo_changes.get(seat, {}).get("before"),
                "elo_after": elo_changes.get(seat, {}).get("after"),
            }
            for seat in range(4)
        ],
    }


def build_forfeit_game_record(
    room: GameRoom,
    forfeiter_seat: int,
    elo_changes: dict[int, EloChange],
) -> GameRecord:
    return {
        "_id": room.game_id,
        "mode": room.mode,
        "difficulty": room.difficulty,
        "duration_seconds": int(time.time() - room.start_time),
        "played_at": datetime.now(timezone.utc),
        "result_type": "forfeit",
        "forfeiter_seat": forfeiter_seat,
        "players": [
            {
                "player_id": room.player_ids.get(seat) if seat in room.human_seats else None,
                "display_name": (
                    elo_changes[seat]["username"]
                    if seat in elo_changes and elo_changes[seat].get("username")
                    else room.player_infos[seat]["name"]
                ),
                "is_bot": seat not in room.human_seats,
                "seat": seat,
                "finish_pos": -1,
                "team_result": "forfeit" if seat == forfeiter_seat else "voided",
                "elo_before": elo_changes.get(seat, {}).get("before"),
                "elo_after": elo_changes.get(seat, {}).get("after"),
            }
            for seat in range(4)
        ],
    }


async def persist_completed_game(
    room: GameRoom,
    rewards: dict[int, int],
    elo_changes: dict[int, EloChange],
) -> None:
    """Save game and apply Elo once per game_id."""
    try:
        inserted = await db.save_game_if_absent(
            build_completed_game_record(room, rewards, elo_changes)
        )
        if not inserted:
            logger.info("Skipping duplicate completed-game persistence for %s", room.game_id)
            return
        for seat, change in elo_changes.items():
            pid = room.player_ids.get(seat)
            if pid:
                await db.update_player_elo(pid, change["after"])
    except Exception:
        logger.exception("DB persistence failed for game %s; game-over still delivered", room.game_id)


async def persist_forfeit_game(
    room: GameRoom,
    forfeiter_seat: int,
    elo_changes: dict[int, EloChange],
) -> None:
    """Save forfeit and apply Elo once per game_id."""
    try:
        inserted = await db.save_game_if_absent(
            build_forfeit_game_record(room, forfeiter_seat, elo_changes)
        )
        if not inserted:
            logger.info("Skipping duplicate forfeit persistence for %s", room.game_id)
            return
        for seat, change in elo_changes.items():
            pid = room.player_ids.get(seat)
            if pid:
                await db.update_player_elo(pid, change["after"])
    except Exception:
        logger.exception("Forfeit DB persistence failed for game %s", room.game_id)
