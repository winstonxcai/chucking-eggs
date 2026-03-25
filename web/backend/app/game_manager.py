"""Manages active game rooms with grace period and idle cleanup."""

from __future__ import annotations

import asyncio
import random
import string
import time
import uuid

from .ai_service import AIService
from .game_room import GameRoom

GRACE_PERIOD = 60      # seconds to keep a disconnected room alive
IDLE_TIMEOUT = 300     # seconds before cleaning up an idle room
CLEANUP_INTERVAL = 30  # seconds between cleanup sweeps


def _generate_room_code() -> str:
    """6-char alphanumeric room code (uppercase)."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


class GameManager:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.rooms: dict[str, GameRoom] = {}
        self.room_codes: dict[str, str] = {}   # code → game_id
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup_loop(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_loop(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = time.time()
            expired = []
            for gid, room in self.rooms.items():
                if room.disconnected_at and (now - room.disconnected_at > GRACE_PERIOD):
                    expired.append(gid)
                elif now - room.last_activity > IDLE_TIMEOUT:
                    expired.append(gid)
            for gid in expired:
                room = self.rooms.pop(gid, None)
                if room and room.room_code:
                    self.room_codes.pop(room.room_code, None)

    # ---------------------------------------------------------------------------
    # Room creation
    # ---------------------------------------------------------------------------

    def create_game(self, difficulty: str) -> GameRoom:
        """Create a solo game room (backwards compat for /api/game/create)."""
        return self.create_room("solo", difficulty)

    def create_room(self, mode: str, difficulty: str) -> GameRoom:
        """Create a game room with the given mode (solo/duo/quad)."""
        game_id = uuid.uuid4().hex[:12]
        room = GameRoom(game_id, mode, difficulty, self.ai_service)
        self.rooms[game_id] = room

        if mode != "solo":
            code = self._unique_room_code()
            room.room_code = code
            self.room_codes[code] = game_id

        return room

    def _unique_room_code(self) -> str:
        for _ in range(100):
            code = _generate_room_code()
            if code not in self.room_codes:
                return code
        raise RuntimeError("Could not generate a unique room code")

    # ---------------------------------------------------------------------------
    # Room joining
    # ---------------------------------------------------------------------------

    def join_room(self, room_code: str) -> tuple[GameRoom, int] | None:
        """
        Find a joinable room by code and return (room, assigned_seat).
        Returns None if not found, already started, or all human seats are taken.
        """
        game_id = self.room_codes.get(room_code.upper())
        if not game_id:
            return None
        room = self.rooms.get(game_id)
        if not room or room.started:
            return None
        # Find the next unassigned human seat (assigned_seats tracks HTTP-level claims)
        available = sorted(room.human_seats - room.assigned_seats)
        if not available:
            return None
        seat = available[0]
        room.assigned_seats.add(seat)
        return room, seat

    # ---------------------------------------------------------------------------
    # Room lookup
    # ---------------------------------------------------------------------------

    def get_room(self, game_id: str) -> GameRoom | None:
        return self.rooms.get(game_id)

    def get_room_by_code(self, room_code: str) -> GameRoom | None:
        game_id = self.room_codes.get(room_code.upper())
        return self.rooms.get(game_id) if game_id else None

    # ---------------------------------------------------------------------------
    # Disconnect / reconnect (per-seat and room-level compat)
    # ---------------------------------------------------------------------------

    def disconnect_seat(self, game_id: str, seat: int) -> None:
        """Mark a specific seat as disconnected."""
        room = self.rooms.get(game_id)
        if room:
            room.connections.pop(seat, None)
            room.disconnected_seats[seat] = time.time()

    def reconnect_seat(self, game_id: str, seat: int, token: str) -> GameRoom | None:
        """Reconnect a specific seat using its token. Returns room or None."""
        room = self.rooms.get(game_id)
        if room and room.reconnect_tokens.get(seat) == token:
            room.disconnected_seats.pop(seat, None)
            room.last_activity = time.time()
            return room
        return None

    # Solo / Phase-1 compat wrappers
    def disconnect_room(self, game_id: str) -> None:
        self.disconnect_seat(game_id, 0)

    def reconnect_room(self, game_id: str, token: str) -> GameRoom | None:
        return self.reconnect_seat(game_id, 0, token)

    def remove_room(self, game_id: str) -> None:
        """Immediately remove a room (e.g. for completed / cancelled games)."""
        room = self.rooms.pop(game_id, None)
        if room and room.room_code:
            self.room_codes.pop(room.room_code, None)
