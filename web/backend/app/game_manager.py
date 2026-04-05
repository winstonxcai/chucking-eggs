"""Manages active game rooms with grace period and idle cleanup."""

from __future__ import annotations

import asyncio
import secrets
import string
import time
import uuid

from .ai_service import AIService
from .game_room import GameRoom

GRACE_PERIOD = 60      # seconds to keep a disconnected room alive
IDLE_TIMEOUT = 300     # seconds before cleaning up an idle room
CLEANUP_INTERVAL = 30  # seconds between cleanup sweeps
LOBBY_TIMEOUT = 300    # seconds before auto-closing an unstarted duo/quad room


_ROOM_CODE_CHARS = string.ascii_uppercase + string.digits


def _generate_room_code() -> str:
    """6-char alphanumeric room code (uppercase), using cryptographically secure PRNG."""
    return "".join(secrets.choice(_ROOM_CODE_CHARS) for _ in range(6))


class GameManager:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.rooms: dict[str, GameRoom] = {}
        self.room_codes: dict[str, str] = {}   # code → game_id
        self._cleanup_task: asyncio.Task | None = None
        self._room_lock = asyncio.Lock()  # prevents duplicate codes under concurrent creates

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
            expired_lobby: list[str] = []
            expired_silent: list[str] = []
            for gid, room in self.rooms.items():
                if not room.started and room.room_code and now - room.created_at > LOBBY_TIMEOUT:
                    expired_lobby.append(gid)
                elif room.disconnected_at and (now - room.disconnected_at > GRACE_PERIOD):
                    expired_silent.append(gid)
                elif now - room.last_activity > IDLE_TIMEOUT:
                    expired_silent.append(gid)
            for gid in expired_lobby:
                room = self.rooms.pop(gid, None)
                if room:
                    if room.room_code:
                        self.room_codes.pop(room.room_code, None)
                    asyncio.create_task(room.broadcast({"type": "room_closed", "reason": "lobby_timeout"}))
            for gid in expired_silent:
                room = self.rooms.pop(gid, None)
                if room and room.room_code:
                    self.room_codes.pop(room.room_code, None)

    # ---------------------------------------------------------------------------
    # Room creation
    # ---------------------------------------------------------------------------

    async def create_game(self, difficulty: str) -> GameRoom:
        """Create a solo game room (backwards compat for /api/game/create)."""
        return await self.create_room("solo", difficulty)

    def _rooms_for_player(self, player_id: str, exclude_game_id: str | None = None) -> list[GameRoom]:
        """Return all rooms where this player holds a seat (by HTTP-time seat_player_ids).
        Must be called while holding self._room_lock."""
        return [
            room for gid, room in self.rooms.items()
            if gid != exclude_game_id and player_id in room.seat_player_ids.values()
        ]

    async def create_room(self, mode: str, difficulty: str, seed: int | None = None, creator_player_id: str | None = None) -> GameRoom:
        """Create a game room with the given mode (solo/duo/quad).

        If creator_player_id is provided, enforces one-room-per-player:
        - Raises ValueError('already_in_game') if the player is in a started room.
        - Auto-closes any unstarted room the player currently owns.
        """
        rooms_to_close: list[GameRoom] = []
        async with self._room_lock:
            if creator_player_id:
                for existing in self._rooms_for_player(creator_player_id):
                    if existing.started:
                        raise ValueError("already_in_game")
                    rooms_to_close.append(existing)
                    self.remove_room(existing.game_id)
            game_id = uuid.uuid4().hex[:12]
            room = GameRoom(game_id, mode, difficulty, self.ai_service, seed=seed)
            if creator_player_id:
                room.seat_player_ids[0] = creator_player_id
            self.rooms[game_id] = room

            if mode != "solo":
                code = self._unique_room_code()
                room.room_code = code
                self.room_codes[code] = game_id

        for old_room in rooms_to_close:
            await old_room.broadcast({"type": "room_closed", "reason": "creator_left"})
        return room

    def _unique_room_code(self) -> str:
        """Must be called while holding self._room_lock."""
        for _ in range(100):
            code = _generate_room_code()
            if code not in self.room_codes:
                return code
        raise RuntimeError("Could not generate a unique room code")

    # ---------------------------------------------------------------------------
    # Room joining
    # ---------------------------------------------------------------------------

    async def join_room(self, room_code: str, joiner_player_id: str | None = None) -> tuple[GameRoom, int] | None:
        """
        Find a joinable room by code and return (room, assigned_seat).
        Returns None if not found, already started, or all human seats are taken.

        If joiner_player_id is provided, enforces one-room-per-player:
        - Raises ValueError('already_in_game') if the player is in a started room.
        - Auto-closes any unstarted room the player owns (excluding the target room).
        """
        rooms_to_close: list[GameRoom] = []
        result: tuple[GameRoom, int] | None = None
        async with self._room_lock:
            game_id = self.room_codes.get(room_code.upper())
            if not game_id:
                return None
            room = self.rooms.get(game_id)
            if not room or room.started:
                return None
            if joiner_player_id:
                # exclude the target room — self-join guard (creator joining own quad room)
                for other in self._rooms_for_player(joiner_player_id, exclude_game_id=game_id):
                    if other.started:
                        raise ValueError("already_in_game")
                    rooms_to_close.append(other)
                    self.remove_room(other.game_id)
            # Find the next unassigned human seat (assigned_seats tracks HTTP-level claims)
            available = sorted(room.human_seats - room.assigned_seats)
            if not available:
                return None
            seat = available[0]
            room.assigned_seats.add(seat)
            if joiner_player_id:
                room.seat_player_ids[seat] = joiner_player_id
            result = (room, seat)
        for old_room in rooms_to_close:
            await old_room.broadcast({"type": "room_closed", "reason": "creator_left"})
        return result

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

    def disconnect_seat(self, game_id: str, seat: int, ws: object | None = None) -> None:
        """Mark a specific seat as disconnected.

        If *ws* is given, only disconnect when that exact WebSocket is still
        the active connection.  This prevents a stale handler (e.g. from a
        page refresh) from killing the replacement connection.
        """
        room = self.rooms.get(game_id)
        if room:
            if ws is not None and room.connections.get(seat) is not ws:
                return
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

    async def create_rematch(self, game_id: str) -> GameRoom | None:
        """Create a new room with the same mode/difficulty as a finished game, then broadcast to all players."""
        old_room = self.rooms.get(game_id)
        if old_room is None or not old_room.env.done:
            return None
        new_room = await self.create_room(old_room.mode, old_room.difficulty)
        # Broadcast to all still-connected players in the old room
        await old_room.broadcast({
            "type": "rematch_created",
            "game_id": new_room.game_id,
            "room_code": new_room.room_code,
        })
        return new_room

    async def leave_room(self, game_id: str, seat: int) -> bool:
        """Remove a player from a room that hasn't started yet. Dissolves room if empty.
        Returns True if the room was dissolved, False otherwise."""
        room = self.rooms.get(game_id)
        if not room or room.started:
            return False
        room.assigned_seats.discard(seat)
        room.connections.pop(seat, None)
        # If no human seats are assigned, dissolve
        if not room.assigned_seats:
            await room.broadcast({"type": "room_closed", "reason": "Host left the room"})
            self.remove_room(game_id)
            return True
        # If host (seat 0) leaves, dissolve for everyone
        if seat == 0:
            await room.broadcast({"type": "room_closed", "reason": "Host left the room"})
            self.remove_room(game_id)
            return True
        return False

    def remove_room(self, game_id: str) -> None:
        """Immediately remove a room (e.g. for completed / cancelled games)."""
        room = self.rooms.pop(game_id, None)
        if room and room.room_code:
            self.room_codes.pop(room.room_code, None)
