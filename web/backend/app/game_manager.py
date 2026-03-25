"""Manages active game rooms with grace period and idle cleanup."""

from __future__ import annotations

import asyncio
import time
import uuid

from .ai_service import AIService
from .game_room import GameRoom

GRACE_PERIOD = 60    # seconds to keep a disconnected room alive
IDLE_TIMEOUT = 300   # seconds before cleaning up an idle room
CLEANUP_INTERVAL = 30  # seconds between cleanup sweeps


class GameManager:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.rooms: dict[str, GameRoom] = {}
        self._cleanup_task: asyncio.Task | None = None

    async def start_cleanup_loop(self) -> None:
        """Background task: expire disconnected and idle rooms."""
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
                # Disconnected past grace period
                if room.disconnected_at and (now - room.disconnected_at > GRACE_PERIOD):
                    expired.append(gid)
                # Idle too long (no messages)
                elif now - room.last_activity > IDLE_TIMEOUT:
                    expired.append(gid)
            for gid in expired:
                self.rooms.pop(gid, None)

    def create_game(self, difficulty: str) -> GameRoom:
        game_id = uuid.uuid4().hex[:12]
        room = GameRoom(game_id, difficulty, self.ai_service)
        self.rooms[game_id] = room
        return room

    def get_room(self, game_id: str) -> GameRoom | None:
        return self.rooms.get(game_id)

    def disconnect_room(self, game_id: str) -> None:
        """Mark room as disconnected but keep alive for grace period."""
        room = self.rooms.get(game_id)
        if room:
            room.ws = None
            room.disconnected_at = time.time()

    def reconnect_room(self, game_id: str, token: str) -> GameRoom | None:
        """Reconnect to a room using its token. Returns room or None."""
        room = self.rooms.get(game_id)
        if room and room.reconnect_token == token:
            room.disconnected_at = None
            room.last_activity = time.time()
            return room
        return None

    def remove_room(self, game_id: str) -> None:
        """Immediately remove a room (for completed games)."""
        self.rooms.pop(game_id, None)
