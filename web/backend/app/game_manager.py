"""Manages active game rooms."""

from __future__ import annotations

import uuid

from .ai_service import AIService
from .game_room import GameRoom


class GameManager:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.rooms: dict[str, GameRoom] = {}

    def create_game(self, difficulty: str) -> GameRoom:
        game_id = uuid.uuid4().hex[:12]
        room = GameRoom(game_id, difficulty, self.ai_service)
        self.rooms[game_id] = room
        return room

    def get_room(self, game_id: str) -> GameRoom | None:
        return self.rooms.get(game_id)

    def remove_room(self, game_id: str) -> None:
        self.rooms.pop(game_id, None)
