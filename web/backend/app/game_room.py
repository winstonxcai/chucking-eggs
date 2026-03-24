"""Single game room: manages env, agents, and WebSocket communication."""

from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket

from guandan.cards import ComboType, Rank
from guandan.game import GuanDanEnv

from .ai_service import AIService
from .card_matcher import find_matching_combo
from .serializer import combo_to_dto, serialize_game_state

HUMAN_SEAT = 0


class GameRoom:
    def __init__(self, game_id: str, difficulty: str, ai_service: AIService):
        self.game_id = game_id
        self.difficulty = difficulty
        self.ai_service = ai_service
        self.env = GuanDanEnv(level_rank=Rank.TWO)
        self.env.reset()
        self.ws: WebSocket | None = None

        # Pick bot personalities
        bots = ai_service.pick_bots(difficulty)
        self.agent = ai_service.get_agent(difficulty)

        # Player infos: seat 0 = human, seats 1,2,3 = bots
        # Seat 2 is partner, seats 1,3 are opponents
        self.player_infos = [
            {"name": "You", "avatar": None, "elo": 1200},
            bots[0],  # seat 1: left opponent
            bots[1],  # seat 2: partner
            bots[2],  # seat 3: right opponent
        ]

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.ws = ws

    async def send(self, data: dict) -> None:
        if self.ws:
            await self.ws.send_text(json.dumps(data))

    async def send_game_state(self) -> None:
        state = serialize_game_state(
            self.env, self.game_id, HUMAN_SEAT, self.player_infos
        )
        await self.send({"type": "game_state", **state})

    async def run_ai_turns(self) -> None:
        """Run AI turns until it's the human's turn or game is over."""
        while not self.env.done and self.env.current_player != HUMAN_SEAT:
            seat = self.env.current_player
            await self.send({"type": "ai_thinking", "seat": seat})
            await asyncio.sleep(0.5)  # natural pacing

            combo = await self.ai_service.get_ai_move(
                self.agent, self.env, seat
            )
            next_player, done = self.env.step(combo)

            await self.send({
                "type": "move_played",
                "seat": seat,
                "combo": combo_to_dto(combo, self.env.level_rank),
                "next_player": next_player,
                "done": done,
            })

        if self.env.done:
            rewards = self.env.get_rewards()
            await self.send({
                "type": "game_over",
                "finish_order": self.env.finish_order,
                "rewards": rewards,
                "players": [
                    {"seat": i, "name": self.player_infos[i]["name"]}
                    for i in self.env.finish_order
                ],
            })
        else:
            await self.send_game_state()

    async def handle_message(self, data: dict) -> None:
        """Handle a message from the human player."""
        msg_type = data.get("type")

        if msg_type == "play_cards":
            await self._handle_play(data.get("card_ids", []))
        elif msg_type == "pass":
            await self._handle_pass()
        else:
            await self.send({"type": "error", "message": f"Unknown message type: {msg_type}"})

    async def _handle_play(self, card_ids: list[str]) -> None:
        if self.env.current_player != HUMAN_SEAT:
            await self.send({"type": "error", "message": "Not your turn"})
            return

        legal = self.env.legal_moves(HUMAN_SEAT)
        matches = find_matching_combo(card_ids, legal)

        if not matches:
            await self.send({"type": "error", "message": "Invalid combo"})
            return

        combo = matches[0]
        next_player, done = self.env.step(combo)

        await self.send({
            "type": "move_played",
            "seat": HUMAN_SEAT,
            "combo": combo_to_dto(combo, self.env.level_rank),
            "next_player": next_player,
            "done": done,
        })

        if not done:
            await self.run_ai_turns()
        else:
            rewards = self.env.get_rewards()
            await self.send({
                "type": "game_over",
                "finish_order": self.env.finish_order,
                "rewards": rewards,
                "players": [
                    {"seat": i, "name": self.player_infos[i]["name"]}
                    for i in self.env.finish_order
                ],
            })

    async def _handle_pass(self) -> None:
        if self.env.current_player != HUMAN_SEAT:
            await self.send({"type": "error", "message": "Not your turn"})
            return

        if self.env.is_leading():
            await self.send({"type": "error", "message": "Cannot pass when leading"})
            return

        # Find the PASS combo from legal moves
        legal = self.env.legal_moves(HUMAN_SEAT)
        pass_combo = next((c for c in legal if c.type == ComboType.PASS), None)
        if pass_combo is None:
            await self.send({"type": "error", "message": "Pass not available"})
            return

        next_player, done = self.env.step(pass_combo)

        await self.send({
            "type": "move_played",
            "seat": HUMAN_SEAT,
            "combo": combo_to_dto(pass_combo, self.env.level_rank),
            "next_player": next_player,
            "done": done,
        })

        if not done:
            await self.run_ai_turns()
        else:
            rewards = self.env.get_rewards()
            await self.send({
                "type": "game_over",
                "finish_order": self.env.finish_order,
                "rewards": rewards,
                "players": [
                    {"seat": i, "name": self.player_infos[i]["name"]}
                    for i in self.env.finish_order
                ],
            })
