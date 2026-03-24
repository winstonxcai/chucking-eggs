"""Single game room: manages env, agents, and WebSocket communication."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import WebSocket

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv

from .ai_service import AIService
from .card_matcher import find_matching_combo
from .serializer import combo_to_dto, serialize_game_state

HUMAN_SEAT = 0
ACTION_PAUSE = 0.9  # seconds each AI play is visible before next turn


class GameRoom:
    def __init__(self, game_id: str, difficulty: str, ai_service: AIService):
        self.game_id = game_id
        self.difficulty = difficulty
        self.ai_service = ai_service
        self.env = GuanDanEnv(level_rank=Rank.TWO)
        self.env.reset()
        self.ws: WebSocket | None = None

        # Track current trick plays: list of (seat, Combo)
        self.trick_plays: list[tuple[int, Combo]] = []
        self._was_leading = True  # track trick boundaries

        self._start_time = time.time()
        self.groups: list[dict] = []

        # Pick bot personalities
        bots = ai_service.pick_bots(difficulty)
        self.agent = ai_service.get_agent(difficulty)

        # Player infos: seat 0 = human, seats 1,2,3 = bots
        self.player_infos = [
            {"name": "You", "avatar": None, "elo": 1200},
            bots[0],  # seat 1: left opponent
            bots[1],  # seat 2: partner
            bots[2],  # seat 3: right opponent
        ]

    def _record_move(self, seat: int, combo: Combo) -> None:
        """Record a move and detect trick boundaries."""
        # If we were leading (no trick) and now someone plays, start a new trick
        if self._was_leading and combo.type != ComboType.PASS:
            self.trick_plays = []
        self.trick_plays.append((seat, combo))
        # Update leading state for next check
        self._was_leading = self.env.is_leading()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.ws = ws

    async def send(self, data: dict) -> None:
        if self.ws:
            await self.ws.send_text(json.dumps(data))

    async def send_game_state(self) -> None:
        state = serialize_game_state(
            self.env, self.game_id, HUMAN_SEAT, self.player_infos,
            trick_plays=self.trick_plays,
            groups=self.groups,
        )
        await self.send({"type": "game_state", **state})

    def _record_game_result(self) -> None:
        fo = self.env.finish_order
        human_pos = fo.index(HUMAN_SEAT) + 1
        rewards = self.env.get_rewards()
        team_result = "win" if rewards[HUMAN_SEAT] > 0 else "loss"
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "difficulty": self.difficulty,
            "agent": self.ai_service.get_agent_name(self.difficulty),
            "finish_order": fo,
            "human_finish_pos": human_pos,
            "team_result": team_result,
            "reward": rewards[HUMAN_SEAT],
            "n_moves": len(self.env.move_history),
            "duration_s": int(time.time() - self._start_time),
            "players": [
                {"seat": i, "name": info["name"], "elo": info.get("elo"), "is_human": i == HUMAN_SEAT}
                for i, info in enumerate(self.player_infos)
            ],
        }
        DATA_DIR.mkdir(exist_ok=True)
        with open(DATA_DIR / "human_games.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    async def _send_game_over(self) -> None:
        self._record_game_result()
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

    async def run_ai_turns(self) -> None:
        """Run AI turns until it's the human's turn or game is over."""
        while not self.env.done and self.env.current_player != HUMAN_SEAT:
            seat = self.env.current_player
            legal = self.env.legal_moves(seat)

            # Auto-pass: skip inference when PASS is the only option
            if len(legal) == 1 and legal[0].type == ComboType.PASS:
                combo = legal[0]
            else:
                await self.send({"type": "ai_thinking", "seat": seat})
                await asyncio.sleep(0.5)
                combo = await self.ai_service.get_ai_move(
                    self.agent, self.env, seat
                )
            next_player, done = self.env.step(combo)
            self._record_move(seat, combo)

            await self.send({
                "type": "move_played",
                "seat": seat,
                "combo": combo_to_dto(combo, self.env.level_rank),
                "next_player": next_player,
                "done": done,
            })

            # Show the played card on the table before moving on
            await self.send_game_state()
            if not done:
                await asyncio.sleep(ACTION_PAUSE)

        if self.env.done:
            await self._send_game_over()
        else:
            await self.send_game_state()

    async def handle_message(self, data: dict) -> None:
        msg_type = data.get("type")
        if msg_type == "play_cards":
            await self._handle_play(data.get("card_ids", []))
        elif msg_type == "pass":
            await self._handle_pass()
        elif msg_type == "create_group":
            await self._handle_create_group(data)
        elif msg_type == "delete_group":
            await self._handle_delete_group(data)
        else:
            await self.send({"type": "error", "message": f"Unknown message type: {msg_type}"})

    async def _handle_create_group(self, data: dict) -> None:
        card_ids = data.get("card_ids", [])
        combo_type = data.get("combo_type", "")
        combo_name = data.get("combo_name", "")
        group_id = f"grp-{len(self.groups)}-{int(time.time() * 1000)}"
        # Remove overlap with existing groups
        id_set = set(card_ids)
        self.groups = [g for g in self.groups
                       if not any(cid in id_set for cid in g["card_ids"])]
        self.groups.append({
            "id": group_id,
            "card_ids": card_ids,
            "combo_type": combo_type,
            "combo_name": combo_name,
        })
        await self.send_game_state()

    async def _handle_delete_group(self, data: dict) -> None:
        group_id = data.get("group_id", "")
        self.groups = [g for g in self.groups if g["id"] != group_id]
        await self.send_game_state()

    async def _do_move(self, combo: Combo, seat: int = HUMAN_SEAT) -> None:
        """Execute a move, record it, broadcast, then run AI or end game."""
        next_player, done = self.env.step(combo)
        self._record_move(seat, combo)
        # Dissolve groups containing played cards
        if seat == HUMAN_SEAT:
            played = {f"{c.rank}-{c.suit}-{c.deck}" for c in combo.cards}
            self.groups = [g for g in self.groups
                           if not any(cid in played for cid in g["card_ids"])]

        await self.send({
            "type": "move_played",
            "seat": seat,
            "combo": combo_to_dto(combo, self.env.level_rank),
            "next_player": next_player,
            "done": done,
        })

        if done:
            await self._send_game_over()
        else:
            await self.run_ai_turns()

    async def _handle_play(self, card_ids: list[str]) -> None:
        if self.env.current_player != HUMAN_SEAT:
            await self.send({"type": "error", "message": "Not your turn"})
            return

        legal = self.env.legal_moves(HUMAN_SEAT)
        matches = find_matching_combo(card_ids, legal, hand=self.env.hands[HUMAN_SEAT])

        if not matches:
            await self.send({"type": "error", "message": "Invalid combo"})
            return

        await self._do_move(matches[0])

    async def _handle_pass(self) -> None:
        if self.env.current_player != HUMAN_SEAT:
            await self.send({"type": "error", "message": "Not your turn"})
            return

        if self.env.is_leading():
            await self.send({"type": "error", "message": "Cannot pass when leading"})
            return

        legal = self.env.legal_moves(HUMAN_SEAT)
        pass_combo = next((c for c in legal if c.type == ComboType.PASS), None)
        if pass_combo is None:
            await self.send({"type": "error", "message": "Pass not available"})
            return

        await self._do_move(pass_combo)
