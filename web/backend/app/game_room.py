"""Single game room: manages env, agents, and WebSocket communication."""

from __future__ import annotations

import asyncio
import json
import secrets
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

ACTION_PAUSE = 0.9   # seconds each AI play is visible before next turn
AI_THINK_PAUSE = 0.5  # seconds for "thinking" animation before AI move


def _human_seats_for_mode(mode: str) -> set[int]:
    """Return which seats are human-controlled based on mode."""
    if mode == "duo":
        return {0, 2}   # partners
    if mode == "quad":
        return {0, 1, 2, 3}
    return {0}  # solo


class GameRoom:
    def __init__(self, game_id: str, mode: str, difficulty: str, ai_service: AIService):
        self.game_id = game_id
        self.mode = mode
        self.difficulty = difficulty
        self.ai_service = ai_service
        self.env = GuanDanEnv(level_rank=Rank.TWO)
        self.env.reset()

        # Multi-human support
        self.human_seats: set[int] = _human_seats_for_mode(mode)
        self.connections: dict[int, WebSocket] = {}
        self.reconnect_tokens: dict[int, str] = {
            seat: secrets.token_urlsafe(16) for seat in self.human_seats
        }
        self.groups_by_seat: dict[int, list[dict]] = {seat: [] for seat in self.human_seats}
        self.started: bool = (mode == "solo")  # solo auto-starts; duo/quad wait for all
        self.room_code: str | None = None  # set by GameManager for non-solo modes
        # Tracks seats that have been handed out (via HTTP create/join), whether or not
        # the holder has actually connected via WebSocket yet.
        self.assigned_seats: set[int] = {0}  # creator always gets seat 0

        # Room lifecycle
        self.last_activity = time.time()
        self.disconnected_seats: dict[int, float] = {}  # seat -> disconnect timestamp

        # Trick tracking
        self.trick_plays: list[tuple[int, Combo]] = []
        self._was_leading = True
        self._start_time = time.time()

        # AI lock — prevents concurrent AI run coroutines
        self._ai_lock = asyncio.Lock()

        # Agent + player infos
        bots = ai_service.pick_bots(difficulty)
        self.agent = ai_service.get_agent(difficulty)
        self.player_infos: list[dict] = [
            {"name": "You", "avatar": None, "elo": 1200},
            bots[0],
            bots[1],
            bots[2],
        ]
        # Rename partner slot for multiplayer
        if mode == "duo":
            self.player_infos[2] = {"name": "Partner", "avatar": None, "elo": 1200}
        elif mode == "quad":
            for seat in [1, 2, 3]:
                self.player_infos[seat] = {"name": f"Player {seat + 1}", "avatar": None, "elo": 1200}

    # ---------------------------------------------------------------------------
    # Backwards-compat properties (solo mode / Phase 1 tests)
    # ---------------------------------------------------------------------------

    @property
    def ws(self) -> WebSocket | None:
        return self.connections.get(0)

    @ws.setter
    def ws(self, value: WebSocket | None) -> None:
        if value is None:
            self.connections.pop(0, None)
        else:
            self.connections[0] = value

    @property
    def disconnected_at(self) -> float | None:
        """Non-None only when ALL human seats are disconnected; returns most-recent time."""
        if len(self.disconnected_seats) < len(self.human_seats):
            return None
        if not self.disconnected_seats:
            return None
        return max(self.disconnected_seats.values())

    @disconnected_at.setter
    def disconnected_at(self, value: float | None) -> None:
        if value is None:
            self.disconnected_seats.clear()
        else:
            for seat in self.human_seats:
                self.disconnected_seats[seat] = value

    @property
    def reconnect_token(self) -> str:
        """Reconnect token for seat 0 (solo mode compat)."""
        return self.reconnect_tokens.get(0, "")

    @property
    def groups(self) -> list[dict]:
        """Groups for seat 0 (solo mode compat)."""
        return self.groups_by_seat.get(0, [])

    # ---------------------------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------------------------

    def is_fully_disconnected(self) -> bool:
        return all(seat not in self.connections for seat in self.human_seats)

    async def connect(self, ws: WebSocket, seat: int = 0) -> None:
        await ws.accept()
        self.connections[seat] = ws
        self.disconnected_seats.pop(seat, None)

    # ---------------------------------------------------------------------------
    # Sending
    # ---------------------------------------------------------------------------

    async def send_to(self, seat: int, data: dict) -> None:
        ws = self.connections.get(seat)
        if ws:
            payload = json.dumps(data)  # serialization errors propagate — don't hide bugs
            try:
                await ws.send_text(payload)
            except Exception:
                pass  # seat disconnected mid-send — expected in multiplayer

    async def broadcast(self, data: dict) -> None:
        for seat in list(self.connections.keys()):
            await self.send_to(seat, data)

    # Backwards-compat alias
    async def send(self, data: dict) -> None:
        await self.broadcast(data)

    async def send_game_state_to(self, seat: int) -> None:
        state = serialize_game_state(
            self.env, self.game_id, seat, self.player_infos,
            trick_plays=self.trick_plays,
            groups=self.groups_by_seat.get(seat, []),
        )
        await self.send_to(seat, {"type": "game_state", **state})

    async def broadcast_game_state(self) -> None:
        for seat in list(self.connections.keys()):
            await self.send_game_state_to(seat)

    # Backwards-compat alias
    async def send_game_state(self) -> None:
        await self.broadcast_game_state()

    # ---------------------------------------------------------------------------
    # Trick recording
    # ---------------------------------------------------------------------------

    def _record_move(self, seat: int, combo: Combo) -> None:
        if self._was_leading and combo.type != ComboType.PASS:
            self.trick_plays = []
        self.trick_plays.append((seat, combo))
        self._was_leading = self.env.is_leading()

    # ---------------------------------------------------------------------------
    # Game result persistence
    # ---------------------------------------------------------------------------

    def _record_game_result(self) -> None:
        primary = min(self.human_seats)
        fo = self.env.finish_order
        human_pos = fo.index(primary) + 1 if primary in fo else -1
        rewards = self.env.get_rewards()
        team_result = "win" if rewards[primary] > 0 else "loss"
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "difficulty": self.difficulty,
            "agent": self.ai_service.get_agent_name(self.difficulty),
            "finish_order": fo,
            "human_finish_pos": human_pos,
            "team_result": team_result,
            "reward": rewards[primary],
            "n_moves": len(self.env.move_history),
            "duration_s": int(time.time() - self._start_time),
            "players": [
                {
                    "seat": i,
                    "name": info["name"],
                    "elo": info.get("elo"),
                    "is_human": i in self.human_seats,
                }
                for i, info in enumerate(self.player_infos)
            ],
        }
        DATA_DIR.mkdir(exist_ok=True)
        with open(DATA_DIR / "human_games.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    async def _send_game_over(self) -> None:
        self._record_game_result()
        rewards = self.env.get_rewards()
        await self.broadcast({
            "type": "game_over",
            "finish_order": self.env.finish_order,
            "rewards": rewards,
            "players": [
                {"seat": i, "name": self.player_infos[i]["name"]}
                for i in self.env.finish_order
            ],
        })

    # ---------------------------------------------------------------------------
    # AI turns
    # ---------------------------------------------------------------------------

    async def run_ai_turns(self) -> None:
        """Run AI turns until it's a human's turn or the game ends."""
        async with self._ai_lock:
            while not self.env.done and self.env.current_player not in self.human_seats:
                seat = self.env.current_player
                legal = self.env.legal_moves(seat)

                if len(legal) == 1 and legal[0].type == ComboType.PASS:
                    combo = legal[0]
                else:
                    await self.broadcast({"type": "ai_thinking", "seat": seat})
                    await asyncio.sleep(AI_THINK_PAUSE)
                    combo = await self.ai_service.get_ai_move(self.agent, self.env, seat)

                next_player, done = self.env.step(combo)
                self._record_move(seat, combo)

                await self.broadcast({
                    "type": "move_played",
                    "seat": seat,
                    "combo": combo_to_dto(combo, self.env.level_rank),
                    "next_player": next_player,
                    "done": done,
                })

                await self.broadcast_game_state()
                if not done:
                    await asyncio.sleep(ACTION_PAUSE)

            if self.env.done:
                await self._send_game_over()
            else:
                await self.broadcast_game_state()

    # ---------------------------------------------------------------------------
    # Message handling
    # ---------------------------------------------------------------------------

    async def handle_message(self, data: dict, seat: int = 0) -> None:
        self.last_activity = time.time()
        msg_type = data.get("type")
        if msg_type == "play_cards":
            await self._handle_play(data.get("card_ids", []), seat)
        elif msg_type == "pass":
            await self._handle_pass(seat)
        elif msg_type == "create_group":
            await self._handle_create_group(data, seat)
        elif msg_type == "delete_group":
            await self._handle_delete_group(data, seat)
        else:
            await self.send_to(seat, {"type": "error", "message": f"Unknown message type: {msg_type}"})

    async def _handle_create_group(self, data: dict, seat: int = 0) -> None:
        card_ids = data.get("card_ids", [])
        combo_type = data.get("combo_type", "")
        combo_name = data.get("combo_name", "")
        groups = self.groups_by_seat.setdefault(seat, [])
        group_id = f"grp-{len(groups)}-{int(time.time() * 1000)}"
        id_set = set(card_ids)
        self.groups_by_seat[seat] = [g for g in groups if not any(cid in id_set for cid in g["cardIds"])]
        self.groups_by_seat[seat].append({
            "id": group_id,
            "cardIds": card_ids,
            "comboType": combo_type,
            "comboName": combo_name,
        })
        await self.send_game_state_to(seat)

    async def _handle_delete_group(self, data: dict, seat: int = 0) -> None:
        group_id = data.get("group_id", "")
        groups = self.groups_by_seat.get(seat, [])
        self.groups_by_seat[seat] = [g for g in groups if g["id"] != group_id]
        await self.send_game_state_to(seat)

    async def _do_move(self, combo: Combo, seat: int) -> None:
        next_player, done = self.env.step(combo)
        self._record_move(seat, combo)

        # Dissolve groups containing played cards
        if seat in self.human_seats:
            played = {f"{c.rank}-{c.suit}-{c.deck}" for c in combo.cards}
            groups = self.groups_by_seat.get(seat, [])
            self.groups_by_seat[seat] = [
                g for g in groups if not any(cid in played for cid in g["cardIds"])
            ]

        await self.broadcast({
            "type": "move_played",
            "seat": seat,
            "combo": combo_to_dto(combo, self.env.level_rank),
            "next_player": next_player,
            "done": done,
        })

        await self.broadcast_game_state()

        if done:
            await self._send_game_over()
        else:
            await asyncio.sleep(ACTION_PAUSE)
            await self.run_ai_turns()

    async def _handle_play(self, card_ids: list[str], seat: int = 0) -> None:
        if not self.started:
            await self.send_to(seat, {"type": "error", "message": "Game not started yet"})
            return

        if self.env.current_player != seat:
            await self.send_to(seat, {"type": "error", "message": "Not your turn"})
            return

        legal = self.env.legal_moves(seat)
        matches = find_matching_combo(card_ids, legal, hand=self.env.hands[seat])

        if not matches:
            await self.send_to(seat, {"type": "error", "message": "Invalid combo"})
            return

        await self._do_move(matches[0], seat)

    async def _handle_pass(self, seat: int = 0) -> None:
        if not self.started:
            await self.send_to(seat, {"type": "error", "message": "Game not started yet"})
            return

        if self.env.current_player != seat:
            await self.send_to(seat, {"type": "error", "message": "Not your turn"})
            return

        if self.env.is_leading():
            await self.send_to(seat, {"type": "error", "message": "Cannot pass when leading"})
            return

        legal = self.env.legal_moves(seat)
        pass_combo = next((c for c in legal if c.type == ComboType.PASS), None)
        if pass_combo is None:
            await self.send_to(seat, {"type": "error", "message": "Pass not available"})
            return

        await self._do_move(pass_combo, seat)
