"""Single game room: manages env, agents, and WebSocket communication."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv

from .ai_service import AIService
from .card_matcher import find_matching_combo
from .serializer import combo_to_dto, serialize_game_state

logger = logging.getLogger(__name__)

ACTION_PAUSE = 0.9    # seconds each AI play is visible before next turn
AI_THINK_PAUSE = 0.5  # seconds for "thinking" animation before AI move

DISCONNECT_TAKEOVER_S = 60   # seconds before AI takes over a disconnected seat
HUMAN_TURN_TIMEOUT_S = 90    # seconds before AFK auto-play kicks in


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
        self.assigned_seats: set[int] = {0}  # creator always gets seat 0

        # Room lifecycle
        self.last_activity = time.time()
        self.disconnected_seats: dict[int, float] = {}  # seat -> disconnect timestamp

        # Trick tracking
        self.trick_plays: list[tuple[int, Combo]] = []
        self._start_time = time.time()

        # AI lock — prevents concurrent AI run coroutines
        self._ai_lock = asyncio.Lock()

        # Player IDs for Elo tracking (set by WS handler on connect)
        self.player_ids: dict[int, str | None] = {seat: None for seat in self.human_seats}

        # Move count per seat (for abort eligibility)
        self.moves_played_by: dict[int, int] = {i: 0 for i in range(4)}

        # AFK rope: absolute monotonic deadline per seat
        self.turn_deadlines: dict[int, float] = {}

        # Disconnect takeover tasks
        self._takeover_tasks: dict[int, asyncio.Task] = {}

        # Agent + player infos
        bots = ai_service.pick_bots(difficulty)
        self.agent = ai_service.get_agent(difficulty)
        self.player_infos: list[dict] = [
            {"name": "You", "avatar": None, "elo": 1200},
            bots[0],
            bots[1],
            bots[2],
        ]
        # Rename partner/player slots for multiplayer
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
    # Disconnect takeover
    # ---------------------------------------------------------------------------

    def schedule_disconnect_takeover(self, seat: int) -> None:
        """Schedule AI takeover if seat stays disconnected for DISCONNECT_TAKEOVER_S."""
        existing = self._takeover_tasks.pop(seat, None)
        if existing:
            existing.cancel()
        if not self.env.done:
            self._takeover_tasks[seat] = asyncio.create_task(
                self._takeover_after_delay(seat)
            )

    def cancel_disconnect_takeover(self, seat: int) -> None:
        task = self._takeover_tasks.pop(seat, None)
        if task:
            task.cancel()

    async def _takeover_after_delay(self, seat: int) -> None:
        await asyncio.sleep(DISCONNECT_TAKEOVER_S)
        if seat not in self.connections:  # still disconnected
            await self._do_ai_takeover(seat)

    async def _do_ai_takeover(self, seat: int) -> None:
        """Remove seat from human control; AI takes over silently."""
        self.human_seats.discard(seat)
        self._takeover_tasks.pop(seat, None)
        await self.broadcast({
            "type": "player_disconnected",
            "seat": seat,
            "username": self.player_infos[seat]["name"],
            "ai_takeover": True,
        })
        if not self.env.done and self.env.current_player == seat:
            await self.run_ai_turns()

    # ---------------------------------------------------------------------------
    # Sending
    # ---------------------------------------------------------------------------

    async def send_to(self, seat: int, data: dict) -> None:
        ws = self.connections.get(seat)
        if ws:
            payload = json.dumps(data)
            try:
                await ws.send_text(payload)
            except Exception:
                pass  # seat disconnected mid-send

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
        # Include AFK deadline when it's this seat's turn
        deadline = self.turn_deadlines.get(seat)
        if deadline is not None and self.env.current_player == seat:
            state["turn_deadline_ms"] = int(deadline * 1000)
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
        if combo.type != ComboType.PASS:
            self.trick_plays = []
        self.trick_plays.append((seat, combo))

    # ---------------------------------------------------------------------------
    # Game result persistence (DB + Elo)
    # ---------------------------------------------------------------------------

    def _record_game_result_local(self) -> None:
        """Append game record to local JSONL file (always runs, no DB dep)."""
        primary = min(self.human_seats) if self.human_seats else 0
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

    async def _persist_to_db(self) -> None:
        """Save game to MongoDB and update Elo for all human seats.

        Wrapped in try/except — a DB failure must NOT block game-over delivery.
        """
        try:
            from . import db as _db
            from .elo import compute_elo_delta, BOT_ELOS

            fo = self.env.finish_order
            rewards = self.env.get_rewards()

            # Fetch current Elo + games_played for each human seat
            human_docs: dict[int, dict] = {}
            for seat in self.human_seats:
                pid = self.player_ids.get(seat)
                if pid:
                    doc = await _db.get_player_by_id(pid)
                    if doc:
                        human_docs[seat] = doc

            # Compute Elo delta per human seat
            # Team A: seats 0+2, Team B: seats 1+3
            def _seat_elo(seat: int) -> int:
                if seat in human_docs:
                    return human_docs[seat]["elo"]
                return self.player_infos[seat].get("elo") or BOT_ELOS.get(self.difficulty, 1500)

            new_elos: dict[int, int] = {}
            for seat in self.human_seats:
                if seat not in human_docs:
                    continue
                partner = 2 if seat == 0 else (0 if seat == 2 else (3 if seat == 1 else 1))
                opps = [s for s in range(4) if s != seat and s != partner]
                delta = compute_elo_delta(
                    player_elo=human_docs[seat]["elo"],
                    partner_elo=_seat_elo(partner),
                    opp1_elo=_seat_elo(opps[0]),
                    opp2_elo=_seat_elo(opps[1]),
                    player_games=human_docs[seat].get("games_played", 0),
                    won=rewards[seat] > 0,
                    reward=rewards[seat],
                )
                new_elos[seat] = human_docs[seat]["elo"] + delta

            # Build game document
            game_doc = {
                "_id": self.game_id,
                "mode": self.mode,
                "difficulty": self.difficulty,
                "duration_seconds": int(time.time() - self._start_time),
                "played_at": datetime.now(timezone.utc),
                "players": [
                    {
                        "player_id": self.player_ids.get(seat) if seat in self.human_seats else None,
                        "display_name": self.player_infos[seat]["name"],
                        "is_bot": seat not in self.human_seats,
                        "seat": seat,
                        "finish_pos": fo.index(seat) + 1 if seat in fo else -1,
                        "team_result": (
                            "win" if rewards[seat] > 0
                            else ("loss" if rewards[seat] < 0 else "neutral")
                        ),
                        "elo_before": human_docs.get(seat, {}).get("elo") if seat in self.human_seats else None,
                        "elo_after": new_elos.get(seat) if seat in self.human_seats else None,
                    }
                    for seat in range(4)
                ],
            }

            await _db.save_game(game_doc)

            # Update each human's Elo
            for seat, new_elo in new_elos.items():
                pid = self.player_ids.get(seat)
                if pid:
                    await _db.update_player_elo(pid, new_elo)

        except Exception:
            logger.exception("DB persistence failed for game %s — game-over still delivered", self.game_id)

    async def _send_game_over(self) -> None:
        self._record_game_result_local()
        # DB persistence runs concurrently — never blocks game-over broadcast
        asyncio.create_task(self._persist_to_db())
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
        """Run AI turns and human auto-passes until a human has real choices or the game ends."""
        async with self._ai_lock:
            while not self.env.done:
                seat = self.env.current_player
                legal = self.env.legal_moves(seat)
                only_pass = len(legal) == 1 and legal[0].type == ComboType.PASS

                if seat not in self.human_seats:
                    # AI turn
                    if only_pass:
                        combo = legal[0]
                    else:
                        await self.broadcast({"type": "ai_thinking", "seat": seat})
                        await asyncio.sleep(AI_THINK_PAUSE)
                        combo = await self.ai_service.get_ai_move(self.agent, self.env, seat)
                else:
                    # Human's turn — set AFK deadline and stop
                    self.turn_deadlines[seat] = asyncio.get_event_loop().time() + HUMAN_TURN_TIMEOUT_S
                    break

                next_player, done = self.env.step(combo)
                self._record_move(seat, combo)
                self.moves_played_by[seat] += 1

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
    # AFK timeout
    # ---------------------------------------------------------------------------

    async def handle_afk_timeout(self, seat: int) -> None:
        """Auto-play least disruptive move when a human's turn times out."""
        if self.env.done or self.env.current_player != seat:
            return
        self.turn_deadlines.pop(seat, None)
        legal = self.env.legal_moves(seat)
        # Prefer PASS; else smallest single card; else first legal move
        pass_combo = next((c for c in legal if c.type == ComboType.PASS), None)
        if pass_combo:
            combo = pass_combo
        else:
            singles = [c for c in legal if c.type == ComboType.SINGLE]
            combo = min(singles, key=lambda c: c.cards[0].rank) if singles else legal[0]
        await self._do_move(combo, seat)

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
        elif msg_type == "abort":
            await self._handle_abort(seat)
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

    async def _handle_abort(self, seat: int) -> None:
        """Abort before first move — AI takes over this seat immediately, no Elo change."""
        if self.moves_played_by.get(seat, 0) > 0:
            await self.send_to(seat, {"type": "error", "message": "Cannot abort after playing your first card"})
            return
        # Mark seat as having aborted — exclude from Elo at game end
        self.player_ids[seat] = None
        await self._do_ai_takeover(seat)

    async def _do_move(self, combo: Combo, seat: int) -> None:
        self.turn_deadlines.pop(seat, None)
        next_player, done = self.env.step(combo)
        self._record_move(seat, combo)
        self.moves_played_by[seat] += 1

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
