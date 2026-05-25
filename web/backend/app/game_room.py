"""Single game room: manages env, agents, and WebSocket communication."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
import time
from datetime import datetime

from fastapi import WebSocket
from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv

from .ai_service import AIService
from .card_matcher import find_matching_combo
from .persistence import (
    compute_elo_changes,
    compute_forfeit_elo_change,
    persist_completed_game,
    persist_forfeit_game,
)
from .serializer import card_to_dto, combo_to_dto, serialize_game_state, sort_hand
from .settings import get_settings
from .tasks import create_logged_task

logger = logging.getLogger(__name__)


SETTINGS = get_settings()
DATA_DIR = SETTINGS.data_dir

ACTION_PAUSE = SETTINGS.action_pause_s    # seconds each AI play is visible before next turn
AI_THINK_PAUSE = SETTINGS.ai_think_pause_s  # seconds for "thinking" animation before AI move

DISCONNECT_TAKEOVER_S = SETTINGS.disconnect_takeover_s
HUMAN_TURN_TIMEOUT_S = SETTINGS.human_turn_timeout_s


def _human_seats_for_mode(mode: str) -> set[int]:
    """Return which seats are human-controlled based on mode."""
    if mode == "duo":
        return {0, 2}   # partners
    if mode == "quad":
        return {0, 1, 2, 3}
    return {0}  # solo


class GameRoom:
    def __init__(self, game_id: str, mode: str, difficulty: str, ai_service: AIService, seed: int | None = None):
        self.game_id = game_id
        self.mode = mode
        self.difficulty = difficulty
        self.ai_service = ai_service
        self.env = GuanDanEnv(level_rank=Rank.TWO)
        self.env.reset(seed=seed)

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
        self.created_at: float = time.time()
        self.seat_player_ids: dict[int, str | None] = {}  # populated at HTTP create/join time
        self.disconnected_seats: dict[int, float] = {}  # seat -> disconnect timestamp

        # Trick tracking — dict persists each seat's last action so played cards
        # stay on the board even after opponents respond (not cleared on new trick)
        self.trick_plays: dict[int, Combo] = {}
        self._start_time = time.time()

        # Trick history — accumulated during game, sent with game_over
        self.trick_history: list[dict] = []
        self.current_trick_num: int = 0
        self.current_trick_plays: list[dict] = []   # ordered sequence of {seat, type, combo?}
        self.current_trick_hands_before: dict[str, list] = {}  # str(seat) -> [card DTOs]

        # AI lock — prevents concurrent AI run coroutines
        self._ai_lock = asyncio.Lock()

        # Player IDs for Elo tracking (set by WS handler on connect)
        self.player_ids: dict[int, str | None] = {seat: None for seat in self.human_seats}

        # Move count per seat (for abort eligibility)
        self.moves_played_by: dict[int, int] = {i: 0 for i in range(4)}

        # AFK rope: absolute monotonic deadline per seat
        self.turn_deadlines: dict[int, float] = {}
        # Cached wall-clock deadline (Unix ms) — avoids jitter from repeated conversion
        self.turn_deadline_wallclock_ms: dict[int, int] = {}
        self._turn_timeout_tasks: dict[int, asyncio.Task] = {}

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

    @property
    def start_time(self) -> float:
        return self._start_time

    def clear_turn_deadline(self, seat: int) -> None:
        self.turn_deadlines.pop(seat, None)
        self.turn_deadline_wallclock_ms.pop(seat, None)
        task = self._turn_timeout_tasks.pop(seat, None)
        if task and task is not asyncio.current_task():
            task.cancel()

    def clear_all_turn_deadlines(self) -> None:
        for seat in set(self.turn_deadlines) | set(self._turn_timeout_tasks):
            self.clear_turn_deadline(seat)

    def ensure_turn_deadline(self, seat: int) -> None:
        """Start or refresh the AFK deadline for the active human seat."""
        self.clear_turn_deadline(seat)
        deadline = asyncio.get_event_loop().time() + HUMAN_TURN_TIMEOUT_S
        self.turn_deadlines[seat] = deadline
        self.turn_deadline_wallclock_ms[seat] = int((time.time() + HUMAN_TURN_TIMEOUT_S) * 1000)
        self._turn_timeout_tasks[seat] = create_logged_task(
            self._auto_play_after_deadline(seat, deadline),
            name=f"turn-timeout:{self.game_id}:{seat}",
        )

    async def _auto_play_after_deadline(self, seat: int, deadline: float) -> None:
        min_wait = 0.05 if HUMAN_TURN_TIMEOUT_S == 0 else 0.5
        await asyncio.sleep(max(min_wait, deadline - asyncio.get_event_loop().time()))
        if self.turn_deadlines.get(seat) != deadline:
            return
        if seat not in self.connections:
            return
        await self.handle_afk_timeout(seat)

    # ---------------------------------------------------------------------------
    # Difficulty
    # ---------------------------------------------------------------------------

    def set_difficulty(self, difficulty: str) -> None:
        """Change bot difficulty before game starts (duo/solo modes)."""
        self.difficulty = difficulty
        bots = self.ai_service.pick_bots(difficulty)
        self.agent = self.ai_service.get_agent(difficulty)
        if self.mode == "solo":
            self.player_infos[1] = bots[0]
            self.player_infos[2] = bots[1]
            self.player_infos[3] = bots[2]
        elif self.mode == "duo":
            # seats 1 and 3 are bots; seat 2 is human partner
            self.player_infos[1] = bots[0]
            self.player_infos[3] = bots[2]

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
            self._takeover_tasks[seat] = create_logged_task(
                self._takeover_after_delay(seat),
                name=f"disconnect-takeover:{self.game_id}:{seat}",
            )

    def cancel_disconnect_takeover(self, seat: int) -> None:
        task = self._takeover_tasks.pop(seat, None)
        if task:
            task.cancel()

    async def _takeover_after_delay(self, seat: int) -> None:
        await asyncio.sleep(DISCONNECT_TAKEOVER_S)
        if seat not in self.connections and not self.env.done:  # still disconnected + game running
            if self.mode in ("duo", "quad"):
                await self.handle_forfeit(seat)
            else:
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
            mode=self.mode,
        )
        # Include cached wall-clock AFK deadline when it's this seat's turn.
        wc_deadline = self.turn_deadline_wallclock_ms.get(seat)
        if wc_deadline is not None and self.env.current_player == seat:
            state["turn_deadline_ms"] = wc_deadline
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
        """Update display state (used by serialize_game_state)."""
        self.trick_plays[seat] = combo

    def _advance(self, combo: Combo, seat: int) -> tuple[int, bool]:
        """Execute env.step() and record into trick history.

        Detects trick boundaries by watching env.current_trick go None→set (new
        trick starting) and set→None (trick ending via passes) or done=True.
        """
        starting_new_trick = self.env.current_trick is None
        if starting_new_trick:
            self.current_trick_num += 1
            self.current_trick_hands_before = {
                str(s): [card_to_dto(c) for c in sort_hand(self.env.hands[s], self.env.level_rank)]
                for s in range(4)
            }
            self.current_trick_plays = []

        next_player, done = self.env.step(combo)

        if combo.type == ComboType.PASS:
            self.current_trick_plays.append({"seat": str(seat), "type": "pass"})
        else:
            self.current_trick_plays.append({
                "seat": str(seat),
                "type": "play",
                "combo": combo_to_dto(combo, self.env.level_rank),
            })

        trick_just_ended = (self.env.current_trick is None and not starting_new_trick) or done
        if trick_just_ended and self.current_trick_plays:
            self.trick_history.append({
                "trick_num": self.current_trick_num,
                "hands_before": self.current_trick_hands_before,
                "plays": list(self.current_trick_plays),
                "winner_seat": self.env.trick_winner,
            })
            self.current_trick_plays = []

        return next_player, done

    def _rotate_trick_snapshot(self, trick: dict, viewer: int) -> dict:
        """Rotate absolute seats in a trick snapshot to viewer-relative."""
        def r(s) -> int | None:
            return None if s is None else (int(s) - viewer + 4) % 4
        rotated_plays = []
        for p in trick["plays"]:
            entry: dict = {"seat": str(r(int(p["seat"]))), "type": p["type"]}
            if "combo" in p:
                entry["combo"] = p["combo"]
            rotated_plays.append(entry)
        return {
            "trick_num": trick["trick_num"],
            "hands_before": {
                str(r(int(s))): cards for s, cards in trick["hands_before"].items()
            },
            "plays": rotated_plays,
            "winner_seat": r(trick["winner_seat"]),
        }

    # ---------------------------------------------------------------------------
    # Game result persistence (DB + Elo)
    # ---------------------------------------------------------------------------

    def _completed_game_rewards(self) -> dict[int, float] | None:
        try:
            return self.env.get_rewards()
        except (ValueError, IndexError):
            logger.debug(
                "Game %s is done with incomplete finish order: %s",
                self.game_id,
                self.env.finish_order,
            )
            return None

    def _record_game_result_local(self) -> None:
        """Append game record to local JSONL file (always runs, no DB dep)."""
        primary = min(self.human_seats) if self.human_seats else 0
        fo = self.env.finish_order
        human_pos = fo.index(primary) + 1 if primary in fo else -1
        rewards = self._completed_game_rewards()
        primary_reward = rewards.get(primary) if rewards else None
        team_result = (
            "win" if primary_reward is not None and primary_reward > 0
            else "loss" if primary_reward is not None
            else "unknown"
        )
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "difficulty": self.difficulty,
            "agent": self.ai_service.get_agent_name(self.difficulty),
            "finish_order": fo,
            "human_finish_pos": human_pos,
            "team_result": team_result,
            "reward": primary_reward,
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "human_games.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    async def _send_game_over(self) -> None:
        self._record_game_result_local()
        rewards = self._completed_game_rewards()
        if rewards is None:
            return
        # Compute Elo before broadcasting so the modal can show the change
        elo_changes = await compute_elo_changes(self, rewards)
        # DB persistence runs in background — never blocks game-over broadcast
        create_logged_task(
            persist_completed_game(self, rewards, elo_changes),
            name=f"persist-completed-game:{self.game_id}",
        )
        # Send per-viewer with rotated seats (frontend always sees itself as seat 0)
        for viewer in list(self.connections.keys()):
            def r(s: int, v: int = viewer) -> int:
                return (s - v + 4) % 4
            await self.send_to(viewer, {
                "type": "game_over",
                "finish_order": [r(s) for s in self.env.finish_order],
                "rewards": {r(s): rv for s, rv in rewards.items()},
                "players": [
                    {"seat": r(i), "name": self.player_infos[i]["name"]}
                    for i in self.env.finish_order
                ],
                "elo_changes": {
                    str(r(seat)): {"delta": v["delta"], "before": v["before"], "after": v["after"]}
                    for seat, v in elo_changes.items()
                },
                "trick_history": [
                    self._rotate_trick_snapshot(t, viewer) for t in self.trick_history
                ],
            })
        self.clear_all_turn_deadlines()

    # ---------------------------------------------------------------------------
    # Forfeit
    # ---------------------------------------------------------------------------

    async def handle_forfeit(self, forfeiter_seat: int) -> None:
        """Handle a player forfeiting. Game ends immediately for all players.

        Only the forfeiter loses ELO; partner and opponents get no change (voided).
        """
        if self.env.done:
            return
        self.env.done = True
        self.clear_all_turn_deadlines()
        forfeiter_name = self.player_infos[forfeiter_seat]["name"]

        elo_changes = await compute_forfeit_elo_change(self, forfeiter_seat)
        await persist_forfeit_game(self, forfeiter_seat, elo_changes)

        # Send per-viewer with rotated seats
        for viewer in list(self.connections.keys()):
            def r(s: int, v: int = viewer) -> int:
                return (s - v + 4) % 4
            await self.send_to(viewer, {
                "type": "game_forfeited",
                "forfeiter_seat": r(forfeiter_seat),
                "forfeiter_name": forfeiter_name,
                "elo_changes": {
                    str(r(seat)): {"delta": v["delta"], "before": v["before"], "after": v["after"]}
                    for seat, v in elo_changes.items()
                },
            })

    # ---------------------------------------------------------------------------
    # AI turns
    # ---------------------------------------------------------------------------

    async def run_ai_turns(self) -> None:
        """Run AI turns until it's a human's turn or the game ends."""
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
                    # Human's turn — set AFK deadline (monotonic for wait_for) and stop
                    if seat in self.connections:
                        self.ensure_turn_deadline(seat)
                    break

                next_player, done = self._advance(combo, seat)
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
        """Auto-play when a human's turn times out.

        Leading (must play): random legal move.
        Not leading (can pass): pass.
        """
        if self.env.done or self.env.current_player != seat:
            return
        self.clear_turn_deadline(seat)
        legal = self.env.legal_moves(seat)
        leading = self.env.is_leading()
        if leading:
            combo = random.choice(legal)
        else:
            combo = next(c for c in legal if c.type == ComboType.PASS)
        await self._do_move(combo, seat)
        await self.send_to(seat, {"type": "auto_played", "leading": leading})

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
        combo_type = str(data.get("combo_type", ""))[:32]
        combo_name = str(data.get("combo_name", ""))[:64]
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
        self.clear_turn_deadline(seat)
        next_player, done = self._advance(combo, seat)
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

        if (
            not isinstance(card_ids, list)
            or any(not isinstance(card_id, str) for card_id in card_ids)
            or len(card_ids) != len(set(card_ids))
        ):
            await self.send_to(seat, {"type": "error", "message": "Invalid combo"})
            return

        legal = self.env.legal_moves(seat)
        matches = find_matching_combo(card_ids, legal, hand=self.env.hands[seat], level_rank=self.env.level_rank)

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
