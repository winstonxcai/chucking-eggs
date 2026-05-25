"""Tests for trick history recording in GameRoom.

Tests call room._advance() and room._record_move() directly (no WS) so we
can verify the data structure without standing up a full game server.
"""

from __future__ import annotations

import pytest
from guandan.cards import ComboType
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plays_by_seat(snap: dict) -> dict:
    """Return last non-pass per seat (or pass if only passed) from ordered plays list."""
    result: dict = {}
    for p in snap["plays"]:
        if p["type"] == "play" or p["seat"] not in result:
            result[p["seat"]] = p
    return result


async def _create_room(client: AsyncClient):
    """POST /api/game/create and return the room object."""
    from app.main import game_manager
    res = await client.post("/api/game/create", json={"difficulty": "greedy"})
    assert res.status_code == 200
    return game_manager.get_room(res.json()["game_id"])


def _play_lead(room) -> tuple[int, object]:
    """Have the current player play any non-pass lead. Returns (seat, combo)."""
    env = room.env
    seat = env.current_player
    combo = next(c for c in env.legal_moves(seat) if c.type != ComboType.PASS)
    room._advance(combo, seat)
    room._record_move(seat, combo)
    return seat, combo


def _pass_until_trick_ends(room) -> list[int]:
    """Have every responding player pass until the trick ends. Returns list of passing seats."""
    env = room.env
    passers: list[int] = []
    while not env.done and env.current_trick is not None:
        seat = env.current_player
        pass_combo = next(c for c in env.legal_moves(seat) if c.type == ComboType.PASS)
        room._advance(pass_combo, seat)
        room._record_move(seat, pass_combo)
        passers.append(seat)
    return passers


def _play_one_trick(room) -> bool:
    """Play one complete trick (lead + passes). Returns True if game ended."""
    _play_lead(room)
    if room.env.done:
        return True
    _pass_until_trick_ends(room)
    return room.env.done


# ---------------------------------------------------------------------------
# Trick history recording
# ---------------------------------------------------------------------------

class TestTrickHistoryRecording:
    @pytest.mark.asyncio
    async def test_empty_before_any_plays(self, client: AsyncClient):
        """New game has empty trick_history."""
        room = await _create_room(client)
        assert room.trick_history == []

    @pytest.mark.asyncio
    async def test_one_entry_after_one_trick(self, client: AsyncClient):
        """After a complete lead+pass trick, exactly one snapshot is recorded."""
        room = await _create_room(client)
        _play_one_trick(room)
        assert len(room.trick_history) == 1

    @pytest.mark.asyncio
    async def test_trick_num_starts_at_1(self, client: AsyncClient):
        """First trick snapshot has trick_num=1."""
        room = await _create_room(client)
        _play_one_trick(room)
        assert room.trick_history[0]["trick_num"] == 1

    @pytest.mark.asyncio
    async def test_trick_num_increments(self, client: AsyncClient):
        """trick_num increases by 1 for each successive trick."""
        room = await _create_room(client)
        for _ in range(3):
            if room.env.done:
                break
            _play_one_trick(room)
        for i, snap in enumerate(room.trick_history):
            assert snap["trick_num"] == i + 1

    @pytest.mark.asyncio
    async def test_snapshot_has_all_seats_in_hands_before(self, client: AsyncClient):
        """hands_before contains all 4 absolute seats (str keys '0'-'3')."""
        room = await _create_room(client)
        _play_one_trick(room)
        snap = room.trick_history[0]
        assert set(snap["hands_before"].keys()) == {"0", "1", "2", "3"}

    @pytest.mark.asyncio
    async def test_hands_before_are_lists_of_card_dtos(self, client: AsyncClient):
        """hands_before values are lists of card DTOs with expected fields."""
        room = await _create_room(client)
        _play_one_trick(room)
        snap = room.trick_history[0]
        for seat_key, hand in snap["hands_before"].items():
            assert isinstance(hand, list), f"seat {seat_key}: expected list, got {type(hand)}"
            assert len(hand) > 0, f"seat {seat_key}: hand is empty"
            first = hand[0]
            assert "id" in first and "rank" in first and "suit" in first

    @pytest.mark.asyncio
    async def test_hands_before_contains_played_card(self, client: AsyncClient):
        """The card played by the leader appears in their hands_before entry."""
        room = await _create_room(client)
        env = room.env
        leader = env.current_player
        lead_combo = next(c for c in env.legal_moves(leader) if c.type != ComboType.PASS)
        played_ids = {f"{c.rank}-{c.suit}-{c.deck}" for c in lead_combo.cards}

        room._advance(lead_combo, leader)
        room._record_move(leader, lead_combo)
        _pass_until_trick_ends(room)

        snap = room.trick_history[0]
        before_ids = {c["id"] for c in snap["hands_before"][str(leader)]}
        assert played_ids.issubset(before_ids), (
            f"Played cards {played_ids} not all found in hands_before {before_ids}"
        )

    @pytest.mark.asyncio
    async def test_winner_seat_is_leader(self, client: AsyncClient):
        """When others pass, the leader wins the trick (absolute seat)."""
        room = await _create_room(client)
        env = room.env
        leader = env.current_player
        _play_lead(room)
        _pass_until_trick_ends(room)

        snap = room.trick_history[0]
        assert snap["winner_seat"] == leader

    @pytest.mark.asyncio
    async def test_passes_recorded_in_plays(self, client: AsyncClient):
        """Passing seats appear in plays dict with type='pass'."""
        room = await _create_room(client)
        _play_lead(room)
        passers = _pass_until_trick_ends(room)

        snap = room.trick_history[0]
        by_seat = _plays_by_seat(snap)
        for passer in passers:
            key = str(passer)
            assert key in by_seat, f"seat {passer} pass not recorded"
            assert by_seat[key]["type"] == "pass"

    @pytest.mark.asyncio
    async def test_lead_play_recorded_with_combo(self, client: AsyncClient):
        """Leader's play is recorded with type='play' and a combo field."""
        room = await _create_room(client)
        env = room.env
        leader = env.current_player
        _play_lead(room)
        _pass_until_trick_ends(room)

        snap = room.trick_history[0]
        by_seat = _plays_by_seat(snap)
        lead_entry = by_seat.get(str(leader))
        assert lead_entry is not None
        assert lead_entry["type"] == "play"
        assert "combo" in lead_entry
        assert len(lead_entry["combo"]["cards"]) > 0

    @pytest.mark.asyncio
    async def test_full_game_produces_multiple_tricks(self, client: AsyncClient):
        """Playing a full game to completion produces >1 trick in history."""
        room = await _create_room(client)
        env = room.env
        while not env.done:
            seat = env.current_player
            combo = env.legal_moves(seat)[0]
            room._advance(combo, seat)
            room._record_move(seat, combo)

        assert len(room.trick_history) > 1


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestTrickHistoryRotation:
    @pytest.mark.asyncio
    async def test_rotation_viewer_0_is_identity(self, client: AsyncClient):
        """_rotate_trick_snapshot with viewer=0 leaves all seats unchanged."""
        room = await _create_room(client)
        snap = {
            "trick_num": 1,
            "hands_before": {"0": [], "1": [], "2": [], "3": []},
            "plays": [{"seat": "0", "type": "play", "combo": {}}, {"seat": "2", "type": "pass"}],
            "winner_seat": 0,
        }
        rotated = room._rotate_trick_snapshot(snap, viewer=0)
        assert rotated["winner_seat"] == 0
        assert set(rotated["hands_before"].keys()) == {"0", "1", "2", "3"}
        assert any(p["seat"] == "0" for p in rotated["plays"])

    @pytest.mark.asyncio
    async def test_rotation_shifts_seats(self, client: AsyncClient):
        """viewer=1 shifts all seats by -1 mod 4 (seat 1 becomes 0, 0 becomes 3)."""
        room = await _create_room(client)
        snap = {
            "trick_num": 1,
            "hands_before": {"0": [{"id": "a"}], "1": [{"id": "b"}], "2": [], "3": []},
            "plays": [{"seat": "1", "type": "play", "combo": {}}],
            "winner_seat": 1,
        }
        rotated = room._rotate_trick_snapshot(snap, viewer=1)
        # Seat 1 → 0, seat 0 → 3, seat 2 → 1, seat 3 → 2
        assert rotated["winner_seat"] == 0
        assert rotated["hands_before"]["0"][0]["id"] == "b"   # was seat 1
        assert rotated["hands_before"]["3"][0]["id"] == "a"   # was seat 0
        assert any(p["seat"] == "0" for p in rotated["plays"])

    @pytest.mark.asyncio
    async def test_rotation_none_winner(self, client: AsyncClient):
        """winner_seat=None in snapshot produces None in rotated output."""
        room = await _create_room(client)
        snap = {
            "trick_num": 1,
            "hands_before": {"0": [], "1": [], "2": [], "3": []},
            "plays": [],
            "winner_seat": None,
        }
        rotated = room._rotate_trick_snapshot(snap, viewer=2)
        assert rotated["winner_seat"] is None

    @pytest.mark.asyncio
    async def test_game_over_message_includes_trick_history(self, client: AsyncClient):
        """After a full game, room.trick_history is non-empty and JSON-serializable."""
        import json
        room = await _create_room(client)
        env = room.env
        while not env.done:
            seat = env.current_player
            combo = env.legal_moves(seat)[0]
            room._advance(combo, seat)
            room._record_move(seat, combo)

        assert len(room.trick_history) > 0
        # Rotated history should be JSON-serializable (same check _send_game_over does)
        rotated = [room._rotate_trick_snapshot(t, viewer=0) for t in room.trick_history]
        try:
            json.dumps(rotated)
        except TypeError as e:
            pytest.fail(f"trick_history not JSON-serializable: {e}")
