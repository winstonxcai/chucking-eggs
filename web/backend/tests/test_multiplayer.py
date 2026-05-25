"""Phase 2 tests: room creation, joining, multi-WS game flow, seat isolation."""

from __future__ import annotations

import pytest
from app.game_room import _human_seats_for_mode
from httpx import AsyncClient

from .conftest import create_game

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def create_room(client: AsyncClient, mode: str = "duo", difficulty: str = "greedy") -> dict:
    res = await client.post("/api/room/create", json={"mode": mode, "difficulty": difficulty})
    assert res.status_code == 200, res.text
    return res.json()


async def join_room(client: AsyncClient, room_code: str) -> dict:
    res = await client.post(f"/api/room/join/{room_code}")
    assert res.status_code == 200, res.text
    return res.json()


# ---------------------------------------------------------------------------
# Human seat assignment
# ---------------------------------------------------------------------------

class TestHumanSeats:
    def test_solo_seats(self):
        assert _human_seats_for_mode("solo") == {0}

    def test_duo_seats(self):
        assert _human_seats_for_mode("duo") == {0, 2}

    def test_quad_seats(self):
        assert _human_seats_for_mode("quad") == {0, 1, 2, 3}

    def test_unknown_mode_defaults_to_solo(self):
        assert _human_seats_for_mode("bogus") == {0}


# ---------------------------------------------------------------------------
# Room creation
# ---------------------------------------------------------------------------

class TestRoomCreate:
    @pytest.mark.asyncio
    async def test_create_solo_room(self, client: AsyncClient):
        """/api/room/create with mode=solo returns no room_code."""
        data = await create_room(client, mode="solo")
        assert data["game_id"]
        assert data["room_code"] is None
        assert data["seat"] == 0
        assert len(data["reconnect_token"]) > 10

    @pytest.mark.asyncio
    async def test_create_duo_room(self, client: AsyncClient):
        """Duo room gets a 6-char room code and seat=0 for creator."""
        data = await create_room(client, mode="duo")
        assert data["room_code"] is not None
        assert len(data["room_code"]) == 6
        assert data["seat"] == 0

    @pytest.mark.asyncio
    async def test_create_quad_room(self, client: AsyncClient):
        """Quad room gets a room code."""
        data = await create_room(client, mode="quad")
        assert data["room_code"] is not None
        assert len(data["room_code"]) == 6
        assert data["seat"] == 0

    @pytest.mark.asyncio
    async def test_create_room_rejects_invalid_mode(self, client: AsyncClient):
        res = await client.post("/api/room/create", json={"mode": "bogus", "difficulty": "greedy"})
        assert res.status_code == 422

    @pytest.mark.asyncio
    async def test_create_room_rejects_invalid_difficulty(self, client: AsyncClient):
        res = await client.post("/api/room/create", json={"mode": "solo", "difficulty": "not-a-bot"})
        assert res.status_code == 400
        assert res.json()["detail"] == "Invalid difficulty: not-a-bot"

    @pytest.mark.asyncio
    async def test_forfeit_requires_player_identity(self, client: AsyncClient):
        data = await create_room(client, mode="solo")
        res = await client.post(f"/api/room/{data['game_id']}/forfeit", json={})
        assert res.status_code == 401
        assert res.json()["detail"] == "Player token required"

    @pytest.mark.asyncio
    async def test_solo_room_auto_started(self, client: AsyncClient):
        """Solo room starts immediately."""
        data = await create_room(client, mode="solo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room is not None
        assert room.started is True

    @pytest.mark.asyncio
    async def test_duo_room_not_started(self, client: AsyncClient):
        """Duo room is not started until both humans connect."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room is not None
        assert room.started is False

    @pytest.mark.asyncio
    async def test_duo_room_has_two_human_seats(self, client: AsyncClient):
        """Duo room has seats 0 and 2 as human seats."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room.human_seats == {0, 2}

    @pytest.mark.asyncio
    async def test_two_rooms_get_different_codes(self, client: AsyncClient):
        """Each duo room gets a unique room code."""
        data1 = await create_room(client, mode="duo")
        data2 = await create_room(client, mode="duo")
        assert data1["room_code"] != data2["room_code"]

    @pytest.mark.asyncio
    async def test_each_human_seat_has_own_token(self, client: AsyncClient):
        """Duo room: seats 0 and 2 each have separate reconnect tokens."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        tokens = list(room.reconnect_tokens.values())
        assert len(tokens) == 2
        assert tokens[0] != tokens[1]


# ---------------------------------------------------------------------------
# Room joining
# ---------------------------------------------------------------------------

class TestRoomJoin:
    @pytest.mark.asyncio
    async def test_join_assigns_seat_2(self, client: AsyncClient):
        """In duo mode, joiner gets seat 2 (partner of seat 0)."""
        creator = await create_room(client, mode="duo")
        joiner = await join_room(client, creator["room_code"])
        assert joiner["seat"] == 2
        assert joiner["game_id"] == creator["game_id"]

    @pytest.mark.asyncio
    async def test_join_returns_room_code(self, client: AsyncClient):
        """Join response echoes the room code."""
        creator = await create_room(client, mode="duo")
        joiner = await join_room(client, creator["room_code"])
        assert joiner["room_code"] == creator["room_code"]

    @pytest.mark.asyncio
    async def test_join_nonexistent_code_404(self, client: AsyncClient):
        """Joining a nonexistent room code returns 404."""
        res = await client.post("/api/room/join/ZZZZZZ")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_join_full_duo_room_404(self, client: AsyncClient):
        """Joining a full duo room (both humans connected) returns 404."""
        creator = await create_room(client, mode="duo")
        await join_room(client, creator["room_code"])
        # Second joiner should fail
        res = await client.post(f"/api/room/join/{creator['room_code']}")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_join_lowercase_code_ok(self, client: AsyncClient):
        """Room codes are case-insensitive."""
        creator = await create_room(client, mode="duo")
        code_lower = creator["room_code"].lower()
        joiner = await join_room(client, code_lower)
        assert joiner["game_id"] == creator["game_id"]

    @pytest.mark.asyncio
    async def test_join_gives_different_token(self, client: AsyncClient):
        """Joiner gets a different reconnect token than creator."""
        creator = await create_room(client, mode="duo")
        joiner = await join_room(client, creator["room_code"])
        assert joiner["reconnect_token"] != creator["reconnect_token"]


# ---------------------------------------------------------------------------
# Room status endpoint
# ---------------------------------------------------------------------------

class TestRoomStatus:
    @pytest.mark.asyncio
    async def test_status_shows_mode(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        res = await client.get(f"/api/room/{data['game_id']}/status")
        assert res.status_code == 200
        status = res.json()
        assert status["mode"] == "duo"

    @pytest.mark.asyncio
    async def test_status_shows_room_code(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        res = await client.get(f"/api/room/{data['game_id']}/status")
        status = res.json()
        assert status["room_code"] == data["room_code"]

    @pytest.mark.asyncio
    async def test_status_shows_4_seats(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        res = await client.get(f"/api/room/{data['game_id']}/status")
        status = res.json()
        assert len(status["seats"]) == 4

    @pytest.mark.asyncio
    async def test_status_marks_human_seats(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        res = await client.get(f"/api/room/{data['game_id']}/status")
        status = res.json()
        human_seat_ids = {s["seat"] for s in status["seats"] if s["is_human"]}
        assert human_seat_ids == {0, 2}

    @pytest.mark.asyncio
    async def test_status_not_started_until_all_joined(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        res = await client.get(f"/api/room/{data['game_id']}/status")
        status = res.json()
        assert status["started"] is False

    @pytest.mark.asyncio
    async def test_status_404_missing_room(self, client: AsyncClient):
        res = await client.get("/api/room/doesnotexist/status")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# GameRoom internals: per-seat isolation
# ---------------------------------------------------------------------------

class TestPerSeatIsolation:
    @pytest.mark.asyncio
    async def test_groups_isolated_per_seat(self, client: AsyncClient):
        """Groups created for seat 0 don't appear for seat 2."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])

        # Add a group to seat 0
        room.groups_by_seat[0].append({
            "id": "g0",
            "cardIds": ["1-0-0"],
            "comboType": "SINGLE",
            "comboName": "Single",
        })

        assert room.groups_by_seat.get(0, []) != room.groups_by_seat.get(2, [])
        assert room.groups_by_seat.get(2, []) == []

    @pytest.mark.asyncio
    async def test_reconnect_tokens_are_per_seat(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        joiner = await join_room(client, data["room_code"])
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room.reconnect_tokens[0] == data["reconnect_token"]
        assert room.reconnect_tokens[2] == joiner["reconnect_token"]

    @pytest.mark.asyncio
    async def test_reconnect_seat_validates_token(self, client: AsyncClient):
        """reconnect_seat succeeds with correct token but fails with wrong one."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        gid = data["game_id"]
        token = data["reconnect_token"]

        game_manager.disconnect_seat(gid, 0)
        assert game_manager.reconnect_seat(gid, 0, token) is not None
        assert game_manager.reconnect_seat(gid, 0, "wrong") is None

    @pytest.mark.asyncio
    async def test_disconnect_seat_marks_only_that_seat(self, client: AsyncClient):
        """Disconnecting seat 0 doesn't mark seat 2 as disconnected."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])

        game_manager.disconnect_seat(data["game_id"], 0)
        assert 0 in room.disconnected_seats
        assert 2 not in room.disconnected_seats

    @pytest.mark.asyncio
    async def test_all_disconnected_triggers_room_expiry(self, client: AsyncClient):
        """Room only shows disconnected_at once ALL human seats disconnect."""
        data = await create_room(client, mode="duo")
        await join_room(client, data["room_code"])
        from app.main import game_manager
        gid = data["game_id"]

        game_manager.disconnect_seat(gid, 0)
        room = game_manager.get_room(gid)
        assert room.disconnected_at is None  # seat 2 still "connected"

        game_manager.disconnect_seat(gid, 2)
        assert room.disconnected_at is not None  # now both disconnected


# ---------------------------------------------------------------------------
# GameManager: room code lookup
# ---------------------------------------------------------------------------

class TestGameManagerRoomCodes:
    @pytest.mark.asyncio
    async def test_get_room_by_code(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room_by_code(data["room_code"])
        assert room is not None
        assert room.game_id == data["game_id"]

    @pytest.mark.asyncio
    async def test_remove_room_clears_code(self, client: AsyncClient):
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        code = data["room_code"]
        game_manager.remove_room(data["game_id"])
        assert game_manager.get_room_by_code(code) is None

    @pytest.mark.asyncio
    async def test_solo_rooms_have_no_code(self, client: AsyncClient):
        data = await create_room(client, mode="solo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room.room_code is None

    @pytest.mark.asyncio
    async def test_join_once_started_fails(self, client: AsyncClient):
        """Once a room is marked started, join attempts fail."""
        data = await create_room(client, mode="duo")
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        room.started = True  # forcibly mark started
        res = await client.post(f"/api/room/join/{data['room_code']}")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Backwards compat: Phase 1 API still works
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    @pytest.mark.asyncio
    async def test_create_game_still_works(self, client: AsyncClient):
        """/api/game/create still returns game_id and reconnect_token."""
        data = await create_game(client)
        assert "game_id" in data
        assert "reconnect_token" in data

    @pytest.mark.asyncio
    async def test_create_game_creates_solo_room(self, client: AsyncClient):
        data = await create_game(client)
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room.mode == "solo"
        assert room.human_seats == {0}

    @pytest.mark.asyncio
    async def test_disconnect_room_compat(self, client: AsyncClient):
        data = await create_game(client)
        from app.main import game_manager
        gid = data["game_id"]
        game_manager.disconnect_room(gid)
        room = game_manager.get_room(gid)
        assert room.disconnected_at is not None
        assert room.ws is None

    @pytest.mark.asyncio
    async def test_reconnect_room_compat(self, client: AsyncClient):
        data = await create_game(client)
        from app.main import game_manager
        gid = data["game_id"]
        token = data["reconnect_token"]
        game_manager.disconnect_room(gid)
        room = game_manager.reconnect_room(gid, token)
        assert room is not None
        assert room.disconnected_at is None


# ---------------------------------------------------------------------------
# Reconnect race condition
# ---------------------------------------------------------------------------

class TestReconnectRace:
    @pytest.mark.asyncio
    async def test_stale_disconnect_does_not_kill_new_connection(self, client: AsyncClient):
        """Old WS handler's disconnect_seat must not remove a newer connection."""
        data = await create_game(client)
        from app.main import game_manager
        gid = data["game_id"]
        room = game_manager.get_room(gid)

        # Simulate: old WS stored, then new WS replaces it
        old_ws = object()  # sentinel
        new_ws = object()  # sentinel
        room.connections[0] = old_ws
        room.connections[0] = new_ws  # new connection overwrites

        # Old handler's finally fires with the old ws reference
        game_manager.disconnect_seat(gid, 0, ws=old_ws)

        # New connection should still be alive
        assert room.connections.get(0) is new_ws
        assert 0 not in room.disconnected_seats

    @pytest.mark.asyncio
    async def test_disconnect_without_ws_still_works(self, client: AsyncClient):
        """Calling disconnect_seat without ws arg still removes unconditionally."""
        data = await create_game(client)
        from app.main import game_manager
        gid = data["game_id"]
        room = game_manager.get_room(gid)

        room.connections[0] = object()
        game_manager.disconnect_seat(gid, 0)

        assert 0 not in room.connections
        assert 0 in room.disconnected_seats
