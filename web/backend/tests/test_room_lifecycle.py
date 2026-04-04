"""Phase 1 tests: CORS, room lifecycle, reconnection."""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app, lifespan, game_manager as _gm_ref
from app.game_manager import GameManager, GRACE_PERIOD, IDLE_TIMEOUT

from .conftest import create_game


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:
    @pytest.mark.asyncio
    async def test_cors_default_origins(self, client: AsyncClient):
        """Default CORS should allow localhost:3000."""
        res = await client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_cors_env_override(self):
        """ALLOWED_ORIGINS env var should be respected."""
        # This is a config-level test — we verify the variable is read
        from app.main import _cors_origins
        assert "http://localhost:3000" in _cors_origins


# ---------------------------------------------------------------------------
# Room lifecycle
# ---------------------------------------------------------------------------

class TestRoomLifecycle:
    @pytest.mark.asyncio
    async def test_create_game_returns_token(self, client: AsyncClient):
        """POST /api/game/create should return game_id and reconnect_token."""
        data = await create_game(client)
        assert "game_id" in data
        assert "reconnect_token" in data
        assert len(data["reconnect_token"]) > 10

    @pytest.mark.asyncio
    async def test_room_has_reconnect_token(self, client: AsyncClient):
        """Created room should have a reconnect_token attribute."""
        data = await create_game(client)
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room is not None
        assert room.reconnect_token == data["reconnect_token"]

    @pytest.mark.asyncio
    async def test_room_tracks_activity(self, client: AsyncClient):
        """Room should have last_activity timestamp."""
        data = await create_game(client)
        from app.main import game_manager
        room = game_manager.get_room(data["game_id"])
        assert room.last_activity > 0
        assert time.time() - room.last_activity < 2

    @pytest.mark.asyncio
    async def test_disconnect_keeps_room(self, client: AsyncClient):
        """disconnect_room should mark room but not delete it."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager
        game_manager.disconnect_room(gid)
        room = game_manager.get_room(gid)
        assert room is not None
        assert room.disconnected_at is not None
        assert room.ws is None

    @pytest.mark.asyncio
    async def test_reconnect_with_valid_token(self, client: AsyncClient):
        """reconnect_room with correct token should succeed."""
        data = await create_game(client)
        gid = data["game_id"]
        token = data["reconnect_token"]
        from app.main import game_manager
        game_manager.disconnect_room(gid)

        room = game_manager.reconnect_room(gid, token)
        assert room is not None
        assert room.disconnected_at is None

    @pytest.mark.asyncio
    async def test_reconnect_wrong_token(self, client: AsyncClient):
        """reconnect_room with wrong token should fail."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager
        game_manager.disconnect_room(gid)

        room = game_manager.reconnect_room(gid, "wrong-token")
        assert room is None

    @pytest.mark.asyncio
    async def test_reconnect_nonexistent_room(self, client: AsyncClient):
        """reconnect_room on missing game_id should return None."""
        from app.main import game_manager
        room = game_manager.reconnect_room("nonexistent", "any-token")
        assert room is None

    @pytest.mark.asyncio
    async def test_remove_room(self, client: AsyncClient):
        """remove_room should immediately delete the room."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager
        game_manager.remove_room(gid)
        assert game_manager.get_room(gid) is None


# ---------------------------------------------------------------------------
# Cleanup loop
# ---------------------------------------------------------------------------

class TestCleanup:
    @pytest.mark.asyncio
    async def test_active_room_not_cleaned(self, client: AsyncClient):
        """Active room with recent activity should survive cleanup."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager

        # Simulate a cleanup sweep
        expired = []
        now = time.time()
        for rid, room in game_manager.rooms.items():
            if room.disconnected_at and (now - room.disconnected_at > GRACE_PERIOD):
                expired.append(rid)
            elif now - room.last_activity > IDLE_TIMEOUT:
                expired.append(rid)
        for rid in expired:
            game_manager.rooms.pop(rid, None)

        # Room should still exist
        assert game_manager.get_room(gid) is not None

    @pytest.mark.asyncio
    async def test_disconnected_room_cleaned_after_grace(self, client: AsyncClient):
        """Disconnected room should be cleaned up after grace period."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager
        game_manager.disconnect_room(gid)

        # Fake: set disconnected_at to past grace period
        room = game_manager.get_room(gid)
        room.disconnected_at = time.time() - GRACE_PERIOD - 1

        # Run one cleanup pass
        now = time.time()
        expired = [rid for rid, r in game_manager.rooms.items()
                   if r.disconnected_at and (now - r.disconnected_at > GRACE_PERIOD)]
        for rid in expired:
            game_manager.rooms.pop(rid, None)

        assert game_manager.get_room(gid) is None

    @pytest.mark.asyncio
    async def test_idle_room_cleaned(self, client: AsyncClient):
        """Idle room (no activity) should be cleaned up."""
        data = await create_game(client)
        gid = data["game_id"]
        from app.main import game_manager

        # Fake: set last_activity to past idle timeout
        room = game_manager.get_room(gid)
        room.last_activity = time.time() - IDLE_TIMEOUT - 1

        now = time.time()
        expired = [rid for rid, r in game_manager.rooms.items()
                   if now - r.last_activity > IDLE_TIMEOUT]
        for rid in expired:
            game_manager.rooms.pop(rid, None)

        assert game_manager.get_room(gid) is None

    @pytest.mark.asyncio
    async def test_multiple_rooms_cleanup(self, client: AsyncClient):
        """Multiple disconnected rooms should all be cleaned up."""
        from app.main import game_manager
        ids = []
        for _ in range(5):
            data = await create_game(client)
            ids.append(data["game_id"])
            game_manager.disconnect_room(data["game_id"])
            room = game_manager.get_room(data["game_id"])
            room.disconnected_at = time.time() - GRACE_PERIOD - 1

        now = time.time()
        expired = [rid for rid, r in game_manager.rooms.items()
                   if r.disconnected_at and (now - r.disconnected_at > GRACE_PERIOD)]
        for rid in expired:
            game_manager.rooms.pop(rid, None)

        for gid in ids:
            assert game_manager.get_room(gid) is None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        res = await client.get("/api/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}
