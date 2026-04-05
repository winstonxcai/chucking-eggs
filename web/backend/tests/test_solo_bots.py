"""Solo bot tests: verify all 4 difficulties create valid games.

HTTP-only tests verify agents load and games create successfully for all
difficulties. The send_to fix (json.dumps outside try/except) ensures
serialization errors surface rather than silently hanging.
"""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

DIFFICULTIES = ["easy", "medium", "hard", "expert"]


# ---------------------------------------------------------------------------
# HTTP smoke tests
# ---------------------------------------------------------------------------

class TestSoloBotHTTP:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    async def test_create_game_returns_200(self, client: AsyncClient, difficulty: str):
        """POST /api/game/create should succeed for all 4 difficulties."""
        res = await client.post("/api/game/create", json={"difficulty": difficulty})
        assert res.status_code == 200, f"{difficulty}: {res.text}"
        data = res.json()
        assert "game_id" in data
        assert "reconnect_token" in data

    @pytest.mark.asyncio
    async def test_expert_agent_loaded(self, client: AsyncClient):
        """Expert difficulty should use the RL agent (or strategic fallback) — not None."""
        from app.main import game_manager
        ai = game_manager.ai_service
        agent_name = ai.get_agent_name("expert")
        assert agent_name is not None
        assert isinstance(agent_name, str)
        assert len(agent_name) > 0

    @pytest.mark.asyncio
    async def test_all_difficulties_have_agents(self, client: AsyncClient):
        """Every difficulty must map to a non-None agent."""
        from app.main import game_manager
        ai = game_manager.ai_service
        for diff in DIFFICULTIES:
            agent = ai.get_agent(diff)
            assert agent is not None, f"No agent for {diff}"


# ---------------------------------------------------------------------------
# Game state serialization tests (no WS — tests the serializer directly)
# ---------------------------------------------------------------------------

class TestGameStateSerialization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    async def test_game_state_is_json_serializable(self, client: AsyncClient, difficulty: str):
        """game_state for every difficulty must be fully JSON-serializable (no numpy types)."""
        from app.main import game_manager
        from app.serializer import serialize_game_state

        res = await client.post("/api/game/create", json={"difficulty": difficulty})
        data = res.json()
        room = game_manager.get_room(data["game_id"])
        assert room is not None

        state = serialize_game_state(
            room.env, room.game_id, 0, room.player_infos,
            trick_plays=room.trick_plays,
            groups=room.groups_by_seat.get(0, []),
        )
        msg = {"type": "game_state", **state}

        # This is what send_to does — must not raise TypeError
        try:
            payload = json.dumps(msg)
        except TypeError as e:
            pytest.fail(f"[{difficulty}] game_state not JSON-serializable: {e}")

        assert len(payload) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    async def test_game_state_has_correct_fields(self, client: AsyncClient, difficulty: str):
        """game_state must have my_seat=0, non-empty hand, and 4 players."""
        from app.main import game_manager
        from app.serializer import serialize_game_state

        res = await client.post("/api/game/create", json={"difficulty": difficulty})
        data = res.json()
        room = game_manager.get_room(data["game_id"])

        state = serialize_game_state(
            room.env, room.game_id, 0, room.player_infos,
        )

        assert state["my_seat"] == 0, f"[{difficulty}] my_seat={state['my_seat']}"
        assert len(state["my_hand"]) > 0, f"[{difficulty}] Hand is empty"
        assert len(state["players"]) == 4, f"[{difficulty}] player count={len(state['players'])}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    async def test_legal_moves_when_human_leads(self, client: AsyncClient, difficulty: str):
        """If current_player is seat 0, legal_moves must be non-empty."""
        from app.main import game_manager
        from app.serializer import serialize_game_state

        res = await client.post("/api/game/create", json={"difficulty": difficulty})
        data = res.json()
        room = game_manager.get_room(data["game_id"])

        state = serialize_game_state(
            room.env, room.game_id, 0, room.player_infos,
        )

        if state["is_my_turn"]:
            assert len(state["legal_moves"]) > 0, (
                f"[{difficulty}] is_my_turn=True but no legal_moves"
            )
