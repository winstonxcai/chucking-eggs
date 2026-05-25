"""Opt-in smoke test against a real MongoDB server."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from app import db
from app.settings import Settings

pytestmark = pytest.mark.asyncio


@pytest.mark.mongo
async def test_real_mongo_player_and_game_round_trip() -> None:
    if os.getenv("RUN_MONGO_SMOKE") != "true":
        pytest.skip("Set RUN_MONGO_SMOKE=true to run against a real MongoDB instance")

    mongodb_url = os.getenv("MONGODB_URL")
    if not mongodb_url:
        pytest.skip("Set MONGODB_URL for real MongoDB smoke testing")

    player_id = f"mongo-smoke-{uuid.uuid4()}"
    game_id = f"mongo-smoke-game-{uuid.uuid4()}"
    settings = Settings(
        mongodb_url=mongodb_url,
        use_mock_db=False,
        app_env="production",
        auth_secret="mongo-smoke-secret",
        allowed_origins=["http://127.0.0.1:3000"],
        data_dir=Path("/tmp/chucking-eggs-mongo-smoke"),
        redis_url="",
        cleanup_interval_s=30,
        lobby_timeout_s=300,
        disconnect_takeover_s=60,
        human_turn_timeout_s=30,
        action_pause_s=0,
        ai_think_pause_s=0,
        profile_game_limit=100,
    )

    await db.close_db()
    try:
        await db.init_db(settings)
        await db.get_db().command("ping")
        player = await db.get_or_create_player(player_id, username=f"mongo-{uuid.uuid4().hex[:8]}", is_test=True)
        assert player["_id"] == player_id

        inserted = await db.save_game_if_absent({
            "_id": game_id,
            "mode": "duo",
            "difficulty": "greedy",
            "duration_seconds": 1,
            "played_at": player["created_at"],
            "players": [
                {
                    "player_id": player_id,
                    "display_name": player["username"],
                    "is_bot": False,
                    "seat": 0,
                    "finish_pos": 1,
                    "team_result": "win",
                    "elo_before": 1200,
                    "elo_after": 1210,
                },
            ],
        })
        assert inserted is True
        duplicate = await db.save_game_if_absent({
            "_id": game_id,
            "mode": "duo",
            "difficulty": "greedy",
            "duration_seconds": 1,
            "played_at": player["created_at"],
            "players": [],
        })
        assert duplicate is False
        stats = await db.get_player_stats(player_id)
        assert len(stats["games"]) == 1
    finally:
        try:
            real_db = db.get_db()
            await real_db.players.delete_one({"_id": player_id})
            await real_db.games.delete_one({"_id": game_id})
        finally:
            await db.close_db()
            await db.init_db()
