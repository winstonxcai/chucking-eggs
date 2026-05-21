"""MongoDB client via Motor. All operations are async."""

from __future__ import annotations

import os
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def get_db() -> AsyncIOMotorDatabase:
    assert _db is not None, "DB not initialized — call init_db() first"
    return _db


async def init_db() -> None:
    global _client, _db
    url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    # Use in-memory mock when MongoDB is unavailable (local dev without mongo).
    if os.getenv("USE_MOCK_DB", "").lower() in ("1", "true", "yes"):
        from mongomock_motor import AsyncMongoMockClient
        _client = AsyncMongoMockClient()  # type: ignore[assignment]
    else:
        _client = AsyncIOMotorClient(url, serverSelectionTimeoutMS=3000)
    _db = _client["chucking_eggs"]
    # Idempotent indexes
    try:
        await _db.players.create_index("username", unique=True)
        await _db.players.create_index("email", unique=True, sparse=True)
        await _db.games.create_index([("players.player_id", 1)])
        await _db.games.create_index([("played_at", -1)])
    except Exception:
        # Fall back to mock DB when real MongoDB is unreachable
        from mongomock_motor import AsyncMongoMockClient
        _client = AsyncMongoMockClient()  # type: ignore[assignment]
        _db = _client["chucking_eggs"]


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

async def is_username_taken(username: str) -> bool:
    return await get_db().players.find_one({"username": username}, {"_id": 1}) is not None


async def get_or_create_player(player_id: str, username: str, email: str | None = None, is_test: bool = False) -> dict:
    """Insert player if not exists. Returns the (possibly pre-existing) doc."""
    db = get_db()
    doc = {
        "_id": player_id,
        "username": username,
        "elo": 1200,
        "games_played": 0,
        "is_test": is_test,
        "created_at": datetime.utcnow(),
    }
    if email:
        doc["email"] = email
    await db.players.update_one(
        {"_id": player_id},
        {"$setOnInsert": doc},
        upsert=True,
    )
    return await db.players.find_one({"_id": player_id})


async def get_player_by_id(player_id: str) -> dict | None:
    return await get_db().players.find_one({"_id": player_id})


async def get_player_by_username(username: str) -> dict | None:
    return await get_db().players.find_one({"username": username})


async def update_player_elo(player_id: str, new_elo: int) -> None:
    await get_db().players.update_one(
        {"_id": player_id},
        {"$set": {"elo": new_elo}, "$inc": {"games_played": 1}},
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

async def save_game(game_doc: dict) -> None:
    await get_db().games.insert_one(game_doc)


async def get_player_stats(player_id: str) -> dict:
    """Returns player doc + recent games for profile page."""
    db = get_db()
    player = await db.players.find_one({"_id": player_id})
    if not player:
        return {}
    games = await db.games.find(
        {"players.player_id": player_id},
        sort=[("played_at", -1)],
    ).to_list(length=None)
    return {"player": player, "games": games}


async def get_leaderboard() -> list[dict]:
    """Top 50 real (non-test) human players by Elo."""
    cursor = get_db().players.find({"is_test": {"$ne": True}}, sort=[("elo", -1)], limit=50)
    return await cursor.to_list(length=50)
