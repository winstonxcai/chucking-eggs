"""MongoDB client via Motor. All operations are async."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .settings import Settings, get_settings
from .types import GameRecord, PlayerDoc

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None
logger = logging.getLogger(__name__)


def _create_mongo_client(settings: Settings) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(settings.mongodb_url, serverSelectionTimeoutMS=3000)


def _create_mock_client() -> AsyncIOMotorClient:
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()  # type: ignore[return-value]


async def _create_indexes(db: Any) -> None:
    await db.players.create_index("username", unique=True)
    await db.players.create_index("email", unique=True, sparse=True)
    await db.games.create_index([("players.player_id", 1)])
    await db.games.create_index([("played_at", -1)])


def _close_client() -> None:
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialized; call init_db() during app startup")
    return _db


async def init_db(settings: Settings | None = None) -> None:
    global _client, _db
    settings = settings or get_settings()
    if settings.use_mock_db:
        logger.info("Using in-memory mock MongoDB because USE_MOCK_DB is enabled")
        _client = _create_mock_client()
    else:
        _client = _create_mongo_client(settings)
    _db = _client["chucking_eggs"]
    try:
        await _create_indexes(_db)
    except Exception as exc:
        if settings.use_mock_db:
            raise
        _close_client()
        if settings.is_production:
            raise RuntimeError("MongoDB initialization failed in production") from exc
        logger.warning("MongoDB unavailable; falling back to in-memory mock DB", exc_info=True)
        _client = _create_mock_client()
        _db = _client["chucking_eggs"]
        await _create_indexes(_db)


async def close_db() -> None:
    _close_client()


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

async def is_username_taken(username: str) -> bool:
    return await get_db().players.find_one({"username": username}, {"_id": 1}) is not None


async def get_or_create_player(
    player_id: str,
    username: str,
    email: str | None = None,
    is_test: bool = False,
) -> PlayerDoc:
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


async def get_player_by_id(player_id: str) -> PlayerDoc | None:
    return await get_db().players.find_one({"_id": player_id})


async def get_player_by_username(username: str) -> PlayerDoc | None:
    return await get_db().players.find_one({"username": username})


async def update_player_elo(player_id: str, new_elo: int) -> None:
    await get_db().players.update_one(
        {"_id": player_id},
        {"$set": {"elo": new_elo}, "$inc": {"games_played": 1}},
    )


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

async def save_game_if_absent(game_doc: GameRecord) -> bool:
    """Insert a game record once. Returns True only for the first insert."""
    result = await get_db().games.update_one(
        {"_id": game_doc["_id"]},
        {"$setOnInsert": game_doc},
        upsert=True,
    )
    return result.upserted_id is not None


async def save_game(game_doc: GameRecord) -> None:
    """Compatibility wrapper for callers that do not need idempotency info."""
    await save_game_if_absent(game_doc)


async def get_player_stats(player_id: str, *, game_limit: int | None = None) -> dict:
    """Returns player doc + recent games for profile page."""
    settings = get_settings()
    limit = game_limit or settings.profile_game_limit
    db = get_db()
    player = await db.players.find_one({"_id": player_id})
    if not player:
        return {}
    games = await db.games.find(
        {"players.player_id": player_id},
        sort=[("played_at", -1)],
        limit=limit,
    ).to_list(length=limit)
    return {"player": player, "games": games}


async def get_leaderboard() -> list[dict]:
    """Top 50 real (non-test) human players by Elo."""
    cursor = get_db().players.find({"is_test": {"$ne": True}}, sort=[("elo", -1)], limit=50)
    return await cursor.to_list(length=50)
