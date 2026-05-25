"""Shared fixtures for web tests."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pytest"
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("USE_MOCK_DB", "true")
os.environ.setdefault("DATA_DIR", str(TEST_DATA_DIR))


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client against the FastAPI app."""
    from app.main import app, lifespan

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def create_game(client: AsyncClient, difficulty: str = "greedy") -> dict:
    """Helper: POST /api/game/create and return response JSON."""
    res = await client.post("/api/game/create", json={"difficulty": difficulty})
    assert res.status_code == 200
    return res.json()
