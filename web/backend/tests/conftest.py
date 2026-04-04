"""Shared fixtures for web tests."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app, lifespan


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client against the FastAPI app."""
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


async def create_game(client: AsyncClient, difficulty: str = "easy") -> dict:
    """Helper: POST /api/game/create and return response JSON."""
    res = await client.post("/api/game/create", json={"difficulty": difficulty})
    assert res.status_code == 200
    return res.json()
