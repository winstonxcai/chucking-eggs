"""FastAPI app for Guan Dan web game."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ai_service import AIService
from .game_manager import GameManager
from .redis_client import close_redis

ai_service: AIService | None = None
game_manager: GameManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service, game_manager
    ai_service = AIService()
    game_manager = GameManager(ai_service)
    await game_manager.start_cleanup_loop()
    print("AI agents loaded, server ready")
    yield
    await game_manager.stop_cleanup_loop()
    await close_redis()
    print("Shutting down")


app = FastAPI(title="Guan Dan", lifespan=lifespan)

_cors_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateGameRequest(BaseModel):
    difficulty: str = "medium"


class CreateGameResponse(BaseModel):
    game_id: str
    reconnect_token: str


@app.post("/api/game/create", response_model=CreateGameResponse)
async def create_game(req: CreateGameRequest):
    assert game_manager is not None
    if req.difficulty not in ("easy", "medium", "hard", "expert"):
        req.difficulty = "medium"
    room = game_manager.create_game(req.difficulty)
    return CreateGameResponse(
        game_id=room.game_id,
        reconnect_token=room.reconnect_token,
    )


@app.websocket("/ws/game/{game_id}")
async def game_websocket(
    ws: WebSocket,
    game_id: str,
    token: str = Query(default=""),
):
    assert game_manager is not None

    # Try reconnection first
    room = None
    if token:
        room = game_manager.reconnect_room(game_id, token)

    # Fall back to normal room lookup
    if room is None:
        room = game_manager.get_room(game_id)

    if room is None:
        await ws.close(code=4004, reason="Game not found")
        return

    await room.connect(ws)

    try:
        # Send initial state
        await room.send_game_state()

        # If it's not the human's turn first, run AI turns
        if room.env.current_player != 0 and not room.env.done:
            await room.run_ai_turns()

        # Main message loop
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            await room.handle_message(data)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error in game {game_id}: {e}")
    finally:
        # Keep room alive for reconnection instead of removing immediately
        game_manager.disconnect_room(game_id)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
