"""FastAPI app for Guan Dan web game."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateGameRequest(BaseModel):
    difficulty: str = "medium"


class CreateGameResponse(BaseModel):
    game_id: str
    reconnect_token: str


class CreateRoomRequest(BaseModel):
    mode: str = "solo"       # "solo" | "duo" | "quad"
    difficulty: str = "medium"


class CreateRoomResponse(BaseModel):
    game_id: str
    room_code: str | None
    seat: int
    reconnect_token: str


class JoinRoomResponse(BaseModel):
    game_id: str
    room_code: str
    seat: int
    reconnect_token: str


class RoomSeatInfo(BaseModel):
    seat: int
    is_human: bool
    connected: bool
    name: str


class RoomStatusResponse(BaseModel):
    game_id: str
    mode: str
    room_code: str | None
    started: bool
    seats: list[RoomSeatInfo]


# ---------------------------------------------------------------------------
# Solo / backwards-compat endpoint
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Phase 2: room create / join / status
# ---------------------------------------------------------------------------

@app.post("/api/room/create", response_model=CreateRoomResponse)
async def create_room(req: CreateRoomRequest):
    assert game_manager is not None
    if req.mode not in ("solo", "duo", "quad"):
        req.mode = "solo"
    if req.difficulty not in ("easy", "medium", "hard", "expert"):
        req.difficulty = "medium"
    room = game_manager.create_room(req.mode, req.difficulty)
    seat = 0  # creator always gets seat 0
    return CreateRoomResponse(
        game_id=room.game_id,
        room_code=room.room_code,
        seat=seat,
        reconnect_token=room.reconnect_tokens[seat],
    )


@app.post("/api/room/join/{room_code}", response_model=JoinRoomResponse)
async def join_room(room_code: str):
    assert game_manager is not None
    result = game_manager.join_room(room_code)
    if result is None:
        raise HTTPException(status_code=404, detail="Room not found or already full")
    room, seat = result
    return JoinRoomResponse(
        game_id=room.game_id,
        room_code=room.room_code or room_code.upper(),
        seat=seat,
        reconnect_token=room.reconnect_tokens[seat],
    )


@app.get("/api/room/{game_id}/status", response_model=RoomStatusResponse)
async def room_status(game_id: str):
    assert game_manager is not None
    room = game_manager.get_room(game_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    seats = []
    for i, info in enumerate(room.player_infos):
        is_human = i in room.human_seats
        seats.append(RoomSeatInfo(
            seat=i,
            is_human=is_human,
            connected=i in room.connections,
            name=info["name"],
        ))
    return RoomStatusResponse(
        game_id=room.game_id,
        mode=room.mode,
        room_code=room.room_code,
        started=room.started,
        seats=seats,
    )


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/game/{game_id}")
async def game_websocket(
    ws: WebSocket,
    game_id: str,
    token: str = Query(default=""),
    seat: int = Query(default=0),
):
    assert game_manager is not None

    # Try reconnection first (seat-specific, then seat-0 compat)
    room = None
    if token:
        room = game_manager.reconnect_seat(game_id, seat, token)

    # Fall back to regular room lookup
    if room is None:
        room = game_manager.get_room(game_id)

    if room is None:
        await ws.accept()
        await ws.close(code=4004, reason="Game not found")
        return

    if seat not in room.human_seats:
        await ws.accept()
        await ws.close(code=4003, reason="Seat not valid")
        return

    await room.connect(ws, seat)

    # Determine whether this connection causes the game to start
    game_just_started = False
    if not room.started and all(s in room.connections for s in room.human_seats):
        room.started = True
        game_just_started = True

    try:
        if game_just_started:
            # Broadcast full state to all connected humans
            await room.broadcast_game_state()
            # Run AI if the first player is an AI
            if room.env.current_player not in room.human_seats and not room.env.done:
                await room.run_ai_turns()
        else:
            # Send state only to this reconnecting/joining seat
            await room.send_game_state_to(seat)
            # Solo mode: run AI if it's not the human's turn
            if (
                room.started
                and room.env.current_player not in room.human_seats
                and not room.env.done
                # Only the lowest-numbered connected seat triggers AI to avoid dups
                and seat == min(room.connections.keys())
            ):
                await room.run_ai_turns()

        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            await room.handle_message(data, seat)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error in game {game_id} seat {seat}: {e}")
    finally:
        game_manager.disconnect_seat(game_id, seat)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}
