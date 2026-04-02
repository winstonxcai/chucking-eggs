"""FastAPI app for Guan Dan web game."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .ai_service import AIService
from .game_manager import GameManager
from .redis_client import close_redis
from . import db
from .elo import BOT_LEADERBOARD_ENTRIES

ai_service: AIService | None = None
game_manager: GameManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_service, game_manager
    ai_service = AIService()
    game_manager = GameManager(ai_service)
    await game_manager.start_cleanup_loop()
    await db.init_db()
    print("AI agents loaded, DB connected, server ready")
    yield
    await game_manager.stop_cleanup_loop()
    await close_redis()
    await db.close_db()
    print("Shutting down")


app = FastAPI(title="Guan Dan", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


class ClaimUsernameRequest(BaseModel):
    username: str
    email: Optional[str] = None
    is_test: bool = False


class SetDifficultyRequest(BaseModel):
    difficulty: str


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/claim")
@limiter.limit("5/minute")
async def claim_username(request: Request, req: ClaimUsernameRequest):
    from .auth import claim_username as _claim
    return await _claim(req.username, req.email, req.is_test)


# ---------------------------------------------------------------------------
# Profile & Leaderboard
# ---------------------------------------------------------------------------

@app.get("/api/profile/{username}")
async def get_profile(username: str):
    player = await db.get_player_by_username(username)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    stats = await db.get_player_stats(player["_id"])
    # Serialize MongoDB ObjectId/_id fields
    player_doc = {k: str(v) if k == "_id" else v for k, v in player.items()}
    games = []
    for g in stats.get("games", []):
        game_doc = {k: str(v) if k == "_id" else v for k, v in g.items()}
        # Convert datetime to ISO string
        if "played_at" in game_doc and hasattr(game_doc["played_at"], "isoformat"):
            game_doc["played_at"] = game_doc["played_at"].isoformat()
        games.append(game_doc)
    return {"player": player_doc, "games": games}


@app.get("/api/leaderboard")
async def get_leaderboard():
    humans = await db.get_leaderboard()
    human_entries = [
        {
            "username": p["username"],
            "elo": p["elo"],
            "games_played": p.get("games_played", 0),
            "is_bot": False,
        }
        for p in humans
    ]
    return {"humans": human_entries, "bots": BOT_LEADERBOARD_ENTRIES}


# ---------------------------------------------------------------------------
# Solo / backwards-compat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/game/create", response_model=CreateGameResponse)
async def create_game(req: CreateGameRequest):
    assert game_manager is not None
    if req.difficulty not in ("easy", "wjsd", "casual", "medium", "competition", "hard",
         "yaoji", "jidan", "expert", "hulalala", "liuzha", "master"):
        req.difficulty = "medium"
    room = await game_manager.create_game(req.difficulty)
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
    if req.difficulty not in ("easy", "wjsd", "casual", "medium", "competition", "hard",
         "yaoji", "jidan", "expert", "hulalala", "liuzha", "master"):
        req.difficulty = "medium"
    room = await game_manager.create_room(req.mode, req.difficulty)
    seat = 0  # creator always gets seat 0
    return CreateRoomResponse(
        game_id=room.game_id,
        room_code=room.room_code,
        seat=seat,
        reconnect_token=room.reconnect_tokens[seat],
    )


@app.post("/api/room/{game_id}/set_difficulty")
async def set_room_difficulty(game_id: str, req: SetDifficultyRequest):
    assert game_manager is not None
    room = game_manager.get_room(game_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.started:
        raise HTTPException(status_code=400, detail="Game already started")
    valid = ("easy", "wjsd", "casual", "medium", "competition", "hard",
             "yaoji", "jidan", "expert", "hulalala", "liuzha", "master")
    if req.difficulty not in valid:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    room.set_difficulty(req.difficulty)
    return {"ok": True}


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
        started=room.started or (room.assigned_seats >= room.human_seats),
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
    player_id: str = Query(default=""),
):
    assert game_manager is not None

    # Try reconnection first (seat-specific, then seat-0 compat)
    room = None
    if token:
        room = game_manager.reconnect_seat(game_id, seat, token)
        if room:
            room.cancel_disconnect_takeover(seat)

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

    # Register player_id and update display name from DB
    if player_id:
        room.player_ids[seat] = player_id
        try:
            player_doc = await db.get_player_by_id(player_id)
            if player_doc:
                room.player_infos[seat]["name"] = player_doc["username"]
                room.player_infos[seat]["elo"] = player_doc.get("elo", 1200)
        except Exception:
            pass  # non-critical; display name stays as default

    # Determine whether this connection causes the game to start
    game_just_started = False
    if not room.started and all(s in room.connections for s in room.human_seats):
        room.started = True
        game_just_started = True

    try:
        if game_just_started:
            await room.broadcast_game_state()
            if room.env.current_player not in room.human_seats and not room.env.done:
                await room.run_ai_turns()
        else:
            await room.send_game_state_to(seat)
            if (
                room.started
                and room.env.current_player not in room.human_seats
                and not room.env.done
                and seat == min(room.connections.keys())
            ):
                await room.run_ai_turns()

        while True:
            # AFK rope: use wait_for when it's this seat's turn and a deadline is set
            deadline = room.turn_deadlines.get(seat)
            if deadline is not None and room.env.current_player == seat:
                remaining = max(0.5, deadline - asyncio.get_event_loop().time())
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=remaining)
                except asyncio.TimeoutError:
                    await room.handle_afk_timeout(seat)
                    continue
            else:
                raw = await ws.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await room.send_to(seat, {"type": "error", "message": "Invalid message"})
                continue
            await room.handle_message(data, seat)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error in game {game_id} seat {seat}: {e}")
    finally:
        game_manager.disconnect_seat(game_id, seat)
        room.schedule_disconnect_takeover(seat)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}
