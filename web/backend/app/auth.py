"""Lightweight auth: UUID-as-token, first-claim-wins username."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from fastapi import Header, HTTPException

from . import db as _db

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{2,20}$")


def get_player_id(x_player_id: Optional[str] = Header(default=None)) -> Optional[str]:
    """FastAPI dependency: reads X-Player-ID header. Returns None if absent."""
    return x_player_id


async def claim_username(username: str, email: Optional[str] = None) -> dict:
    """Claim a username and create an account. Raises 409 if taken.

    Returns: {"player_id": str, "username": str, "elo": int}
    """
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 2–20 characters (letters, numbers, _ or -).",
        )
    if await _db.is_username_taken(username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    player_id = str(uuid.uuid4())
    player = await _db.get_or_create_player(player_id, username, email)
    return {"player_id": player_id, "username": username, "elo": player["elo"]}
