"""Lightweight signed player identity for the web app."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import uuid

from fastapi import Header, HTTPException

from . import db as _db
from .settings import Settings, get_settings

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{2,20}$")
_TOKEN_VERSION = "v1"


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _sign_payload(payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(digest)


def issue_player_token(player_id: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    payload = f"{_TOKEN_VERSION}.{player_id}"
    signature = _sign_payload(payload, settings.auth_secret)
    return f"{payload}.{signature}"


def verify_player_token(token: str, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    try:
        version, player_id, signature = token.split(".", 2)
    except ValueError:
        return None
    if version != _TOKEN_VERSION or not player_id:
        return None
    payload = f"{version}.{player_id}"
    expected = _sign_payload(payload, settings.auth_secret)
    if not hmac.compare_digest(signature, expected):
        return None
    return player_id


def resolve_player_id(
    player_token: str | None,
    legacy_player_id: str | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Resolve signed identity.

    Unsigned legacy IDs are accepted only outside production so existing local
    smoke/E2E flows can keep running without a full account bootstrap.
    """
    settings = settings or get_settings()
    if player_token:
        player_id = verify_player_token(player_token, settings)
        if player_id is None:
            raise HTTPException(status_code=401, detail="Invalid player token")
        if legacy_player_id and legacy_player_id != player_id:
            raise HTTPException(status_code=401, detail="Player token does not match player id")
        return player_id
    if legacy_player_id and not settings.is_production:
        return legacy_player_id
    if legacy_player_id:
        raise HTTPException(status_code=401, detail="Player token required")
    return None


def get_player_id(
    x_player_token: str | None = Header(default=None),
    x_player_id: str | None = Header(default=None),
) -> str | None:
    """FastAPI dependency: resolves signed identity from headers."""
    return resolve_player_id(x_player_token, legacy_player_id=x_player_id)


async def claim_username(username: str, email: str | None = None, is_test: bool = False) -> dict:
    """Claim a username and create an account. Raises 409 if taken."""
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 2-20 characters (letters, numbers, _ or -).",
        )
    if await _db.is_username_taken(username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    player_id = str(uuid.uuid4())
    player = await _db.get_or_create_player(player_id, username, email, is_test=is_test)
    return {
        "player_id": player_id,
        "player_token": issue_player_token(player_id),
        "username": username,
        "elo": player["elo"],
    }
