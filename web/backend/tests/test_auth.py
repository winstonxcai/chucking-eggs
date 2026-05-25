"""Auth token tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.auth import issue_player_token, resolve_player_id, verify_player_token
from app.settings import Settings
from fastapi import HTTPException


def _settings(is_production: bool = False) -> Settings:
    return Settings(
        mongodb_url="mongodb://test",
        use_mock_db=True,
        app_env="production" if is_production else "development",
        auth_secret="test-secret",
        allowed_origins=["http://test"],
        data_dir=Path("/tmp/chucking-eggs-test"),
        redis_url="",
        cleanup_interval_s=30,
        lobby_timeout_s=300,
        disconnect_takeover_s=60,
        human_turn_timeout_s=30,
        action_pause_s=0.3,
        ai_think_pause_s=0.2,
        profile_game_limit=100,
    )


def test_player_token_round_trips() -> None:
    token = issue_player_token("player-1", _settings())
    assert verify_player_token(token, _settings()) == "player-1"


def test_invalid_player_token_is_rejected() -> None:
    token = issue_player_token("player-1", _settings())
    assert verify_player_token(token + "x", _settings()) is None


def test_resolve_player_id_rejects_mismatched_legacy_id() -> None:
    token = issue_player_token("player-1", _settings())
    with pytest.raises(HTTPException):
        resolve_player_id(token, legacy_player_id="player-2", settings=_settings())


def test_unsigned_legacy_id_is_rejected_in_production() -> None:
    with pytest.raises(HTTPException):
        resolve_player_id(None, legacy_player_id="player-1", settings=_settings(is_production=True))


def test_unsigned_legacy_id_is_allowed_outside_production() -> None:
    assert resolve_player_id(None, legacy_player_id="player-1", settings=_settings()) == "player-1"
