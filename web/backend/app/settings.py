"""Application settings parsed from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_data_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent / "data" / "web_backend"
    return Path.cwd() / "data" / "web_backend"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false', got {value!r}")
    return normalized == "true"


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}, got {parsed:g}")
    return parsed


def _env_csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    mongodb_url: str
    use_mock_db: bool
    app_env: str
    auth_secret: str
    allowed_origins: list[str]
    data_dir: Path
    redis_url: str
    cleanup_interval_s: float
    lobby_timeout_s: float
    disconnect_takeover_s: int
    human_turn_timeout_s: int
    action_pause_s: float
    ai_think_pause_s: float
    profile_game_limit: int

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"


def get_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development")
    auth_secret = os.getenv("AUTH_SECRET")
    if app_env.strip().lower() == "production" and not auth_secret:
        raise ValueError("AUTH_SECRET must be set when APP_ENV=production")
    return Settings(
        mongodb_url=os.getenv("MONGODB_URL", "mongodb://localhost:27017"),
        use_mock_db=_env_bool("USE_MOCK_DB"),
        app_env=app_env,
        auth_secret=auth_secret or "dev-only-auth-secret",
        allowed_origins=_env_csv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ),
        data_dir=Path(os.getenv("DATA_DIR", str(_default_data_dir()))),
        redis_url=os.getenv("REDIS_URL", ""),
        cleanup_interval_s=_env_float("CLEANUP_INTERVAL_S", 30, minimum=0.1),
        lobby_timeout_s=_env_float("LOBBY_TIMEOUT_S", 300, minimum=1),
        disconnect_takeover_s=_env_int("DISCONNECT_TAKEOVER_S", 60, minimum=1),
        human_turn_timeout_s=_env_int("HUMAN_TURN_TIMEOUT_S", 30, minimum=0),
        action_pause_s=_env_float("ACTION_PAUSE_S", 0.3, minimum=0),
        ai_think_pause_s=_env_float("AI_THINK_PAUSE_S", 0.2, minimum=0),
        profile_game_limit=_env_int("PROFILE_GAME_LIMIT", 100, minimum=1),
    )
