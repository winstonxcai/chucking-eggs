"""Agent wrapper for the released DART checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

from ..combos import Combo
from ..game import GuanDanEnv
from .base import Agent

DEFAULT_DART_CHECKPOINT = Path(
    "ml/runs/dart_l4/release_1_25m/update_01250000.pt"
)
DART_CHECKPOINT_ENV = "DART_CHECKPOINT"
DART_DEVICE_ENV = "DART_DEVICE"
DART_WEB_ENABLED_ENV = "DART_WEB_ENABLED"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_dart_checkpoint(checkpoint: str | Path | None = None) -> Path:
    raw = Path(checkpoint or os.getenv(DART_CHECKPOINT_ENV) or DEFAULT_DART_CHECKPOINT)
    raw = raw.expanduser()
    if raw.is_absolute():
        return raw

    cwd_path = Path.cwd() / raw
    if cwd_path.exists():
        return cwd_path
    return _repo_root() / raw


def is_dart_available(checkpoint: str | Path | None = None) -> bool:
    return resolve_dart_checkpoint(checkpoint).exists()


def _is_production_runtime() -> bool:
    if os.getenv("FLY_APP_NAME") or os.getenv("FLY_MACHINE_ID"):
        return True

    for env_name in ("APP_ENV", "ENV", "NODE_ENV"):
        if os.getenv(env_name, "").strip().lower() == "production":
            return True

    return False


def is_local_dart_enabled() -> bool:
    if _is_production_runtime():
        return False

    explicit = os.getenv(DART_WEB_ENABLED_ENV)
    if explicit is not None:
        return explicit.strip().lower() in _TRUE_VALUES

    return True


class DartBot(Agent):
    label = "DART"
    description = "1.25M-update partner-visible DART checkpoint."
    source = "In-house"
    color = "#111111"

    def __init__(
        self,
        level_rank: int | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        del level_rank
        resolved = resolve_dart_checkpoint(checkpoint)
        if not resolved.exists():
            raise FileNotFoundError(
                f"DART checkpoint not found at {resolved}. "
                f"Set {DART_CHECKPOINT_ENV} to a valid checkpoint path."
            )

        from guandan.dart.agent import DartBot as _CheckpointDartBot

        self.checkpoint = resolved
        self.device = device or os.getenv(DART_DEVICE_ENV, "cpu")
        self._bot = _CheckpointDartBot.load(resolved, device=self.device)

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        return self._bot.act(env, player)


__all__ = [
    "DART_CHECKPOINT_ENV",
    "DART_DEVICE_ENV",
    "DART_WEB_ENABLED_ENV",
    "DEFAULT_DART_CHECKPOINT",
    "DartBot",
    "is_dart_available",
    "is_local_dart_enabled",
    "resolve_dart_checkpoint",
]
