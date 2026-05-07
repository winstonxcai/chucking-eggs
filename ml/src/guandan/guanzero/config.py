"""Training configuration for GuanZero.

``TrainConfig`` is the single configuration object threaded through the
training entry point, learner process, actor processes, and inference server.
It is serialised to ``config.json`` in each run directory and embedded in
every checkpoint file.

``from_flat_dict`` provides YAML and checkpoint compatibility: it accepts both
the current flat format and any future nested sub-config format, and silently
ignores fields that were removed in past refactors.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any


# Fields removed in past refactors. ``from_flat_dict`` drops these silently so
# existing YAML configs and old checkpoints still load without KeyErrors.
_REMOVED_FIELDS: frozenset[str] = frozenset({
    "episodes",
    "learn_every_episodes",
    "log_every_episodes",
    "checkpoint_every_episodes",
    "history_window",
    "max_legal_actions",
    "env_lanes_per_actor",
    "compile_actor",
})


@dataclasses.dataclass
class TrainConfig:
    """Unified configuration for GuanZero training.

    Flat for now; fields will be grouped into ``QNetConfig``, ``EpsilonConfig``,
    and ``InferenceConfig`` sub-configs in the Phase I refactor. The
    ``from_flat_dict`` classmethod handles both the current flat form and the
    future nested form so checkpoints and YAMLs remain compatible across the
    migration.
    """

    seed: int = 0

    gamma: float = 1.0
    batch_size: int = 512
    lr: float = 1e-4

    epsilon_start: float = 0.1
    epsilon_final: float = 0.01
    epsilon_decay_episodes: int = 15_000

    buffer_capacity_per_player: int = 50_000
    buffer_min_size: int = 1_000

    hidden_lstm: int = 256
    hidden_mlp: int = 1024
    n_mlp_layers: int = 6
    dropout: float = 0.0

    use_oracle_others_hand: bool = True

    device: str = "cpu"
    run_dir: str = ""   # empty → auto-generate timestamped name via resolved_run_dir

    # ── Distributed actor-learner ──────────────────────────
    n_actors: int = 1
    sync_interval_episodes: int = 20
    actor_push_batch_size: int = 512
    sample_queue_maxsize: int = 64
    max_drain_batches_per_loop: int = 32
    publish_interval_updates: int = 100
    checkpoint_every_updates: int = 5_000
    total_updates_target: int = 0    # 0 = run until stopped; >0 = stop after this many updates
    log_every_updates: int = 200
    updates_per_learner_step: int = 1

    # ── CUDA throughput knobs ──────────────────────────────
    use_bf16_learner: bool = False   # BF16 autocast in Learner.update (cuda only)
    compile_mode: str = "default"    # passed to torch.compile(mode=...)

    # ── Shared-GPU inference server ────────────────────────
    use_inference_server:            bool  = False
    inference_device:                str   = "cuda"
    inference_batch_max_requests:    int   = 32
    inference_batch_max_action_rows: int   = 4096
    inference_batch_timeout_ms:      float = 5.0
    inference_n_slots:               int   = 512
    inference_max_actions:           int   = 512    # per-slot buffer capacity; ≥ max K post-dedup
    inference_timeout_s:             float = 60.0
    inference_weight_refresh_s:      float = 5.0

    # ── Replay-ratio controller ────────────────────────────
    target_replay_ratio:             float = 0.0    # 0 = disabled
    max_replay_ratio:                float = 4.0
    max_throttle_sleep_s:            float = 0.05

    @property
    def resolved_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/guanzero_m0_{ts}"

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Construct a ``TrainConfig`` from a flat or partially-nested dict.

        Accepts:
        * Current flat form: ``{"hidden_lstm": 256, ...}``
        * Future nested form (Phase I): ``{"qnet": {...}, "epsilon": {...}, ...}``
          — nested keys are flattened before construction.
        * Old checkpoint dicts: removed fields are silently dropped.

        This is the preferred loader in ``GuanZeroBot.load()`` and anywhere
        a checkpoint ``config`` dict is deserialised.
        """
        flat: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                # Future nested sub-configs: flatten into top-level namespace.
                for sub_k, sub_v in v.items():
                    flat[sub_k] = sub_v
            else:
                flat[k] = v

        valid = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in flat.items()
                    if k in valid and k not in _REMOVED_FIELDS}
        return cls(**filtered)


def load_config_from_yaml(path: str | Path) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file.

    CLI flags should be applied on top of this via ``dataclasses.replace``.
    """
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return TrainConfig.from_flat_dict(raw)


__all__ = ["TrainConfig", "load_config_from_yaml"]
