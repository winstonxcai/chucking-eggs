"""Training configuration for GuanZero.

``TrainConfig`` is the single configuration object threaded through the
training entry point, learner process, actor processes, and inference server.
It is serialised to ``config.json`` in each run directory and embedded in
every checkpoint file.

Three sub-configs group related knobs:
- ``QNetConfig``       — architecture (5 fields, must match across all processes)
- ``EpsilonConfig``    — exploration schedule (3 fields)
- ``InferenceConfig``  — optional shared-GPU inference server (9 fields)

``from_flat_dict`` provides YAML and checkpoint compatibility: it accepts both
the current flat YAML format and the nested format produced by
``dataclasses.asdict``, and silently ignores fields removed in past refactors.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Sub-configs ─────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class QNetConfig:
    """Q-network architecture. All fields must match across every process
    that constructs or loads the networks (learner, actors, inference server).
    """
    hidden_lstm: int = 256
    hidden_mlp: int = 1024
    n_mlp_layers: int = 6
    dropout: float = 0.0
    use_oracle_others_hand: bool = True
    history_encoder: str = "lstm"       # "lstm" | "transformer"
    transformer_nhead: int = 4          # heads; hidden_lstm / nhead = dim/head
    transformer_layers: int = 2
    transformer_ff_dim: int = 1024      # feedforward dim inside each transformer layer


@dataclasses.dataclass(frozen=True)
class EpsilonConfig:
    """Parameters for the linear epsilon-greedy decay schedule."""
    start: float = 0.1
    final: float = 0.01
    decay_updates: int = 5_000


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    """Shared-GPU inference server. Set ``enabled=True`` to route actor
    action-selection through a dedicated GPU process instead of per-actor
    CPU forward passes.
    """
    enabled: bool = False
    device: str = "cuda"
    batch_max_requests: int = 32
    batch_max_action_rows: int = 4096
    batch_timeout_ms: float = 5.0
    n_slots: int = 512
    max_actions: int = 512   # per-slot buffer capacity; ≥ max K post-dedup (measured: 379)
    timeout_s: float = 60.0
    weight_refresh_s: float = 5.0


# ─── Removed flat field names ─────────────────────────────────
# from_flat_dict drops these silently so existing YAML configs and old
# checkpoints still load without KeyErrors.
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

# Flat YAML key → EpsilonConfig field name
_EPSILON_FLAT_MAP: dict[str, str] = {
    "epsilon_start":          "start",
    "epsilon_final":          "final",
    "epsilon_decay_updates":  "decay_updates",
    "epsilon_decay_episodes": "decay_updates",  # backward compat YAML alias
    "decay_episodes":         "decay_updates",  # backward compat nested-dict alias (old checkpoints)
}

# Flat YAML key → InferenceConfig field name
_INFERENCE_FLAT_MAP: dict[str, str] = {
    "use_inference_server":            "enabled",
    "inference_device":                "device",
    "inference_batch_max_requests":    "batch_max_requests",
    "inference_batch_max_action_rows": "batch_max_action_rows",
    "inference_batch_timeout_ms":      "batch_timeout_ms",
    "inference_n_slots":               "n_slots",
    "inference_max_actions":           "max_actions",
    "inference_timeout_s":             "timeout_s",
    "inference_weight_refresh_s":      "weight_refresh_s",
}

_QNET_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(QNetConfig)
)


# ─── TrainConfig ─────────────────────────────────────────────


@dataclasses.dataclass
class TrainConfig:
    """Unified configuration for GuanZero training.

    Network architecture, exploration schedule, and inference-server settings
    live in the ``qnet``, ``epsilon``, and ``inference`` sub-configs respectively.
    All other knobs are flat fields.

    Use ``from_flat_dict`` to load from YAML files or old checkpoint dicts;
    it handles both the flat and nested serialisation formats transparently.
    """

    # ── Sub-configs ─────────────────────────────────────────
    qnet:      QNetConfig      = dataclasses.field(default_factory=QNetConfig)
    epsilon:   EpsilonConfig   = dataclasses.field(default_factory=EpsilonConfig)
    inference: InferenceConfig = dataclasses.field(default_factory=InferenceConfig)

    # ── Core hypers ─────────────────────────────────────────
    model_type: str = "seat_nets"
    seed: int = 0
    gamma: float = 1.0
    batch_size: int = 512
    lr: float = 1e-4
    max_grad_norm: float = 10.0

    # ── Buffer ──────────────────────────────────────────────
    buffer_capacity_per_player: int = 50_000
    buffer_capacity: int = 0
    buffer_min_size: int = 1_000

    # ── Shared-head architecture ────────────────────────────
    shared_head_role_d_model: int = 128
    shared_head_history_hidden: int = 256
    shared_head_global_hidden: int = 128
    shared_head_action_hidden: int = 128
    shared_head_trunk_hidden: int = 1024
    shared_head_trunk_layers: int = 4

    # ── Runtime ─────────────────────────────────────────────
    device: str = "cpu"
    run_dir: str = ""   # empty → auto-generate timestamped name via resolved_run_dir

    # ── Distributed actor-learner ────────────────────────────
    n_actors: int = 1
    sync_interval_episodes: int = 20      # how often actors check `latest.txt` for new weights
    max_version_lag_updates: int = 0      # only reload weights when ≥N updates behind (0 = always)
    actor_push_batch_size: int = 512
    sample_queue_maxsize: int = 64
    max_drain_batches_per_loop: int = 32
    publish_interval_updates: int = 100
    checkpoint_every_updates: int = 5_000
    total_updates_target: int = 0   # 0 = run until stopped; >0 = stop after this many
    log_every_updates: int = 200
    updates_per_learner_step: int = 1

    # ── CUDA throughput knobs ────────────────────────────────
    use_bf16_learner: bool = False   # BF16 autocast in Learner.update (cuda only)
    compile_mode: str = "default"    # passed to torch.compile(mode=...)

    # ── Replay-ratio controller ──────────────────────────────
    target_replay_ratio: float = 0.0   # 0 = disabled
    max_replay_ratio: float = 4.0
    max_throttle_sleep_s: float = 0.05

    def __post_init__(self) -> None:
        if self.model_type not in ("seat_nets", "shared_heads"):
            raise ValueError(
                f"Unknown model_type {self.model_type!r}; expected 'seat_nets' or 'shared_heads'"
            )

    @property
    def resolved_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/guanzero_m0_{ts}"

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Construct a ``TrainConfig`` from a flat or nested dict.

        Accepts:
        * Flat YAML form:   ``{"hidden_lstm": 256, "epsilon_start": 0.1, ...}``
        * Nested form:      ``{"qnet": {...}, "epsilon": {...}, ...}``
          (produced by ``dataclasses.asdict`` on a nested ``TrainConfig``)
        * Old checkpoint dicts: removed fields are silently dropped.

        This is the preferred loader in ``GuanZeroBot.load()`` and anywhere a
        checkpoint or YAML ``config`` dict is deserialised.
        """
        if dataclasses.is_dataclass(d.get("qnet")):
            d = {
                **d,
                "qnet": dataclasses.asdict(d["qnet"]),
                "epsilon": (
                    dataclasses.asdict(d["epsilon"])
                    if dataclasses.is_dataclass(d.get("epsilon"))
                    else d.get("epsilon")
                ),
                "inference": (
                    dataclasses.asdict(d["inference"])
                    if dataclasses.is_dataclass(d.get("inference"))
                    else d.get("inference")
                ),
            }

        if isinstance(d.get("qnet"), dict):
            # ── Nested form (new checkpoints / round-tripped asdict) ──
            qnet      = QNetConfig(**d["qnet"])
            epsilon_kw: dict[str, Any] = {}
            inference_kw: dict[str, Any] = {}
            for k, v in d.items():
                if k in _EPSILON_FLAT_MAP:
                    epsilon_kw[_EPSILON_FLAT_MAP[k]] = v
                elif k in _INFERENCE_FLAT_MAP:
                    inference_kw[_INFERENCE_FLAT_MAP[k]] = v
            if isinstance(d.get("epsilon"), dict):
                for k, v in d["epsilon"].items():
                    epsilon_kw[_EPSILON_FLAT_MAP.get(k, k)] = v
            if isinstance(d.get("inference"), dict):
                inference_kw.update(d["inference"])
            epsilon   = EpsilonConfig(**epsilon_kw)
            inference = InferenceConfig(**inference_kw)
            skip = {"qnet", "epsilon", "inference"} | _REMOVED_FIELDS
            valid_top = {f.name for f in dataclasses.fields(cls)} - {"qnet", "epsilon", "inference"}
            top = {k: v for k, v in d.items() if k in valid_top and k not in skip}
            return cls(qnet=qnet, epsilon=epsilon, inference=inference, **top)

        # ── Flat form (YAML files, old checkpoints) ──
        qnet_kw:      dict[str, Any] = {}
        epsilon_kw:   dict[str, Any] = {}
        inference_kw: dict[str, Any] = {}
        valid_top = {f.name for f in dataclasses.fields(cls)} - {"qnet", "epsilon", "inference"}
        top_kw: dict[str, Any] = {}

        for k, v in d.items():
            if k in _REMOVED_FIELDS:
                continue
            if k in _QNET_FIELDS:
                qnet_kw[k] = v
            elif k in _EPSILON_FLAT_MAP:
                epsilon_kw[_EPSILON_FLAT_MAP[k]] = v
            elif k in _INFERENCE_FLAT_MAP:
                inference_kw[_INFERENCE_FLAT_MAP[k]] = v
            elif k in valid_top:
                top_kw[k] = v
            # else: silently ignore unknown keys

        return cls(
            qnet      = QNetConfig(**qnet_kw),
            epsilon   = EpsilonConfig(**epsilon_kw),
            inference = InferenceConfig(**inference_kw),
            **top_kw,
        )


def load_config_from_yaml(path: str | Path) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file.

    CLI flags should be applied on top of this via ``dataclasses.replace``.
    """
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return TrainConfig.from_flat_dict(raw)


def shared_head_qnet_config(cfg: TrainConfig):
    """Assemble SharedHeadQNetConfig from TrainConfig flat fields."""
    from .q_network import SharedHeadQNetConfig

    return SharedHeadQNetConfig(
        role_d_model=cfg.shared_head_role_d_model,
        history_hidden=cfg.shared_head_history_hidden,
        global_hidden=cfg.shared_head_global_hidden,
        action_hidden=cfg.shared_head_action_hidden,
        trunk_hidden=cfg.shared_head_trunk_hidden,
        trunk_layers=cfg.shared_head_trunk_layers,
        dropout=cfg.qnet.dropout,
    )


def load_config_from_cli(
    yaml_path: str | Path | None,
    quick: bool,
    quick_overrides: dict[str, Any],
    **cli_overrides: Any,
) -> TrainConfig:
    """Load config from YAML plus quick-mode and explicit CLI overrides.

    ``quick_overrides`` is applied after YAML, and explicit CLI values are
    applied last. ``None`` CLI values are ignored so optional argparse flags do
    not erase values loaded from the config file.
    """
    raw: dict[str, Any] = {}
    if yaml_path:
        import yaml
        raw.update(yaml.safe_load(Path(yaml_path).read_text()) or {})
    if quick:
        raw.update(quick_overrides)
    raw.update({k: v for k, v in cli_overrides.items() if v is not None})
    return TrainConfig.from_flat_dict(raw)


__all__ = [
    "TrainConfig",
    "QNetConfig",
    "EpsilonConfig",
    "InferenceConfig",
    "shared_head_qnet_config",
    "load_config_from_yaml",
    "load_config_from_cli",
]
