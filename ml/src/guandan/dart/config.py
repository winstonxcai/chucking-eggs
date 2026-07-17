"""Training configuration for Dart.

``TrainConfig`` is the single configuration object threaded through the
training entry point, learner process, and actor processes.
It is serialised to ``config.json`` in each run directory and embedded in
every checkpoint file.

Two sub-configs group related knobs:
- ``QNetConfig``       — architecture (5 fields, must match across all processes)
- ``EpsilonConfig``    — exploration schedule (3 fields)

``from_flat_dict`` accepts both the flat YAML format and the nested format
produced by ``dataclasses.asdict`` (the checkpoint serialisation form).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from .constants import NUM_PLAYERS

ModelType = Literal["GuanZero", "Dart"]
MODEL_TYPE_GUANZERO: ModelType = "GuanZero"
MODEL_TYPE_DART: ModelType = "Dart"
CheckpointSaveType = Literal["weight", "full"]
QueueFullPolicy = Literal["stop", "block"]
BatchSizeSemantics = Literal["total", "per_seat"]
CONFIG_SCHEMA_VERSION = 1


# ─── Sub-configs ─────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class QNetConfig:
    """Q-network architecture. All fields must match across every process
    that constructs or loads the networks (learner and actors).
    """
    hidden_lstm: int = 256
    hidden_mlp: int = 1024
    n_mlp_layers: int = 6
    dropout: float = 0.0
    is_partner_visible: bool = True
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

    def __post_init__(self) -> None:
        for name in ("start", "final"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"epsilon.{name} must be in [0, 1]; got {value}")
        if self.start < self.final:
            raise ValueError(
                "epsilon.start must be >= epsilon.final for linear decay; "
                f"got {self.start} < {self.final}"
            )
        if self.decay_updates < 0:
            raise ValueError(
                f"epsilon.decay_updates must be >= 0; got {self.decay_updates}"
            )


@dataclasses.dataclass
class EpisodeMixConfig:
    self_play: float = 1.0
    frozen_pool: float = 0.0
    hard_bot: float = 0.0

    def __post_init__(self) -> None:
        for name in ("self_play", "frozen_pool", "hard_bot"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"opponents.episode_mix.{name} must be in [0, 1]; got {value}")
        total = self.self_play + self.frozen_pool + self.hard_bot
        if total > 1.0 + 1e-9:
            raise ValueError(
                "opponents.episode_mix values must sum to <= 1; "
                f"got {self.self_play} + {self.frozen_pool} + {self.hard_bot}"
            )


@dataclasses.dataclass
class FrozenPoolConfig:
    checkpoints: tuple[str, ...] = ()
    epsilon: float = 0.0
    sampling: str = "uniform"

    def __post_init__(self) -> None:
        if isinstance(self.checkpoints, list):
            self.checkpoints = tuple(self.checkpoints)
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"opponents.frozen_pool.epsilon must be in [0, 1]; got {self.epsilon}")
        if self.sampling != "uniform":
            raise ValueError("opponents.frozen_pool.sampling currently supports only 'uniform'")


SamplingType = Literal["uniform", "weighted", "pair_weighted"]


@dataclasses.dataclass
class OpponentSamplingConfig:
    type: SamplingType = "uniform"
    weights: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in ("uniform", "weighted", "pair_weighted"):
            raise ValueError(
                "opponents.hard_bot.sampling.type must be one of "
                "'uniform', 'weighted', or 'pair_weighted'"
            )


@dataclasses.dataclass
class HardBotConfig:
    bots: tuple[str, ...] = ()
    sampling: OpponentSamplingConfig = dataclasses.field(default_factory=OpponentSamplingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.bots, list):
            self.bots = tuple(self.bots)
        if isinstance(self.sampling, dict):
            self.sampling = OpponentSamplingConfig(**self.sampling)

        if self.sampling.type == "uniform":
            self.sampling.weights = {}
            return

        if not self.sampling.weights:
            raise ValueError(
                f"opponents.hard_bot.sampling.weights must be non-empty for {self.sampling.type!r}"
            )
        if any(w <= 0 for w in self.sampling.weights.values()):
            raise ValueError(
                f"opponents.hard_bot.sampling.weights must be positive; got {self.sampling.weights}"
            )

        if self.sampling.type == "weighted":
            unknown = set(self.sampling.weights) - set(self.bots)
            if unknown:
                raise ValueError(
                    f"opponents.hard_bot.sampling.weights keys not in bots: {sorted(unknown)}"
                )
        else:
            bot_names = set(self.bots)
            for key in self.sampling.weights:
                parts = key.split("_")
                if len(parts) != 2:
                    raise ValueError(
                        f"opponents.hard_bot.sampling.weights key {key!r} must be '<a>_<b>'"
                    )
                a, b = parts
                if a > b:
                    raise ValueError(
                        f"opponents.hard_bot.sampling.weights key {key!r} must be alphabetized "
                        f"(use {b}_{a} instead)"
                    )
                unknown = {a, b} - bot_names
                if unknown:
                    raise ValueError(
                        f"opponents.hard_bot.sampling.weights key {key!r} references bots "
                        f"not in bots: {sorted(unknown)}"
                    )

        total = sum(self.sampling.weights.values())
        self.sampling.weights = {k: v / total for k, v in self.sampling.weights.items()}


@dataclasses.dataclass
class OpponentConfig:
    latest_learner_team_odd_probability: float = 0.5
    episode_mix: EpisodeMixConfig = dataclasses.field(default_factory=EpisodeMixConfig)
    frozen_pool: FrozenPoolConfig = dataclasses.field(default_factory=FrozenPoolConfig)
    hard_bot: HardBotConfig = dataclasses.field(default_factory=HardBotConfig)

    def __post_init__(self) -> None:
        if isinstance(self.episode_mix, dict):
            self.episode_mix = EpisodeMixConfig(**self.episode_mix)
        if isinstance(self.frozen_pool, dict):
            self.frozen_pool = FrozenPoolConfig(**self.frozen_pool)
        if isinstance(self.hard_bot, dict):
            self.hard_bot = HardBotConfig(**self.hard_bot)
        if not 0.0 <= self.latest_learner_team_odd_probability <= 1.0:
            raise ValueError(
                "opponents.latest_learner_team_odd_probability must be in [0, 1]; "
                f"got {self.latest_learner_team_odd_probability}"
            )


@dataclasses.dataclass
class EvalConfig:
    enabled: bool = False
    opponents: str | tuple[str, ...] = "all"
    n_eval_games_per_opponent: int = 200
    every_updates: int = 0
    workers: int = 0
    lanes: int = 0
    device: str = "cpu"
    max_wait_s: float = 300.0

    def __post_init__(self) -> None:
        if isinstance(self.opponents, list):
            self.opponents = tuple(self.opponents)
        if isinstance(self.opponents, tuple):
            self.opponents = tuple(str(name) for name in self.opponents)
            if self.opponents == ("all",):
                self.opponents = "all"
            elif not self.opponents:
                raise ValueError("eval.opponents must be 'all' or a non-empty list")
        elif self.opponents != "all":
            raise ValueError("eval.opponents must be 'all' or a list of agent names")
        if self.n_eval_games_per_opponent <= 0:
            raise ValueError(
                "eval.n_eval_games_per_opponent must be positive; "
                f"got {self.n_eval_games_per_opponent}"
            )
        if self.n_eval_games_per_opponent % 2 != 0:
            raise ValueError(
                "eval.n_eval_games_per_opponent must be even for paired fixed-deck eval; "
                f"got {self.n_eval_games_per_opponent}"
            )
        for name in ("every_updates", "workers", "lanes"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"eval.{name} must be >= 0; got {value}")
        if self.max_wait_s <= 0:
            raise ValueError(f"eval.max_wait_s must be > 0; got {self.max_wait_s}")


# Flat YAML key → EpsilonConfig field name
_EPSILON_FLAT_MAP: dict[str, str] = {
    "epsilon_start":         "start",
    "epsilon_final":         "final",
    "epsilon_decay_updates": "decay_updates",
}

_QNET_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(QNetConfig)
)

_REMOVED_OPPONENT_KEYS: frozenset[str] = frozenset({
    "population_pool",
    "hard_bot_pool",
    "hard_bot_sampling",
    "hard_bot_pair_sampling",
    "latest_vs_latest_frac",
    "latest_vs_hard_bot_frac",
    "latest_odd_probability",
    "epsilon_frozen",
})

_REMOVED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "updates_per_learner_step",
})

_TOP_LEVEL_ALIASES: dict[str, str] = {
    "max_forced_k1_replay_frac": "max_forced_pass_replay_frac",
}

_OPPONENT_ALIASES: dict[str, str] = {
    "latest_team_odd_probability": "latest_learner_team_odd_probability",
}


def _reject_unknown_keys(section: str, raw: dict[str, Any], valid: set[str]) -> None:
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"{section} has unknown key(s): {sorted(unknown)}")


def _dataclass_field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _opponent_config_from_raw(raw: Any) -> OpponentConfig:
    if raw is None:
        return OpponentConfig()
    if isinstance(raw, OpponentConfig):
        return raw
    if dataclasses.is_dataclass(raw):
        raw = dataclasses.asdict(cast(Any, raw))
    if not isinstance(raw, dict):
        raise TypeError(f"opponents must be a dict or OpponentConfig; got {type(raw).__name__}")
    raw = dict(raw)
    for old_key, new_key in _OPPONENT_ALIASES.items():
        if old_key in raw:
            if new_key in raw and raw[new_key] != raw[old_key]:
                raise ValueError(
                    f"Both opponents.{old_key!r} and opponents.{new_key!r} "
                    "were provided with different values"
                )
            raw[new_key] = raw.pop(old_key)
    _reject_unknown_keys(
        "opponents",
        raw,
        {"latest_learner_team_odd_probability", "episode_mix", "frozen_pool", "hard_bot"},
    )
    return OpponentConfig(
        latest_learner_team_odd_probability=raw.get(
            "latest_learner_team_odd_probability", 0.5
        ),
        episode_mix=raw.get("episode_mix", {}),
        frozen_pool=raw.get("frozen_pool", {}),
        hard_bot=raw.get("hard_bot", {}),
    )


def _eval_config_from_raw(raw: Any) -> EvalConfig:
    if raw is None:
        return EvalConfig()
    if isinstance(raw, EvalConfig):
        return raw
    if dataclasses.is_dataclass(raw):
        raw = dataclasses.asdict(cast(Any, raw))
    if not isinstance(raw, dict):
        raise TypeError(f"eval must be a dict or EvalConfig; got {type(raw).__name__}")
    _reject_unknown_keys(
        "eval",
        raw,
        {
            "enabled",
            "opponents",
            "n_eval_games_per_opponent",
            "every_updates",
            "workers",
            "lanes",
            "device",
            "max_wait_s",
        },
    )
    return EvalConfig(**raw)


# ─── TrainConfig ─────────────────────────────────────────────


@dataclasses.dataclass
class TrainConfig:
    """Unified configuration for Dart training.

    Network architecture and exploration schedule live in the ``qnet`` and
    ``epsilon`` sub-configs respectively.
    All other knobs are flat fields.

    Use ``from_flat_dict`` to load from YAML files or old checkpoint dicts;
    it handles both the flat and nested serialisation formats transparently.
    """

    # ── Sub-configs ─────────────────────────────────────────
    qnet:      QNetConfig      = dataclasses.field(default_factory=QNetConfig)
    epsilon:   EpsilonConfig   = dataclasses.field(default_factory=EpsilonConfig)
    opponents: OpponentConfig = dataclasses.field(default_factory=OpponentConfig)
    eval:      EvalConfig      = dataclasses.field(default_factory=EvalConfig)
    config_schema_version: int = CONFIG_SCHEMA_VERSION

    # ── Core hypers ─────────────────────────────────────────
    model_type: ModelType = MODEL_TYPE_DART
    seed: int = 0
    gamma: float = 1.0
    batch_size: int = 512
    # DART has one shared learner batch. GuanZero historically interpreted
    # this as a per-seat batch; keep that behavior by default for old configs
    # and opt into total-batch semantics explicitly for fair comparisons.
    batch_size_semantics: BatchSizeSemantics = "per_seat"
    lr: float = 1e-4
    max_grad_norm: float = 10.0

    # ── Buffer ──────────────────────────────────────────────
    buffer_capacity_per_player: int = 50_000
    buffer_capacity: int = 0
    buffer_min_size: int = 1_000

    # ── Dart architecture ───────────────────────────────────
    dart_role_d_model: int = 128
    dart_history_hidden: int = 256
    dart_global_hidden: int = 128
    dart_action_hidden: int = 128
    dart_trunk_hidden: int = 1024
    dart_trunk_layers: int = 4

    # ── Runtime ─────────────────────────────────────────────
    device: str = "cpu"
    run_dir: str = ""   # empty → auto-generate timestamped name via resolved_run_dir

    # ── Distributed actor-learner ────────────────────────────
    n_actors: int = 1
    # Self-play envs interleaved per actor for batched local Q-forward.
    actor_batch_lanes: int = 1
    sync_interval_updates: int = 20       # actor reloads weights when ≥N learner updates behind
    sync_jitter_updates: int = 0          # per-actor uniform jitter ±J around the threshold (0 = no jitter)
    actor_push_batch_size: int = 512
    sample_queue_maxsize: int = 64
    max_drain_batches_per_loop: int = 32
    actor_queue_full_policy: QueueFullPolicy = "stop"  # "stop" = fail fast, "block" = backpressure
    actor_queue_put_timeout_s: float = 5.0
    actor_queue_full_log_every: int = 12
    publish_interval_updates: int = 100
    checkpoint_every_updates: int = 5_000
    checkpoint_save_type: CheckpointSaveType = "full"
    total_updates_target: int = 0   # 0 = run until stopped; >0 = stop after this many
    max_train_seconds: int = 0      # 0 = no duration limit; >0 = stop after this many seconds
    log_every_updates: int = 200

    # ── CUDA throughput knobs ────────────────────────────────
    use_bf16_learner: bool = False   # BF16 autocast in Learner.update (cuda only)
    compile_mode: str = "default"    # passed to torch.compile(mode=...)

    # ── CPU actor inference knobs ────────────────────────────
    use_int8_actor: bool = False     # int8 dynamic quantization of actor Q-nets (cpu only)

    # ── Replay-ratio controller ──────────────────────────────
    target_replay_ratio: float = 0.0   # 0 = disabled
    max_replay_ratio: float = 4.0
    max_throttle_sleep_s: float = 0.05

    # Stratified replay: sample proportionally from per-bucket sub-populations.
    # Keys are bucket names; values are weights (normalized at load time).
    # Empty dict → uniform balanced sampling. Non-empty values are consumed by
    # DartLearner via RoleAwareReplayBuffer.sample_batch_stratified().
    replay_mix: dict[str, float] = dataclasses.field(default_factory=dict)
    # Cap forced-pass samples at this fraction of each batch. 1.0 = no
    # cap (sample uniformly across all samples in the buffer). E.g. 0.05 means
    # 5% of every batch is forced pass, 95% is K>1 (real decisions).
    max_forced_pass_replay_frac: float = 1.0
    # Actor-side coordination bucket boundaries for hard-bot episodes. Samples
    # are marked coordination-endgame when either team is near finish or the
    # episode has reached this trailing fraction.
    coordination_bucket_card_threshold: int = 5
    coordination_bucket_final_fraction: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if self.config_schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported config_schema_version "
                f"{self.config_schema_version}; expected {CONFIG_SCHEMA_VERSION}"
            )
        if self.model_type not in (MODEL_TYPE_GUANZERO, MODEL_TYPE_DART):
            raise ValueError(
                f"Unknown model_type {self.model_type!r}; expected "
                f"{MODEL_TYPE_GUANZERO!r} or {MODEL_TYPE_DART!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.batch_size_semantics not in ("total", "per_seat"):
            raise ValueError(
                "batch_size_semantics must be 'total' or 'per_seat'; "
                f"got {self.batch_size_semantics!r}"
            )
        if (
            self.model_type == MODEL_TYPE_GUANZERO
            and self.batch_size_semantics == "total"
            and self.batch_size % NUM_PLAYERS != 0
        ):
            raise ValueError(
                "GuanZero total batch_size must be divisible by the number of seats "
                f"({NUM_PLAYERS}); got {self.batch_size}"
            )
        if self.actor_batch_lanes < 1:
            raise ValueError(
                f"actor_batch_lanes must be >= 1; got {self.actor_batch_lanes}"
            )
        if self.checkpoint_save_type not in ("weight", "full"):
            raise ValueError(
                "checkpoint_save_type must be 'weight' or 'full'; "
                f"got {self.checkpoint_save_type!r}"
            )
        if self.actor_queue_full_policy not in ("stop", "block"):
            raise ValueError(
                "actor_queue_full_policy must be 'stop' or 'block'; "
                f"got {self.actor_queue_full_policy!r}"
            )
        if self.actor_queue_put_timeout_s <= 0:
            raise ValueError(
                "actor_queue_put_timeout_s must be > 0; "
                f"got {self.actor_queue_put_timeout_s}"
            )
        if self.actor_queue_full_log_every < 1:
            raise ValueError(
                "actor_queue_full_log_every must be >= 1; "
                f"got {self.actor_queue_full_log_every}"
            )
        if self.max_train_seconds < 0:
            raise ValueError(
                f"max_train_seconds must be >= 0; got {self.max_train_seconds}"
            )
        self.opponents = _opponent_config_from_raw(self.opponents)
        self.eval = _eval_config_from_raw(self.eval)
        if self.replay_mix:
            valid_keys = {
                "general", "hard_bot_general", "coordination_endgame",
                "pivotal_qgap", "partner_active_coordination", "bomb_decision",
            }
            unknown = set(self.replay_mix) - valid_keys
            if unknown:
                raise ValueError(f"replay_mix has unknown key(s): {sorted(unknown)}. Valid: {sorted(valid_keys)}")
            if any(w <= 0 for w in self.replay_mix.values()):
                raise ValueError(f"replay_mix weights must be positive; got {self.replay_mix}")
            total = sum(self.replay_mix.values())
            self.replay_mix = {k: v / total for k, v in self.replay_mix.items()}
        if not 0.0 <= self.max_forced_pass_replay_frac <= 1.0:
            raise ValueError(
                f"max_forced_pass_replay_frac must be in [0, 1]; "
                f"got {self.max_forced_pass_replay_frac}"
            )
        if self.coordination_bucket_card_threshold < 0:
            raise ValueError(
                "coordination_bucket_card_threshold must be >= 0; "
                f"got {self.coordination_bucket_card_threshold}"
            )
        if not 0.0 <= self.coordination_bucket_final_fraction <= 1.0:
            raise ValueError(
                "coordination_bucket_final_fraction must be in [0, 1]; "
                f"got {self.coordination_bucket_final_fraction}"
            )

    @property
    def batch_size_per_seat(self) -> int:
        """Number of examples sampled for each GuanZero seat update.

        DART uses a single shared batch, so this property returns ``batch_size``
        for DART and is only used by the per-seat GuanZero learner.
        """
        if self.model_type == MODEL_TYPE_GUANZERO and self.batch_size_semantics == "total":
            return self.batch_size // NUM_PLAYERS
        return self.batch_size

    @property
    def learner_samples_per_update(self) -> int:
        """Total learner examples consumed by one optimizer update."""
        if (
            self.model_type == MODEL_TYPE_GUANZERO
            and self.batch_size_semantics == "per_seat"
        ):
            return NUM_PLAYERS * self.batch_size
        return self.batch_size

    @property
    def resolved_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/dart_{ts}"

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Construct a ``TrainConfig`` from a flat or nested dict.

        Accepts:
        * Flat YAML form:   ``{"hidden_lstm": 256, "epsilon_start": 0.1, ...}``
        * Nested form:      ``{"qnet": {...}, "epsilon": {...}, ...}``
          (produced by ``dataclasses.asdict`` on a ``TrainConfig`` — the
          format used by checkpoint serialisation)

        This is the preferred loader in ``DartBot.load()`` and anywhere a
        checkpoint or YAML ``config`` dict is deserialised.
        """
        d = dict(d)
        for old_key, new_key in _TOP_LEVEL_ALIASES.items():
            if old_key in d:
                if new_key in d and d[new_key] != d[old_key]:
                    raise ValueError(
                        f"Both {old_key!r} and {new_key!r} were provided with "
                        "different values"
                    )
                d[new_key] = d.pop(old_key)
        for removed_key in sorted(set(d) & _REMOVED_TOP_LEVEL_KEYS):
            d.pop(removed_key)
        if "inference" in d and isinstance(d.get("qnet"), dict):
            # Older checkpoints carried inference-server settings that are not
            # part of the local batched actor config.
            d.pop("inference")
        removed = sorted(set(d) & _REMOVED_OPPONENT_KEYS)
        if removed:
            raise ValueError(
                f"Removed opponent config key(s): {removed}. Use the nested 'opponents' block."
            )
        if dataclasses.is_dataclass(d.get("qnet")):
            d = {
                **d,
                "qnet": dataclasses.asdict(d["qnet"]),
                "epsilon": (
                    dataclasses.asdict(d["epsilon"])
                    if dataclasses.is_dataclass(d.get("epsilon"))
                    else d.get("epsilon")
                ),
                "opponents": (
                    dataclasses.asdict(d["opponents"])
                    if dataclasses.is_dataclass(d.get("opponents"))
                    else d.get("opponents")
                ),
                "eval": (
                    dataclasses.asdict(d["eval"])
                    if dataclasses.is_dataclass(d.get("eval"))
                    else d.get("eval")
                ),
            }

        if isinstance(d.get("qnet"), dict):
            # ── Nested form (checkpoints / round-tripped asdict) ──
            _reject_unknown_keys("qnet", d["qnet"], set(_QNET_FIELDS))
            qnet_d = dict(d["qnet"])
            qnet      = QNetConfig(**qnet_d)
            nested_epsilon_kw: dict[str, Any] = {}
            valid_top = {f.name for f in dataclasses.fields(cls)} - {"qnet", "epsilon", "opponents", "eval"}
            valid_nested_top = (
                valid_top
                | {"qnet", "epsilon", "opponents", "eval"}
                | set(_EPSILON_FLAT_MAP)
            )
            _reject_unknown_keys("TrainConfig", d, valid_nested_top)
            for k, v in d.items():
                if k in _EPSILON_FLAT_MAP:
                    nested_epsilon_kw[_EPSILON_FLAT_MAP[k]] = v
            if isinstance(d.get("epsilon"), dict):
                _reject_unknown_keys(
                    "epsilon",
                    d["epsilon"],
                    _dataclass_field_names(EpsilonConfig) | set(_EPSILON_FLAT_MAP),
                )
                for k, v in d["epsilon"].items():
                    nested_epsilon_kw[_EPSILON_FLAT_MAP.get(k, k)] = v
            epsilon   = EpsilonConfig(**nested_epsilon_kw)
            top = {k: v for k, v in d.items() if k in valid_top}
            return cls(
                qnet=qnet,
                epsilon=epsilon,
                opponents=_opponent_config_from_raw(d.get("opponents")),
                eval=_eval_config_from_raw(d.get("eval")),
                **top,
            )

        # ── Flat form (YAML files) ──
        qnet_kw:      dict[str, Any] = {}
        epsilon_kw:   dict[str, Any] = {}
        valid_top = {f.name for f in dataclasses.fields(cls)} - {"qnet", "epsilon", "opponents", "eval"}
        valid_flat = (
            valid_top
            | set(_QNET_FIELDS)
            | set(_EPSILON_FLAT_MAP)
            | {"opponents", "eval"}
        )
        _reject_unknown_keys("TrainConfig", d, valid_flat)
        top_kw: dict[str, Any] = {}
        opponents = _opponent_config_from_raw(d.get("opponents"))
        eval_cfg = _eval_config_from_raw(d.get("eval"))

        for k, v in d.items():
            if k in _QNET_FIELDS:
                qnet_kw[k] = v
            elif k in _EPSILON_FLAT_MAP:
                epsilon_kw[_EPSILON_FLAT_MAP[k]] = v
            elif k in valid_top:
                top_kw[k] = v

        return cls(
            qnet      = QNetConfig(**qnet_kw),
            epsilon   = EpsilonConfig(**epsilon_kw),
            opponents = opponents,
            eval      = eval_cfg,
            **top_kw,
        )


def load_config_from_yaml(path: str | Path) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file.

    CLI flags should be applied on top of this via ``dataclasses.replace``.
    """
    return TrainConfig.from_flat_dict(_load_yaml_dict(path))


def _merge_yaml_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge config mappings, with ``override`` taking precedence."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_yaml_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_dict(path: str | Path, *, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML mapping, recursively resolving relative ``base_config`` refs."""
    import yaml

    resolved = Path(path).resolve()
    if resolved in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, resolved))
        raise ValueError(f"base_config cycle detected: {chain}")
    if not resolved.exists():
        raise FileNotFoundError(f"config file does not exist: {resolved}")

    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config YAML must contain a mapping: {resolved}")

    raw = dict(raw)
    base_ref = raw.pop("base_config", None)
    if base_ref is None:
        return raw
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f"base_config must be a non-empty relative path: {resolved}")

    base_path = Path(base_ref)
    if base_path.is_absolute():
        raise ValueError(f"base_config must be a relative path: {resolved}")
    base_path = resolved.parent / base_path
    base = _load_yaml_dict(base_path, _stack=(*_stack, resolved))
    return _merge_yaml_dicts(base, raw)


def dart_qnet_config(cfg: TrainConfig):
    """Assemble ``DartQNetConfig`` from ``TrainConfig`` flat fields."""
    from .model.q_network import DartQNetConfig

    return DartQNetConfig(
        role_d_model=cfg.dart_role_d_model,
        history_hidden=cfg.dart_history_hidden,
        global_hidden=cfg.dart_global_hidden,
        action_hidden=cfg.dart_action_hidden,
        trunk_hidden=cfg.dart_trunk_hidden,
        trunk_layers=cfg.dart_trunk_layers,
        dropout=cfg.qnet.dropout,
    )


def load_config_from_cli(
    yaml_path: str | Path | None,
    **cli_overrides: Any,
) -> TrainConfig:
    """Load config from YAML plus explicit CLI overrides.

    ``None`` CLI values are ignored so optional argparse flags do not erase
    values loaded from the config file.
    """
    raw: dict[str, Any] = {}
    if yaml_path:
        raw.update(_load_yaml_dict(yaml_path))
    raw.update({k: v for k, v in cli_overrides.items() if v is not None})
    return TrainConfig.from_flat_dict(raw)


__all__ = [
    "TrainConfig",
    "QNetConfig",
    "EpsilonConfig",
    "CheckpointSaveType",
    "QueueFullPolicy",
    "BatchSizeSemantics",
    "EpisodeMixConfig",
    "FrozenPoolConfig",
    "HardBotConfig",
    "OpponentConfig",
    "OpponentSamplingConfig",
    "EvalConfig",
    "MODEL_TYPE_GUANZERO",
    "MODEL_TYPE_DART",
    "CONFIG_SCHEMA_VERSION",
    "ModelType",
    "dart_qnet_config",
    "load_config_from_yaml",
    "load_config_from_cli",
]
