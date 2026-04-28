"""PPO training run config for train_pvguan.py."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclasses.dataclass
class PPORunConfig:
    # ── ablation ─────────────────────────────────────────────────────────────
    critic:  Literal["pv", "ptie"] = "pv"   # only difference between the two runs
    seed:    int                   = 0

    # ── architecture ─────────────────────────────────────────────────────────
    hidden:    int = 256
    warmstart: str | None = None             # path to distilled checkpoint

    # ── training budget ───────────────────────────────────────────────────────
    total_decisions:    int        = 6_000_000
    iter_decisions:     int        = 4_096
    snapshot_interval:  int        = 1_000_000   # save checkpoint every N decisions

    # ── PPO loss ──────────────────────────────────────────────────────────────
    clip_eps:    float = 0.2    # trust-region clip
    c_value:     float = 0.5    # value loss coefficient
    c_entropy:   float = 0.01   # entropy bonus (prevents collapse)
    lr:          float = 3e-4
    weight_decay: float = 0.0   # intentionally 0 — see priv-slot diagnostic
    n_epochs:    int   = 4      # PPO epochs per rollout batch
    mini_batch:  int   = 256

    # ── GAE ───────────────────────────────────────────────────────────────────
    gamma: float = 1.0    # no discounting (episodic terminal reward)
    lam:   float = 0.95   # bias/variance tradeoff

    # ── warmstart regularization ─────────────────────────────────────────────
    c_kl_init:            float = 0.05   # KL coeff during critic warmup + actor unfreeze
    kl_decay_actor_iters: int   = 100    # iters to decay c_kl → 0 after actor unfreezes
    critic_warmup_iters:  int   = 30     # iters actor is frozen (critic trains only)

    # ── sampling temperature ─────────────────────────────────────────────────
    temperature:      float       = 1.0
    init_temperature: float | None = None  # override for iter-0 only; falls back to temperature

    # ── infrastructure ───────────────────────────────────────────────────────
    workers:         int   = 4    # legacy; unused by current trainer
    rollout_workers: int   = 1    # >1 enables multiprocess rollout collector
    resume:          str | None = None    # path to checkpoint to resume from
    run_dir:         str | None = None    # defaults to ml/runs/pvguan_{critic}_seed{seed}

    # ── mixed-opponent rollout ───────────────────────────────────────────────
    # Per hand: with prob `selfplay_frac` all 4 seats are net (current behavior).
    # Otherwise pick a random bot from `opponent_mix` and put it at seats {1,3};
    # net plays {0,2}. Only seat-0/2 decisions enter the PPO buffer.
    opponent_mix:    tuple[str, ...] = ()  # e.g. ("jidan", "yaoji", "strategic")
    selfplay_frac:   float = 1.0           # 1.0 = pure self-play (legacy)

    # ── validation eval ──────────────────────────────────────────────────────
    val_every:     int  = 100             # iters between validation paired evals (0 = off)
    val_games:     int  = 200             # games per opponent (= n_decks × 4 rotations)
    val_opponents: tuple[str, ...] = ("jidan", "yaoji")
    val_bootstrap: int  = 200             # bootstrap resamples for CI
    val_patience:  int  = 0              # stop if val metric doesn't improve for N evals (0 = off)

    # ── derived ───────────────────────────────────────────────────────────────

    @property
    def resolved_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/pvguan_{self.critic}_seed{self.seed}_{ts}"

    @property
    def resolved_init_temperature(self) -> float:
        return self.init_temperature if self.init_temperature is not None else self.temperature

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PPORunConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in fields}
        # JSON lists → keep as list (snapshot_decisions)
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PPORunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── CLI integration ───────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "PPORunConfig":
        return cls(
            critic=args.critic,
            seed=args.seed,
            hidden=args.hidden,
            warmstart=args.warmstart,
            total_decisions=args.total_decisions,
            iter_decisions=args.iter_decisions,
            clip_eps=args.clip_eps,
            c_value=args.c_value,
            c_entropy=args.c_entropy,
            lr=args.lr,
            weight_decay=0.0,
            n_epochs=args.n_epochs,
            mini_batch=args.mini_batch,
            gamma=args.gamma,
            lam=args.lam,
            c_kl_init=args.kl_init,
            kl_decay_actor_iters=args.kl_decay_actor_iters,
            critic_warmup_iters=args.critic_warmup,
            temperature=args.temperature,
            init_temperature=args.init_temperature,
            workers=getattr(args, "workers", 4),
            rollout_workers=getattr(args, "rollout_workers", 1),
            resume=getattr(args, "resume", None),
            run_dir=args.run_dir,
            snapshot_interval=getattr(args, "snapshot_interval", 1_000_000),
            val_every=getattr(args, "val_every", 50),
            val_games=getattr(args, "val_games", 200),
            val_opponents=tuple(
                getattr(args, "val_opponents", "jidan,yaoji").split(",")
            ) if isinstance(getattr(args, "val_opponents", None), str)
              else getattr(args, "val_opponents", ("jidan", "yaoji")),
            val_bootstrap=getattr(args, "val_bootstrap", 200),
            val_patience=getattr(args, "val_patience", 0),
            opponent_mix=tuple(
                s.strip() for s in getattr(args, "opponent_mix", "").split(",")
                if s.strip()
            ),
            selfplay_frac=getattr(args, "selfplay_frac", 1.0),
        )

    def __str__(self) -> str:
        lines = [
            f"PPORunConfig:",
            f"  critic          = {self.critic}  seed={self.seed}  hidden={self.hidden}",
            f"  warmstart       = {self.warmstart}",
            f"  budget          = {self.total_decisions:,} decisions  "
                f"iter={self.iter_decisions}  snap_every={self.snapshot_interval:,}",
            f"  resume          = {self.resume}",
            f"  rollout_workers = {self.rollout_workers}",
            f"  PPO             = clip={self.clip_eps}  c_V={self.c_value}  "
                f"c_H={self.c_entropy}  epochs={self.n_epochs}  mb={self.mini_batch}",
            f"  GAE             = γ={self.gamma}  λ={self.lam}",
            f"  optim           = lr={self.lr}  wd={self.weight_decay}",
            f"  KL warmstart    = c_kl={self.c_kl_init}  "
                f"decay_iters={self.kl_decay_actor_iters}  "
                f"critic_warmup={self.critic_warmup_iters}",
            f"  temperature     = τ={self.temperature}  τ_init={self.resolved_init_temperature}",
            f"  validation      = every {self.val_every} iters  "
                f"games={self.val_games} (={self.val_games // 4} decks × 4 rot)  "
                f"opponents={list(self.val_opponents)}  "
                f"patience={self.val_patience or 'off'}",
            f"  rollout opp_mix = {list(self.opponent_mix) or 'none (pure self-play)'}  "
                f"selfplay_frac={self.selfplay_frac}",
            f"  run_dir         = {self.resolved_run_dir}  (auto-timestamped)",
        ]
        return "\n".join(lines)
