"""Distillation config for Stage 0 (distill_pvguan.py)."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path


@dataclasses.dataclass
class DistillConfig:
    # Supervisor
    supervisor:              str         = "jidan"
    supervisor_search:       bool        = False
    supervisor_pimc_n_det:   int         = 16
    supervisor_checkpoint:   str | None  = None

    # Data collection
    n_decisions:  int   = 500_000
    workers:      int   = 4
    seed:         int   = 0

    # Training
    epochs:           int   = 8
    batch_size:       int   = 512
    lr:               float = 3e-4
    hidden:           int   = 256
    label_mode:       str   = "hard"   # hard | soft | hybrid
    label_smoothing:  float = 0.1
    val_split:        float = 0.10
    patience:         int   = 3

    # Output
    output: str = "ml/checkpoints/pvguan_distilled_{supervisor}.pt"

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def resolved_output(self) -> str:
        return self.output.format(supervisor=self.supervisor)

    @property
    def resolved_run_dir(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/distill_{self.supervisor}_{ts}"

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DistillConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DistillConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── CLI integration ───────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> "DistillConfig":
        return cls(
            supervisor=args.supervisor,
            supervisor_search=args.supervisor_search == "on",
            supervisor_pimc_n_det=args.supervisor_pimc_n_det,
            supervisor_checkpoint=args.supervisor_checkpoint,
            n_decisions=args.n_decisions,
            workers=args.workers,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            label_mode=args.label_mode,
            label_smoothing=args.label_smoothing,
            output=args.output,
        )

    def __str__(self) -> str:
        lines = [
            f"DistillConfig:",
            f"  supervisor      = {self.supervisor}  search={self.supervisor_search}  n_det={self.supervisor_pimc_n_det}",
            f"  n_decisions     = {self.n_decisions:,}  workers={self.workers}  seed={self.seed}",
            f"  epochs          = {self.epochs}  batch={self.batch_size}  lr={self.lr}",
            f"  hidden          = {self.hidden}  label_mode={self.label_mode}  smoothing={self.label_smoothing}",
            f"  output          = {self.resolved_output}",
            f"  run_dir         = {self.resolved_run_dir}  (auto-timestamped)",
        ]
        return "\n".join(lines)
