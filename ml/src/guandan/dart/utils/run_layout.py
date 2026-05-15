"""On-disk layout for a Dart training run.

Centralizes the filenames that show up in multiple processes so renames are
a one-line change. Picklable for spawn-context subprocess passing.

Weight publishing has its own protocol (``latest.txt`` + ``weights_{ver}.pt``
under ``weight_dir``) and is intentionally not modeled here — that path can
be redirected via ``DART_WEIGHT_DIR`` independent of ``run_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path

    @property
    def config_json(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def train_log(self) -> Path:
        return self.run_dir / "train.log"

    @property
    def learner_log(self) -> Path:
        return self.run_dir / "learner.log"

    @property
    def inference_server_log(self) -> Path:
        return self.run_dir / "inference_server.log"

    @property
    def metrics_jsonl(self) -> Path:
        return self.run_dir / "metrics_learner.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def final_checkpoint(self) -> Path:
        return self.checkpoints_dir / "final.pt"

    def update_checkpoint(self, n: int) -> Path:
        return self.checkpoints_dir / f"update_{n:08d}.pt"


__all__ = ["RunLayout"]
