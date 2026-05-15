"""Smoke test for the train() orchestrator.

Exercises the full spawn → counter-driven progress → clean-shutdown path
with a tiny network and CPU device. Verifies that the metrics file is
schema-stamped and that train.log captures both subprocess and main events.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from guandan.dart.config import EpsilonConfig, QNetConfig, TrainConfig
from guandan.dart.utils.metrics import METRICS_SCHEMA_VERSION
from guandan.dart.runtime.train import train


pytestmark = pytest.mark.slow


def _smoke_cfg(run_dir: Path) -> TrainConfig:
    return TrainConfig(
        qnet=QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2),
        epsilon=EpsilonConfig(start=0.05, final=0.05, decay_updates=0),
        n_actors=1,
        batch_size=16,
        buffer_min_size=16,
        buffer_capacity_per_player=2_000,
        actor_push_batch_size=16,
        sample_queue_maxsize=16,
        max_drain_batches_per_loop=8,
        publish_interval_updates=5,
        checkpoint_every_updates=5,
        log_every_updates=5,
        total_updates_target=5,
        device="cpu",
        run_dir=str(run_dir),
    )


def test_train_smoke_runs_to_completion(tmp_path):
    run_dir = tmp_path / "smoke"
    cfg = _smoke_cfg(run_dir)

    train(cfg)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "learner.log").exists()
    assert (run_dir / "checkpoints" / "final.pt").exists()

    metrics_path = run_dir / "metrics_learner.jsonl"
    assert metrics_path.exists()
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    assert rows, "expected at least one metrics row"
    assert all(r["_schema"] == METRICS_SCHEMA_VERSION for r in rows)
    assert rows[-1]["updates"] >= 5

    train_log = (run_dir / "train.log").read_text()
    assert "Learner started" in train_log
    assert "done" in train_log.lower()
