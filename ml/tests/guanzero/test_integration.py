"""Integration tests that span multiple modules or spawn subprocesses."""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path


def test_actor_loop_single_episode():
    """Actor pushes at least one batch to the queue within a short window."""
    import multiprocessing as mp

    from guandan.guanzero.config import QNetConfig, TrainConfig
    from guandan.guanzero.learner import publish_weights
    from guandan.guanzero.q_network import init_seat_nets
    from guandan.guanzero.worker import actor_loop

    cfg = TrainConfig(
        qnet=QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2),
        actor_push_batch_size=1,
        sync_interval_episodes=1,
    )
    cfg_dict = dataclasses.asdict(cfg)

    q_nets = init_seat_nets(cfg.qnet)

    ctx = mp.get_context("spawn")

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets, weight_dir, version=0)

        queue      = ctx.Queue(maxsize=10)
        stop_event = ctx.Event()

        proc = ctx.Process(
            target=actor_loop,
            args=(0, cfg_dict, queue, stop_event, weight_dir),
            daemon=True,
        )
        proc.start()

        deadline = time.time() + 30
        msg = None
        while time.time() < deadline:
            if not queue.empty():
                msg = queue.get(timeout=1)
                break
            time.sleep(0.2)

        stop_event.set()
        proc.join(timeout=10)

        assert msg is not None, "Actor produced no samples within 30s"
        assert {"actor_id", "version", "stacked", "players", "returns"} <= msg.keys()
        n = len(msg["players"])
        assert n >= 1
        assert len(msg["returns"]) == n
        for k, arr in msg["stacked"].items():
            assert arr.shape[0] == n, f"stacked[{k!r}] has shape[0]={arr.shape[0]}, expected {n}"


def test_learner_drains_queue_and_updates():
    """Push 2 batches of samples to the queue; learner drains and updates > 0 times."""
    import multiprocessing as mp

    import numpy as np

    from guandan.combos import Combo  # noqa: F401 — ensures game engine is importable
    from guandan.game import GuanDanEnv
    from guandan.guanzero.config import QNetConfig, TrainConfig
    from guandan.guanzero.encoder import StateActionEncoder
    from guandan.guanzero.learner import learner_loop, publish_weights
    from guandan.guanzero.q_network import init_seat_nets

    cfg = TrainConfig(
        qnet=QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2),
        buffer_min_size=2,
        batch_size=2,
        max_drain_batches_per_loop=10,
        publish_interval_updates=1,
        log_every_updates=1,
    )
    cfg_dict = dataclasses.asdict(cfg)

    encoder = StateActionEncoder()
    env = GuanDanEnv()
    env.reset(seed=7)
    legal = env.legal_moves(env.current_player)
    dummy_encoded = encoder.encode_all(env, env.current_player, legal)[0]

    N = 8
    stacked = {k: np.stack([dummy_encoded[k]] * N, axis=0) for k in dummy_encoded}
    batch_msg = {
        "actor_id": 0,
        "version":  0,
        "stacked":  stacked,
        "players":  np.array([0, 1, 2, 3] * 2, dtype=np.int8),
        "returns":  np.full(N, 1.0, dtype=np.float32),
    }

    ctx = mp.get_context("spawn")

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        run_dir    = Path(td) / "run"
        run_dir.mkdir()

        queue      = ctx.Queue(maxsize=10)
        stop_event = ctx.Event()

        queue.put(batch_msg)
        queue.put(batch_msg)

        proc = ctx.Process(
            target=learner_loop,
            args=(cfg_dict, queue, stop_event, weight_dir, run_dir),
            daemon=True,
        )
        proc.start()

        deadline = time.time() + 30
        metrics_path = run_dir / "metrics_learner.jsonl"
        got_update = False
        while time.time() < deadline:
            if metrics_path.exists() and metrics_path.stat().st_size > 0:
                got_update = True
                break
            time.sleep(0.2)

        stop_event.set()
        proc.join(timeout=10)

        assert got_update, "Learner never logged a metrics row"
        assert (weight_dir / "latest.txt").exists(), "Learner never published weights"
