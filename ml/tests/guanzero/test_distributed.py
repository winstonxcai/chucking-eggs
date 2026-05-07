"""Smoke tests for faithful persistent actor-learner DMC components."""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path

import torch


# ─── publish / load round-trip ────────────────────────────


def test_publish_and_load_weights():
    from guandan.guanzero.learner import load_latest_weights, publish_weights
    from guandan.guanzero.q_network import init_position_nets

    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"

        # Before any publish — should return (None, None)
        ver, sds = load_latest_weights(weight_dir)
        assert ver is None and sds is None

        publish_weights(q_nets, weight_dir, version=0)

        assert (weight_dir / "latest.txt").exists()
        assert (weight_dir / "weights_0.pt").exists()
        assert not (weight_dir / "weights_0.tmp").exists()   # tmp cleaned up

        ver, sds = load_latest_weights(weight_dir)
        assert ver == 0
        assert set(sds.keys()) == {0, 1, 2, 3}

        # Second publish — version increments, latest.txt updates atomically
        publish_weights(q_nets, weight_dir, version=1)
        ver2, _ = load_latest_weights(weight_dir)
        assert ver2 == 1
        assert (weight_dir / "weights_1.pt").exists()


# ─── maybe_sync_weights ───────────────────────────────────


def test_maybe_sync_weights_no_op_when_not_newer():
    from guandan.guanzero.worker import maybe_sync_weights
    from guandan.guanzero.learner import publish_weights
    from guandan.guanzero.q_network import init_position_nets

    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets, weight_dir, version=5)

        # local_version == published version → no update
        result = maybe_sync_weights(q_nets, weight_dir, local_version=5)
        assert result == 5

        # local_version > published version → no update (shouldn't happen in practice)
        result = maybe_sync_weights(q_nets, weight_dir, local_version=6)
        assert result == 6


def test_maybe_sync_weights_loads_newer():
    from guandan.guanzero.worker import maybe_sync_weights
    from guandan.guanzero.learner import publish_weights
    from guandan.guanzero.q_network import init_position_nets

    q_nets_pub  = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)
    q_nets_actor = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)

    # Perturb published nets so we can verify the actor's nets got updated
    for net in q_nets_pub.values():
        for p in net.parameters():
            p.data.fill_(99.0)

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        publish_weights(q_nets_pub, weight_dir, version=3)

        new_ver = maybe_sync_weights(q_nets_actor, weight_dir, local_version=-1)
        assert new_ver == 3

        # Actor's nets should now reflect the published weights
        for p in range(4):
            for param in q_nets_actor[p].parameters():
                assert (param.data == 99.0).all()


# ─── actor_loop: single episode → queue ──────────────────


def test_actor_loop_single_episode():
    """Actor pushes at least one batch to the queue within a short window."""
    import multiprocessing as mp

    from guandan.guanzero.worker import actor_loop
    from guandan.guanzero.learner import publish_weights
    from guandan.guanzero.q_network import init_position_nets
    from guandan.guanzero.config import TrainConfig

    cfg = TrainConfig(
        hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2,
        actor_push_batch_size=1,   # push after every episode
        sync_interval_episodes=1,
    )
    cfg_dict = dataclasses.asdict(cfg)

    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)

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

        # Wait up to 30s for the actor to produce something
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
        # All channels stacked along axis 0 with the same N
        for k, arr in msg["stacked"].items():
            assert arr.shape[0] == n, f"stacked[{k!r}] has shape[0]={arr.shape[0]}, expected {n}"


# ─── learner_loop: drains queue + updates ────────────────


def test_learner_drains_queue_and_updates():
    """Push 2 batches of samples to the queue; learner drains and updates > 0 times."""
    import multiprocessing as mp

    import numpy as np

    from guandan.guanzero.learner import learner_loop, publish_weights
    from guandan.guanzero.q_network import init_position_nets
    from guandan.guanzero.config import TrainConfig

    cfg = TrainConfig(
        hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2,
        buffer_min_size=2,
        batch_size=2,
        max_drain_batches_per_loop=10,
        publish_interval_updates=1,
        log_every_updates=1,
    )
    cfg_dict = dataclasses.asdict(cfg)

    # Build minimal encoded dicts matching the encoder's output shape
    from guandan.guanzero.encoder import StateActionEncoder
    encoder = StateActionEncoder()
    from guandan.game import GuanDanEnv
    from guandan.combos import Combo
    env = GuanDanEnv()
    env.reset(seed=7)
    legal = env.legal_moves(env.current_player)
    dummy_encoded = encoder.encode_all(env, env.current_player, legal)[0]

    # Build a pre-stacked batch message — one sample for each of the 4 players
    # so the learner's per-position warmup gate (all buffers ≥ buffer_min_size) opens.
    # buffer_min_size=2 → push 2 samples per player, twice.
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

        # Pre-load queue before learner starts
        queue.put(batch_msg)
        queue.put(batch_msg)

        proc = ctx.Process(
            target=learner_loop,
            args=(cfg_dict, queue, stop_event, weight_dir, run_dir),
            daemon=True,
        )
        proc.start()

        # Give learner time to drain + update
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
