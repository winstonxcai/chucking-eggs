from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from guandan.guanzero.data.buffer import RoleAwareReplayBuffer
from guandan.guanzero.config import QNetConfig
from guandan.guanzero.model.encoding.role_encoder import ROLE_ENCODE_CHANNEL_SHAPES
from guandan.guanzero.runtime.learner import (
    SharedHeadLearner,
    load_latest_weights,
    publish_weights,
    publish_weights_shared,
)
from guandan.guanzero.model.q_network import SharedHeadQNet, SharedHeadQNetConfig, init_seat_nets
from guandan.guanzero.runtime.worker import maybe_sync_weights_shared


def _small_shared_cfg() -> SharedHeadQNetConfig:
    return SharedHeadQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    )


def _role_stacked(n: int) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for key, shape in ROLE_ENCODE_CHANNEL_SHAPES.items():
        if key == "seat_id":
            stacked[key] = np.asarray([i % 4 for i in range(n)], dtype=np.int8)
        else:
            arr = np.zeros((n, *shape), dtype=np.uint8)
            if key == "candidate_action":
                arr[:, 0] = 1
            if key == "global_features":
                arr[:, 0] = 1
            stacked[key] = arr
    return stacked


def _role_buffer(n: int = 32, tags: dict | None = None) -> RoleAwareReplayBuffer:
    buf = RoleAwareReplayBuffer(capacity=n)
    buf.push_stacked(
        _role_stacked(n),
        np.linspace(-1.0, 1.0, n, dtype=np.float32),
        tags=tags,
    )
    return buf


def _full_tags(n: int) -> dict:
    """Build a tag dict that exercises every grid + marginal cell at least once."""
    return {
        "phase_self":        np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "trick_role":        np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "phase_partner":     np.asarray([i % 4 for i in range(n)], dtype=np.int8),
        "action_type":       np.asarray([i % 17 for i in range(n)], dtype=np.int8),
        "is_pass":           np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "is_bomb":           np.asarray([(i + 1) % 2 for i in range(n)], dtype=np.int8),
        "num_legal_actions": np.asarray([(i % 25) + 1 for i in range(n)], dtype=np.int16),
        "q_gap":             np.asarray([np.nan if i % 4 == 0 else (i % 10) * 0.05 for i in range(n)], dtype=np.float32),
        "chosen_by_epsilon": np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "episode_mode":      np.asarray([i % 3 for i in range(n)], dtype=np.int8),
        "opponent_id":       np.asarray([i % 6 for i in range(n)], dtype=np.int8),
        "latest_team":       np.asarray([i % 2 for i in range(n)], dtype=np.int8),
        "terminal_reward":   np.asarray([float((i % 7) - 3) for i in range(n)], dtype=np.float32),
    }


def test_publish_and_load_weights():
    q_nets = init_seat_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"

        assert load_latest_weights(weight_dir) is None

        publish_weights(q_nets, weight_dir, version=0)

        assert (weight_dir / "latest.txt").exists()
        assert (weight_dir / "weights_0.pt").exists()
        assert not (weight_dir / "weights_0.tmp").exists()

        snapshot = load_latest_weights(weight_dir)
        assert snapshot is not None
        assert snapshot.version == 0
        assert set(snapshot.state_dicts.keys()) == {0, 1, 2, 3}

        publish_weights(q_nets, weight_dir, version=1)
        snapshot = load_latest_weights(weight_dir)
        assert snapshot is not None
        assert snapshot.version == 1
        assert (weight_dir / "weights_1.pt").exists()


def test_shared_learner_update_step_changes_params_and_returns_metrics():
    torch.manual_seed(0)
    net = SharedHeadQNet(_small_shared_cfg())
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    learner = SharedHeadLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)

    metrics = learner.update(_role_buffer(32), batch_size=16)

    assert metrics is not None
    assert "loss" in metrics
    for seat in range(4):
        assert f"loss_seat_{seat}" in metrics
        assert f"q_mean_seat_{seat}" in metrics
        assert f"grad_norm_head_{seat}" in metrics
        assert metrics[f"sample_count_seat_{seat}"] == 4.0
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())


def test_shared_learner_skips_when_seat_short():
    buf = RoleAwareReplayBuffer(capacity=8)
    stacked = _role_stacked(8)
    stacked["seat_id"][:] = 0
    buf.push_stacked(stacked, np.zeros(8, dtype=np.float32))
    learner = SharedHeadLearner(SharedHeadQNet(_small_shared_cfg()), lr=1e-3)

    assert learner.update(buf, batch_size=8) is None


def test_publish_weights_shared_roundtrip_and_sync():
    torch.manual_seed(1)
    q_net = SharedHeadQNet(_small_shared_cfg())
    actor_net = SharedHeadQNet(_small_shared_cfg())

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        assert maybe_sync_weights_shared(actor_net, weight_dir, local_version=5) == (5, 0, 0)

        publish_weights_shared(q_net, weight_dir, version=7)
        snapshot = load_latest_weights(weight_dir)
        assert snapshot is not None
        assert snapshot.version == 7
        assert set(snapshot.state_dicts) == {"shared"}

        new_version, local_updates, global_updates = maybe_sync_weights_shared(
            actor_net, weight_dir, local_version=6,
        )
        assert new_version == 7
        assert local_updates == 0
        assert global_updates == 0
        for key, value in q_net.state_dict().items():
            assert torch.equal(value, actor_net.state_dict()[key])


def test_per_bucket_loss_emitted_all_grids():
    torch.manual_seed(2)
    net = SharedHeadQNet(_small_shared_cfg())
    learner = SharedHeadLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 512
    buf = _role_buffer(n, tags=_full_tags(n))

    metrics = learner.update(buf, batch_size=128)

    assert metrics is not None
    # All five grids: phase_role(9), phase_pair(12), source_phase(9), opp_phase(18), action_phase(18)
    for prefix, n_cells in [
        ("phase_role", 9), ("phase_pair", 12), ("source_phase", 9),
        ("opp_phase", 18), ("action_phase", 18),
    ]:
        for c in range(n_cells):
            assert f"{prefix}_{c}_n" in metrics
            assert f"{prefix}_{c}_frac" in metrics
            assert f"{prefix}_{c}_loss" in metrics
    # All seven marginals
    for prefix, n_cells in [
        ("epsilon", 2), ("is_pass", 2), ("is_bomb", 2),
        ("k_bucket", 4), ("q_gap", 4), ("team", 2), ("reward", 4),
    ]:
        for c in range(n_cells):
            assert f"{prefix}_{c}_n" in metrics
            assert f"{prefix}_{c}_frac" in metrics
            assert f"{prefix}_{c}_loss" in metrics


def test_sparse_cell_emits_null_loss():
    torch.manual_seed(3)
    net = SharedHeadQNet(_small_shared_cfg())
    learner = SharedHeadLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 64
    tags = _full_tags(n)
    # Force every sample to land in the same phase_role cell (a=0, b=0); other
    # cells will be empty so their loss must be None.
    tags["phase_self"][:] = 0
    tags["trick_role"][:] = 0
    buf = _role_buffer(n, tags=tags)

    metrics = learner.update(buf, batch_size=32)
    assert metrics is not None
    # Cell 0 (a=0, b=0) is populated; cells 1..8 are empty → None loss.
    assert metrics["phase_role_0_n"] > 0
    for c in range(1, 9):
        assert metrics[f"phase_role_{c}_n"] == 0
        assert metrics[f"phase_role_{c}_loss"] is None


def test_q_gap_nan_handled():
    torch.manual_seed(4)
    net = SharedHeadQNet(_small_shared_cfg())
    learner = SharedHeadLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 128
    tags = _full_tags(n)
    # Set every q_gap to NaN; should land them all in bucket 0 without crashing.
    tags["q_gap"][:] = np.nan
    buf = _role_buffer(n, tags=tags)

    metrics = learner.update(buf, batch_size=64)
    assert metrics is not None
    assert metrics["q_gap_0_n"] > 0
    # Other q_gap buckets should be empty.
    for c in range(1, 4):
        assert metrics[f"q_gap_{c}_n"] == 0
