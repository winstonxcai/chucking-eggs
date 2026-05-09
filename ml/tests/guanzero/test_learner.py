from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from guandan.guanzero.buffer import RoleAwareReplayBuffer
from guandan.guanzero.config import QNetConfig
from guandan.guanzero.encoding.role_encoder import ROLE_ENCODE_CHANNEL_SHAPES
from guandan.guanzero.learner import (
    SharedHeadLearner,
    load_latest_weights,
    publish_weights,
    publish_weights_shared,
)
from guandan.guanzero.q_network import SharedHeadQNet, SharedHeadQNetConfig, init_seat_nets
from guandan.guanzero.worker import maybe_sync_weights_shared


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


def _role_buffer(n: int = 32) -> RoleAwareReplayBuffer:
    buf = RoleAwareReplayBuffer(capacity=n)
    buf.push_stacked(_role_stacked(n), np.linspace(-1.0, 1.0, n, dtype=np.float32))
    return buf


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
