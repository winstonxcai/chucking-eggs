from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from guandan.dart.config import MODEL_TYPE_GUANZERO, QNetConfig, TrainConfig
from guandan.dart.data.buffer import RoleAwareReplayBuffer
from guandan.dart.model.encoding.role_encoder import ROLE_ENCODE_CHANNEL_SHAPES
from guandan.dart.model.q_network import DartQNet, DartQNetConfig, init_guanzero_nets
from guandan.dart.runtime.actor.runtime import maybe_sync_weights
from guandan.dart.runtime.learner import StepResult, _SeatAdapter
from guandan.dart.runtime.learners.dart import DartLearner
from guandan.dart.runtime.learners.loss_bucket_schema import (
    LOSS_BUCKET_GRIDS,
    LOSS_BUCKET_MARGINALS,
    LOSS_BUCKET_SCHEMA,
)
from guandan.dart.runtime.weights import (
    load_latest_weights,
    publish_weights,
    publish_weights_dart,
    read_latest_metadata,
)


def _small_dart_cfg() -> DartQNetConfig:
    return DartQNetConfig(
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
        if key == "trick_head_id":
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


def test_guanzero_total_batch_adapter_samples_per_seat(monkeypatch):
    cfg = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        qnet=QNetConfig(hidden_lstm=8, hidden_mlp=16, n_mlp_layers=1),
        batch_size=4096,
        batch_size_semantics="total",
        device="cpu",
    )
    adapter = _SeatAdapter(cfg)
    observed: list[int] = []

    def fake_update(_buffer, batch_size):
        observed.append(batch_size)
        return {0: 0.0}

    monkeypatch.setattr(adapter._learner, "update", fake_update)
    assert adapter.step() is not None
    assert observed == [1024]


def test_effective_batch_size_drives_learner_sample_metrics():
    cfg = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        qnet=QNetConfig(hidden_lstm=8, hidden_mlp=16, n_mlp_layers=1),
        batch_size=4096,
        batch_size_semantics="total",
        device="cpu",
    )
    adapter = _SeatAdapter(cfg)
    timing = {
        "updates": 2,
        "version": 0,
        "interval_dt": 1.0,
        "interval_upd": 2,
        "session_start_updates": 0,
        "elapsed_s": 1.0,
        "queue_depth": 0,
        "queue_put_timeouts_total": 0,
        "gpu_mem_gb": None,
        "drained_since_log": 0,
        "throttle_drained_since_log": 0,
        "cumulative_drained": 8192,
        "fresh_samples_total": 8192,
        "ema_actor_rate": 8192.0,
        "interval_replay": 1.0,
        "cum_replay": 1.0,
        "n_throttle_sleeps": 0,
        "throttle_sleep_total_s": 0.0,
        "actor_update_lag": {},
        "actor_update_lag_max": None,
        "actor_update_lag_mean": None,
        "actor_version_lag": {},
        "actor_version_lag_max": None,
    }
    row = adapter.format_log_row(StepResult(loss={"0": 0.0}, phase={}), timing)

    assert row["learner_samples_per_update"] == 4096
    assert row["learner_samples_total"] == 8192
    assert row["samples_per_sec"] == 8192.0
    assert row["wall_clock_s"] == 1.0


def test_publish_and_load_weights():
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))

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


def test_read_latest_metadata_warns_on_corrupt_file(caplog):
    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        weight_dir.mkdir()
        (weight_dir / "latest.txt").write_text("not-a-version")

        assert read_latest_metadata(weight_dir) is None
        assert "corrupted latest.txt" in caplog.text


def test_dart_learner_update_step_changes_params_and_returns_metrics():
    torch.manual_seed(0)
    net = DartQNet(_small_dart_cfg())
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)

    metrics = learner.update(_role_buffer(32), batch_size=16)

    assert metrics is not None
    assert "loss" in metrics
    for head in range(4):
        assert f"loss_trick_head_{head}" in metrics
        assert f"q_mean_trick_head_{head}" in metrics
        assert f"grad_norm_head_{head}" in metrics
        assert metrics[f"sample_count_trick_head_{head}"] == 4.0
    assert any(not torch.equal(before[k], v) for k, v in net.state_dict().items())


def test_dart_learner_resume_state_is_equivalent_for_next_update():
    torch.manual_seed(11)
    net = DartQNet(_small_dart_cfg())
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    buf = _role_buffer(128, tags=_full_tags(128))

    assert learner.update(buf, batch_size=64) is not None
    checkpoint_q = {k: v.detach().clone() for k, v in net.state_dict().items()}
    checkpoint_learner = copy.deepcopy(learner.state_dict())
    checkpoint_buffer = buf.state_dict()

    metrics_continued = learner.update(buf, batch_size=64)
    continued = {k: v.detach().clone() for k, v in net.state_dict().items()}

    resumed_net = DartQNet(_small_dart_cfg())
    resumed_net.load_state_dict(checkpoint_q)
    resumed_learner = DartLearner(
        resumed_net,
        lr=1e-3,
        device="cpu",
        max_grad_norm=10.0,
    )
    resumed_learner.load_state_dict(checkpoint_learner)
    resumed_buf = _role_buffer(128, tags=_full_tags(128))
    resumed_buf.load_state_dict(checkpoint_buffer)

    metrics_resumed = resumed_learner.update(resumed_buf, batch_size=64)

    assert metrics_continued is not None
    assert metrics_resumed is not None
    assert metrics_resumed["loss"] == pytest.approx(metrics_continued["loss"])
    for key, value in continued.items():
        assert torch.allclose(value, resumed_net.state_dict()[key], atol=1e-6, rtol=1e-6)


def test_dart_learner_nonfinite_target_error_includes_batch_context():
    net = DartQNet(_small_dart_cfg())
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    tags = _full_tags(32)
    buf = _role_buffer(32, tags=tags)
    buf.returns[:buf.size()] = np.nan

    with pytest.raises(RuntimeError, match="episode_mode_counts"):
        learner.update(buf, batch_size=16)


def test_dart_learner_skips_when_head_short():
    buf = RoleAwareReplayBuffer(capacity=8)
    stacked = _role_stacked(8)
    stacked["trick_head_id"][:] = 0
    buf.push_stacked(stacked, np.zeros(8, dtype=np.float32))
    learner = DartLearner(DartQNet(_small_dart_cfg()), lr=1e-3)

    assert learner.update(buf, batch_size=8) is None


def test_publish_weights_dart_roundtrip_and_sync():
    torch.manual_seed(1)
    q_net = DartQNet(_small_dart_cfg())
    actor_net = DartQNet(_small_dart_cfg())

    with tempfile.TemporaryDirectory() as td:
        weight_dir = Path(td) / "weights"
        assert maybe_sync_weights(actor_net, weight_dir, local_version=5) == (5, 0, 0)

        publish_weights_dart(q_net, weight_dir, version=7)
        snapshot = load_latest_weights(weight_dir)
        assert snapshot is not None
        assert snapshot.version == 7
        assert set(snapshot.state_dicts) == {"q_net"}

        new_version, local_updates, global_updates = maybe_sync_weights(
            actor_net, weight_dir, local_version=6,
        )
        assert new_version == 7
        assert local_updates == 0
        assert global_updates == 0
        for key, value in q_net.state_dict().items():
            assert torch.equal(value, actor_net.state_dict()[key])


def test_per_bucket_loss_emitted_all_grids():
    torch.manual_seed(2)
    net = DartQNet(_small_dart_cfg())
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 512
    buf = _role_buffer(n, tags=_full_tags(n))

    metrics = learner.update(buf, batch_size=128)

    assert metrics is not None
    for spec in LOSS_BUCKET_GRIDS:
        for a in range(len(spec.axis_a.labels)):
            for b in range(len(spec.axis_b.labels)):
                assert spec.key(a, b, "n") in metrics
                assert spec.key(a, b, "frac") in metrics
                assert spec.key(a, b, "loss") in metrics
    for spec in LOSS_BUCKET_MARGINALS:
        for level in range(len(spec.axis.labels)):
            assert spec.key(level, "n") in metrics
            assert spec.key(level, "frac") in metrics
            assert spec.key(level, "loss") in metrics


def test_sparse_cell_emits_null_loss():
    torch.manual_seed(3)
    net = DartQNet(_small_dart_cfg())
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 64
    tags = _full_tags(n)
    # Force every sample to land in the same phase_role cell (a=0, b=0); other
    # cells will be empty so their loss must be None.
    tags["phase_self"][:] = 0
    tags["trick_role"][:] = 0
    buf = _role_buffer(n, tags=tags)

    metrics = learner.update(buf, batch_size=32)
    assert metrics is not None
    phase_role = next(spec for spec in LOSS_BUCKET_GRIDS if spec.name == "phase_role")
    assert metrics[phase_role.key(0, 0, "n")] > 0
    for a in range(len(phase_role.axis_a.labels)):
        for b in range(len(phase_role.axis_b.labels)):
            if (a, b) == (0, 0):
                continue
            assert metrics[phase_role.key(a, b, "n")] == 0
            assert metrics[phase_role.key(a, b, "loss")] is None


def test_q_gap_nan_handled():
    torch.manual_seed(4)
    net = DartQNet(_small_dart_cfg())
    learner = DartLearner(net, lr=1e-3, device="cpu", max_grad_norm=10.0)
    n = 128
    tags = _full_tags(n)
    # Set every q_gap to NaN; should land them all in bucket 0 without crashing.
    tags["q_gap"][:] = np.nan
    buf = _role_buffer(n, tags=tags)

    metrics = learner.update(buf, batch_size=64)
    assert metrics is not None
    q_gap = next(spec for spec in LOSS_BUCKET_MARGINALS if spec.name == "q_gap")
    assert metrics[q_gap.key(0, "n")] > 0
    for c in range(1, len(q_gap.axis.labels)):
        assert metrics[q_gap.key(c, "n")] == 0


def test_loss_bucket_schema_documents_named_keys():
    phase_role = next(spec for spec in LOSS_BUCKET_GRIDS if spec.name == "phase_role")
    assert phase_role.key(0, 0, "loss") == "phase_role_opening__leading_loss"
    assert LOSS_BUCKET_SCHEMA["phase_role"]["axis_a"] == ("opening", "midgame", "endgame")
    assert LOSS_BUCKET_SCHEMA["phase_role"]["axis_b"] == (
        "leading",
        "following_partner_alive",
        "following_partner_out",
    )
