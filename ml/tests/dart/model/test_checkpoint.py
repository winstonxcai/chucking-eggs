from __future__ import annotations

import numpy as np
import torch
from guandan.dart.config import (
    MODEL_TYPE_DART,
    MODEL_TYPE_GUANZERO,
    QNetConfig,
    TrainConfig,
)
from guandan.dart.data.buffer import RoleAwareReplayBuffer
from guandan.dart.model.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_checkpoint_dart,
)
from guandan.dart.model.encoding.role_encoder import ROLE_ENCODE_CHANNEL_SHAPES
from guandan.dart.model.q_network import DartQNet, DartQNetConfig, init_guanzero_nets
from guandan.dart.runtime.learners.dart import DartLearner


def test_checkpoint_roundtrip_saves_config_and_all_seat_weights(tmp_path):
    cfg = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        qnet=QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2),
        batch_size=4096,
        batch_size_semantics="total",
    )
    q_nets = init_guanzero_nets(cfg.qnet)
    for param in q_nets[0].parameters():
        param.data.fill_(3.0)

    path = tmp_path / "checkpoints" / "ckpt.pt"
    save_checkpoint(path, q_nets, cfg, episode_or_update=12)

    ckpt = load_checkpoint(path)
    assert ckpt["episode"] == 12
    assert ckpt["config"]["qnet"]["hidden_lstm"] == 16
    assert ckpt["config"]["batch_size_semantics"] == "total"
    assert set(ckpt["q_nets"]) == {0, 1, 2, 3}

    loaded = init_guanzero_nets(cfg.qnet)
    loaded[0].load_state_dict(ckpt["q_nets"][0])
    for param in loaded[0].parameters():
        assert torch.all(param == 3.0)


def _small_dart_cfg() -> DartQNetConfig:
    return DartQNetConfig(
        role_d_model=8,
        history_hidden=8,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=16,
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
            stacked[key] = arr
    return stacked


def test_dart_checkpoint_persists_resume_state(tmp_path):
    cfg = TrainConfig(model_type=MODEL_TYPE_DART, batch_size=8, buffer_capacity=32)
    q_net = DartQNet(_small_dart_cfg())
    learner = DartLearner(q_net, lr=1e-3)
    buffer = RoleAwareReplayBuffer(capacity=32, seed=123)
    buffer.push_stacked(_role_stacked(16), np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    metrics = learner.update(buffer, batch_size=8)
    assert metrics is not None

    path = tmp_path / "checkpoints" / "dart.pt"
    save_checkpoint_dart(
        path,
        q_net,
        cfg,
        episode_or_update=17,
        learner_state=learner.state_dict(),
        replay_state=buffer.state_dict(),
        rng_state={"cpu": torch.get_rng_state()},
        actor_rng_states={3: {"actor_random": ("state",)}},
    )

    ckpt = load_checkpoint(path)
    assert ckpt["checkpoint_format_version"] == 2
    assert ckpt["total_updates"] == 17
    assert "optimizer" in ckpt["learner_state"]
    assert ckpt["replay_state"]["size"] == 16
    assert ckpt["actor_rng_states"][3]["actor_random"] == ("state",)

    restored_buffer = RoleAwareReplayBuffer(capacity=32, seed=999)
    restored_buffer.load_state_dict(ckpt["replay_state"])
    assert restored_buffer.size() == buffer.size()
    assert np.array_equal(restored_buffer.returns[:16], buffer.returns[:16])

    restored_learner = DartLearner(DartQNet(_small_dart_cfg()), lr=1e-3)
    restored_learner.load_state_dict(ckpt["learner_state"])
    assert restored_learner.opt.state_dict()["state"]


def test_dart_checkpoint_weight_only_omits_resume_state(tmp_path):
    cfg = TrainConfig(model_type=MODEL_TYPE_DART, batch_size=8, buffer_capacity=32)
    q_net = DartQNet(_small_dart_cfg())

    path = tmp_path / "checkpoints" / "dart_weights.pt"
    save_checkpoint_dart(
        path,
        q_net,
        cfg,
        episode_or_update=23,
        learner_state={"optimizer": {"state": {"expensive": True}}},
        replay_state={"size": 16},
        rng_state={"cpu": torch.get_rng_state()},
        actor_rng_states={0: {"actor_random": ("state",)}},
        save_type="weight",
    )

    ckpt = load_checkpoint(path)
    assert ckpt["checkpoint_save_type"] == "weight"
    assert ckpt["total_updates"] == 23
    assert "q_net" in ckpt
    assert "config" in ckpt
    assert "learner_state" not in ckpt
    assert "replay_state" not in ckpt
    assert "rng_state" not in ckpt
    assert "actor_rng_states" not in ckpt


def test_invalid_checkpoint_save_type_fails_fast(tmp_path):
    cfg = TrainConfig(model_type=MODEL_TYPE_DART)
    q_net = DartQNet(_small_dart_cfg())

    path = tmp_path / "checkpoints" / "bad.pt"
    try:
        save_checkpoint_dart(path, q_net, cfg, episode_or_update=1, save_type="weights")
    except ValueError as exc:
        assert "checkpoint_save_type" in str(exc)
    else:
        raise AssertionError("invalid save_type should raise")
