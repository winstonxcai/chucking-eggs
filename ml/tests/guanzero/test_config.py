from __future__ import annotations

import dataclasses

import yaml

from guandan.guanzero.config import (
    EpsilonConfig,
    InferenceConfig,
    QNetConfig,
    TrainConfig,
    load_config_from_cli,
    load_config_from_yaml,
    shared_head_qnet_config,
)
from guandan.guanzero.q_network import SharedHeadQNetConfig


def test_from_flat_dict_routes_flat_keys_and_ignores_removed_fields():
    cfg = TrainConfig.from_flat_dict({
        "hidden_lstm": 64,
        "hidden_mlp": 128,
        "epsilon_start": 0.2,
        "epsilon_final": 0.03,
        "use_inference_server": True,
        "inference_n_slots": 32,
        "n_actors": 3,
        "max_legal_actions": 320,
        "env_lanes_per_actor": 4,
        "compile_actor": True,
    })

    assert cfg.qnet.hidden_lstm == 64
    assert cfg.qnet.hidden_mlp == 128
    assert cfg.epsilon.start == 0.2
    assert cfg.epsilon.final == 0.03
    assert cfg.inference.enabled is True
    assert cfg.inference.n_slots == 32
    assert cfg.n_actors == 3


def test_from_flat_dict_accepts_nested_dicts_and_dataclass_instances():
    nested = dataclasses.asdict(TrainConfig(
        qnet=QNetConfig(hidden_lstm=32),
        epsilon=EpsilonConfig(decay_updates=7),
        inference=InferenceConfig(max_actions=99),
        batch_size=11,
    ))

    cfg = TrainConfig.from_flat_dict(nested)
    assert cfg.qnet.hidden_lstm == 32
    assert cfg.epsilon.decay_updates == 7
    assert cfg.inference.max_actions == 99
    assert cfg.batch_size == 11

    cfg = TrainConfig.from_flat_dict({
        "qnet": QNetConfig(hidden_mlp=256),
        "epsilon": EpsilonConfig(final=0.02),
        "inference": InferenceConfig(enabled=True),
    })
    assert cfg.qnet.hidden_mlp == 256
    assert cfg.epsilon.final == 0.02
    assert cfg.inference.enabled is True


def test_yaml_and_cli_loading_apply_precedence(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "hidden_lstm": 16,
        "epsilon_start": 0.5,
        "n_actors": 4,
    }))

    from_yaml = load_config_from_yaml(path)
    assert from_yaml.qnet.hidden_lstm == 16
    assert from_yaml.epsilon.start == 0.5
    assert from_yaml.n_actors == 4

    cfg = load_config_from_cli(
        path,
        quick=True,
        quick_overrides={
            "qnet": QNetConfig(hidden_lstm=24, hidden_mlp=48),
            "buffer_min_size": 5,
        },
        n_actors=2,
        device=None,
    )
    assert cfg.qnet.hidden_lstm == 24
    assert cfg.qnet.hidden_mlp == 48
    assert cfg.epsilon.start == 0.5
    assert cfg.buffer_min_size == 5
    assert cfg.n_actors == 2


def test_model_type_default_and_invalid():
    assert TrainConfig().model_type == "seat_nets"

    try:
        TrainConfig(model_type="bad")
    except ValueError as exc:
        assert "Unknown model_type" in str(exc)
    else:
        raise AssertionError("invalid model_type should raise")


def test_model_type_shared_heads_loads_and_factory(tmp_path):
    path = tmp_path / "shared.yaml"
    path.write_text(yaml.safe_dump({
        "model_type": "shared_heads",
        "shared_head_role_d_model": 64,
        "shared_head_history_hidden": 96,
        "shared_head_global_hidden": 32,
        "shared_head_action_hidden": 48,
        "shared_head_trunk_hidden": 128,
        "shared_head_trunk_layers": 2,
        "dropout": 0.25,
    }))

    cfg = load_config_from_yaml(path)
    qcfg = shared_head_qnet_config(cfg)

    assert cfg.model_type == "shared_heads"
    assert isinstance(qcfg, SharedHeadQNetConfig)
    assert qcfg.role_d_model == 64
    assert qcfg.history_hidden == 96
    assert qcfg.global_hidden == 32
    assert qcfg.action_hidden == 48
    assert qcfg.trunk_hidden == 128
    assert qcfg.trunk_layers == 2
    assert qcfg.dropout == 0.25


def test_buffer_capacity_default_for_shared_mode():
    cfg = TrainConfig(model_type="shared_heads", buffer_capacity=0, buffer_capacity_per_player=123)
    capacity = cfg.buffer_capacity or 4 * cfg.buffer_capacity_per_player
    assert capacity == 492
