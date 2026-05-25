from __future__ import annotations

import dataclasses
import glob
from pathlib import Path

import pytest
import yaml
from guandan.dart.config import (
    MODEL_TYPE_DART,
    MODEL_TYPE_GUANZERO,
    EpisodeMixConfig,
    EpsilonConfig,
    OpponentConfig,
    OpponentSamplingConfig,
    QNetConfig,
    TrainConfig,
    dart_qnet_config,
    load_config_from_cli,
    load_config_from_yaml,
)
from guandan.dart.model.q_network import DartQNetConfig


def test_from_flat_dict_accepts_nested_dicts_and_dataclass_instances():
    nested = dataclasses.asdict(TrainConfig(
        qnet=QNetConfig(hidden_lstm=32),
        epsilon=EpsilonConfig(decay_updates=7),
        batch_size=11,
    ))

    cfg = TrainConfig.from_flat_dict(nested)
    assert cfg.qnet.hidden_lstm == 32
    assert cfg.epsilon.decay_updates == 7
    assert cfg.batch_size == 11

    cfg = TrainConfig.from_flat_dict({
        "qnet": QNetConfig(hidden_mlp=256),
        "epsilon": EpsilonConfig(final=0.02),
    })
    assert cfg.qnet.hidden_mlp == 256
    assert cfg.epsilon.final == 0.02


def test_yaml_and_cli_loading_apply_precedence(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "hidden_lstm": 16,
        "epsilon_start": 0.5,
        "checkpoint_save_type": "weight",
        "n_actors": 4,
    }))

    from_yaml = load_config_from_yaml(path)
    assert from_yaml.qnet.hidden_lstm == 16
    assert from_yaml.epsilon.start == 0.5
    assert from_yaml.checkpoint_save_type == "weight"
    assert from_yaml.n_actors == 4

    cfg = load_config_from_cli(
        path,
        n_actors=2,
        device=None,
    )
    assert cfg.qnet.hidden_lstm == 16
    assert cfg.epsilon.start == 0.5
    assert cfg.n_actors == 2


def test_cpu_smoke_config_is_tiny_and_portable():
    cfg = load_config_from_yaml(
        Path("ml/src/guandan/dart/configs/dart_cpu_smoke.yaml")
    )

    assert cfg.device == "cpu"
    assert cfg.n_actors == 1
    assert cfg.actor_batch_lanes == 1
    assert cfg.batch_size == 16
    assert cfg.buffer_min_size == 16
    assert cfg.checkpoint_every_updates == 5
    assert cfg.eval.enabled is False


def test_unknown_flat_config_key_fails_fast():
    with pytest.raises(ValueError, match="TrainConfig has unknown key"):
        TrainConfig.from_flat_dict({"batch_szie": 128})


def test_legacy_config_aliases_and_removed_keys_are_migrated():
    cfg = TrainConfig.from_flat_dict({
        "max_forced_k1_replay_frac": 0.25,
        "updates_per_learner_step": 99,
    })
    assert cfg.max_forced_pass_replay_frac == 0.25

    with pytest.raises(ValueError, match="Both 'max_forced_k1_replay_frac'"):
        TrainConfig.from_flat_dict({
            "max_forced_k1_replay_frac": 0.25,
            "max_forced_pass_replay_frac": 0.5,
        })


def test_config_schema_version_rejects_unknown_future_schema():
    with pytest.raises(ValueError, match="Unsupported config_schema_version"):
        TrainConfig(config_schema_version=999)


def test_unknown_nested_config_key_fails_fast():
    with pytest.raises(ValueError, match="qnet has unknown key"):
        TrainConfig.from_flat_dict({"qnet": {"hidden_lstm": 16, "hidden_lstn": 32}})


def test_model_type_default_and_invalid():
    assert TrainConfig().model_type == MODEL_TYPE_DART

    try:
        TrainConfig(model_type="bad")
    except ValueError as exc:
        assert "Unknown model_type" in str(exc)
    else:
        raise AssertionError("invalid model_type should raise")


def test_checkpoint_save_type_validation():
    assert TrainConfig(checkpoint_save_type="weight").checkpoint_save_type == "weight"
    assert TrainConfig(checkpoint_save_type="full").checkpoint_save_type == "full"

    with pytest.raises(ValueError, match="checkpoint_save_type"):
        TrainConfig(checkpoint_save_type="weights")


def test_eval_timeout_config_parses_and_validates():
    cfg = TrainConfig.from_flat_dict({"eval": {"enabled": True, "max_wait_s": 12.5}})
    assert cfg.eval.max_wait_s == 12.5

    with pytest.raises(ValueError, match="eval.max_wait_s"):
        TrainConfig.from_flat_dict({"eval": {"max_wait_s": 0}})


def test_model_type_dart_loads_and_factory(tmp_path):
    path = tmp_path / "dart.yaml"
    path.write_text(yaml.safe_dump({
        "model_type": MODEL_TYPE_DART,
        "dart_role_d_model": 64,
        "dart_history_hidden": 96,
        "dart_global_hidden": 32,
        "dart_action_hidden": 48,
        "dart_trunk_hidden": 128,
        "dart_trunk_layers": 2,
        "dropout": 0.25,
    }))

    cfg = load_config_from_yaml(path)
    qcfg = dart_qnet_config(cfg)

    assert cfg.model_type == MODEL_TYPE_DART
    assert isinstance(qcfg, DartQNetConfig)
    assert qcfg.role_d_model == 64
    assert qcfg.history_hidden == 96
    assert qcfg.global_hidden == 32
    assert qcfg.action_hidden == 48
    assert qcfg.trunk_hidden == 128
    assert qcfg.trunk_layers == 2
    assert qcfg.dropout == 0.25


def test_buffer_capacity_default_for_dart_mode():
    cfg = TrainConfig(model_type=MODEL_TYPE_DART, buffer_capacity=0, buffer_capacity_per_player=123)
    capacity = cfg.buffer_capacity or 4 * cfg.buffer_capacity_per_player
    assert capacity == 492


# ── nested opponent config validation ─────────────────────────────────


def test_nested_opponent_config_parses_and_normalizes_weights():
    cfg = TrainConfig.from_flat_dict({
        "opponents": {
            "latest_learner_team_odd_probability": 0.25,
            "episode_mix": {"self_play": 0.4, "frozen_pool": 0.2, "hard_bot": 0.3},
            "frozen_pool": {"checkpoints": ["a.pt"], "epsilon": 0.05},
            "hard_bot": {
                "bots": ["strategic", "yaoji", "jidan"],
                "sampling": {
                    "type": "pair_weighted",
                    "weights": {"yaoji_yaoji": 2.0, "jidan_yaoji": 2.0},
                },
            },
        },
    })

    assert cfg.opponents.latest_learner_team_odd_probability == 0.25
    assert cfg.opponents.episode_mix.self_play == 0.4
    assert cfg.opponents.frozen_pool.checkpoints == ("a.pt",)
    weights = cfg.opponents.hard_bot.sampling.weights
    assert sum(weights.values()) == 1.0
    assert weights["yaoji_yaoji"] == 0.5
    assert weights["jidan_yaoji"] == 0.5


def test_legacy_opponent_probability_alias_is_migrated():
    cfg = TrainConfig.from_flat_dict({
        "opponents": {
            "latest_team_odd_probability": 0.75,
        },
    })
    assert cfg.opponents.latest_learner_team_odd_probability == 0.75

    with pytest.raises(ValueError, match="Both opponents.'latest_team_odd_probability'"):
        TrainConfig.from_flat_dict({
            "opponents": {
                "latest_team_odd_probability": 0.25,
                "latest_learner_team_odd_probability": 0.75,
            },
        })


def test_removed_flat_opponent_keys_fail_fast():
    with pytest.raises(ValueError, match="Removed opponent config key"):
        TrainConfig.from_flat_dict({"hard_bot_pool": ["strategic"]})


def test_unknown_nested_opponent_key_fails_fast():
    with pytest.raises(ValueError, match="opponents has unknown key"):
        TrainConfig.from_flat_dict({"opponents": {"episode_mxi": {}}})


def test_episode_mix_rejects_total_over_one():
    with pytest.raises(ValueError, match="sum to <= 1"):
        TrainConfig(opponents=OpponentConfig(
            episode_mix=EpisodeMixConfig(self_play=0.7, frozen_pool=0.3, hard_bot=0.1)
        ))


def test_coordination_bucket_config_validates_ranges():
    with pytest.raises(ValueError, match="coordination_bucket_card_threshold"):
        TrainConfig(coordination_bucket_card_threshold=-1)
    with pytest.raises(ValueError, match="coordination_bucket_final_fraction"):
        TrainConfig(coordination_bucket_final_fraction=1.5)


def test_hard_bot_weighted_sampling_rejects_unknown_bot():
    with pytest.raises(ValueError, match="keys not in bots"):
        TrainConfig.from_flat_dict({
            "opponents": {
                "hard_bot": {
                    "bots": ["strategic", "yaoji"],
                    "sampling": {
                        "type": "weighted",
                        "weights": {"unknown": 1.0},
                    },
                },
            },
        })


def test_hard_bot_pair_sampling_rejects_unordered_pair_key():
    with pytest.raises(ValueError, match="alphabetized"):
        TrainConfig.from_flat_dict({
            "opponents": {
                "hard_bot": {
                    "bots": ["strategic", "yaoji", "jidan"],
                    "sampling": {
                        "type": "pair_weighted",
                        "weights": {"yaoji_jidan": 1.0},
                    },
                },
            },
        })


def test_hard_bot_sampling_rejects_non_positive_weights():
    with pytest.raises(ValueError, match="weights must be positive"):
        TrainConfig(opponents=OpponentConfig(
            hard_bot={
                "bots": ["strategic", "yaoji"],
                "sampling": OpponentSamplingConfig(
                    type="weighted",
                    weights={"strategic": -0.5},
                ),
            },
        ))


# ── YAML config roundtrip ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "yaml_path",
    sorted(glob.glob("ml/src/guandan/dart/configs/*.yaml")),
)
def test_all_config_yamls_parse(yaml_path):
    cfg = load_config_from_yaml(yaml_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.n_actors >= 1
    assert cfg.model_type in (MODEL_TYPE_GUANZERO, MODEL_TYPE_DART)
