from __future__ import annotations

import dataclasses
import glob
import sys
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
from guandan.dart.runtime.train import _parse_args


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


def test_cli_checkpoint_interval_override(tmp_path, monkeypatch):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"checkpoint_every_updates": 25_000}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guandan.dart",
            "--config",
            str(path),
            "--checkpoint-every-updates",
            "1000",
        ],
    )

    cfg, resume = _parse_args()

    assert cfg.checkpoint_every_updates == 1000
    assert resume is None


def test_cli_actor_batch_lanes_override(tmp_path, monkeypatch):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"actor_batch_lanes": 32}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guandan.dart",
            "--config",
            str(path),
            "--actor-batch-lanes",
            "64",
        ],
    )

    cfg, resume = _parse_args()

    assert cfg.actor_batch_lanes == 64
    assert resume is None


def test_yaml_base_config_deep_merges_and_child_wins(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "n_actors": 4,
        "device": "cpu",
        "eval": {"enabled": False, "n_eval_games_per_opponent": 1000},
        "opponents": {"episode_mix": {"self_play": 0.5}},
    }))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({
        "base_config": "base.yaml",
        "n_actors": 2,
        "eval": {"n_eval_games_per_opponent": 2000},
        "opponents": {"episode_mix": {"hard_bot": 0.5}},
    }))

    cfg = load_config_from_yaml(child)

    assert cfg.n_actors == 2
    assert cfg.device == "cpu"
    assert cfg.eval.enabled is False
    assert cfg.eval.n_eval_games_per_opponent == 2000
    assert cfg.opponents.episode_mix.self_play == 0.5
    assert cfg.opponents.episode_mix.hard_bot == 0.5


def test_yaml_base_config_rejects_missing_base(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("base_config: missing.yaml\n")

    with pytest.raises(FileNotFoundError, match="config file does not exist"):
        load_config_from_yaml(child)


def test_yaml_base_config_rejects_cycles(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n")
    second.write_text("base_config: first.yaml\n")

    with pytest.raises(ValueError, match="base_config cycle detected"):
        load_config_from_yaml(first)


def test_l4_ablation_configs_share_all_non_experimental_settings():
    paths = [
        Path("ml/src/guandan/dart/configs/ablation_l4_dart_partner_visible_10h.yaml"),
        Path("ml/src/guandan/dart/configs/ablation_l4_dart_partner_hidden_10h.yaml"),
        Path("ml/src/guandan/dart/configs/ablation_l4_guanzero_partner_visible_10h.yaml"),
        Path("ml/src/guandan/dart/configs/ablation_l4_guanzero_partner_hidden_10h.yaml"),
    ]
    configs = [load_config_from_yaml(path) for path in paths]

    assert {cfg.model_type for cfg in configs} == {MODEL_TYPE_DART, MODEL_TYPE_GUANZERO}
    assert {cfg.qnet.is_partner_visible for cfg in configs} == {True, False}
    assert all(cfg.learner_samples_per_update == 4096 for cfg in configs)
    assert all(cfg.eval.enabled for cfg in configs)
    assert all(cfg.eval.n_eval_games_per_opponent == 1000 for cfg in configs)

    def shared_config(cfg):
        raw = dataclasses.asdict(cfg)
        raw.pop("model_type")
        raw["qnet"].pop("is_partner_visible")
        return raw

    assert all(shared_config(cfg) == shared_config(configs[0]) for cfg in configs[1:])

    serialized = dataclasses.asdict(configs[0])
    assert "base_config" not in serialized


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


def test_a800_config_preserves_learning_recipe_and_scales_runtime():
    cfg = load_config_from_yaml(
        Path("ml/src/guandan/dart/configs/dart_a800.yaml")
    )

    assert cfg.device == "cuda"
    assert cfg.n_actors == 112
    assert cfg.actor_batch_lanes == 32
    assert cfg.batch_size == 4096
    assert cfg.lr == pytest.approx(3.0e-5)
    assert cfg.buffer_capacity == 400_000
    assert cfg.target_replay_ratio == 1.0
    assert cfg.use_bf16_learner is True
    assert cfg.compile_mode == "reduce-overhead"
    assert cfg.checkpoint_save_type == "weight"
    assert cfg.eval.enabled is True
    assert cfg.eval.workers == 32
    assert cfg.eval.lanes == 64
    assert cfg.eval.n_eval_games_per_opponent == 1000


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


def test_batch_semantics_normalize_learner_samples():
    dart = TrainConfig(model_type=MODEL_TYPE_DART, batch_size=4096)
    assert dart.batch_size_per_seat == 4096
    assert dart.learner_samples_per_update == 4096

    guanzero_total = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        batch_size=4096,
        batch_size_semantics="total",
    )
    assert guanzero_total.batch_size_per_seat == 1024
    assert guanzero_total.learner_samples_per_update == 4096

    guanzero_legacy = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        batch_size=4096,
        batch_size_semantics="per_seat",
    )
    assert guanzero_legacy.batch_size_per_seat == 4096
    assert guanzero_legacy.learner_samples_per_update == 16384


def test_guanzero_total_batch_must_be_divisible_by_four():
    with pytest.raises(ValueError, match="divisible by the number of seats"):
        TrainConfig(
            model_type=MODEL_TYPE_GUANZERO,
            batch_size=4095,
            batch_size_semantics="total",
        )


def test_batch_semantics_roundtrip_through_nested_config():
    cfg = TrainConfig(
        model_type=MODEL_TYPE_GUANZERO,
        batch_size=4096,
        batch_size_semantics="total",
    )

    roundtripped = TrainConfig.from_flat_dict(dataclasses.asdict(cfg))
    assert roundtripped.batch_size_semantics == "total"
    assert roundtripped.batch_size_per_seat == 1024
    assert roundtripped.learner_samples_per_update == 4096


def test_checkpoint_save_type_validation():
    assert TrainConfig(checkpoint_save_type="weight").checkpoint_save_type == "weight"
    assert TrainConfig(checkpoint_save_type="full").checkpoint_save_type == "full"

    with pytest.raises(ValueError, match="checkpoint_save_type"):
        TrainConfig(checkpoint_save_type="weights")


def test_actor_queue_full_config_validation():
    assert TrainConfig(actor_queue_full_policy="block").actor_queue_full_policy == "block"

    with pytest.raises(ValueError, match="actor_queue_full_policy"):
        TrainConfig(actor_queue_full_policy="drop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="actor_queue_put_timeout_s"):
        TrainConfig(actor_queue_put_timeout_s=0)
    with pytest.raises(ValueError, match="actor_queue_full_log_every"):
        TrainConfig(actor_queue_full_log_every=0)


def test_eval_timeout_config_parses_and_validates():
    cfg = TrainConfig.from_flat_dict({"eval": {"enabled": True, "max_wait_s": 12.5}})
    assert cfg.eval.max_wait_s == 12.5

    with pytest.raises(ValueError, match="eval.max_wait_s"):
        TrainConfig.from_flat_dict({"eval": {"max_wait_s": 0}})


def test_max_train_seconds_parses_and_validates():
    cfg = TrainConfig.from_flat_dict({"max_train_seconds": 36000})
    assert cfg.max_train_seconds == 36000

    with pytest.raises(ValueError, match="max_train_seconds must be >= 0"):
        TrainConfig(max_train_seconds=-1)


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
