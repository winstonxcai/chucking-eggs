from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from guandan.dart.agent import DartBot
from guandan.dart.model.encoding.role_encoder import RoleAwareStateActionEncoder
from guandan.dart.model.q_network import DartQNet, DartQNetConfig


_EVAL_WORKER_PATH = Path(__file__).resolve().parents[3] / "scripts/eval/_eval_worker.py"
_SPEC = importlib.util.spec_from_file_location("_eval_worker_for_test", _EVAL_WORKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EVAL_WORKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL_WORKER)
play_n_games_dart_lanes = _EVAL_WORKER.play_n_games_dart_lanes


def _small_dart_bot() -> DartBot:
    torch.manual_seed(123)
    net = DartQNet(DartQNetConfig(
        role_d_model=16,
        history_hidden=16,
        global_hidden=8,
        action_hidden=8,
        trunk_hidden=32,
        trunk_layers=1,
    ))
    return DartBot(net, RoleAwareStateActionEncoder(), device="cpu")


def test_eval_lanes_match_single_lane_even_seating():
    bot = _small_dart_bot()
    seeds = [101, 102, 103, 104]

    single = play_n_games_dart_lanes(
        bot,
        "strategic",
        len(seeds),
        seeds=seeds,
        dart_seats=(0, 2),
        lanes=1,
    )
    batched = play_n_games_dart_lanes(
        bot,
        "strategic",
        len(seeds),
        seeds=seeds,
        dart_seats=(0, 2),
        lanes=3,
    )

    assert batched["n_games"] == single["n_games"] == len(seeds)
    assert batched["wins"] == single["wins"]
    assert batched["losses"] == single["losses"]


def test_eval_lanes_match_single_lane_odd_seating():
    bot = _small_dart_bot()
    seeds = [201, 202, 203, 204]

    single = play_n_games_dart_lanes(
        bot,
        "strategic",
        len(seeds),
        seeds=seeds,
        dart_seats=(1, 3),
        lanes=1,
    )
    batched = play_n_games_dart_lanes(
        bot,
        "strategic",
        len(seeds),
        seeds=seeds,
        dart_seats=(1, 3),
        lanes=2,
    )

    assert batched["n_games"] == single["n_games"] == len(seeds)
    assert batched["wins"] == single["wins"]
    assert batched["losses"] == single["losses"]
