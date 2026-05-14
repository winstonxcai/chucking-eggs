"""MC-return computation: per-player, team-signed, gamma=1 default."""

from __future__ import annotations

import numpy as np

from guandan.guanzero.returns import (
    TERMINAL_REWARD_SCALE,
    compute_mc_returns,
    normalize_terminal_rewards,
)


def _step(player: int) -> dict:
    # Encoded payload doesn't matter for return computation.
    return {"player": player, "encoded": {"k": np.zeros(1, dtype=np.float32)}}


def test_normalization_scales_to_unit_range():
    rewards = {0: 3.0, 1: -3.0, 2: 3.0, 3: -3.0}
    norm = normalize_terminal_rewards(rewards)
    assert norm == {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0}
    assert TERMINAL_REWARD_SCALE == 3.0


def test_gamma_one_propagates_terminal_to_every_step():
    traj = [_step(0), _step(1), _step(0), _step(2), _step(1), _step(3)]
    rewards = {0: 3.0, 1: -3.0, 2: 3.0, 3: -3.0}  # team {0,2} wins big
    samples = compute_mc_returns(traj, rewards, gamma=1.0)

    # Every step on a winner's trajectory has G_t = +1; loser = -1.
    assert [s.mc_return for s in samples] == [1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    assert [s.player for s in samples] == [0, 1, 0, 2, 1, 3]


def test_gamma_decays_intermediate_steps():
    traj = [_step(0), _step(0), _step(0)]  # three steps for player 0
    rewards = {0: 3.0, 1: -3.0, 2: 3.0, 3: -3.0}
    samples = compute_mc_returns(traj, rewards, gamma=0.5)

    # Walk backward: last step has G = 1.0; prior has 0.5; prior-prior 0.25.
    assert [round(s.mc_return, 4) for s in samples] == [0.25, 0.5, 1.0]


def test_player_with_no_steps_is_skipped():
    traj = [_step(0), _step(2), _step(0), _step(2)]  # only seats 0 and 2
    rewards = {0: 3.0, 1: -3.0, 2: 3.0, 3: -3.0}
    samples = compute_mc_returns(traj, rewards, gamma=1.0)
    assert all(s.player in (0, 2) for s in samples)
    assert all(s.mc_return == 1.0 for s in samples)
