"""Reward mapping tests.

Asserts finish_order → team reward per the project's LEVEL_CHANGE table,
and that normalized reward r/3 ∈ [-1, 1].

Finish order is the list of seats in the order they went out.
Reward for player: env.get_rewards()[player].
Normalized training reward: get_rewards()[player] / 3.
"""

from __future__ import annotations

import pytest

from guandan.cards import Rank
from guandan.game import GuanDanEnv


def _env_with_finish(finish_order: list[int]) -> GuanDanEnv:
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=1)
    env.finish_order = list(finish_order)
    env.is_out = [True, True, True, True]
    env.done = True
    return env


# ─── Finish pattern → expected reward (player 0 perspective) ─────────────────

@pytest.mark.parametrize("finish_order,team02_reward", [
    # Team {0,2} finishes 1st + 2nd → +3
    ([0, 2, 1, 3], 3.0),
    ([0, 2, 3, 1], 3.0),
    ([2, 0, 1, 3], 3.0),
    ([2, 0, 3, 1], 3.0),
    # Team {0,2} finishes 1st + 3rd → +2
    ([0, 1, 2, 3], 2.0),
    ([0, 3, 2, 1], 2.0),
    ([2, 1, 0, 3], 2.0),
    ([2, 3, 0, 1], 2.0),
    # Team {1,3} finishes 1st + 2nd → team {0,2} gets -3
    ([1, 3, 0, 2], -3.0),
    ([1, 3, 2, 0], -3.0),
    ([3, 1, 0, 2], -3.0),
    ([3, 1, 2, 0], -3.0),
    # Team {1,3} finishes 1st + 3rd → team {0,2} gets -2
    ([1, 0, 3, 2], -2.0),
    ([1, 2, 3, 0], -2.0),
    ([3, 0, 1, 2], -2.0),
    ([3, 2, 1, 0], -2.0),
    # 1-4: team member finishes 1st and 4th → +1 / -1
    ([1, 2, 0, 3], -1.0),  # team {0,2} has 0 at 3rd, 2 at 2nd
    ([0, 1, 3, 2], 1.0),   # team {0,2} has 0 at 1st, 2 at 4th
])
def test_reward_by_finish_order(finish_order, team02_reward):
    env = _env_with_finish(finish_order)
    rewards = env.get_rewards()
    # Team {0,2} should both get the same reward
    assert rewards[0] == team02_reward, \
        f"finish={finish_order}: player 0 reward {rewards[0]} != {team02_reward}"
    assert rewards[2] == team02_reward, \
        f"finish={finish_order}: player 2 reward {rewards[2]} != {team02_reward}"
    # Opponent team should get the negative
    assert rewards[1] == -team02_reward
    assert rewards[3] == -team02_reward


# ─── Normalized reward ∈ [-1, 1] ─────────────────────────────────────────────

def test_normalized_reward_range():
    """Training reward r/3 must be in [-1, +1]."""
    all_perms = [
        [0,2,1,3], [0,2,3,1], [2,0,1,3], [2,0,3,1],
        [0,1,2,3], [0,3,2,1],
        [1,3,0,2], [3,1,0,2],
        [1,0,3,2], [3,0,1,2],
        [0,1,3,2], [1,2,0,3],
    ]
    for fo in all_perms:
        env = _env_with_finish(fo)
        rewards = env.get_rewards()
        for p, r in rewards.items():
            norm = r / 3.0
            assert -1.0 <= norm <= 1.0, \
                f"finish={fo} player={p}: normalized reward {norm} out of [-1,1]"


# ─── Promotion differential == mean(get_rewards) ─────────────────────────────

def test_promotion_differential_sign():
    """Positive reward <=> team won more levels than they gave away."""
    env = _env_with_finish([0, 2, 1, 3])   # 1-2 win for team {0,2}
    rewards = env.get_rewards()
    assert rewards[0] > 0
    assert rewards[2] > 0
    assert rewards[1] < 0
    assert rewards[3] < 0


def test_zero_sum():
    """Sum of all rewards must be 0 (symmetric rules)."""
    for fo in [[0,2,1,3], [0,1,2,3], [1,3,0,2], [1,0,3,2], [0,1,3,2]]:
        env = _env_with_finish(fo)
        rewards = env.get_rewards()
        assert sum(rewards.values()) == 0.0, \
            f"finish={fo}: rewards sum to {sum(rewards.values())}"
