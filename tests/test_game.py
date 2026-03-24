"""Tests for game.py — GuanDanEnv full game loop."""

import random

from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv


def test_random_games_complete(n: int = 100):
    """Random games should complete without crashes."""
    env = GuanDanEnv()
    crashes = 0
    for _ in range(n):
        env.reset()
        steps = 0
        while not env.done and steps < 1000:
            legal = env.legal_moves()
            assert len(legal) > 0, f"No legal moves for player {env.current_player}"
            move = random.choice(legal)
            env.step(move)
            steps += 1

        if not env.done:
            crashes += 1
            continue

        assert len(env.finish_order) == 4, (
            f"Expected 4 finishers, got {len(env.finish_order)}"
        )
        assert set(env.finish_order) == {0, 1, 2, 3}

    assert crashes == 0, f"{crashes}/{n} games failed to complete"


def test_game_ends_when_three_out():
    """The game should end once 3 players have gone out."""
    env = GuanDanEnv()
    for _ in range(100):
        env.reset()
        while not env.done:
            legal = env.legal_moves()
            env.step(random.choice(legal))

        assert sum(env.is_out) >= 3
        assert len(env.finish_order) == 4


def test_team_balance(n: int = 500):
    """Random play should give roughly 50% winrate to each team."""
    team_02_wins = 0
    env = GuanDanEnv()
    for _ in range(n):
        env.reset()
        while not env.done:
            env.step(random.choice(env.legal_moves()))
        # Winning team = team of the first finisher
        first = env.finish_order[0]
        if first in (0, 2):
            team_02_wins += 1

    winrate = team_02_wins / n
    # Should be within a reasonable range of 50%
    assert 0.38 <= winrate <= 0.62, f"Team {{0,2}} winrate: {winrate:.1%}"


def test_counterclockwise():
    """Play direction should be counterclockwise (0 → 3 → 2 → 1 → 0)."""
    env = GuanDanEnv()
    env.reset()
    env.current_player = 0
    env.current_trick = None  # free lead

    legal = env.legal_moves()
    # Play a non-pass move
    non_pass = [m for m in legal if m.type != ComboType.PASS]
    assert non_pass, "Should have at least one non-pass lead"
    env.step(non_pass[0])

    if not env.done:
        expected_next = 3  # counterclockwise from 0
        assert env.current_player == expected_next or env.is_out[expected_next], (
            f"Expected player {expected_next} next (CCW from 0), got {env.current_player}"
        )


def test_rewards_zero_sum():
    """Rewards should be zero-sum: teammates get same, opponents get negated."""
    env = GuanDanEnv()
    for _ in range(100):
        env.reset()
        while not env.done:
            env.step(random.choice(env.legal_moves()))

        rewards = env.get_rewards()
        total = sum(rewards.values())
        assert total == 0.0, f"Rewards not zero-sum: {rewards} (total={total})"


def test_finish_order_has_all_players():
    """Finish order should contain all 4 players exactly once."""
    env = GuanDanEnv()
    for _ in range(100):
        env.reset()
        while not env.done:
            env.step(random.choice(env.legal_moves()))

        assert len(env.finish_order) == 4
        assert set(env.finish_order) == {0, 1, 2, 3}


def test_no_cards_left_when_out():
    """Players who are out should have empty hands."""
    env = GuanDanEnv()
    for _ in range(100):
        env.reset()
        while not env.done:
            env.step(random.choice(env.legal_moves()))

        # The first two out should have empty hands
        # (remaining players may still have cards when game ends)
        for p in env.finish_order[:2]:
            assert len(env.hands[p]) == 0, (
                f"Player {p} (finished) still has {len(env.hands[p])} cards"
            )


def test_legal_moves_always_available():
    """Current player should always have legal moves (at minimum PASS when following)."""
    env = GuanDanEnv()
    for _ in range(50):
        env.reset()
        steps = 0
        while not env.done and steps < 1000:
            legal = env.legal_moves()
            assert len(legal) > 0, (
                f"Player {env.current_player} has no legal moves "
                f"(hand size={len(env.hands[env.current_player])}, "
                f"leading={env.is_leading()})"
            )
            env.step(random.choice(legal))
            steps += 1
