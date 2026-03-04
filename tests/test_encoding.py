"""Tests for encoding.py — state/action tensor dimensions and sanity."""

import random

import numpy as np

from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.encoding import (
    ACTION_DIM,
    STATE_DIM,
    cards_to_matrix,
    encode_action,
    encode_state,
)
from guandan.game import GuanDanEnv


def test_state_dim():
    env = GuanDanEnv()
    env.reset()
    state = encode_state(env, 0)
    assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {state.shape}"


def test_action_dim():
    env = GuanDanEnv()
    env.reset()
    legal = env.legal_moves()
    action = encode_action(legal[0])
    assert action.shape == (ACTION_DIM,), f"Expected ({ACTION_DIM},), got {action.shape}"


def test_state_values_bounded():
    """State values should be reasonable (not NaN, not huge)."""
    env = GuanDanEnv()
    env.reset()
    state = encode_state(env, 0)
    assert not np.any(np.isnan(state))
    assert np.all(np.abs(state) <= 10.0)


def test_action_pass():
    """PASS action should encode cleanly."""
    pass_combo = Combo(ComboType.PASS, 0, [])
    action = encode_action(pass_combo)
    assert action.shape == (ACTION_DIM,)
    assert not np.any(np.isnan(action))


def test_cards_to_matrix_shape():
    env = GuanDanEnv()
    env.reset()
    mat = cards_to_matrix(env.hands[0])
    assert mat.shape == (15, 4)
    assert mat.sum() == 27  # 27 cards in hand, each counted once


def test_state_changes_after_play():
    """State should change after a move is made."""
    env = GuanDanEnv()
    env.reset()
    state_before = encode_state(env, 0)
    legal = env.legal_moves()
    non_pass = [m for m in legal if m.type != ComboType.PASS]
    if non_pass:
        env.step(non_pass[0])
        state_after = encode_state(env, 0)
        assert not np.array_equal(state_before, state_after)


def test_encode_during_game():
    """Encoding should work at every step of a game."""
    env = GuanDanEnv()
    env.reset()
    steps = 0
    while not env.done and steps < 200:
        player = env.current_player
        state = encode_state(env, player)
        assert state.shape == (STATE_DIM,)
        assert not np.any(np.isnan(state))

        legal = env.legal_moves()
        for m in legal:
            action = encode_action(m)
            assert action.shape == (ACTION_DIM,)
            assert not np.any(np.isnan(action))

        env.step(random.choice(legal))
        steps += 1
