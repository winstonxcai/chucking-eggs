"""Tests for encoding.py — state/action/history tensor dimensions and sanity."""

import random

import numpy as np
import torch

from guandan.cards import ComboType, Rank
from guandan.combos import Combo
from guandan.training.encoding import (
    ACTION_DIM,
    D_MOVE,
    MAX_HISTORY,
    STATE_DIM,
    cards_to_matrix,
    encode_action,
    encode_history,
    encode_move_event,
    encode_state,
)
from guandan.game import GuanDanEnv
from guandan.training.q_network import QNetworkLSTM


def test_state_dim():
    env = GuanDanEnv()
    env.reset()
    state = encode_state(env, 0)
    assert state.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {state.shape}"


def test_action_dim():
    env = GuanDanEnv()
    env.reset()
    legal = env.legal_moves()
    hand = env.hands[0]
    action = encode_action(legal[0], hand, env.level_rank)
    assert action.shape == (ACTION_DIM,), f"Expected ({ACTION_DIM},), got {action.shape}"


def test_action_dim_is_160():
    assert ACTION_DIM == 160


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
    action = encode_action(pass_combo, set(), Rank.TWO)
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
        hand = env.hands[player]
        for m in legal:
            action = encode_action(m, hand, env.level_rank)
            assert action.shape == (ACTION_DIM,)
            assert not np.any(np.isnan(action))

        env.step(random.choice(legal))
        steps += 1


def test_remaining_after_play():
    """After playing cards, remaining matrix should have fewer cards than hand.

    Note: wild card substitution can create Card objects not in the hand set,
    so we only test combos whose cards are all found in the hand.
    """
    random.seed(42)  # deterministic seed where combo cards are in hand
    env = GuanDanEnv()
    env.reset()
    legal = env.legal_moves()
    hand = env.hands[0]
    non_pass = [m for m in legal if m.type != ComboType.PASS and len(m.cards) > 1]
    # Find a combo whose cards are actually in the hand (no wild substitution)
    valid = [m for m in non_pass if all(c in hand for c in m.cards)]
    if valid:
        combo = valid[0]
        action = encode_action(combo, hand, env.level_rank)
        remaining_count_norm = action[-3]  # len(remaining)/27.0
        assert remaining_count_norm < 1.0


# ─── History encoding tests ──────────────────────────


def test_move_event_dim():
    """encode_move_event should produce D_MOVE-dim vector."""
    assert D_MOVE == 83
    pass_combo = Combo(ComboType.PASS, 0, [])
    event = encode_move_event(0, pass_combo, Rank.TWO)
    assert event.shape == (D_MOVE,)
    assert not np.any(np.isnan(event))


def test_move_event_pass_flag():
    """Pass move should have is_pass=1, card matrix all zeros."""
    pass_combo = Combo(ComboType.PASS, 0, [])
    event = encode_move_event(2, pass_combo, Rank.TWO)
    # actor_oh[2] should be 1
    assert event[2] == 1.0
    # card matrix (indices 4:64) should be all zeros
    assert event[4:64].sum() == 0.0
    # is_pass (index 64) should be 1
    assert event[64] == 1.0


def test_encode_history_empty():
    """Before any moves, history should be a single zero vector."""
    env = GuanDanEnv()
    env.reset()
    history, length = encode_history(env, 0, env.level_rank)
    assert history.shape == (1, D_MOVE)
    assert length == 1
    assert history.sum() == 0.0


def test_encode_history_grows():
    """History length should grow as moves are played."""
    env = GuanDanEnv()
    env.reset()

    for i in range(5):
        player = env.current_player
        legal = env.legal_moves()
        env.step(legal[0])
        if env.done:
            break

    history, length = encode_history(env, 0, env.level_rank)
    assert length == min(len(env.move_history), MAX_HISTORY)
    assert history.shape == (length, D_MOVE)
    assert not np.any(np.isnan(history))


def test_encode_history_caps_at_max():
    """History should cap at MAX_HISTORY moves."""
    env = GuanDanEnv()
    env.reset()
    steps = 0
    while not env.done and steps < 50:
        legal = env.legal_moves()
        env.step(random.choice(legal))
        steps += 1

    history, length = encode_history(env, 0, env.level_rank)
    assert length <= MAX_HISTORY
    assert history.shape[0] == length


def test_history_relative_actors():
    """Actor indices should be relative to the player."""
    env = GuanDanEnv()
    env.reset()

    # Play a few moves
    for _ in range(4):
        if env.done:
            break
        legal = env.legal_moves()
        env.step(legal[0])

    if len(env.move_history) > 0:
        for player in range(4):
            history, length = encode_history(env, player, env.level_rank)
            # Each event's actor_oh should sum to 1
            for i in range(length):
                assert history[i, :4].sum() == 1.0


def test_encode_history_full_game():
    """History encoding should work throughout a complete game."""
    env = GuanDanEnv()
    env.reset()
    while not env.done:
        player = env.current_player
        history, length = encode_history(env, player, env.level_rank)
        assert history.shape == (length, D_MOVE)
        assert not np.any(np.isnan(history))
        legal = env.legal_moves()
        env.step(random.choice(legal))


# ─── QNetworkLSTM tests ──────────────────────────────


def test_q_network_lstm_forward():
    """QNetworkLSTM forward pass with random tensors."""
    net = QNetworkLSTM(lstm_hidden=32, hidden=64, n_layers=2)
    B = 8
    T = 5
    s = torch.randn(B, STATE_DIM)
    a = torch.randn(B, ACTION_DIM)
    h = torch.randn(B, T, D_MOVE)
    hl = torch.randint(1, T + 1, (B,))
    out = net(s, a, h, hl)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_q_network_lstm_single_sample():
    """QNetworkLSTM should work with batch size 1."""
    net = QNetworkLSTM(lstm_hidden=32, hidden=64)
    s = torch.randn(1, STATE_DIM)
    a = torch.randn(1, ACTION_DIM)
    h = torch.randn(1, 3, D_MOVE)
    hl = torch.tensor([3])
    out = net(s, a, h, hl)
    assert out.shape == (1,)
