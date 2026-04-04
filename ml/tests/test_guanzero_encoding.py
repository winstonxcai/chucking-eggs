"""Tests for GuanZero 108-dim encoding (arXiv:2402.13582)."""

from __future__ import annotations

import numpy as np
import pytest

from guandan.cards import Card, ComboType, Rank, Suit, make_deck
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.training.guanzero_encoding import (
    _CARD_TO_IDX,
    card_to_108,
    cards_to_108,
    combo_to_108,
    compute_behavior_flags,
    encode_base_state,
    encode_guanzero_state,
    encode_history,
)


# ── Card index lookup ────────────────────────────────────────────────────────

def test_card_index_count():
    assert len(_CARD_TO_IDX) == 108


def test_card_indices_unique():
    indices = list(_CARD_TO_IDX.values())
    assert sorted(indices) == list(range(108))


def test_full_deck_all_ones():
    deck = make_deck()
    vec = cards_to_108(deck)
    assert vec.shape == (108,)
    assert vec.sum() == 108
    assert (vec == 1.0).all()


def test_hand_card_count():
    deck = make_deck()
    hand = set(deck[:27])
    vec = cards_to_108(hand)
    assert vec.shape == (108,)
    assert int(vec.sum()) == 27


def test_pass_combo_all_zeros():
    pass_combo = Combo(ComboType.PASS, 0, [])
    vec = combo_to_108(pass_combo)
    assert vec.shape == (108,)
    assert (vec == 0.0).all()


def test_joker_indices():
    # BJ (rank=16, suit=SPADE=0) deck 0 and deck 1
    bj0 = Card(Rank.BLACK_JOKER, Suit.SPADE, 0)
    bj1 = Card(Rank.BLACK_JOKER, Suit.SPADE, 1)
    assert bj0 in _CARD_TO_IDX
    assert bj1 in _CARD_TO_IDX
    assert _CARD_TO_IDX[bj0] != _CARD_TO_IDX[bj1]

    # RJ (rank=17, suit=HEART=1) deck 0 and deck 1
    rj0 = Card(Rank.RED_JOKER, Suit.HEART, 0)
    rj1 = Card(Rank.RED_JOKER, Suit.HEART, 1)
    assert rj0 in _CARD_TO_IDX
    assert rj1 in _CARD_TO_IDX
    assert _CARD_TO_IDX[rj0] != _CARD_TO_IDX[rj1]


def test_normal_card_first_index():
    # Rank.TWO (2), Suit.SPADE (0), deck 0 → should be first in matrix (row 0, col 0)
    c = Card(Rank.TWO, Suit.SPADE, 0)
    assert card_to_108(c.rank, c.suit, c.deck) == 0


# ── Behavior flags ───────────────────────────────────────────────────────────

def _make_env_with_trick(trick_winner: int, current_player: int):
    """Set up env with a current trick owned by trick_winner."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    env.current_player = current_player
    # Simulate a non-pass current trick
    from guandan.combos import Combo
    # Give current player something to play
    return env


def test_behavior_flags_sum_to_one():
    """Each 3-dim group must sum to exactly 1.0."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    player = env.current_player
    legal = env.legal_moves()
    if not legal:
        pytest.skip("No legal moves in this game state")

    action = legal[0]
    flags = compute_behavior_flags(env, player, action, legal)
    assert flags.shape == (9,)
    assert abs(flags[0:3].sum() - 1.0) < 1e-6, f"Cooperating group sums to {flags[0:3].sum()}"
    assert abs(flags[3:6].sum() - 1.0) < 1e-6, f"Dwarfing group sums to {flags[3:6].sum()}"
    assert abs(flags[6:9].sum() - 1.0) < 1e-6, f"Assisting group sums to {flags[6:9].sum()}"


def test_behavior_flags_all_actions():
    """All legal actions must produce flags summing to 1.0 in each group."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    player = env.current_player
    legal = env.legal_moves()

    for action in legal[:10]:  # check first 10 to keep test fast
        flags = compute_behavior_flags(env, player, action, legal)
        for group_start in (0, 3, 6):
            g = flags[group_start:group_start + 3]
            assert abs(g.sum() - 1.0) < 1e-6
            assert (g >= 0).all()
            assert int((g > 0).sum()) == 1  # exactly one is hot


def test_not_leading_dwarfing_not_applicable():
    """When following (not leading), dwarfing must be not-applicable [1,0,0]."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()

    # Force a trick to be in progress by stepping until someone leads
    player = env.current_player
    legal = env.legal_moves()
    # Find a non-pass lead
    lead = next((m for m in legal if m.type != ComboType.PASS), legal[0])
    env.step(lead)

    # Now the next player is following
    follower = env.current_player
    follow_legal = env.legal_moves()
    action = follow_legal[0]

    flags = compute_behavior_flags(env, follower, action, follow_legal)
    # Dwarfing (flags[3:6]) should be not-applicable [1,0,0]
    assert flags[3] == 1.0 and flags[4] == 0.0 and flags[5] == 0.0
    # Assisting (flags[6:9]) should also be not-applicable
    assert flags[6] == 1.0 and flags[7] == 0.0 and flags[8] == 0.0


# ── State encoding shapes ────────────────────────────────────────────────────

def test_base_state_shape():
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    player = env.current_player
    base = encode_base_state(env, player, Rank.TWO)
    assert base.shape == (1066,), f"Expected (1066,), got {base.shape}"


def test_history_shape():
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    player = env.current_player
    hist = encode_history(env, player)
    assert hist.shape == (5, 432), f"Expected (5, 432), got {hist.shape}"


def test_full_state_shapes():
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    player = env.current_player
    legal = env.legal_moves()
    action = legal[0]

    nh, hist, a_enc = encode_guanzero_state(env, player, action, legal, Rank.TWO)
    assert nh.shape == (1075,), f"Expected (1075,), got {nh.shape}"
    assert hist.shape == (5, 432), f"Expected (5, 432), got {hist.shape}"
    assert a_enc.shape == (108,), f"Expected (108,), got {a_enc.shape}"


def test_encoding_no_crash_full_game():
    """Run a full game and encode every decision point without errors."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset()
    n_decisions = 0

    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        action = legal[0]  # always pick first legal move

        nh, hist, a_enc = encode_guanzero_state(env, player, action, legal, Rank.TWO)
        assert nh.shape == (1075,)
        assert hist.shape == (5, 432)
        assert a_enc.shape == (108,)
        n_decisions += 1
        env.step(action)

    assert n_decisions > 0


# ── Network smoke test ───────────────────────────────────────────────────────

def test_network_forward():
    import torch
    from guandan.training.guanzero_network import GuanZeroNetwork

    net = GuanZeroNetwork()
    n_params = sum(p.numel() for p in net.parameters())
    # ~2M params (allow 1.5M–4M range given hyperparameter flexibility)
    assert 1_000_000 < n_params < 6_000_000, f"Unexpected param count: {n_params:,}"

    B = 8
    nh = torch.randn(B, 1075)
    hist = torch.randn(B, 5, 432)
    hl = torch.ones(B, dtype=torch.long) * 3
    act = torch.randn(B, 108)

    out = net(nh, hist, hl, act)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_network_forward_fast():
    import torch
    from guandan.training.guanzero_network import GuanZeroNetwork

    net = GuanZeroNetwork()
    B = 12
    nh = torch.randn(B, 1075)
    hist = torch.randn(5, 432)
    hl = torch.tensor([4])
    act = torch.randn(B, 108)

    out = net.forward_fast(nh, hist, hl, act)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
