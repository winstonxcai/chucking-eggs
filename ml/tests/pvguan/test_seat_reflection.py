"""Seat-reflection tests for pvguan encoders.

Tests both coordinate bases used by encode_state_features:
  - Team-fixed basis (Groups 1–3): teammate_lo/hi, opp_l/opp_r are canonical seats
  - Self-relative basis (Groups 4–6): {self, partner, opp_l, opp_r} from actor POV

Also tests:
  - acting_teammate_flag correctness
  - finish_position_one_hot ordering
  - pass_sequence_one_hot per-seat convention
  - action round-trip via canonical signature after reflection
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from guandan.cards import Card, ComboType, Rank, Suit, is_wild
from guandan.combos import Combo, generate_all_leads, generate_responses
from guandan.game import GuanDanEnv
from guandan.pvguan.encoders import (
    ACTING_FLAG,
    FINISH_POSITION,
    HAND_COUNTS,
    INITIAL_HAND_SIZE,
    OPP_L_PLAYED,
    OPP_R_PLAYED,
    PASS_SEQUENCE,
    TEAMMATE_HI_HAND,
    TEAMMATE_HI_PLAYED,
    TEAMMATE_LO_HAND,
    TEAMMATE_LO_PLAYED,
    TRICK_OWNER_REL,
    encode_state_features,
)


def _fresh_env(seed: int = 42) -> GuanDanEnv:
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=seed)
    return env


def card_id(c: Card) -> tuple:
    return (c.rank, c.suit, c.deck)


def card_multiset(cards) -> tuple:
    return tuple(sorted(Counter(card_id(c) for c in cards).items()))


def action_signature(combo: Combo, level_rank: int) -> tuple:
    from guandan.pvguan.encoders import _BOMB_TIER_INDEX, _seq_length, _kicker_rank
    return (
        card_multiset(combo.cards),
        int(combo.type),
        combo.key,
        tuple(sorted(int(c.rank) for c in combo.cards if is_wild(c, level_rank))),
        _BOMB_TIER_INDEX.get(combo.type, -1),
        _kicker_rank(combo, level_rank),
        _seq_length(combo),
    )


# ─── Team-fixed basis (Groups 1–3) ───────────────────────────────────────────

def test_teammate_lo_hand_is_seat0():
    """When player=0 acts, teammate_lo_hand should equal seat 0's hand."""
    env = _fresh_env()
    env.current_player = 0
    s = encode_state_features(env, 0)

    from guandan.training.encoding import cards_to_matrix
    expected = cards_to_matrix(env.hands[0]).flatten()
    assert np.array_equal(s[TEAMMATE_LO_HAND], expected)


def test_teammate_hi_hand_is_seat2():
    env = _fresh_env()
    s = encode_state_features(env, 0)

    from guandan.training.encoding import cards_to_matrix
    expected = cards_to_matrix(env.hands[2]).flatten()
    assert np.array_equal(s[TEAMMATE_HI_HAND], expected)


def test_opp_l_played_is_seat1():
    """opp_l_played should always be seat 1's played set, regardless of acting player."""
    env = _fresh_env()
    for player in (0, 2):
        s = encode_state_features(env, player)
        from guandan.training.encoding import cards_to_matrix
        expected = cards_to_matrix(env.played[1]).flatten()
        assert np.array_equal(s[OPP_L_PLAYED], expected), f"player={player}"


def test_opp_r_played_is_seat3():
    env = _fresh_env()
    for player in (0, 2):
        s = encode_state_features(env, player)
        from guandan.training.encoding import cards_to_matrix
        expected = cards_to_matrix(env.played[3]).flatten()
        assert np.array_equal(s[OPP_R_PLAYED], expected), f"player={player}"


# ─── Acting teammate flag ─────────────────────────────────────────────────────

def test_acting_flag_lo_acts():
    env = _fresh_env()
    s = encode_state_features(env, 0)   # player 0 = lo
    flag = s[ACTING_FLAG]
    assert np.array_equal(flag, [1.0, 0.0]), f"Got {flag}"


def test_acting_flag_hi_acts():
    env = _fresh_env()
    s = encode_state_features(env, 2)   # player 2 = hi
    flag = s[ACTING_FLAG]
    assert np.array_equal(flag, [0.0, 1.0]), f"Got {flag}"


# ─── Hand counts normalized ───────────────────────────────────────────────────

def test_hand_counts_order():
    """Hand counts in order [lo, hi, oL, oR] = [seat0, seat2, seat1, seat3]."""
    env = _fresh_env()
    s = encode_state_features(env, 0)
    counts = s[HAND_COUNTS]
    expected = np.array([
        len(env.hands[0]) / INITIAL_HAND_SIZE,
        len(env.hands[2]) / INITIAL_HAND_SIZE,
        len(env.hands[1]) / INITIAL_HAND_SIZE,
        len(env.hands[3]) / INITIAL_HAND_SIZE,
    ], dtype=np.float32)
    assert np.allclose(counts, expected)


# ─── Finish position one-hot ──────────────────────────────────────────────────

def test_finish_position_all_still_in():
    env = _fresh_env()
    s = encode_state_features(env, 0)
    finish_oh = s[FINISH_POSITION].reshape(4, 5)
    # All players still in → bucket 4 (still_in) set for all
    for slot in range(4):
        assert finish_oh[slot, 4] == 1.0, f"Slot {slot}: {finish_oh[slot]}"
        assert finish_oh[slot, :4].sum() == 0.0


def test_finish_position_after_out():
    """After player 0 goes out, lo-slot should show 1st-out."""
    env = _fresh_env()
    # Manually mark player 0 as first out
    env.finish_order = [0]
    env.is_out[0] = True
    env.hands[0] = set()

    s = encode_state_features(env, 2)  # lo=0 is out; player 2 acts
    finish_oh = s[FINISH_POSITION].reshape(4, 5)
    # Slot 0 = lo (seat 0): should be 1st-out (pos=0)
    assert finish_oh[0, 0] == 1.0, f"lo finish: {finish_oh[0]}"
    # Slot 1 = hi (seat 2): still_in
    assert finish_oh[1, 4] == 1.0


# ─── Self-relative basis: trick owner ─────────────────────────────────────────

def test_trick_owner_self_relative():
    """Trick winner expressed relative to acting player."""
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=5)
    env.current_player = 0

    legal = generate_all_leads(env.hands[0], env.level_rank)
    lead = next(c for c in legal if c.type != ComboType.PASS)
    env.step(lead)

    assert env.trick_winner == 0
    # From player 0: trick_winner=0 → self → index 0
    s0 = encode_state_features(env, 0)
    owner = s0[TRICK_OWNER_REL]
    assert owner.argmax() == 0, f"Expected self(0), got {owner.argmax()}"

    # From player 2: trick_winner=0 → partner → index 1
    s2 = encode_state_features(env, 2)
    owner2 = s2[TRICK_OWNER_REL]
    assert owner2.argmax() == 1, f"Expected partner(1), got {owner2.argmax()}"


# ─── Action round-trip via canonical signature ───────────────────────────────

def test_action_roundtrip_after_reflection():
    """An action on reflected env maps to a signature in original env's legal set."""
    from guandan.agents.partner_oracle_bot import _reflect_env

    env = _fresh_env(seed=21)
    # Reflect so that team {1,3} becomes team {0,2}
    reflected = _reflect_env(env)

    # Legal moves from original player 1 (now canonical seat 0)
    original_legal = generate_all_leads(env.hands[1], env.level_rank)
    reflected_legal = generate_all_leads(reflected.hands[0], reflected.level_rank)

    orig_sigs = {action_signature(a, env.level_rank) for a in original_legal}
    refl_sigs = {action_signature(a, reflected.level_rank) for a in reflected_legal}

    # All reflected signatures must exist in original (cards are same, just env swapped)
    assert refl_sigs == orig_sigs, \
        f"Reflected sigs not matching: extra={refl_sigs - orig_sigs}"


# ─── Pass sequence per-seat convention ───────────────────────────────────────

def test_pass_sequence_order():
    """Pass sequence in order [lo, hi, oL, oR] = [seat0, seat2, seat1, seat3].

    After P0 leads, the CCW order is P3 → P2 → P1. We have P3 pass, then check.
    """
    for seed in range(200):
        env = GuanDanEnv(level_rank=Rank.TWO)
        env.reset(seed=seed)
        env.current_player = 0

        legal = generate_all_leads(env.hands[0], env.level_rank)
        lead = next((c for c in legal if c.type != ComboType.PASS), None)
        if lead is None:
            continue
        env.step(lead)  # P0 leads → next CCW = P3

        if env.current_trick is None or env.done:
            continue

        # The next player should be P3 (or someone in CCW order)
        # We want exactly one player to pass, then check the counts.
        p = env.current_player
        r = generate_responses(env.hands[p], env.level_rank, env.current_trick)
        pass_m = next((c for c in r if c.type == ComboType.PASS), None)
        if pass_m is None:
            continue
        env.step(pass_m)  # seat p passes

        if env.current_trick is None or env.done:
            continue

        s = encode_state_features(env, 0)
        pass_seq = s[PASS_SEQUENCE].reshape(4, 4)

        # seat 0 (lo, slot 0): 0 passes → bucket 0
        assert pass_seq[0, 0] == 1.0, f"lo: {pass_seq[0]}, seed={seed}"

        # seat p passed once → check the right slot
        # Slot order: [lo=seat0→slot0, hi=seat2→slot1, oL=seat1→slot2, oR=seat3→slot3]
        seat_to_slot = {0: 0, 2: 1, 1: 2, 3: 3}
        passing_slot = seat_to_slot[p]
        assert pass_seq[passing_slot, 1] == 1.0, \
            f"seat {p} (slot {passing_slot}): expected bucket 1, got {pass_seq[passing_slot]}, seed={seed}"

        # All other slots: bucket 0
        for seat, slot in seat_to_slot.items():
            if seat != 0 and seat != p:
                assert pass_seq[slot, 0] == 1.0, \
                    f"seat {seat} (slot {slot}): expected bucket 0, got {pass_seq[slot]}"
        return

    pytest.skip("No valid scenario found in 200 seeds")


# ─── Invariance of state groups 1–3 across teammates ─────────────────────────

def test_card_zones_invariant_to_acting_teammate():
    """Groups 1–3 (card zones + per-seat status) same for both teammates.

    Only acting_flag differs.
    """
    env = _fresh_env()
    s0 = encode_state_features(env, 0)
    s2 = encode_state_features(env, 2)

    # Groups 1+2+3 up to acting flag (bytes 0..459)
    assert np.array_equal(s0[:460], s2[:460]), "Card zones/status differ"
    # level_rank and wild flags (462:478) same
    assert np.array_equal(s0[462:478], s2[462:478])
    # Acting flag differs
    assert not np.array_equal(s0[ACTING_FLAG], s2[ACTING_FLAG])
