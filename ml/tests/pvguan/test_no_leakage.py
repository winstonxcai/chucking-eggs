"""Information-boundary tests for pvguan encoders.

Constructs two env states identical except for opponent-hand allocation.
Asserts:
  - Actor pair-features invariant to opponent repartition (matched by signature)
  - Action encodings invariant
  - PV-AC critic privileged slots [755:875] all zero and state invariant
  - PV-PTIE critic privileged slots equal opponent encoding
  - Critic state is action-independent (structural + runtime guard)
"""

from __future__ import annotations

import copy
import inspect
from collections import Counter

import numpy as np
import pytest

from guandan.cards import Card, ComboType, Rank, Suit, is_wild
from guandan.combos import Combo, generate_all_leads
from guandan.game import GuanDanEnv
from guandan.pvguan.encoders import (
    STATE_FEATURES_DIM,
    cards_to_matrix,
    encode_action,
    encode_actor_pair_features,
    encode_critic_state,
    encode_state_features,
)


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


def _clone_env(env: GuanDanEnv) -> GuanDanEnv:
    new = GuanDanEnv(level_rank=env.level_rank)
    new.hands        = [set(h) for h in env.hands]
    new.played       = [set(p) for p in env.played]
    new.is_out       = list(env.is_out)
    new.finish_order = list(env.finish_order)
    new.current_player = env.current_player
    new.current_trick  = env.current_trick
    new.trick_winner   = env.trick_winner
    new.consecutive_passes = env.consecutive_passes
    new.done           = env.done
    new.move_history   = list(env.move_history)
    return new


def _build_repartitioned_pair(seed: int = 42) -> tuple[GuanDanEnv, GuanDanEnv, int]:
    """Build two envs identical except opponent-hand allocation.

    Returns (env_A, env_B, acting_player=0).
    Invariants:
      - env_A.hands[0] == env_B.hands[0]  (own hand same)
      - env_A.hands[2] == env_B.hands[2]  (partner hand same)
      - len(env_A.hands[1]) == len(env_B.hands[1])
      - len(env_A.hands[3]) == len(env_B.hands[3])
      - multiset(env_A.hands[1] | env_A.hands[3]) == multiset(env_B.hands[1] | env_B.hands[3])
    """
    env_A = GuanDanEnv(level_rank=Rank.TWO)
    env_A.reset(seed=seed)
    env_A.current_player = 0

    env_B = _clone_env(env_A)

    # Swap some cards between opponents 1 and 3 while preserving sizes
    opp1 = list(env_A.hands[1])
    opp3 = list(env_A.hands[3])
    # Swap equal-sized prefixes to keep sizes identical
    n_swap = min(5, len(opp1), len(opp3))
    new_opp1 = set(opp3[:n_swap]) | set(opp1[n_swap:])
    new_opp3 = set(opp1[:n_swap]) | set(opp3[n_swap:])

    env_B.hands[1] = new_opp1
    env_B.hands[3] = new_opp3

    # Verify invariants
    assert env_A.hands[0] == env_B.hands[0]
    assert env_A.hands[2] == env_B.hands[2]
    assert len(env_A.hands[1]) == len(env_B.hands[1])
    assert len(env_A.hands[3]) == len(env_B.hands[3])
    agg_A = Counter(card_id(c) for c in env_A.hands[1] | env_A.hands[3])
    agg_B = Counter(card_id(c) for c in env_B.hands[1] | env_B.hands[3])
    assert agg_A == agg_B

    return env_A, env_B, 0


# ─── Actor pair-features invariance ──────────────────────────────────────────

def test_actor_invariant_to_opponent_repartition():
    env_A, env_B, player = _build_repartitioned_pair()

    legal_A = generate_all_leads(env_A.hands[player], env_A.level_rank)
    legal_B = generate_all_leads(env_B.hands[player], env_B.level_rank)

    sig_A = sorted(action_signature(a, env_A.level_rank) for a in legal_A)
    sig_B = sorted(action_signature(a, env_B.level_rank) for a in legal_B)
    assert sig_A == sig_B, "Legal sets differ by signature"

    sig_to_a_A = {action_signature(a, env_A.level_rank): a for a in legal_A}
    sig_to_a_B = {action_signature(a, env_B.level_rank): a for a in legal_B}

    hand_A = list(env_A.hands[player])
    hand_B = list(env_B.hands[player])

    for sig in sig_A:
        a_A = sig_to_a_A[sig]
        a_B = sig_to_a_B[sig]

        pair_A = encode_actor_pair_features(env_A, player, a_A, legal_A)
        pair_B = encode_actor_pair_features(env_B, player, a_B, legal_B)
        assert np.array_equal(pair_A, pair_B), \
            f"Actor features differ for sig={sig}"

        act_A = encode_action(a_A, hand_A, env_A.level_rank)
        act_B = encode_action(a_B, hand_B, env_B.level_rank)
        assert np.array_equal(act_A, act_B), \
            f"Action encodings differ for sig={sig}"


# ─── Critic structural API guard ─────────────────────────────────────────────

def test_critic_no_action_parameter():
    sig = inspect.signature(encode_critic_state)
    assert "action" not in sig.parameters
    assert "legal_moves" not in sig.parameters


def test_critic_does_not_call_behavior_flags(monkeypatch):
    from guandan.azguan import behavior_flags as bf_mod

    called = {"flag": False}

    def _raise(*a, **kw):
        called["flag"] = True
        raise AssertionError("encode_critic_state called compute_behavior_flags")

    monkeypatch.setattr(bf_mod, "compute_behavior_flags", _raise)
    env_A, env_B, player = _build_repartitioned_pair()
    encode_critic_state(env_A, player, mode="pv")
    encode_critic_state(env_A, player, mode="ptie")
    assert not called["flag"]


# ─── Critic state prefix == state features ───────────────────────────────────

def test_critic_state_prefix_eq_state_features():
    env_A, _, player = _build_repartitioned_pair()
    state = encode_state_features(env_A, player)
    for mode in ("pv", "ptie"):
        c = encode_critic_state(env_A, player, mode=mode)
        assert np.array_equal(c[:STATE_FEATURES_DIM], state), f"mode={mode}"


# ─── PV-AC privileged slots ───────────────────────────────────────────────────

def test_pv_ac_privileged_slots_zero_and_invariant():
    env_A, env_B, player = _build_repartitioned_pair()

    c_A = encode_critic_state(env_A, player, mode="pv")
    c_B = encode_critic_state(env_B, player, mode="pv")

    assert np.array_equal(c_A[755:875], np.zeros(120)), "PV-AC priv slots not zero"
    assert np.array_equal(c_A, c_B), "PV-AC state must be invariant to opponent allocation"


# ─── PV-PTIE privileged slots ────────────────────────────────────────────────

def test_ptie_privileged_slots_differ_with_repartition():
    env_A, env_B, player = _build_repartitioned_pair()

    c_A = encode_critic_state(env_A, player, mode="ptie")
    c_B = encode_critic_state(env_B, player, mode="ptie")

    # State portion should still be identical
    assert np.array_equal(c_A[:STATE_FEATURES_DIM], c_B[:STATE_FEATURES_DIM]), \
        "State portion should be equal"

    # Privileged portion should differ (we repartitioned opponents)
    assert not np.array_equal(c_A[755:875], c_B[755:875]), \
        "PV-PTIE priv slots should differ after repartition"


def test_ptie_privileged_slots_equal_opp_hands():
    env_A, _, player = _build_repartitioned_pair()
    c = encode_critic_state(env_A, player, mode="ptie")

    expected_opp_l = cards_to_matrix(env_A.hands[1]).flatten()
    expected_opp_r = cards_to_matrix(env_A.hands[3]).flatten()

    assert np.array_equal(c[755:815], expected_opp_l), "opp_l hand mismatch"
    assert np.array_equal(c[815:875], expected_opp_r), "opp_r hand mismatch"


# ─── Critic action-independence at runtime ────────────────────────────────────

def test_critic_state_action_independent():
    """Critic state must be identical for all K candidates at the same decision."""
    env_A, _, player = _build_repartitioned_pair()
    legal = generate_all_leads(env_A.hands[player], env_A.level_rank)

    for mode in ("pv", "ptie"):
        states = [encode_critic_state(env_A, player, mode=mode) for _ in legal]
        ref = states[0]
        for i, s in enumerate(states[1:], 1):
            assert np.array_equal(ref, s), \
                f"mode={mode}: critic state differs at candidate {i}"
