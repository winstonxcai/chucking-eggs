"""Tests for pvguan/encoders.py.

Covers:
- Dimension assertions for actor, action, critic state
- ComboType bomb-tier enum completeness
- Sequence-length one-hot per combo type
- Bomb-tier one-hot per bomb type
- Kicker rank encoding for full houses
- Move-history mean-pool over actual T
- Current-trick zero when leading
- Current-trick owner semantics (lead → pass → beat → pass)
- Pass-sequence reset on beat
- Action encoding injectivity (canonical signatures)
"""

from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pytest

from guandan.cards import BOMB_TYPES, Card, ComboType, Rank, Suit, is_wild
from guandan.combos import Combo, generate_all_leads, generate_responses
from guandan.game import GuanDanEnv
from guandan.pvguan.encoders import (
    ACTOR_DIM,
    ACTION_DIM,
    BOMB_TIER_DIM,
    CRITIC_DIM,
    STATE_FEATURES_DIM,
    _BOMB_TIER_INDEX,
    _BOMB_TIER_ORDER,
    encode_action,
    encode_actor_pair_features,
    encode_critic_state,
    encode_state_features,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fresh_env(seed: int = 42) -> GuanDanEnv:
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=seed)
    return env


def card_id(c: Card) -> tuple:
    return (c.rank, c.suit, c.deck)


def card_multiset(cards) -> tuple:
    return tuple(sorted(Counter(card_id(c) for c in cards).items()))


def _combo_kicker_rank_from_cards(combo: Combo, level_rank: int) -> int | None:
    if combo.type != ComboType.FULL_HOUSE:
        return None
    natural_non_triple = [
        c.rank for c in combo.cards
        if not is_wild(c, level_rank) and c.rank != combo.key
    ]
    if not natural_non_triple:
        return level_rank
    return natural_non_triple[0]


def action_signature(combo: Combo, level_rank: int) -> tuple:
    from guandan.pvguan.encoders import _BOMB_TIER_INDEX, _seq_length
    kicker = _combo_kicker_rank_from_cards(combo, level_rank)
    bomb_tier = _BOMB_TIER_INDEX.get(combo.type, -1)
    seq_len = _seq_length(combo)
    return (
        card_multiset(combo.cards),
        int(combo.type),
        combo.key,
        tuple(sorted(int(c.rank) for c in combo.cards if is_wild(c, level_rank))),
        bomb_tier,
        kicker,
        seq_len,
    )


# ─── Dimension tests ──────────────────────────────────────────────────────────

def test_state_features_dim():
    env = _fresh_env()
    s = encode_state_features(env, 0)
    assert s.shape == (STATE_FEATURES_DIM,), f"Got {s.shape}"
    assert STATE_FEATURES_DIM == 755


def test_actor_pair_features_dim():
    env = _fresh_env()
    legal = generate_all_leads(env.hands[0], env.level_rank)
    pair = encode_actor_pair_features(env, 0, legal[0], legal)
    assert pair.shape == (ACTOR_DIM,), f"Got {pair.shape}"
    assert ACTOR_DIM == 764


def test_action_dim():
    env = _fresh_env()
    legal = generate_all_leads(env.hands[0], env.level_rank)
    hand = list(env.hands[0])
    a = encode_action(legal[0], hand, env.level_rank)
    assert a.shape == (ACTION_DIM,), f"Got {a.shape}"
    assert ACTION_DIM == 198


def test_critic_state_dim_pv():
    env = _fresh_env()
    c = encode_critic_state(env, 0, mode="pv")
    assert c.shape == (CRITIC_DIM,), f"Got {c.shape}"
    assert CRITIC_DIM == 875


def test_critic_state_dim_ptie():
    env = _fresh_env()
    c = encode_critic_state(env, 0, mode="ptie")
    assert c.shape == (CRITIC_DIM,)


# ─── State features shared prefix ────────────────────────────────────────────

def test_actor_state_prefix_equals_state_features():
    env = _fresh_env()
    legal = generate_all_leads(env.hands[0], env.level_rank)
    state = encode_state_features(env, 0)
    pair = encode_actor_pair_features(env, 0, legal[0], legal)
    assert np.array_equal(pair[:STATE_FEATURES_DIM], state)


def test_critic_state_prefix_equals_state_features():
    env = _fresh_env()
    state = encode_state_features(env, 0)
    for mode in ("pv", "ptie"):
        c = encode_critic_state(env, 0, mode=mode)
        assert np.array_equal(c[:STATE_FEATURES_DIM], state), f"mode={mode}"


# ─── Critic API action-independence ──────────────────────────────────────────

def test_critic_state_no_action_parameter():
    sig = inspect.signature(encode_critic_state)
    assert "action" not in sig.parameters
    assert "legal_moves" not in sig.parameters


def test_critic_state_does_not_call_behavior_flags(monkeypatch):
    from guandan.azguan import behavior_flags as bf_mod

    def _raise(*a, **kw):
        raise AssertionError("encode_critic_state must not call compute_behavior_flags")

    monkeypatch.setattr(bf_mod, "compute_behavior_flags", _raise)
    env = _fresh_env()
    encode_critic_state(env, 0, mode="pv")
    encode_critic_state(env, 0, mode="ptie")


# ─── PV-AC privileged slots ───────────────────────────────────────────────────

def test_pv_ac_privileged_slots_zero():
    env = _fresh_env()
    c = encode_critic_state(env, 0, mode="pv")
    assert np.array_equal(c[755:875], np.zeros(120))


def test_ptie_privileged_slots_not_zero():
    env = _fresh_env()
    c = encode_critic_state(env, 0, mode="ptie")
    # At least opp hands are non-zero (they have 27 cards)
    assert not np.array_equal(c[755:875], np.zeros(120))


# ─── ComboType bomb-tier enum completeness ────────────────────────────────────

def test_bomb_tier_enum_complete():
    expected = {
        "BOMB_4", "BOMB_5", "BOMB_6", "BOMB_7", "BOMB_8", "BOMB_9", "BOMB_10",
        "STRAIGHT_FLUSH", "BOMB_JOKER",
    }
    present = {ct.name for ct in ComboType if ct in BOMB_TYPES}
    assert expected <= present, f"Missing: {expected - present}"
    assert BOMB_TIER_DIM == 9
    assert len(_BOMB_TIER_ORDER) == 9
    assert all(ct in BOMB_TYPES for ct in _BOMB_TIER_ORDER)


# ─── Sequence length per combo type ──────────────────────────────────────────

def test_seq_length_per_type():
    from guandan.pvguan.encoders import _seq_length

    cases = {
        ComboType.PASS:        None,
        ComboType.SINGLE:      1,
        ComboType.PAIR:        1,
        ComboType.TRIPLE:      1,
        ComboType.FULL_HOUSE:  1,
        ComboType.STRAIGHT:    5,
        ComboType.TUBE:        3,
        ComboType.PLATE:       2,
        ComboType.STRAIGHT_FLUSH: 5,
        ComboType.BOMB_JOKER:  4,
    }
    # Use synthetic combos with just a type and enough cards
    def _fake(ct, n_cards=1, wild=0):
        return Combo(ct, key_rank=3, cards=[Card(3, Suit.SPADE, 0)] * n_cards, wild_count=wild)

    for ct, expected in cases.items():
        combo = _fake(ct, n_cards=4 if ct == ComboType.BOMB_JOKER else 1)
        assert _seq_length(combo) == expected, f"{ct}: expected {expected}"

    # N-of-a-kind bombs
    for n, ct in [(4, ComboType.BOMB_4), (5, ComboType.BOMB_5), (6, ComboType.BOMB_6),
                  (7, ComboType.BOMB_7), (8, ComboType.BOMB_8), (10, ComboType.BOMB_10)]:
        combo = _fake(ct, n_cards=n)
        assert _seq_length(combo) == n, f"{ct}: expected {n}"


# ─── Bomb tier index per type ─────────────────────────────────────────────────

def test_bomb_tier_index():
    for ct in BOMB_TYPES:
        assert ct in _BOMB_TIER_INDEX, f"{ct} not in BOMB_TIER_INDEX"

    # Ordering: BOMB_4 at 0, BOMB_JOKER at 8
    assert _BOMB_TIER_INDEX[ComboType.BOMB_4] == 0
    assert _BOMB_TIER_INDEX[ComboType.BOMB_JOKER] == 8
    assert _BOMB_TIER_INDEX[ComboType.STRAIGHT_FLUSH] == 7


# ─── Bomb tier one-hot in action vector ──────────────────────────────────────

def test_action_bomb_tier_one_hot():
    env = _fresh_env()
    hand = list(env.hands[0])
    legal = generate_all_leads(env.hands[0], env.level_rank)

    for combo in legal:
        a = encode_action(combo, hand, env.level_rank)
        bomb_oh = a[167:176]
        if combo.type in _BOMB_TIER_INDEX:
            idx = _BOMB_TIER_INDEX[combo.type]
            expected = np.zeros(9, dtype=np.float32)
            expected[idx] = 1.0
            assert np.array_equal(bomb_oh, expected), f"{combo.type}"
        else:
            assert np.array_equal(bomb_oh, np.zeros(9)), f"{combo.type} should be zero"


# ─── Move history mean-pool over actual T ─────────────────────────────────────

def test_move_history_mean_zero_when_empty():
    env = _fresh_env()
    s = encode_state_features(env, 0)
    hist_mean = s[672:755]
    assert np.array_equal(hist_mean, np.zeros(83))


def test_move_history_mean_over_actual_T():
    from guandan.pvguan.encoders import _MAX_HISTORY
    from guandan.azguan.encoding import encode_move_event

    env = _fresh_env()
    # Play a few moves manually
    for _ in range(3):
        legal = env.legal_moves(env.current_player)
        env.step(legal[0])
        if env.done:
            break

    if not env.move_history:
        pytest.skip("No moves in history after 3 steps")

    player = 0
    recent = env.move_history[-_MAX_HISTORY:]
    T = len(recent)
    expected_mean = np.stack([
        encode_move_event((player - actor) % 4, combo, env.level_rank)
        for actor, combo in recent
    ]).mean(axis=0)

    s = encode_state_features(env, player)
    hist_mean = s[672:755]
    assert np.allclose(hist_mean, expected_mean, atol=1e-6), \
        f"max diff: {np.abs(hist_mean - expected_mean).max()}"


# ─── Current-trick semantics ──────────────────────────────────────────────────

def test_current_trick_zero_when_leading():
    env = _fresh_env()
    # Before any play, current_trick is None — player is leading
    assert env.current_trick is None
    s = encode_state_features(env, 0)

    # is_self_leader = 1 when player is acting and no trick
    is_leader = s[478]
    if env.current_player == 0:
        assert is_leader == 1.0
    else:
        assert is_leader == 0.0  # player 0 is not the current player

    # Trick type/key/cards all zero
    assert np.array_equal(s[483:576], np.zeros(93))
    # Trick owner all zero
    assert np.array_equal(s[479:483], np.zeros(4))


def test_trick_owner_after_beat():
    """lead → (some passes) → beat: trick_owner should be the beater."""
    # Play from a known state where we can control who leads and who beats.
    # Use seeds to find a case where player 0 leads and player 2 beats.
    for seed in range(200):
        env = GuanDanEnv(level_rank=Rank.TWO)
        env.reset(seed=seed)
        env.current_player = 0

        legal0 = generate_all_leads(env.hands[0], env.level_rank)
        leads = [c for c in legal0 if c.type not in (ComboType.PASS,)]
        if not leads:
            continue
        lead = leads[0]
        env.step(lead)  # P0 leads; next player = (0-1)%4 = 3 (CCW)

        if env.current_trick is None or env.done:
            continue

        # Have all players between P0 and P2 (i.e. P3) pass
        # CCW order: 0→3→2→1→0
        passes_done = 0
        while env.current_player != 2 and not env.done and env.current_trick is not None:
            p = env.current_player
            resp = generate_responses(env.hands[p], env.level_rank, env.current_trick)
            pass_m = next((c for c in resp if c.type == ComboType.PASS), None)
            if pass_m is None:
                break
            env.step(pass_m)
            passes_done += 1
            if passes_done > 4:
                break

        if env.done or env.current_trick is None or env.current_player != 2:
            continue

        # P2 beats
        resp2 = generate_responses(env.hands[2], env.level_rank, env.current_trick)
        beats = [c for c in resp2 if c.type != ComboType.PASS]
        if not beats:
            continue
        env.step(beats[0])

        if env.current_trick is None or env.trick_winner != 2:
            continue

        # trick_winner is player 2 = partner of player 0
        s = encode_state_features(env, 0)
        owner_oh = s[479:483]
        # Relative to player 0: {self=0, partner=1, opp_l=2, opp_r=3}
        # trick_winner=2 → partner → index 1
        assert owner_oh.argmax() == 1, f"Expected partner(1), got {owner_oh.argmax()}, seed={seed}"
        return  # found a valid scenario

    pytest.skip("Could not find a valid lead→pass→beat scenario in 200 seeds")


def test_pass_sequence_reset_on_beat():
    """Pass counts reset to zero after a beat."""
    for seed in range(200):
        env = GuanDanEnv(level_rank=Rank.TWO)
        env.reset(seed=seed)
        env.current_player = 0

        legal0 = generate_all_leads(env.hands[0], env.level_rank)
        lead = next((c for c in legal0 if c.type != ComboType.PASS), None)
        if lead is None:
            continue
        env.step(lead)

        if env.current_trick is None or env.done:
            continue

        # Have next player (CCW) pass
        p = env.current_player
        r = generate_responses(env.hands[p], env.level_rank, env.current_trick)
        pass_m = next((c for c in r if c.type == ComboType.PASS), None)
        if pass_m is None:
            continue
        env.step(pass_m)

        if env.current_trick is None or env.done:
            continue

        # Next player beats
        p2 = env.current_player
        r2 = generate_responses(env.hands[p2], env.level_rank, env.current_trick)
        beats = [c for c in r2 if c.type != ComboType.PASS]
        if not beats:
            continue
        env.step(beats[0])

        if env.current_trick is None or env.done:
            continue

        # After the beat, all pass counts should be 0
        s = encode_state_features(env, 0)
        pass_seq = s[444:460].reshape(4, 4)
        all_zero = all(pass_seq[slot, 0] == 1.0 for slot in range(4))
        if all_zero:
            return  # passed
        else:
            assert False, f"Pass counts not reset after beat, seed={seed}: {pass_seq}"

    pytest.skip("Could not find valid scenario in 200 seeds")


# ─── Action encoding injectivity ─────────────────────────────────────────────

def test_action_encoding_injectivity():
    """Distinct canonical signatures → distinct action vectors."""
    env = _fresh_env(seed=17)
    legal = generate_all_leads(env.hands[0], env.level_rank)
    hand = list(env.hands[0])

    sig_to_enc: dict[tuple, np.ndarray] = {}
    for combo in legal:
        sig = action_signature(combo, env.level_rank)
        enc = encode_action(combo, hand, env.level_rank)
        if sig in sig_to_enc:
            # Same signature → encodings must be identical
            assert np.array_equal(sig_to_enc[sig], enc), \
                f"Same sig {sig} but different encoding"
        else:
            # Different signature → check uniqueness later
            sig_to_enc[sig] = enc

    # All distinct signatures must have distinct encodings
    all_encs = list(sig_to_enc.values())
    for i in range(len(all_encs)):
        for j in range(i + 1, len(all_encs)):
            if np.array_equal(all_encs[i], all_encs[j]):
                # Find which signatures collide
                sigs = list(sig_to_enc.keys())
                raise AssertionError(
                    f"Encoding collision: sig[{i}]={sigs[i]} and sig[{j}]={sigs[j]}"
                )


# ─── Pass action encoding ──────────────────────────────────────────────────────

def test_pass_action_encoding():
    env = _fresh_env()
    env.current_player = 0
    # Lead a card first so we can pass
    lead = generate_all_leads(env.hands[0], env.level_rank)[0]
    env.step(lead)
    env.current_player = 1
    responses = generate_responses(env.hands[1], env.level_rank, env.current_trick)
    pass_move = next(c for c in responses if c.type == ComboType.PASS)
    hand = list(env.hands[1])
    a = encode_action(pass_move, hand, env.level_rank)
    assert a.shape == (ACTION_DIM,)
    # is_pass_flag should be 1
    assert a[193] == 1.0
    # played_cards should be zero
    assert np.array_equal(a[0:60], np.zeros(60))
