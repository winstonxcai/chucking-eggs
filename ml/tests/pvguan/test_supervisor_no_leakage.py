"""Stage-0 supervisor no-leakage tests.

Asserts that:
1. PartnerVisibleInfoState contains no opponent-hand fields.
2. PartnerVisibleInfoState.from_env produces identical results for two envs
   differing only in opponent-hand allocation (same own/partner hands, same
   unknown aggregate, different opp seat 1 vs seat 3 split).
3. The azguan team-centric encoder (used by PartnerOracleBot) never reads
   opponent hands — its output is identical for both repartitioned envs.

PartnerOracleBot with use_search=False uses encode_state_team_with_flags,
which only reads team cards and played piles (no opponent hands). This test
confirms the architecture-level leakage guarantee independent of PIMC.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# distill_pvguan is in ml/scripts/train — add scripts to path so we can import it
_SCRIPTS_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "train"
sys.path.insert(0, str(_SCRIPTS_PATH))

from guandan.azguan.encoding import encode_state_team_with_flags
from guandan.cards import Card, ComboType, Rank, Suit
from guandan.game import GuanDanEnv


# ─── Env builder helpers ──────────────────────────────────────────────────────

def _card_id(c: Card) -> tuple:
    return (c.rank, c.suit, c.deck)


def _clone_env(env: GuanDanEnv) -> GuanDanEnv:
    new = GuanDanEnv(level_rank=env.level_rank)
    new.hands           = [set(h) for h in env.hands]
    new.played          = [set(p) for p in env.played]
    new.is_out          = list(env.is_out)
    new.finish_order    = list(env.finish_order)
    new.current_player  = env.current_player
    new.current_trick   = env.current_trick
    new.trick_winner    = env.trick_winner
    new.consecutive_passes = env.consecutive_passes
    new.done            = env.done
    new.move_history    = list(env.move_history)
    return new


def _build_repartitioned_pair(seed: int = 42) -> tuple[GuanDanEnv, GuanDanEnv, int]:
    """Two envs identical except opponent-hand allocation.

    Returns (env_A, env_B, player=0).
    Invariants: own/partner same, opp sizes same, aggregate unknown same.
    """
    env_A = GuanDanEnv(level_rank=Rank.TWO)
    env_A.reset(seed=seed)
    env_A.current_player = 0
    env_B = _clone_env(env_A)

    opp1 = list(env_A.hands[1])
    opp3 = list(env_A.hands[3])
    n_swap = min(5, len(opp1), len(opp3))
    env_B.hands[1] = set(opp3[:n_swap]) | set(opp1[n_swap:])
    env_B.hands[3] = set(opp1[:n_swap]) | set(opp3[n_swap:])

    assert env_A.hands[0] == env_B.hands[0]
    assert env_A.hands[2] == env_B.hands[2]
    agg_A = Counter(_card_id(c) for c in env_A.hands[1] | env_A.hands[3])
    agg_B = Counter(_card_id(c) for c in env_B.hands[1] | env_B.hands[3])
    assert agg_A == agg_B
    return env_A, env_B, 0


# ─── PartnerVisibleInfoState structure ───────────────────────────────────────

def test_info_state_has_no_opponent_hand_fields():
    """PartnerVisibleInfoState must not have any opponent-hand attributes."""
    from distill_pvguan import PartnerVisibleInfoState
    field_names = {f.name for f in dataclasses.fields(PartnerVisibleInfoState)}
    for bad in ("opp_l_hand", "opp_r_hand", "opponent_hand", "hands"):
        assert bad not in field_names, \
            f"PartnerVisibleInfoState has forbidden field '{bad}'"


def test_info_state_has_required_fields():
    """PartnerVisibleInfoState must include own_hand, partner_hand, legal_moves."""
    from distill_pvguan import PartnerVisibleInfoState
    field_names = {f.name for f in dataclasses.fields(PartnerVisibleInfoState)}
    for required in ("own_hand", "partner_hand", "legal_moves", "public_history"):
        assert required in field_names, f"Missing required field: {required}"


def test_info_state_is_frozen():
    """PartnerVisibleInfoState must be immutable (frozen dataclass)."""
    from distill_pvguan import PartnerVisibleInfoState
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=1)
    info = PartnerVisibleInfoState.from_env(env, 0)
    with pytest.raises((TypeError, dataclasses.FrozenInstanceError)):
        info.own_hand = frozenset()  # type: ignore[misc]


# ─── from_env invariance to opponent repartition ─────────────────────────────

def test_info_state_own_hand_invariant_to_opp_repartition():
    """own_hand in the info state must not change when only opp cards are swapped."""
    from distill_pvguan import PartnerVisibleInfoState
    for seed in range(10):
        env_A, env_B, player = _build_repartitioned_pair(seed=seed)
        info_A = PartnerVisibleInfoState.from_env(env_A, player)
        info_B = PartnerVisibleInfoState.from_env(env_B, player)
        assert info_A.own_hand == info_B.own_hand, \
            f"seed={seed}: own_hand changed with opp repartition"


def test_info_state_partner_hand_invariant_to_opp_repartition():
    from distill_pvguan import PartnerVisibleInfoState
    for seed in range(10):
        env_A, env_B, player = _build_repartitioned_pair(seed=seed)
        info_A = PartnerVisibleInfoState.from_env(env_A, player)
        info_B = PartnerVisibleInfoState.from_env(env_B, player)
        assert info_A.partner_hand == info_B.partner_hand, \
            f"seed={seed}: partner_hand changed with opp repartition"


def test_info_state_hand_counts_invariant_to_opp_repartition():
    """hand_counts (sizes) are preserved because repartition preserves sizes."""
    from distill_pvguan import PartnerVisibleInfoState
    for seed in range(10):
        env_A, env_B, player = _build_repartitioned_pair(seed=seed)
        info_A = PartnerVisibleInfoState.from_env(env_A, player)
        info_B = PartnerVisibleInfoState.from_env(env_B, player)
        assert info_A.hand_counts == info_B.hand_counts, \
            f"seed={seed}: hand_counts changed with opp repartition"


def test_info_state_legal_moves_same_count():
    """Legal moves depend only on own hand — must have same count in both states."""
    from distill_pvguan import PartnerVisibleInfoState
    for seed in range(10):
        env_A, env_B, player = _build_repartitioned_pair(seed=seed)
        info_A = PartnerVisibleInfoState.from_env(env_A, player)
        info_B = PartnerVisibleInfoState.from_env(env_B, player)
        assert len(info_A.legal_moves) == len(info_B.legal_moves), \
            f"seed={seed}: legal move count differs"


# ─── Azguan team encoder invariance (PartnerOracleBot architecture) ──────────

def test_team_encoder_invariant_to_opp_repartition():
    """encode_state_team_with_flags must be identical for both repartitioned envs.

    PartnerOracleBot (use_search=False) uses this encoder. If it's invariant,
    the bot architecturally cannot observe true opponent hands.
    """
    for seed in range(10):
        env_A, env_B, player = _build_repartitioned_pair(seed=seed)

        legal_A = env_A.legal_moves(player)
        legal_B = env_B.legal_moves(player)
        if not legal_A or not legal_B:
            continue

        enc_A = encode_state_team_with_flags(env_A, player, legal_A[0], legal_A)
        enc_B = encode_state_team_with_flags(env_B, player, legal_B[0], legal_B)

        assert np.array_equal(enc_A, enc_B), \
            f"seed={seed}: team encoder differs for repartitioned envs — " \
            f"max diff = {np.abs(enc_A - enc_B).max()}"


# ─── combo key extraction ─────────────────────────────────────────────────────

def test_distill_combo_key_structure():
    """_combo_key produces a deterministic tuple from combo attributes only."""
    from distill_pvguan import _combo_key
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=1)
    legal = env.legal_moves(0)
    if not legal:
        pytest.skip("No legal moves")

    for combo in legal[:5]:
        key = _combo_key(combo)
        assert isinstance(key, tuple), "combo key must be a tuple"
        # Same combo must yield the same key
        assert _combo_key(combo) == _combo_key(combo)


def test_distill_combo_key_differs_for_distinct_combos():
    """_combo_key must distinguish semantically different combos."""
    from distill_pvguan import _combo_key
    env = GuanDanEnv(level_rank=Rank.TWO)
    env.reset(seed=1)
    legal = env.legal_moves(0)
    if len(legal) < 2:
        pytest.skip("Need at least 2 legal moves")

    keys = [_combo_key(a) for a in legal]
    # Not all keys should be identical (distinct legal moves have distinct keys)
    assert len(set(keys)) > 1, "All legal move keys are identical — _combo_key is broken"
