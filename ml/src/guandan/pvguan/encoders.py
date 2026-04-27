"""State and action encoders for the partner-visible PTIE experiment.

All public constants are the single source of truth — tests, the network, and
distillation scripts import slices from here.

Coordinate convention (always applied after _reflect_env in the caller):
  teammate_lo = seat 0, teammate_hi = seat 2
  opp_l = seat 1,  opp_r = seat 3

Both ablations (PV-AC, PV-PTIE) share the same 875-dim critic width.
The only difference is whether privileged slots [755:875] carry opponent cards.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

import numpy as np

from ..cards import BOMB_TYPES, Card, ComboType, Rank, is_wild
from ..game import GuanDanEnv
from ..azguan.encoding import (  # noqa: PLC2701
    INITIAL_HAND_SIZE,
    MAX_BOMBS,
    MAX_COMBO_SIZE,
    MAX_SINGLES,
    NUM_COMBO_TYPES,
    NUM_RANKS,
    _FULL_DECK,
    _encode_combo_type,
    _rank_index,
    cards_to_matrix,
    encode_move_event,
)
from ..azguan.behavior_flags import FLAG_DIM, compute_behavior_flags

# ─── Dimension constants ──────────────────────────────────────────────────────

STATE_FEATURES_DIM = 755   # Groups 1–6, action-independent
ACTOR_DIM = 764            # 755 + 9 behavior flags
ACTION_DIM = 198
CRITIC_DIM = 875           # 755 + 120 privileged slots

_PRIV_DIM = 120            # 2 × 60 opponent hand vectors
_HISTORY_DIM = 83          # encode_move_event output dimension
_MAX_HISTORY = 15          # last N moves to pool

# ─── Named slice constants (Groups 1–6) ──────────────────────────────────────
# Group 1 — Card zones (420 dims)
TEAMMATE_LO_HAND    = slice(0,   60)
TEAMMATE_HI_HAND    = slice(60,  120)
TEAMMATE_LO_PLAYED  = slice(120, 180)
TEAMMATE_HI_PLAYED  = slice(180, 240)
OPP_L_PLAYED        = slice(240, 300)
OPP_R_PLAYED        = slice(300, 360)
UNKNOWN_REMAINING   = slice(360, 420)

# Group 2 — Per-seat status (40 dims): [lo, hi, oL, oR]
HAND_COUNTS         = slice(420, 424)   # 4 (normalized)
FINISH_POSITION     = slice(424, 444)   # 20 = 4 seats × 5 positions (1st..4th, still_in)
PASS_SEQUENCE       = slice(444, 460)   # 16 = 4 seats × 4 buckets (0,1,2,3+)

# Group 3 — Acting context (18 dims)
ACTING_FLAG         = slice(460, 462)   # [1,0]=lo acts, [0,1]=hi acts
LEVEL_RANK_OH       = slice(462, 475)   # over ranks 2..A (13)
TEAM_WILD_FLAGS     = slice(475, 478)   # [lo_wild, hi_wild, team_ge2]

# Group 4 — Active trick (98 dims)
IS_SELF_LEADER      = slice(478, 479)   # 1 if no active trick
TRICK_OWNER_REL     = slice(479, 483)   # one-hot {self, partner, opp_l, opp_r}
TRICK_TYPE_OH       = slice(483, 500)   # 17 (ComboType)
TRICK_KEY_OH        = slice(500, 515)   # 15 (primary rank)
TRICK_IS_BOMB       = slice(515, 516)   # convenience flag
TRICK_CARDS         = slice(516, 576)   # 60 count vector

# Group 5 — Last non-pass action (96 dims)
LAST_ACTOR_REL      = slice(576, 580)   # one-hot {self, partner, opp_l, opp_r}
LAST_TYPE_OH        = slice(580, 597)   # 17 (ComboType)
LAST_KEY_OH         = slice(597, 612)   # 15 (primary rank)
LAST_CARDS          = slice(612, 672)   # 60 count vector

# Group 6 — Move-history summary (83 dims)
MOVE_HISTORY_MEAN   = slice(672, 755)   # mean of encode_move_event over last T

# Group 7 — Per-action behavior flags (9 dims, actor only)
BEHAVIOR_FLAGS      = slice(755, 764)

# Critic privileged tail
CRITIC_PRIV         = slice(755, 875)   # [755:815] = opp_l hand, [815:875] = opp_r hand
CRITIC_PRIV_OPP_L   = slice(755, 815)
CRITIC_PRIV_OPP_R   = slice(815, 875)

# ─── Bomb tier ordering (9 tiers) ────────────────────────────────────────────

_BOMB_TIER_ORDER = [
    ComboType.BOMB_4, ComboType.BOMB_5, ComboType.BOMB_6, ComboType.BOMB_7,
    ComboType.BOMB_8, ComboType.BOMB_9, ComboType.BOMB_10,
    ComboType.STRAIGHT_FLUSH, ComboType.BOMB_JOKER,
]
_BOMB_TIER_INDEX: dict[ComboType, int] = {ct: i for i, ct in enumerate(_BOMB_TIER_ORDER)}
BOMB_TIER_DIM = 9

# ─── Seat index helpers ───────────────────────────────────────────────────────

# Canonical team-fixed seat order used for 4-element per-seat arrays
# [0]=lo(seat 0), [1]=hi(seat 2), [2]=oL(seat 1), [3]=oR(seat 3)
_SEAT_ORDER = [0, 2, 1, 3]
_SEAT_TO_SLOT = {0: 0, 2: 1, 1: 2, 3: 3}


def _relative_seat_g4g5(seat: int, player: int) -> int:
    """Map absolute seat → {0=self, 1=partner, 2=opp_l, 3=opp_r}.

    Uses team-fixed canonical labelling: opp_l=seat 1, opp_r=seat 3.
    Only valid after _reflect_env (player ∈ {0, 2}).
    """
    if seat == player:
        return 0
    if seat == (player + 2) % 4:
        return 1  # partner
    if seat == 1:
        return 2  # canonical opp_l
    return 3       # canonical opp_r (seat 3)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _pass_counts_since_winning_play(env: GuanDanEnv) -> dict[int, int]:
    """Per-seat passes since the last non-pass in move_history."""
    counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for actor, combo in reversed(env.move_history):
        if combo.type == ComboType.PASS:
            counts[actor] += 1
        else:
            break
    return counts


def _last_non_pass(env: GuanDanEnv) -> tuple[int, object] | None:
    """Most recent non-pass (actor, combo) from move_history, or None."""
    for actor, combo in reversed(env.move_history):
        if combo.type != ComboType.PASS:
            return (actor, combo)
    return None


def _kicker_rank(combo, level_rank: int) -> int | None:
    """For FULL_HOUSE: derive the pair (kicker) rank from physical cards."""
    if combo.type != ComboType.FULL_HOUSE:
        return None
    natural_non_triple = [
        c.rank for c in combo.cards
        if not is_wild(c, level_rank) and c.rank != combo.key
    ]
    if not natural_non_triple:
        return level_rank  # pair is all wilds; declared as level_rank
    return natural_non_triple[0]


def _seq_length(combo) -> int | None:
    """Declared sequence length for the action vector encoding."""
    t = combo.type
    if t == ComboType.PASS:
        return None
    if t in (ComboType.SINGLE, ComboType.PAIR, ComboType.TRIPLE, ComboType.FULL_HOUSE):
        return 1
    if t == ComboType.STRAIGHT:
        return 5
    if t == ComboType.TUBE:
        return 3
    if t == ComboType.PLATE:
        return 2
    if t == ComboType.STRAIGHT_FLUSH:
        return 5
    if t == ComboType.BOMB_JOKER:
        return 4
    # N-of-a-kind bombs (BOMB_4..BOMB_10): use actual card count
    return len(combo.cards)


# ─── Public API ───────────────────────────────────────────────────────────────


def encode_state_features(env: GuanDanEnv, player: int) -> np.ndarray:
    """Action-independent state features (Groups 1–6). 755 dims.

    Single source of truth used by encode_actor_pair_features and
    encode_critic_state. Assumes player ∈ {0, 2} (after _reflect_env).
    """
    LO, HI, OL, OR = 0, 2, 1, 3

    # ── Group 1: Card zones (420 dims) ───────────────────────────────────────
    hand_lo     = cards_to_matrix(env.hands[LO]).flatten()       # 60
    hand_hi     = cards_to_matrix(env.hands[HI]).flatten()       # 60
    played_lo   = cards_to_matrix(env.played[LO]).flatten()      # 60
    played_hi   = cards_to_matrix(env.played[HI]).flatten()      # 60
    played_opl  = cards_to_matrix(env.played[OL]).flatten()      # 60
    played_opr  = cards_to_matrix(env.played[OR]).flatten()      # 60

    all_played: set = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[LO] | env.hands[HI] | all_played
    unknown_remaining = cards_to_matrix(
        [c for c in _FULL_DECK if c not in known]
    ).flatten()                                                   # 60

    # ── Group 2: Per-seat status (40 dims) ───────────────────────────────────
    hand_counts = np.array([
        len(env.hands[s]) / INITIAL_HAND_SIZE for s in _SEAT_ORDER
    ], dtype=np.float32)                                          # 4

    # finish_position_one_hot: 4 seats × 5 positions (1st,2nd,3rd,4th,still_in)
    finish_oh = np.zeros(20, dtype=np.float32)
    for slot, seat in enumerate(_SEAT_ORDER):
        if seat in env.finish_order:
            pos = env.finish_order.index(seat)   # 0=1st, 1=2nd, 2=3rd, 3=4th
            finish_oh[slot * 5 + pos] = 1.0
        else:
            finish_oh[slot * 5 + 4] = 1.0        # still_in

    # pass_sequence_one_hot: 4 seats × 4 buckets ({0,1,2,3+} passes since trick)
    pass_seq_oh = np.zeros(16, dtype=np.float32)
    if env.current_trick is not None:
        pc = _pass_counts_since_winning_play(env)
        for slot, seat in enumerate(_SEAT_ORDER):
            bucket = min(pc[seat], 3)
            pass_seq_oh[slot * 4 + bucket] = 1.0

    # ── Group 3: Acting context (18 dims) ────────────────────────────────────
    acting_flag = np.zeros(2, dtype=np.float32)
    if player == LO:
        acting_flag[0] = 1.0
    elif player == HI:
        acting_flag[1] = 1.0

    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0

    wilds_lo = sum(1 for c in env.hands[LO] if is_wild(c, env.level_rank))
    wilds_hi = sum(1 for c in env.hands[HI] if is_wild(c, env.level_rank))
    wild_flags = np.array([
        float(wilds_lo >= 1),
        float(wilds_hi >= 1),
        float(wilds_lo + wilds_hi >= 2),
    ], dtype=np.float32)                                          # 3

    # ── Group 4: Active trick (98 dims) ──────────────────────────────────────
    is_leader_flag = np.array(
        [float(env.current_trick is None and env.current_player == player)],
        dtype=np.float32,
    )

    trick_owner_oh = np.zeros(4, dtype=np.float32)
    trick_type_oh  = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    trick_key_oh   = np.zeros(NUM_RANKS, dtype=np.float32)
    trick_bomb_flag = np.zeros(1, dtype=np.float32)
    trick_cards_vec = np.zeros(60, dtype=np.float32)

    if env.current_trick is not None and env.trick_winner is not None:
        rel = _relative_seat_g4g5(env.trick_winner, player)
        trick_owner_oh[rel] = 1.0
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_bomb_flag[0] = float(env.current_trick.type in BOMB_TYPES)
        trick_cards_vec = cards_to_matrix(env.current_trick.cards).flatten()

    # ── Group 5: Last non-pass action (96 dims) ───────────────────────────────
    last_actor_oh  = np.zeros(4, dtype=np.float32)
    last_type_oh   = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    last_key_oh    = np.zeros(NUM_RANKS, dtype=np.float32)
    last_cards_vec = np.zeros(60, dtype=np.float32)

    last = _last_non_pass(env)
    if last is not None:
        last_actor, last_combo = last
        rel = _relative_seat_g4g5(last_actor, player)
        last_actor_oh[rel] = 1.0
        last_type_oh[last_combo.type] = 1.0
        if 2 <= last_combo.key <= 17:
            last_key_oh[_rank_index(last_combo.key)] = 1.0
        last_cards_vec = cards_to_matrix(last_combo.cards).flatten()

    # ── Group 6: Move-history summary (83 dims) ───────────────────────────────
    recent = env.move_history[-_MAX_HISTORY:]
    if recent:
        events = np.stack([
            encode_move_event((player - actor) % 4, combo, env.level_rank)
            for actor, combo in recent
        ])
        history_mean = events.mean(axis=0)
    else:
        history_mean = np.zeros(_HISTORY_DIM, dtype=np.float32)

    return np.concatenate([
        # Group 1 (420)
        hand_lo, hand_hi, played_lo, played_hi, played_opl, played_opr,
        unknown_remaining,
        # Group 2 (40)
        hand_counts, finish_oh, pass_seq_oh,
        # Group 3 (18)
        acting_flag, level_oh, wild_flags,
        # Group 4 (98)
        is_leader_flag, trick_owner_oh, trick_type_oh, trick_key_oh,
        trick_bomb_flag, trick_cards_vec,
        # Group 5 (96)
        last_actor_oh, last_type_oh, last_key_oh, last_cards_vec,
        # Group 6 (83)
        history_mean,
    ])
    # Total: 420 + 40 + 18 + 98 + 96 + 83 = 755


def encode_actor_pair_features(
    env: GuanDanEnv,
    player: int,
    action,
    legal_moves: list,
) -> np.ndarray:
    """State features + 9-dim per-action behavior flags. 764 dims.

    The first 755 dims are exactly encode_state_features(env, player).
    """
    state = encode_state_features(env, player)
    flags = compute_behavior_flags(env, player, action, legal_moves)
    return np.concatenate([state, flags])


def encode_action(
    combo,
    hand: Sequence[Card],
    level_rank: int,
) -> np.ndarray:
    """Per-action 198-dim vector.

    Hand must be a sequence (not a set) to preserve two-deck duplicates.
    The remaining-after-play computation uses Counter subtraction.
    """
    is_pass = combo.type == ComboType.PASS

    # played_cards [60]
    played_vec = cards_to_matrix(combo.cards).flatten() if not is_pass \
        else np.zeros(60, dtype=np.float32)

    # remaining_after_play [60]
    if not is_pass:
        played_counts = Counter((c.rank, c.suit, c.deck) for c in combo.cards)
        hand_counts   = Counter((c.rank, c.suit, c.deck) for c in hand)
        remaining_cards = []
        for key, cnt in hand_counts.items():
            left = cnt - played_counts.get(key, 0)
            for _ in range(left):
                remaining_cards.append(Card(*key))
        remaining_vec = cards_to_matrix(remaining_cards).flatten()
    else:
        remaining_vec = cards_to_matrix(list(hand)).flatten()

    # combo_type_one_hot [17]
    type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    if not is_pass:
        type_oh[combo.type] = 1.0

    # combo_key_one_hot [15]
    key_oh = np.zeros(NUM_RANKS, dtype=np.float32)
    if not is_pass and 2 <= combo.key <= 17:
        key_oh[_rank_index(combo.key)] = 1.0

    # combo_kicker_one_hot [15]
    kicker_oh = np.zeros(NUM_RANKS, dtype=np.float32)
    kr = _kicker_rank(combo, level_rank)
    if kr is not None and 2 <= kr <= 17:
        kicker_oh[_rank_index(kr)] = 1.0

    # bomb_tier_one_hot [9]
    bomb_tier_oh = np.zeros(BOMB_TIER_DIM, dtype=np.float32)
    if not is_pass and combo.type in _BOMB_TIER_INDEX:
        bomb_tier_oh[_BOMB_TIER_INDEX[combo.type]] = 1.0

    # combo_seq_length_one_hot [13] (lengths 1..13 → indices 0..12)
    seq_len_oh = np.zeros(13, dtype=np.float32)
    slen = _seq_length(combo)
    if slen is not None and 1 <= slen <= 13:
        seq_len_oh[slen - 1] = 1.0

    # wild_count_one_hot [3]: {0, 1, 2+}
    wild_oh = np.zeros(3, dtype=np.float32)
    wc = combo.wild_count if not is_pass else 0
    wild_oh[min(wc, 2)] = 1.0

    # scalar flags
    is_bomb_f = np.array([float(not is_pass and combo.type in BOMB_TYPES)], dtype=np.float32)
    is_pass_f = np.array([float(is_pass)], dtype=np.float32)
    num_cards_f = np.array(
        [len(combo.cards) / MAX_COMBO_SIZE if not is_pass else 0.0],
        dtype=np.float32,
    )

    # post-play hand summary stats
    if not is_pass:
        by_rank: dict[int, int] = {}
        for c in remaining_cards:
            by_rank[c.rank] = by_rank.get(c.rank, 0) + 1
        n_singles = sum(1 for cnt in by_rank.values() if cnt == 1)
        n_bombs   = sum(1 for cnt in by_rank.values() if cnt >= 4)
        n_post    = len(remaining_cards)
    else:
        n_singles = n_bombs = 0
        n_post    = sum(1 for _ in hand)

    post_singles_f = np.array([n_singles / MAX_SINGLES], dtype=np.float32)
    post_bombs_f   = np.array([n_bombs   / MAX_BOMBS],   dtype=np.float32)
    post_size_f    = np.array([n_post    / INITIAL_HAND_SIZE], dtype=np.float32)

    return np.concatenate([
        played_vec, remaining_vec,                           # 120
        type_oh, key_oh, kicker_oh,                          # 47
        bomb_tier_oh, seq_len_oh, wild_oh,                   # 25
        is_bomb_f, is_pass_f,                                # 2
        num_cards_f, post_singles_f, post_bombs_f, post_size_f,  # 4
    ])
    # Total: 120 + 47 + 25 + 2 + 4 = 198


def encode_critic_state(
    env: GuanDanEnv,
    player: int,
    mode: Literal["pv", "ptie"],
) -> np.ndarray:
    """Action-independent critic state. 875 dims.

    mode="pv"  : privileged slots [755:875] are all zeros.
    mode="ptie": privileged slots carry opponent hand count vectors.

    MUST NOT be called with a candidate-action argument.
    The state-only portion [0:755] is bit-identical to encode_state_features.
    """
    state = encode_state_features(env, player)           # [755]

    if mode == "ptie":
        priv_l = cards_to_matrix(env.hands[1]).flatten()  # opp_l = seat 1
        priv_r = cards_to_matrix(env.hands[3]).flatten()  # opp_r = seat 3
        priv = np.concatenate([priv_l, priv_r])           # [120]
    else:
        priv = np.zeros(_PRIV_DIM, dtype=np.float32)      # [120]

    return np.concatenate([state, priv])                  # [875]
