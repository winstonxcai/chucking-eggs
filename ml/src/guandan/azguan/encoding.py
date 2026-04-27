"""State and action encoding for the Q-network and partner-visible policy."""

from __future__ import annotations

import numpy as np

from ..cards import BOMB_TYPES, Card, ComboType, Rank, is_wild, make_deck
from ..combos import Combo
from ..game import GuanDanEnv

NUM_RANKS = 15  # 2..A(14), BJ, RJ → indices 0..14
NUM_COLS = 4    # one per suit
NUM_COMBO_TYPES = 17  # ComboType enum values 0..16

# Normalizer constants for feature encoding
INITIAL_HAND_SIZE = 27  # cards dealt per player
MAX_COMBO_SIZE = 10     # max cards in a single combo (for normalization)
MAX_SINGLES = 10        # reasonable upper bound on isolated singles
MAX_BOMBS = 3           # reasonable upper bound on bombs in one hand


def _rank_index(rank: int) -> int:
    """Map card rank to matrix row index (0-14)."""
    if rank <= 14:
        return rank - 2  # 2→0, 3→1, ..., A(14)→12
    if rank == Rank.BLACK_JOKER:
        return 13
    if rank == Rank.RED_JOKER:
        return 14
    raise ValueError(f"Unknown rank {rank}")


def cards_to_matrix(cards) -> np.ndarray:
    """Cards → [15, 4] count matrix (suit-level counts, 0/1/2)."""
    mat = np.zeros((NUM_RANKS, NUM_COLS), dtype=np.float32)
    for c in cards:
        ri = _rank_index(c.rank)
        si = c.suit if c.rank <= 14 else (0 if c.rank == Rank.BLACK_JOKER else 1)
        mat[ri, si] = min(mat[ri, si] + 1, 2)
    return mat


# Pre-build the full deck for unknown-card computation
_FULL_DECK = make_deck()


def encode_state(env: GuanDanEnv, player: int) -> np.ndarray:
    """State features for player (relative perspective). ~417 dims."""
    partner = env.partner(player)
    opp_l = (player - 1) % 4  # left opponent
    opp_r = (player + 1) % 4  # right opponent

    hand = cards_to_matrix(env.hands[player]).flatten()              # 60
    played_me = cards_to_matrix(env.played[player]).flatten()        # 60
    played_par = cards_to_matrix(env.played[partner]).flatten()      # 60
    played_opl = cards_to_matrix(env.played[opp_l]).flatten()        # 60
    played_opr = cards_to_matrix(env.played[opp_r]).flatten()        # 60

    # Cards unaccounted for (not in our hand, not played by anyone)
    all_played: set[Card] = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[player] | all_played
    unknown = [c for c in _FULL_DECK if c not in known]
    remaining = cards_to_matrix(unknown).flatten()                   # 60

    # Card counts (normalized, relative order: me, right, partner, left)
    counts = np.array(
        [len(env.hands[(player + i) % 4]) / INITIAL_HAND_SIZE for i in range(4)],
        dtype=np.float32,
    )                                                                 # 4

    # Out flags (relative order)
    out_flags = np.array(
        [float(env.is_out[(player + i) % 4]) for i in range(4)],
        dtype=np.float32,
    )                                                                 # 4

    # Level rank (one-hot over 13 normal ranks: 2..A)
    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0                               # 13

    # Wild cards in hand
    wilds_in_hand = sum(1 for c in env.hands[player] if is_wild(c, env.level_rank))
    wild_flags = np.array(
        [float(wilds_in_hand >= 1), float(wilds_in_hand >= 2)],
        dtype=np.float32,
    )                                                                 # 2

    # Current trick info
    is_leader = np.array([float(env.current_trick is None)], dtype=np.float32)  # 1
    trick_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)      # 17
    trick_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)             # 15
    trick_is_bomb = np.array([0.0], dtype=np.float32)                # 1

    if env.current_trick is not None:
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_is_bomb[0] = float(env.current_trick.type in BOMB_TYPES)

    return np.concatenate([
        hand, played_me, played_par, played_opl, played_opr,  # 300
        remaining, counts, out_flags,                          # 68
        level_oh, wild_flags, is_leader,                       # 16
        trick_type_oh, trick_key_oh, trick_is_bomb,            # 33
    ])
    # Total: 417


STATE_DIM = 417
OPP_CARDS_DIM = 60


def encode_opponent_cards(env: GuanDanEnv, player: int) -> np.ndarray:
    """Oracle target: combined opponent cards as [60] binary vector.

    Used as auxiliary training signal — forces LSTM to learn card counting.
    Only available during episode collection (oracle info), not at inference.
    """
    opp_l = (player - 1) % 4
    opp_r = (player + 1) % 4
    return np.clip(
        cards_to_matrix(env.hands[opp_l] | env.hands[opp_r]).flatten(), 0, 1
    )


OPP_CARDS_DIM_SPLIT = 120  # 60 (opp seat 1) + 60 (opp seat 3)


def encode_opponent_cards_split_team(env: GuanDanEnv) -> np.ndarray:
    """Oracle target: per-opponent cards, team-centric. [120] binary vector.

    Returns [opp_seat1 | opp_seat3] concatenation. Unlike encode_opponent_cards
    (the 60-dim union), this is not recoverable algebraically from state +
    partner hand — the network must attribute each unknown card to a specific
    opponent using move history.

    Team-centric: opp_L = seat 1, opp_R = seat 3 (fixed). Invariant to
    which teammate is currently acting.
    """
    l = np.clip(cards_to_matrix(env.hands[1]).flatten(), 0, 1)  # 60
    r = np.clip(cards_to_matrix(env.hands[3]).flatten(), 0, 1)  # 60
    return np.concatenate([l, r])  # 120


def encode_action(combo: Combo, hand: set, level_rank: int) -> np.ndarray:
    """Action features WITH remaining_after_play. 160 dims."""
    cards_played = cards_to_matrix(combo.cards).flatten()              # 60

    # Remaining hand after this play
    remaining = hand - set(combo.cards)
    remaining_matrix = cards_to_matrix(remaining).flatten()            # 60

    combo_type_oh, is_bomb_val = _encode_combo_type(combo)             # 17, scalar
    combo_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)              # 15
    if 2 <= combo.key <= 17:
        combo_key_oh[_rank_index(combo.key)] = 1.0
    num_cards = np.array([len(combo.cards) / MAX_COMBO_SIZE], dtype=np.float32) # 1
    is_bomb = np.array([is_bomb_val], dtype=np.float32)               # 1
    wild_oh = np.zeros(3, dtype=np.float32)                           # 3
    wild_oh[min(combo.wild_count, 2)] = 1.0

    # Summary stats about remaining hand
    n_singles_after = 0
    n_bombs_after = 0
    by_rank: dict[int, int] = {}
    for c in remaining:
        by_rank[c.rank] = by_rank.get(c.rank, 0) + 1
    for count in by_rank.values():
        if count == 1:
            n_singles_after += 1
        if count >= 4:
            n_bombs_after += 1

    hand_stats = np.array([
        len(remaining) / INITIAL_HAND_SIZE,
        n_singles_after / MAX_SINGLES,
        n_bombs_after / MAX_BOMBS,
    ], dtype=np.float32)                                               # 3

    return np.concatenate([
        cards_played, remaining_matrix, combo_type_oh, combo_key_oh,
        num_cards, is_bomb, wild_oh, hand_stats,
    ])
    # Total: 60 + 60 + 17 + 15 + 1 + 1 + 3 + 3 = 160


ACTION_DIM = 160

# ─── History encoding ────────────────────────────────

D_MOVE = 83  # per-move event vector size
MAX_HISTORY = 15  # last N moves to encode


def _encode_combo_type(combo: Combo) -> tuple[np.ndarray, float]:
    """Encode combo type one-hot and bomb flag. Returns (type_oh[17], is_bomb)."""
    combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)
    if combo.type != ComboType.PASS:
        combo_type_oh[combo.type] = 1.0
    return combo_type_oh, float(combo.type in BOMB_TYPES)


def encode_move_event(actor_relative: int, combo: Combo, level_rank: int) -> np.ndarray:
    """Encode a single move as an 83-dim vector.

    actor_relative: 0=me, 1=right(CCW), 2=partner, 3=left
    """
    out = np.zeros(D_MOVE, dtype=np.float32)
    out[actor_relative] = 1.0                                             # [0:4] actor

    if combo.type != ComboType.PASS:
        out[4:64] = cards_to_matrix(combo.cards).flatten()                # [4:64] cards
        # is_pass stays 0                                                 # [64]
        combo_type_oh, is_bomb = _encode_combo_type(combo)
        out[65:82] = combo_type_oh                                        # [65:82] type
        out[82] = is_bomb                                                 # [82]
    else:
        out[64] = 1.0                                                     # is_pass

    return out
    # Total: 4 + 60 + 1 + 17 + 1 = 83


D_GLOBAL = 310  # global state dimension for QMIX mixer


def encode_global_state(env: GuanDanEnv, team: tuple[int, int] = (0, 2)) -> np.ndarray:
    """Global (oracle) state for QMIX mixer. Sees all 4 hands.

    Centralized training only — not used during decentralized execution.

    Layout:
        4 hands × 60 dims    = 240  (oracle: all hands visible)
        played_all            = 60   (cards played by anyone)
        hand_counts (4)       = 4    (normalized)
        out_flags (4)         = 4
        trick_active          = 1
        trick_winner_team     = 1    (1 if team won last trick, else 0)
        Total                 = 310
    """
    # All 4 hands (absolute order: players 0,1,2,3)
    hands = np.concatenate([
        cards_to_matrix(env.hands[p]).flatten() for p in range(4)
    ])  # 240

    # All played cards (union)
    all_played: set = set()
    for p in range(4):
        all_played |= env.played[p]
    played_all = cards_to_matrix(all_played).flatten()  # 60

    # Hand counts (normalized)
    counts = np.array(
        [len(env.hands[p]) / INITIAL_HAND_SIZE for p in range(4)],
        dtype=np.float32,
    )  # 4

    # Out flags
    out_flags = np.array(
        [float(env.is_out[p]) for p in range(4)],
        dtype=np.float32,
    )  # 4

    # Trick active
    trick_active = np.array([float(env.current_trick is not None)], dtype=np.float32)  # 1

    # Did team win the last trick? (heuristic: trick winner's team == team)
    trick_winner_team = np.array([0.0], dtype=np.float32)  # 1
    if env.trick_winner is not None and env.trick_winner in team:
        trick_winner_team[0] = 1.0

    return np.concatenate([
        hands, played_all, counts, out_flags, trick_active, trick_winner_team,
    ])
    # Total: 240 + 60 + 4 + 4 + 1 + 1 = 310


def encode_history(
    env: GuanDanEnv, player: int, level_rank: int
) -> tuple[np.ndarray, int]:
    """Encode the last MAX_HISTORY moves from player's perspective.

    Returns (history [T, D_MOVE], length int).
    """
    recent = env.move_history[-MAX_HISTORY:]

    if len(recent) == 0:
        return np.zeros((1, D_MOVE), dtype=np.float32), 1

    num_events = len(recent)
    history = np.zeros((num_events, D_MOVE), dtype=np.float32)
    for i, (actor, combo) in enumerate(recent):
        rel = (player - actor) % 4  # relative: 0=me, 1=right, 2=partner, 3=left
        history[i] = encode_move_event(rel, combo, level_rank)

    return history, num_events


# ─── Partner-visible state encoding ──────────────────────────────────────────
#
# Seat convention (team {0, 2} vs opponents {1, 3}):
#   lo  = 0  (low-index teammate, fixed)
#   hi  = 2  (high-index teammate, fixed)
#   oL  = 1  (left opponent, fixed)
#   oR  = 3  (right opponent, fixed)

# Partner hand is 60 dims (15 ranks x 4 suits count matrix, values 0/1/2).
PARTNER_HAND_DIM = 60
PARTNER_INSERT_POS = 60  # after own hand (positions 0:60)

STATE_DIM_PARTNER = STATE_DIM + PARTNER_HAND_DIM  # 417 + 60 = 477


def encode_state_partner(env: GuanDanEnv, player: int) -> np.ndarray:
    """State features with partner hand visibility. 477 dims.

    Wraps encode_state (417) and inserts partner's hand (60) at position 60,
    right after the player's own hand.
    """
    base = encode_state(env, player)  # [417]
    partner = env.partner(player)
    partner_hand = cards_to_matrix(env.hands[partner]).flatten()  # [60]
    return np.insert(base, PARTNER_INSERT_POS, partner_hand)  # [477]


# Team-centric encoding (480 dims):
#   [  0: 60] team_hand_lo         (60)
#   [ 60:120] team_hand_hi         (60)
#   [120:180] team_played_lo       (60)
#   [180:240] team_played_hi       (60)
#   [240:300] played_opp_l         (60)
#   [300:360] played_opp_r         (60)
#   [360:420] remaining            (60)
#   [420:424] counts [lo,hi,oL,oR] (4)
#   [424:428] out_flags            (4)
#   [428:430] acting_seat_flag     (2) ← [1,0] for lo acts, [0,1] for hi acts
#   [430:433] wilds_in_team        (3) ← [lo>=1, hi>=1, total>=2]
#   [433:446] level_oh             (13)
#   [446:447] is_leader            (1)
#   [447:464] trick_type_oh        (17)
#   [464:479] trick_key_oh         (15)
#   [479:480] trick_is_bomb        (1)

STATE_DIM_TEAM = 480
ACTING_FLAG_START = 428
ACTING_FLAG_END = 430


def encode_state_team(env: GuanDanEnv, player: int) -> np.ndarray:
    """Team-centric partner-visible state encoding. 480 dims.

    Invariant across teammates: encode_state_team(env, 0) and
    encode_state_team(env, 2) produce the same output except for
    the acting_seat_flag at positions [428:430].

    Assumes team = {0, 2}, opponents = {1, 3}.
    """
    LO, HI, OL, OR = 0, 2, 1, 3

    hand_lo = cards_to_matrix(env.hands[LO]).flatten()        # 60
    hand_hi = cards_to_matrix(env.hands[HI]).flatten()        # 60
    played_lo = cards_to_matrix(env.played[LO]).flatten()     # 60
    played_hi = cards_to_matrix(env.played[HI]).flatten()     # 60
    played_opl = cards_to_matrix(env.played[OL]).flatten()    # 60
    played_opr = cards_to_matrix(env.played[OR]).flatten()    # 60

    # Unknown cards (not in any team hand, not played)
    all_played: set = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[LO] | env.hands[HI] | all_played
    unknown = [c for c in _FULL_DECK if c not in known]
    remaining = cards_to_matrix(unknown).flatten()            # 60

    counts = np.array([
        len(env.hands[LO]) / INITIAL_HAND_SIZE,
        len(env.hands[HI]) / INITIAL_HAND_SIZE,
        len(env.hands[OL]) / INITIAL_HAND_SIZE,
        len(env.hands[OR]) / INITIAL_HAND_SIZE,
    ], dtype=np.float32)                                      # 4

    out_flags = np.array([
        float(env.is_out[LO]),
        float(env.is_out[HI]),
        float(env.is_out[OL]),
        float(env.is_out[OR]),
    ], dtype=np.float32)                                      # 4

    # Acting seat flag: [1,0] if lo acts, [0,1] if hi acts
    acting_flag = np.zeros(2, dtype=np.float32)
    if player == LO:
        acting_flag[0] = 1.0
    elif player == HI:
        acting_flag[1] = 1.0

    # Wilds across team
    wilds_lo = sum(1 for c in env.hands[LO] if is_wild(c, env.level_rank))
    wilds_hi = sum(1 for c in env.hands[HI] if is_wild(c, env.level_rank))
    wild_flags = np.array([
        float(wilds_lo >= 1),
        float(wilds_hi >= 1),
        float(wilds_lo + wilds_hi >= 2),
    ], dtype=np.float32)                                      # 3

    # Level rank one-hot
    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0                        # 13

    # Current trick info (absolute, not acting-seat relative)
    is_leader = np.array([float(env.current_trick is None)], dtype=np.float32)  # 1
    trick_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)  # 17
    trick_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)      # 15
    trick_is_bomb = np.array([0.0], dtype=np.float32)         # 1

    if env.current_trick is not None:
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_is_bomb[0] = float(env.current_trick.type in BOMB_TYPES)

    return np.concatenate([
        hand_lo, hand_hi, played_lo, played_hi, played_opl, played_opr,  # 360
        remaining,                                                        # 60
        counts, out_flags,                                                # 8
        acting_flag,                                                      # 2
        wild_flags,                                                       # 3
        level_oh,                                                         # 13
        is_leader, trick_type_oh, trick_key_oh, trick_is_bomb,            # 34
    ])
    # Total: 360 + 60 + 8 + 2 + 3 + 13 + 34 = 480


# ─── Per-action wrapper with team-coordination flags ──────────────────────────

from .behavior_flags import FLAG_DIM, compute_behavior_flags  # noqa: E402

STATE_DIM_TEAM_WITH_FLAGS = STATE_DIM_TEAM + FLAG_DIM  # 489


def encode_state_team_with_flags(
    env: GuanDanEnv, player: int, action, legal_moves: list
) -> np.ndarray:
    """480-dim team state + 9-dim per-action behavior flags. Total 489."""
    base = encode_state_team(env, player)
    flags = compute_behavior_flags(env, player, action, legal_moves)
    return np.concatenate([base, flags])
