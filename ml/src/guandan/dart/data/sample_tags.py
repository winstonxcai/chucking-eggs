"""Per-sample diagnostic tag registries and bucket functions.

Constants and helpers shared across the actor (which emits tags), the worker
(which sets per-episode tags), the buffer (which stores them), and the learner
(which aggregates loss by tag). Kept in a single module to avoid circular
imports and to keep the bucket boundaries discoverable.

The 14-field tag set is defined in ``data/buffer.py`` (``_TAG_FIELDS``);
this module is the single source of truth for opponent IDs, episode modes,
action-class collapse, and bucket boundaries.
"""

from __future__ import annotations

import math

import numpy as np

from ...cards import ComboType
from ...game import GuanDanEnv

# ── Per-step tag computation ─────────────────────────────────────────────
# Compute the diagnostic tags stored on each TrajectoryStep / TrainSample.
# Mirrored in rollout.rs (phase_bucket, phase_with_out, trick_role) for the
# Rust episode loop — keep the two implementations in sync.


def phase_bucket(hand_size: int) -> int:
    """Bucket a player's current hand size into opening / midgame / endgame."""
    if hand_size >= 20:
        return 0
    if hand_size >= 10:
        return 1
    return 2


def phase_with_out(env: GuanDanEnv, seat: int) -> int:
    """phase_bucket for ``seat``; returns 3 if the player has already finished."""
    if env.is_out[seat]:
        return 3
    return phase_bucket(len(env.hands[seat]))


def trick_role(env: GuanDanEnv, partner_seat: int) -> int:
    """0 = leading a new trick, 1 = following (partner alive), 2 = following (partner out)."""
    if env.is_leading():
        return 0
    if env.is_out[partner_seat]:
        return 2
    return 1


# ── Episode mode ─────────────────────────────────────────────────────────

EPISODE_MODE_SELF_PLAY     = 0
EPISODE_MODE_VS_CHECKPOINT = 1
EPISODE_MODE_VS_HARD_BOT   = 2

EPISODE_MODE_NAMES = {
    EPISODE_MODE_SELF_PLAY:     "self_play",
    EPISODE_MODE_VS_CHECKPOINT: "vs_checkpoint",
    EPISODE_MODE_VS_HARD_BOT:   "vs_hard_bot",
}


# ── Opponent registry ────────────────────────────────────────────────────
# Hard-coded bot IDs 1..9; checkpoint pool entries start at 10.

OPPONENT_NONE      = 0   # pure self-play
OPPONENT_STRATEGIC = 1
OPPONENT_YAOJI     = 2
OPPONENT_JIDAN     = 3
OPPONENT_HEURISTIC = 4
OPPONENT_XINGDREAM = 5
OPPONENT_GREEDY    = 6
OPPONENT_RANDOM    = 7
OPPONENT_CHECKPOINT_BASE = 10  # checkpoint pool: BASE + pool_idx (0..63)

OPPONENT_BY_NAME: dict[str, int] = {
    "strategic": OPPONENT_STRATEGIC,
    "yaoji":     OPPONENT_YAOJI,
    "jidan":     OPPONENT_JIDAN,
    "heuristic": OPPONENT_HEURISTIC,
    "xingdream": OPPONENT_XINGDREAM,
    "greedy":    OPPONENT_GREEDY,
    "random":    OPPONENT_RANDOM,
}

# Top opponents to emit grid cells for (capped at 6 for tractability).
# Used by learner aggregation: `opp_phase_*` cells are indexed 0..5 over this
# list rather than over the full int8 range.
OPP_GRID_TOP = [
    OPPONENT_NONE,       # self-play
    OPPONENT_STRATEGIC,
    OPPONENT_YAOJI,
    OPPONENT_JIDAN,
    OPPONENT_HEURISTIC,
    OPPONENT_XINGDREAM,
]


# ── Action-type collapse ─────────────────────────────────────────────────
# ComboType has 17 values (0..16); collapse to 6 classes for tractable grids.
# Raw action_type is still stored per-sample so post-hoc re-bucketing is
# possible.

ACTION_CLASS_PASS       = 0
ACTION_CLASS_SINGLE     = 1
ACTION_CLASS_PAIR_TRIPLE = 2
ACTION_CLASS_MULTI      = 3   # FULL_HOUSE, STRAIGHT, TUBE, PLATE
ACTION_CLASS_BOMB       = 4   # BOMB_4..BOMB_10, STRAIGHT_FLUSH
ACTION_CLASS_JOKER_BOMB = 5

ACTION_CLASS_NAMES = {
    ACTION_CLASS_PASS:        "pass",
    ACTION_CLASS_SINGLE:      "single",
    ACTION_CLASS_PAIR_TRIPLE: "pair_triple",
    ACTION_CLASS_MULTI:       "multi",
    ACTION_CLASS_BOMB:        "bomb",
    ACTION_CLASS_JOKER_BOMB:  "joker_bomb",
}


def collapse_action_type(combo_type: int) -> int:
    """Map a ComboType int to one of the 6 action classes."""
    if combo_type == ComboType.PASS:
        return ACTION_CLASS_PASS
    if combo_type == ComboType.SINGLE:
        return ACTION_CLASS_SINGLE
    if combo_type in (ComboType.PAIR, ComboType.TRIPLE):
        return ACTION_CLASS_PAIR_TRIPLE
    if combo_type in (
        ComboType.FULL_HOUSE, ComboType.STRAIGHT,
        ComboType.TUBE, ComboType.PLATE,
    ):
        return ACTION_CLASS_MULTI
    if combo_type == ComboType.BOMB_JOKER:
        return ACTION_CLASS_JOKER_BOMB
    # All other bombs (BOMB_4..BOMB_10, STRAIGHT_FLUSH) map to bomb.
    return ACTION_CLASS_BOMB


# Lookup table form for fast tensor indexing in the learner. Index by raw
# ComboType int (0..16), get the collapsed class. Pad to length 32 (covers
# any future ComboType expansion).
ACTION_CLASS_LOOKUP = np.zeros(32, dtype=np.int8)
for ct in ComboType:
    ACTION_CLASS_LOOKUP[int(ct)] = collapse_action_type(int(ct))


# ── Bucket functions for marginals ───────────────────────────────────────

def k_bucket(k: int) -> int:
    """Bucket number of legal actions into 4 tiers."""
    if k == 1:    return 0   # K=1 (forced)
    if k <= 5:    return 1   # narrow choice
    if k <= 20:   return 2   # moderate
    return 3                 # wide branching

K_BUCKET_NAMES = {0: "k=1", 1: "k=2-5", 2: "k=6-20", 3: "k>20"}


def q_gap_bucket(q_gap: float) -> int:
    """Bucket the q_max − q_second gap.

    NaN bucket includes K=1, epsilon-random, and inference-server paths
    where q_gap is undefined.
    """
    if math.isnan(q_gap):    return 0  # undefined
    if q_gap < 0.05:         return 1  # pivotal (model uncertain)
    if q_gap < 0.20:         return 2  # moderate
    return 3                            # confident

Q_GAP_BUCKET_NAMES = {0: "nan", 1: "pivotal", 2: "moderate", 3: "confident"}


def reward_bucket(raw_reward: float) -> int:
    """Bucket raw (pre-normalization) terminal reward.

    Engine reward is in {-3,-2,-1,+1,+2,+3} (team-signed finish quality).
    """
    if raw_reward <= -2:  return 0   # big loss (double-down)
    if raw_reward <  0:   return 1   # small loss
    if raw_reward <= 1:   return 2   # small win
    return 3                          # big win (sweep)

REWARD_BUCKET_NAMES = {
    0: "big_loss", 1: "small_loss", 2: "small_win", 3: "big_win",
}


# ── Numeric vectorized helpers (used in learner aggregation) ─────────────

def k_bucket_array(k_arr: np.ndarray) -> np.ndarray:
    """Vectorized K bucket — accepts an int array, returns int8 buckets."""
    out = np.zeros_like(k_arr, dtype=np.int8)
    out[(k_arr >= 2) & (k_arr <= 5)] = 1
    out[(k_arr >= 6) & (k_arr <= 20)] = 2
    out[k_arr > 20] = 3
    return out


def q_gap_bucket_array(q_gap_arr: np.ndarray) -> np.ndarray:
    """Vectorized q_gap bucket."""
    out = np.zeros_like(q_gap_arr, dtype=np.int8)
    nan_mask = np.isnan(q_gap_arr)
    out[nan_mask] = 0
    out[~nan_mask & (q_gap_arr < 0.05)] = 1
    out[~nan_mask & (q_gap_arr >= 0.05) & (q_gap_arr < 0.20)] = 2
    out[~nan_mask & (q_gap_arr >= 0.20)] = 3
    return out


def reward_bucket_array(r_arr: np.ndarray) -> np.ndarray:
    """Vectorized reward bucket — accepts a float array."""
    out = np.zeros_like(r_arr, dtype=np.int8)
    out[r_arr <= -2] = 0
    out[(r_arr > -2) & (r_arr < 0)] = 1
    out[(r_arr >= 0) & (r_arr <= 1)] = 2
    out[r_arr > 1] = 3
    return out


__all__ = [
    # Per-step tag computation
    "phase_bucket", "phase_with_out", "trick_role",
    # Episode mode
    "EPISODE_MODE_SELF_PLAY", "EPISODE_MODE_VS_CHECKPOINT",
    "EPISODE_MODE_VS_HARD_BOT", "EPISODE_MODE_NAMES",
    # Opponents
    "OPPONENT_NONE", "OPPONENT_STRATEGIC", "OPPONENT_YAOJI", "OPPONENT_JIDAN",
    "OPPONENT_HEURISTIC", "OPPONENT_XINGDREAM", "OPPONENT_GREEDY",
    "OPPONENT_RANDOM", "OPPONENT_CHECKPOINT_BASE",
    "OPPONENT_BY_NAME", "OPP_GRID_TOP",
    # Action classes
    "ACTION_CLASS_PASS", "ACTION_CLASS_SINGLE", "ACTION_CLASS_PAIR_TRIPLE",
    "ACTION_CLASS_MULTI", "ACTION_CLASS_BOMB", "ACTION_CLASS_JOKER_BOMB",
    "ACTION_CLASS_NAMES", "ACTION_CLASS_LOOKUP", "collapse_action_type",
    # Buckets
    "k_bucket", "q_gap_bucket", "reward_bucket",
    "K_BUCKET_NAMES", "Q_GAP_BUCKET_NAMES", "REWARD_BUCKET_NAMES",
    "k_bucket_array", "q_gap_bucket_array", "reward_bucket_array",
]
