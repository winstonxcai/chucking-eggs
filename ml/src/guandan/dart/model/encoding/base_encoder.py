"""GuanZero comparison state-action encoder.

Outputs a dict of channels keyed by ``ENCODE_CHANNEL_KEYS``. Paper-faithful
layout: behavior flags live in the state branch; the action branch is just the
candidate card multi-hot.

Channel shapes:

    own_hand                  (108,)
    others_hand               (108,)   # union of 3 non-self hands (paper)
    recent_action_each_player (4, 108)
    played_cards_others       (3, 108)
    remaining_counts_others   (3, 27)
    level                     (13,)
    behavior                  (9,)     # state-level coordination flags
    history                   (20, 108)
    candidate_action          (108,)

State-level ``behavior`` layout (paper-faithful):
    [0:3] cooperating [N/A, has_doing_option, has_refusing_option]
    [3:6] dwarfing    [N/A, has_doing_option, has_refusing_option]
    [6:9] assisting   [N/A, has_doing_option, has_refusing_option]
"""

from __future__ import annotations

import numpy as np

from ...constants import NUM_PLAYERS, OTHER_PLAYER_OFFSETS
from ....cards import CARD_ID_DIM, ComboType
from ....combos import Combo
from ....game import GuanDanEnv
from ..encoder import (
    BEHAVIOR_DIM,
    HISTORY_LEN,
    LEVEL_DIM,
    RANK_BUCKETS,
    _multi_hot,
    compute_state_behavior_flags,
)


ENCODE_CHANNEL_SHAPES: dict[str, tuple[int, ...]] = {
    "own_hand":                  (108,),
    "others_hand":               (108,),
    "recent_action_each_player": (4, 108),
    "played_cards_others":       (3, 108),
    "remaining_counts_others":   (3, 27),
    "level":                     (13,),
    "behavior":                  (9,),
    "history":                   (20, 108),
    "candidate_action":          (108,),
}

ENCODE_CHANNEL_KEYS: tuple[str, ...] = tuple(ENCODE_CHANNEL_SHAPES.keys())

ENCODE_STATE_KEYS: tuple[str, ...] = (
    "own_hand",
    "others_hand",
    "recent_action_each_player",
    "played_cards_others",
    "remaining_counts_others",
    "level",
    "behavior",
    "history",
)
ENCODE_ACTION_KEYS: tuple[str, ...] = ("candidate_action",)


def _last_action_per_seat(env: GuanDanEnv) -> np.ndarray:
    """(4, 108) — most recent non-pass action by each absolute seat."""
    out = np.zeros((NUM_PLAYERS, CARD_ID_DIM), dtype=np.float32)
    seen = [False] * NUM_PLAYERS
    for actor, combo in reversed(env.move_history):
        if seen[actor] or combo.type == ComboType.PASS:
            continue
        out[actor] = _multi_hot(combo.cards)
        seen[actor] = True
        if all(seen):
            break
    return out


def _played_cards_others(env: GuanDanEnv, player: int) -> np.ndarray:
    """(3, 108) — cumulative played cards for each non-self seat, in
    relative order (right=+1, partner=+2, left=+3). Reads cached multihot."""
    out = np.zeros((len(OTHER_PLAYER_OFFSETS), CARD_ID_DIM), dtype=np.float32)
    for i, offset in enumerate(OTHER_PLAYER_OFFSETS):
        seat = (player + offset) % NUM_PLAYERS
        out[i] = env.played_multihot[seat]
    return out


def _remaining_counts_others(env: GuanDanEnv, player: int) -> np.ndarray:
    """(3, 27) — one-hot remaining-card-count for each non-self seat."""
    out = np.zeros((len(OTHER_PLAYER_OFFSETS), RANK_BUCKETS), dtype=np.float32)
    for i, offset in enumerate(OTHER_PLAYER_OFFSETS):
        seat = (player + offset) % NUM_PLAYERS
        n = min(len(env.hands[seat]), RANK_BUCKETS - 1)
        out[i, n] = 1.0
    return out


def _level_one_hot(level_rank: int) -> np.ndarray:
    out = np.zeros(LEVEL_DIM, dtype=np.float32)
    out[level_rank - 2] = 1.0
    return out


def _history_window(env: GuanDanEnv) -> np.ndarray:
    """(20, 108) — last 20 non-pass moves as card multi-hots. PASS → zeros."""
    out = np.zeros((HISTORY_LEN, CARD_ID_DIM), dtype=np.float32)
    recent = env.move_history[-HISTORY_LEN:]
    offset = HISTORY_LEN - len(recent)
    for i, (_actor, combo) in enumerate(recent):
        if combo.type == ComboType.PASS:
            continue
        out[offset + i] = _multi_hot(combo.cards)
    return out


def static_dim(is_partner_visible: bool = True) -> int:
    """Total flat input dim for ``GuanZeroQNet`` (everything except history)."""
    return (
        CARD_ID_DIM                 # own_hand
        + CARD_ID_DIM               # others_hand
        + NUM_PLAYERS * CARD_ID_DIM # recent_action_each_player
        + len(OTHER_PLAYER_OFFSETS) * CARD_ID_DIM   # played_cards_others
        + len(OTHER_PLAYER_OFFSETS) * RANK_BUCKETS  # remaining_counts_others
        + LEVEL_DIM                 # level
        + BEHAVIOR_DIM              # behavior (now in state)
        + CARD_ID_DIM               # candidate_action
    )


class StateActionEncoder:
    """Configurable encoder; threadsafe; no per-call mutable state.

    ``is_partner_visible=True`` fills the ``others_hand`` channel with the
    union of all 3 non-self hands (paper-faithful). Set False to zero the
    channel entirely (full POMDP). Opponents' individual hands are never
    decomposed here — that's the role-aware encoder's job.
    """

    def __init__(self, is_partner_visible: bool = True) -> None:
        self.is_partner_visible = is_partner_visible

    @property
    def static_dim(self) -> int:
        return static_dim(self.is_partner_visible)

    def _encode_state(
        self,
        env: GuanDanEnv,
        player: int,
        legal_moves: list[Combo],
    ) -> dict[str, np.ndarray]:
        """Compute the 8 state channels that are shared across all legal actions."""
        own_hand = env.hand_multihot[player].astype(np.float32)

        if self.is_partner_visible:
            others_hand = np.zeros(CARD_ID_DIM, dtype=np.float32)
            for seat in range(NUM_PLAYERS):
                if seat != player:
                    others_hand += env.hand_multihot[seat]
            np.clip(others_hand, 0.0, 1.0, out=others_hand)
        else:
            others_hand = np.zeros(CARD_ID_DIM, dtype=np.float32)

        behavior = compute_state_behavior_flags(env, player, legal_moves)

        return {
            "own_hand": own_hand,
            "others_hand": others_hand,
            "recent_action_each_player": _last_action_per_seat(env),
            "played_cards_others": _played_cards_others(env, player),
            "remaining_counts_others": _remaining_counts_others(env, player),
            "level": _level_one_hot(env.level_rank),
            "behavior": behavior,
            "history": _history_window(env),
        }

    def encode_all(
        self,
        env: GuanDanEnv,
        player: int,
        legal_moves: list[Combo],
    ) -> list[dict[str, np.ndarray]]:
        """Encode all legal moves, sharing the state channels."""
        state = self._encode_state(env, player, legal_moves)
        if __debug__:
            for arr in state.values():
                arr.setflags(write=False)
        result = []
        for action in legal_moves:
            enc = dict(state)
            enc["candidate_action"] = _multi_hot(action.cards)
            result.append(enc)
        return result

    def encode_one(
        self,
        env: GuanDanEnv,
        player: int,
        action: Combo,
        legal_moves: list[Combo],
    ) -> dict[str, np.ndarray]:
        """Encode one selected action without materializing every candidate."""
        state = self._encode_state(env, player, legal_moves)
        if __debug__:
            for arr in state.values():
                arr.setflags(write=False)
        enc = dict(state)
        enc["candidate_action"] = _multi_hot(action.cards)
        return enc


__all__ = [
    "StateActionEncoder",
    "ENCODE_CHANNEL_SHAPES",
    "ENCODE_CHANNEL_KEYS",
    "ENCODE_STATE_KEYS",
    "ENCODE_ACTION_KEYS",
    "static_dim",
]
