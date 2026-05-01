"""Paper-faithful state-action encoder for GuanZero (M0).

Outputs a *dict* of channels (not a flat vector) so M1–M4 can swap
individual channels (transformer history, set encoder, role-aware,
belief) without rewriting the rest of the pipeline.

Channel shapes match the paper's 3343-dim total when concatenated:

    own_hand                  (108,)
    others_hand               (108,)         # oracle channel; can be zeroed
    recent_action_each_player (4, 108)
    played_cards_others       (3, 108)
    remaining_counts_others   (3, 27)
    level                     (13,)
    history                   (20, 108)
    behavior                  (9,)
    candidate_action          (108,)

The 9-dim behavior channel reuses ``azguan.behavior_flags`` (cooperation /
dwarfing / assisting × {N/A, doing, refusing}) — that file already
implements the paper's behavior-status definition exactly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..azguan.behavior_flags import FLAG_DIM, compute_behavior_flags
from ..cards import Card, ComboType, Rank, make_deck
from ..combos import Combo
from ..game import GuanDanEnv

# ─── Card-id mapping ─────────────────────────────────────

CARD_ID_DIM = 108
HISTORY_LEN = 20
LEVEL_DIM = 13           # ranks 2..A (level rank can only be one of these)
RANK_BUCKETS = 27        # 0..25 + jokers; one-hot count over 0..26 capped at 26
BEHAVIOR_DIM = FLAG_DIM  # 9


def card_to_id(card: Card) -> int:
    """Stable id in [0, 108) for each physical card.

    Layout:
        ranks 2..A (13 ranks) × 4 suits × 2 deck copies = 104  → ids 0..103
        BLACK_JOKER deck 0/1 → 104, 105
        RED_JOKER   deck 0/1 → 106, 107
    """
    if card.rank == Rank.BLACK_JOKER:
        return 104 + card.deck
    if card.rank == Rank.RED_JOKER:
        return 106 + card.deck
    rank_idx = card.rank - 2          # 0..12
    return rank_idx * 8 + card.suit * 2 + card.deck


_FULL_DECK = make_deck()
_ID_TO_CARD: list[Card] = [None] * CARD_ID_DIM  # type: ignore[list-item]
for _c in _FULL_DECK:
    _ID_TO_CARD[card_to_id(_c)] = _c


def id_to_card(card_id: int) -> Card:
    return _ID_TO_CARD[card_id]


# ─── Channel builders ────────────────────────────────────


def _multi_hot(cards: Iterable[Card]) -> np.ndarray:
    out = np.zeros(CARD_ID_DIM, dtype=np.float32)
    for c in cards:
        out[card_to_id(c)] = 1.0
    return out


def _last_action_per_seat(env: GuanDanEnv) -> np.ndarray:
    """(4, 108) — most recent non-pass action by each absolute seat. Zero
    for seats that haven't played a non-pass yet."""
    out = np.zeros((4, CARD_ID_DIM), dtype=np.float32)
    seen = [False] * 4
    for actor, combo in reversed(env.move_history):
        if seen[actor]:
            continue
        if combo.type == ComboType.PASS:
            continue
        out[actor] = _multi_hot(combo.cards)
        seen[actor] = True
        if all(seen):
            break
    return out


def _played_cards_others(env: GuanDanEnv, player: int) -> np.ndarray:
    """(3, 108) — cumulative played cards for each non-self seat, in
    relative order (right=+1, partner=+2, left=+3)."""
    out = np.zeros((3, CARD_ID_DIM), dtype=np.float32)
    for i, offset in enumerate((1, 2, 3)):
        seat = (player + offset) % 4
        out[i] = _multi_hot(env.played[seat])
    return out


def _remaining_counts_others(env: GuanDanEnv, player: int) -> np.ndarray:
    """(3, 27) — one-hot over remaining-card-count buckets [0..26]
    for each non-self seat, in relative order. The initial deal is 27,
    so 27 buckets covers 0..26 inclusive."""
    out = np.zeros((3, RANK_BUCKETS), dtype=np.float32)
    for i, offset in enumerate((1, 2, 3)):
        seat = (player + offset) % 4
        n = min(len(env.hands[seat]), RANK_BUCKETS - 1)
        out[i, n] = 1.0
    return out


def _level_one_hot(level_rank: int) -> np.ndarray:
    out = np.zeros(LEVEL_DIM, dtype=np.float32)
    out[level_rank - 2] = 1.0
    return out


def _history_window(env: GuanDanEnv) -> np.ndarray:
    """(HISTORY_LEN, 108) — last 20 moves as multi-hot card vectors.
    PASS encodes as all-zero. Older moves on top, newest on bottom; pad
    with leading zeros if fewer than 20 moves so far."""
    out = np.zeros((HISTORY_LEN, CARD_ID_DIM), dtype=np.float32)
    recent = env.move_history[-HISTORY_LEN:]
    offset = HISTORY_LEN - len(recent)
    for i, (_actor, combo) in enumerate(recent):
        if combo.type == ComboType.PASS:
            continue
        out[offset + i] = _multi_hot(combo.cards)
    return out


# ─── Public encoder ──────────────────────────────────────

# Sum of static (non-history) channel sizes — used by the Q-network to
# size the MLP input. Keep in sync with the channel table above.
def static_dim(use_oracle_others_hand: bool = True) -> int:
    return (
        CARD_ID_DIM                 # own_hand
        + CARD_ID_DIM               # others_hand (zeroed if oracle off)
        + 4 * CARD_ID_DIM           # recent_action_each_player
        + 3 * CARD_ID_DIM           # played_cards_others
        + 3 * RANK_BUCKETS          # remaining_counts_others
        + LEVEL_DIM                 # level
        + BEHAVIOR_DIM              # behavior
        + CARD_ID_DIM               # candidate_action
    )


class StateActionEncoder:
    """Stateless encoder. Holds config so we can ablate channels later.

    For M0, ``use_oracle_others_hand=True`` reproduces the paper. M4 will
    flip it to False and replace with a belief-state channel.
    """

    def __init__(self, use_oracle_others_hand: bool = True) -> None:
        self.use_oracle_others_hand = use_oracle_others_hand

    @property
    def static_dim(self) -> int:
        return static_dim(self.use_oracle_others_hand)

    def _encode_state(
        self,
        env: GuanDanEnv,
        player: int,
    ) -> dict[str, np.ndarray]:
        """Compute the 7 state channels that are shared across all legal actions."""
        own_hand = _multi_hot(env.hands[player])

        if self.use_oracle_others_hand:
            others: set = set()
            for p in range(4):
                if p != player:
                    others |= env.hands[p]
            others_hand = _multi_hot(others)
        else:
            others_hand = np.zeros(CARD_ID_DIM, dtype=np.float32)

        return {
            "own_hand": own_hand,
            "others_hand": others_hand,
            "recent_action_each_player": _last_action_per_seat(env),
            "played_cards_others": _played_cards_others(env, player),
            "remaining_counts_others": _remaining_counts_others(env, player),
            "level": _level_one_hot(env.level_rank),
            "history": _history_window(env),
        }

    def encode_all(
        self,
        env: GuanDanEnv,
        player: int,
        legal_moves: list[Combo],
    ) -> list[dict[str, np.ndarray]]:
        """Encode all legal moves for a step, computing shared state channels once.

        Replaces [encode(env, a, p, legal) for a in legal] — the 7 state
        channels are computed once and referenced (not copied) by each result
        dict, since they are read-only numpy arrays during forward pass.
        """
        state = self._encode_state(env, player)
        result = []
        for action in legal_moves:
            enc = dict(state)  # shallow copy — shared arrays are read-only
            enc["behavior"] = compute_behavior_flags(env, player, action, legal_moves)
            enc["candidate_action"] = _multi_hot(action.cards)
            result.append(enc)
        return result

    def encode(
        self,
        env: GuanDanEnv,
        action: Combo,
        player: int,
        legal_moves: list[Combo] | None = None,
    ) -> dict[str, np.ndarray]:
        if legal_moves is None:
            legal_moves = env.legal_moves(player)
        state = self._encode_state(env, player)
        state["behavior"] = compute_behavior_flags(env, player, action, legal_moves)
        state["candidate_action"] = _multi_hot(action.cards)
        return state


# Module-level convenience: paper-faithful default.
_DEFAULT = StateActionEncoder(use_oracle_others_hand=True)


def encode(
    env: GuanDanEnv,
    action: Combo,
    player: int,
    legal_moves: list[Combo] | None = None,
) -> dict[str, np.ndarray]:
    """Paper-faithful default encoder. For ablations construct your own
    ``StateActionEncoder(use_oracle_others_hand=False)``."""
    return _DEFAULT.encode(env, action, player, legal_moves)
