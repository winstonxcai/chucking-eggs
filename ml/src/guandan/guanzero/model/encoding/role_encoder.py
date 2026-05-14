"""Role-normalized state-action encoder for the shared-head GuanZero model.

Two head-routing schemes are supported via the encoder's ``head_scheme`` param:

* ``"absolute_seat"`` (default, legacy) — emits ``seat_id`` (absolute seat 0-3)
  for the model's per-seat head selection. ``player_blocks`` shape is (4, 252).

* ``"trick_relative"`` (new) — emits ``trick_head_id``, the actor's position
  relative to the current trick leader: 0 = leading a new trick, 1 = first
  responder, 2 = across from leader, 3 = last responder. ``player_blocks``
  shape is (4, 256) with a 4-dim trick-position one-hot appended per role.

The formula:
    trick_head_id = 0                              if env.current_trick is None
                  = (env.trick_winner - actor) % 4 otherwise

Play is counterclockwise (``_next_seat = (p-1) % 4``), so the table is:
    actor = leader      → 0  (head_0: leading)
    actor = leader - 1  → 1  (head_1: first responder)
    actor = leader - 2  → 2  (head_2: across)
    actor = leader - 3  → 3  (head_3: last responder)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ....cards import ComboType
from ....combos import Combo
from ....game import GuanDanEnv
from ..encoder import (
    BEHAVIOR_DIM,
    CARD_ID_DIM,
    HISTORY_LEN,
    LEVEL_DIM,
    RANK_BUCKETS,
    card_to_id,
    compute_state_behavior_flags,
)

REL_SELF: int = 0
REL_NEXT_OPP: int = 1
REL_PARTNER: int = 2
REL_PREV_OPP: int = 3

# State channel keys are identical across both head schemes; only the
# action-side head id and the player_blocks width differ.
ROLE_ENCODE_STATE_KEYS: tuple[str, ...] = (
    "own_hand",
    "partner_hand",
    "others_hand",
    "player_blocks",
    "global_features",
    "behavior",
    "history_actions",
    "history_roles",
    "history_is_pass",
)

# ── absolute_seat (legacy) schema ──────────────────────────────────
# player_blocks slot layout (per role, 252 dims):
#   [0:108]   played multi-hot
#   [108:216] last non-pass action multi-hot
#   [216:243] remaining-card count one-hot (27)
#   [243:252] bomb tier histogram (9 tiers: BOMB_4..BOMB_JOKER), uint8 counts capped at 3
ROLE_ENCODE_ACTION_KEYS: tuple[str, ...] = (
    "candidate_action",
    "seat_id",
)
ROLE_ENCODE_CHANNEL_SHAPES: dict[str, tuple[int, ...]] = {
    "own_hand": (CARD_ID_DIM,),
    "partner_hand": (CARD_ID_DIM,),
    "others_hand": (CARD_ID_DIM,),
    "player_blocks": (4, 252),
    "global_features": (LEVEL_DIM,),
    "behavior": (BEHAVIOR_DIM,),
    "history_actions": (HISTORY_LEN, CARD_ID_DIM),
    "history_roles": (HISTORY_LEN, 4),
    "history_is_pass": (HISTORY_LEN, 1),
    "candidate_action": (CARD_ID_DIM,),
    "seat_id": (),
}
ROLE_ENCODE_CHANNEL_KEYS: tuple[str, ...] = tuple(ROLE_ENCODE_CHANNEL_SHAPES.keys())

# ── trick_relative (new) schema ─────────────────────────────────────
# player_blocks gains a 4-dim trick-position one-hot per role at [252:256]:
#   [0:108]   played multi-hot
#   [108:216] last non-pass action multi-hot
#   [216:243] remaining-card count one-hot (27)
#   [243:252] bomb tier histogram
#   [252:256] trick-relative position one-hot (this role's pos from leader)
ROLE_ENCODE_TRICK_ACTION_KEYS: tuple[str, ...] = (
    "candidate_action",
    "trick_head_id",
)
ROLE_ENCODE_TRICK_CHANNEL_SHAPES: dict[str, tuple[int, ...]] = {
    "own_hand": (CARD_ID_DIM,),
    "partner_hand": (CARD_ID_DIM,),
    "others_hand": (CARD_ID_DIM,),
    "player_blocks": (4, 256),
    "global_features": (LEVEL_DIM,),
    "behavior": (BEHAVIOR_DIM,),
    "history_actions": (HISTORY_LEN, CARD_ID_DIM),
    "history_roles": (HISTORY_LEN, 4),
    "history_is_pass": (HISTORY_LEN, 1),
    "candidate_action": (CARD_ID_DIM,),
    "trick_head_id": (),
}
ROLE_ENCODE_TRICK_CHANNEL_KEYS: tuple[str, ...] = tuple(
    ROLE_ENCODE_TRICK_CHANNEL_SHAPES.keys()
)

HeadScheme = Literal["absolute_seat", "trick_relative"]


def relative_role(abs_player: int, actor: int) -> int:
    """Return abs_player's role from actor's perspective."""
    return (abs_player - actor) % 4


def absolute_player(actor: int, role: int) -> int:
    """Return absolute seat for role from actor's perspective."""
    return (actor + role) % 4


def _multi_hot(cards) -> np.ndarray:
    """Encode cards as a 108-dim uint8 multi-hot vector."""
    out = np.zeros(CARD_ID_DIM, dtype=np.uint8)
    for card in cards:
        out[card_to_id(card)] = 1
    return out


def _level_one_hot(level_rank: int) -> np.ndarray:
    """Encode level rank as a 13-dim uint8 one-hot vector."""
    out = np.zeros(LEVEL_DIM, dtype=np.uint8)
    out[level_rank - 2] = 1
    return out


def _count_bucket(n: int) -> np.ndarray:
    """Encode a remaining-card count into the capped 27-bucket convention."""
    out = np.zeros(RANK_BUCKETS, dtype=np.uint8)
    out[min(n, RANK_BUCKETS - 1)] = 1
    return out


def trick_head_id(env: GuanDanEnv, actor: int) -> int:
    """Compute trick-relative head id for the given actor.

    0 = actor is leading a new trick (no current trick).
    1..3 = actor's position counterclockwise after the trick leader.
    """
    if env.current_trick is None:
        return 0
    return (env.trick_winner - actor) % 4


class RoleAwareStateActionEncoder:
    """Encodes state + candidate action into role-normalized tensors.

    Two head-routing schemes are supported, selected at construction time.
    The default ``"absolute_seat"`` preserves the legacy output verbatim.
    """

    def __init__(
        self,
        is_partner_visible: bool = True,
        head_scheme: HeadScheme = "absolute_seat",
    ) -> None:
        if head_scheme not in ("absolute_seat", "trick_relative"):
            raise ValueError(
                f"head_scheme must be 'absolute_seat' or 'trick_relative'; "
                f"got {head_scheme!r}"
            )
        self.is_partner_visible = is_partner_visible
        self.head_scheme = head_scheme
        # Schema-dependent constants, exposed for buffer/collate consumers.
        if head_scheme == "absolute_seat":
            self.channel_shapes = ROLE_ENCODE_CHANNEL_SHAPES
            self.channel_keys = ROLE_ENCODE_CHANNEL_KEYS
            self.state_keys = ROLE_ENCODE_STATE_KEYS
            self.action_keys = ROLE_ENCODE_ACTION_KEYS
            self.head_field = "seat_id"
            self._player_block_width = 252
        else:
            self.channel_shapes = ROLE_ENCODE_TRICK_CHANNEL_SHAPES
            self.channel_keys = ROLE_ENCODE_TRICK_CHANNEL_KEYS
            self.state_keys = ROLE_ENCODE_STATE_KEYS
            self.action_keys = ROLE_ENCODE_TRICK_ACTION_KEYS
            self.head_field = "trick_head_id"
            self._player_block_width = 256

    def encode_all(
        self,
        env: GuanDanEnv,
        player: int,
        legal_moves: list[Combo],
    ) -> list[dict[str, np.ndarray]]:
        """Encode all legal actions, sharing decision-level arrays."""
        state = self._encode_state(env, player, legal_moves)
        head_val = self._head_value(env, player)
        if __debug__:
            for arr in state.values():
                arr.setflags(write=False)
        result: list[dict[str, np.ndarray]] = []
        for action in legal_moves:
            enc = dict(state)
            enc["candidate_action"] = _multi_hot(action.cards)
            enc[self.head_field] = head_val
            result.append(enc)
        return result

    def encode_one(
        self,
        env: GuanDanEnv,
        player: int,
        action: Combo,
        legal_moves: list[Combo],
    ) -> dict[str, np.ndarray]:
        """Encode one selected action without materializing all candidates."""
        enc = self._encode_state(env, player, legal_moves)
        if __debug__:
            for arr in enc.values():
                arr.setflags(write=False)
        enc = dict(enc)
        enc["candidate_action"] = _multi_hot(action.cards)
        enc[self.head_field] = self._head_value(env, player)
        return enc

    def _head_value(self, env: GuanDanEnv, player: int) -> np.ndarray:
        """Compute the head-routing id appropriate to this encoder's mode."""
        if self.head_scheme == "absolute_seat":
            return np.asarray(player, dtype=np.int64)
        return np.asarray(trick_head_id(env, player), dtype=np.int64)

    def _encode_state(
        self,
        env: GuanDanEnv,
        player: int,
        legal_moves: list[Combo],
    ) -> dict[str, np.ndarray]:
        # player_blocks holds only public information per role: played cards,
        # last non-pass action, and remaining count. Hands are decomposed into
        # dedicated top-level channels (own_hand, partner_hand, others_hand)
        # so the visibility gate is explicit at the channel level.
        blocks = np.zeros((4, self._player_block_width), dtype=np.uint8)
        last_action = self._last_action_by_role(env, player)
        for role in range(4):
            seat = absolute_player(player, role)
            blocks[role, 0:108] = env.played_multihot[seat]
            blocks[role, 108:216] = last_action[role]
            blocks[role, 216:243] = _count_bucket(len(env.hands[seat]))
            # Bomb tier histogram. Cap each cell at 3 so values stay small and
            # bounded for the network (duplicate-tier plays beyond 3 are rare).
            np.minimum(env.bombs_played[seat], 3, out=blocks[role, 243:252])

        if self.head_scheme == "trick_relative" and env.current_trick is not None:
            # Per-role trick-position onehot: each role's distance from the
            # leader, in the same counterclockwise convention as trick_head_id.
            leader_role = relative_role(env.trick_winner, player)
            for role in range(4):
                trick_pos = (leader_role - role) % 4
                blocks[role, 252 + trick_pos] = 1

        own_hand = env.hand_multihot[player].copy()
        partner_seat = (player + 2) % 4
        if self.is_partner_visible:
            partner_hand = env.hand_multihot[partner_seat].copy()
        else:
            partner_hand = np.zeros(CARD_ID_DIM, dtype=np.uint8)

        # The 2 opponents only (partner is decomposed separately in partner_hand).
        next_opp_seat = (player + 1) % 4
        prev_opp_seat = (player + 3) % 4
        others_hand = env.hand_multihot[next_opp_seat] | env.hand_multihot[prev_opp_seat]

        behavior = compute_state_behavior_flags(env, player, legal_moves).astype(
            np.uint8,
            copy=False,
        )

        history_actions, history_roles, history_is_pass = self._history(env, player)
        return {
            "own_hand": own_hand,
            "partner_hand": partner_hand,
            "others_hand": others_hand,
            "player_blocks": blocks,
            "global_features": _level_one_hot(env.level_rank),
            "behavior": behavior,
            "history_actions": history_actions,
            "history_roles": history_roles,
            "history_is_pass": history_is_pass,
        }

    def _last_action_by_role(self, env: GuanDanEnv, player: int) -> np.ndarray:
        """Return latest non-pass action for each relative role."""
        out = np.zeros((4, CARD_ID_DIM), dtype=np.uint8)
        seen = [False] * 4
        for actor, combo in reversed(env.move_history):
            role = relative_role(actor, player)
            if seen[role] or combo.type == ComboType.PASS:
                continue
            out[role] = _multi_hot(combo.cards)
            seen[role] = True
            if all(seen):
                break
        return out

    def _history(
        self,
        env: GuanDanEnv,
        player: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode recent actions plus actor role and pass bit."""
        actions = np.zeros((HISTORY_LEN, CARD_ID_DIM), dtype=np.uint8)
        roles = np.zeros((HISTORY_LEN, 4), dtype=np.uint8)
        is_pass = np.zeros((HISTORY_LEN, 1), dtype=np.uint8)
        recent = env.move_history[-HISTORY_LEN:]
        offset = HISTORY_LEN - len(recent)
        for i, (actor, combo) in enumerate(recent):
            row = offset + i
            roles[row, relative_role(actor, player)] = 1
            if combo.type == ComboType.PASS:
                is_pass[row, 0] = 1
            else:
                actions[row] = _multi_hot(combo.cards)
        return actions, roles, is_pass

    @staticmethod
    def decode_card_counts(encoded: dict) -> tuple[int, int, int, int]:
        """Return (own, partner, next_opp, prev_opp) card counts from an encoded sample.

        Uses the count one-hot in player_blocks (cols 216:243 per role) and own_hand
        directly. Callers should use this rather than indexing player_blocks directly
        so that layout changes only require updating this one method.
        """
        own_count = int(encoded["own_hand"].sum())
        partner_count = int(np.argmax(encoded["player_blocks"][REL_PARTNER, 216:243]))
        next_opp_count = int(np.argmax(encoded["player_blocks"][REL_NEXT_OPP, 216:243]))
        prev_opp_count = int(np.argmax(encoded["player_blocks"][REL_PREV_OPP, 216:243]))
        return own_count, partner_count, next_opp_count, prev_opp_count


__all__ = [
    "RoleAwareStateActionEncoder",
    "relative_role",
    "absolute_player",
    "trick_head_id",
    "REL_SELF",
    "REL_NEXT_OPP",
    "REL_PARTNER",
    "REL_PREV_OPP",
    "HeadScheme",
    # Legacy (absolute_seat) schema constants
    "ROLE_ENCODE_CHANNEL_KEYS",
    "ROLE_ENCODE_STATE_KEYS",
    "ROLE_ENCODE_ACTION_KEYS",
    "ROLE_ENCODE_CHANNEL_SHAPES",
    # New trick_relative schema constants
    "ROLE_ENCODE_TRICK_CHANNEL_KEYS",
    "ROLE_ENCODE_TRICK_ACTION_KEYS",
    "ROLE_ENCODE_TRICK_CHANNEL_SHAPES",
]
