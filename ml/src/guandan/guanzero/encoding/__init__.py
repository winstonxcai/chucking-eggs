"""Encoding APIs for GuanZero base and role-aware models."""

from ..encoder import (
    BEHAVIOR_DIM,
    CARD_ID_DIM,
    HISTORY_LEN,
    LEVEL_DIM,
    RANK_BUCKETS,
    card_to_id,
    id_to_card,
)
from .base_encoder import (
    ENCODE_ACTION_KEYS,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    ENCODE_STATE_KEYS,
    StateActionEncoder,
    static_dim,
)
from .role_encoder import (
    REL_NEXT_OPP,
    REL_PARTNER,
    REL_PREV_OPP,
    REL_SELF,
    ROLE_ENCODE_ACTION_KEYS,
    ROLE_ENCODE_CHANNEL_KEYS,
    ROLE_ENCODE_CHANNEL_SHAPES,
    ROLE_ENCODE_STATE_KEYS,
    RoleAwareStateActionEncoder,
    absolute_player,
    relative_role,
)

__all__ = [
    "StateActionEncoder",
    "CARD_ID_DIM",
    "HISTORY_LEN",
    "LEVEL_DIM",
    "RANK_BUCKETS",
    "BEHAVIOR_DIM",
    "ENCODE_CHANNEL_KEYS",
    "ENCODE_STATE_KEYS",
    "ENCODE_ACTION_KEYS",
    "ENCODE_CHANNEL_SHAPES",
    "static_dim",
    "card_to_id",
    "id_to_card",
    "RoleAwareStateActionEncoder",
    "relative_role",
    "absolute_player",
    "REL_SELF",
    "REL_NEXT_OPP",
    "REL_PARTNER",
    "REL_PREV_OPP",
    "ROLE_ENCODE_CHANNEL_KEYS",
    "ROLE_ENCODE_STATE_KEYS",
    "ROLE_ENCODE_ACTION_KEYS",
    "ROLE_ENCODE_CHANNEL_SHAPES",
]
