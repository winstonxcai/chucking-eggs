"""Tier 1 partner-visibility encoding (Direction C)."""

from .behavior_flags import FLAG_DIM, compute_behavior_flags
from .encoding import (
    ACTING_FLAG_END,
    ACTING_FLAG_START,
    PARTNER_HAND_DIM,
    STATE_DIM_TIER1,
    STATE_DIM_TIER1_TEAM,
    STATE_DIM_TIER1_TEAM_WITH_FLAGS,
    encode_state_tier1,
    encode_state_tier1_team,
    encode_state_tier1_team_with_flags,
)

__all__ = [
    "ACTING_FLAG_END",
    "ACTING_FLAG_START",
    "FLAG_DIM",
    "PARTNER_HAND_DIM",
    "STATE_DIM_TIER1",
    "STATE_DIM_TIER1_TEAM",
    "STATE_DIM_TIER1_TEAM_WITH_FLAGS",
    "compute_behavior_flags",
    "encode_state_tier1",
    "encode_state_tier1_team",
    "encode_state_tier1_team_with_flags",
]
