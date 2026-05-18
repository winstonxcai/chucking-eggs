"""Replay buffer + sample shape + return computation.

Pure data-structures and per-sample diagnostic tagging — no model or runtime
dependencies. Knows about encoder output shapes (via ..encoding) to size
its uint8 storage.
"""

from .buffer import (
    BUCKET_NAMES,
    ReplayBuffer,
    RoleAwareReplayBuffer,
    collate_base_encoded,
    collate_role_encoded,
)
from .returns import (
    TERMINAL_REWARD_SCALE,
    TrainSample,
    TrajectoryStep,
    compute_mc_returns,
    normalize_terminal_rewards,
)
from .sample_tags import (
    ACTION_CLASS_LOOKUP,
    ACTION_CLASS_NAMES,
    EPISODE_MODE_NAMES,
    K_BUCKET_NAMES,
    OPP_GRID_TOP,
    OPPONENT_CHECKPOINT_BASE,
    Q_GAP_BUCKET_NAMES,
    REWARD_BUCKET_NAMES,
    collapse_action_type,
    k_bucket,
    k_bucket_array,
    q_gap_bucket,
    q_gap_bucket_array,
    reward_bucket,
    reward_bucket_array,
)

__all__ = [
    # buffer
    "BUCKET_NAMES",
    "ReplayBuffer",
    "RoleAwareReplayBuffer",
    "collate_base_encoded",
    "collate_role_encoded",
    # returns
    "TERMINAL_REWARD_SCALE",
    "TrainSample",
    "TrajectoryStep",
    "compute_mc_returns",
    "normalize_terminal_rewards",
    # sample_tags
    "ACTION_CLASS_LOOKUP",
    "ACTION_CLASS_NAMES",
    "EPISODE_MODE_NAMES",
    "K_BUCKET_NAMES",
    "OPP_GRID_TOP",
    "OPPONENT_CHECKPOINT_BASE",
    "Q_GAP_BUCKET_NAMES",
    "REWARD_BUCKET_NAMES",
    "collapse_action_type",
    "k_bucket",
    "k_bucket_array",
    "q_gap_bucket",
    "q_gap_bucket_array",
    "reward_bucket",
    "reward_bucket_array",
]
