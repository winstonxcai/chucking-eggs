"""Neural network architectures + serialization.

Q-networks (per-seat and shared-head variants), the legacy state/action
encoder shared constants, and checkpoint save/load helpers. The
``encoding/`` subpackage hosts the active per-decision feature builders.
"""

from .checkpoint import (
    WeightSnapshot,
    load_checkpoint,
    load_frozen_shared_qnet,
    load_frozen_trick_qnet,
    save_checkpoint_base,
    save_checkpoint_shared,
)
from .encoder import (
    BEHAVIOR_DIM,
    HISTORY_LEN,
    LEVEL_DIM,
    RANK_BUCKETS,
    compute_state_behavior_flags,
)
from .q_network import (
    GuanZeroQNet,
    SharedHeadQNet,
    SharedHeadQNetConfig,
    SharedTrickHeadQNet,
    SharedTrickHeadQNetConfig,
    init_seat_nets,
)

__all__ = [
    # q_network
    "GuanZeroQNet",
    "SharedHeadQNet",
    "SharedHeadQNetConfig",
    "SharedTrickHeadQNet",
    "SharedTrickHeadQNetConfig",
    "init_seat_nets",
    # encoder constants
    "BEHAVIOR_DIM",
    "HISTORY_LEN",
    "LEVEL_DIM",
    "RANK_BUCKETS",
    "compute_state_behavior_flags",
    # checkpoint
    "WeightSnapshot",
    "load_checkpoint",
    "load_frozen_shared_qnet",
    "load_frozen_trick_qnet",
    "save_checkpoint_base",
    "save_checkpoint_shared",
]
