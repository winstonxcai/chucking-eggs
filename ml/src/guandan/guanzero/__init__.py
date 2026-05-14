"""GuanZero — Deep Monte Carlo training for Guan Dan.

Replicates *GuanZero: Mastering the Game of Guandan with Deep RL and
Behavior Regulating* (arXiv:2402.13582). See ``config.TrainConfig`` for
the hyperparameter surface and ``train.train`` for the entry point.
"""

from .actor import play_episode
from .buffer import (
    ReplayBuffer,
    RoleAwareReplayBuffer,
    collate_base_encoded,
    collate_grouped_encoded,
    collate_role_encoded,
)
from .config import (
    EpsilonConfig,
    InferenceConfig,
    QNetConfig,
    TrainConfig,
    shared_head_qnet_config,
)
from .encoder import (
    CARD_ID_DIM,
    HISTORY_LEN,
    card_to_id,
    id_to_card,
)
from .encoding.base_encoder import (
    ENCODE_ACTION_KEYS,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    ENCODE_STATE_KEYS,
    StateActionEncoder,
    static_dim,
)
from .encoding.role_encoder import (
    REL_NEXT_OPP,
    REL_PARTNER,
    REL_PREV_OPP,
    REL_SELF,
    RoleAwareStateActionEncoder,
    absolute_player,
    relative_role,
)
from .q_network import GuanZeroQNet, SharedHeadQNet, SharedHeadQNetConfig, init_seat_nets
from .returns import compute_mc_returns

__all__ = [
    # config
    "TrainConfig",
    "QNetConfig",
    "EpsilonConfig",
    "InferenceConfig",
    "shared_head_qnet_config",
    # encoder
    "StateActionEncoder",
    "ENCODE_CHANNEL_SHAPES",
    "ENCODE_CHANNEL_KEYS",
    "ENCODE_STATE_KEYS",
    "ENCODE_ACTION_KEYS",
    "CARD_ID_DIM",
    "HISTORY_LEN",
    "card_to_id",
    "id_to_card",
    "static_dim",
    "RoleAwareStateActionEncoder",
    "relative_role",
    "absolute_player",
    "REL_SELF",
    "REL_NEXT_OPP",
    "REL_PARTNER",
    "REL_PREV_OPP",
    # network
    "GuanZeroQNet",
    "SharedHeadQNet",
    "SharedHeadQNetConfig",
    "init_seat_nets",
    # data
    "ReplayBuffer",
    "RoleAwareReplayBuffer",
    "collate_base_encoded",
    "collate_grouped_encoded",
    "collate_role_encoded",
    "compute_mc_returns",
    # actor
    "play_episode",
]
