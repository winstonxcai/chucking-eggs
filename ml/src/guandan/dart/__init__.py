"""Dart — Deep Monte Carlo training for Guan Dan.

Replicates *Dart: Mastering the Game of Guandan with Deep RL and
Behavior Regulating* (arXiv:2402.13582). See ``config.TrainConfig`` for
the hyperparameter surface and ``train.train`` for the entry point.
"""

from .runtime.actor import (
    LaneConfig,
    SeatPolicy,
    all_latest_seats,
    play_episode,
    play_episodes_batched,
)
from .constants import NUM_PLAYERS, OTHER_PLAYER_OFFSETS, PARTNER_OFFSET
from .data.buffer import (
    ReplayBuffer,
    RoleAwareReplayBuffer,
    collate_base_encoded,
    collate_grouped_encoded,
    collate_role_encoded,
)
from .config import (
    MODEL_TYPE_DART,
    MODEL_TYPE_GUANZERO,
    dart_qnet_config,
    EpsilonConfig,
    QNetConfig,
    TrainConfig,
)
from .model.encoder import (
    CARD_ID_DIM,
    HISTORY_LEN,
    card_to_id,
    id_to_card,
)
from .model.encoding.base_encoder import (
    ENCODE_ACTION_KEYS,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    ENCODE_STATE_KEYS,
    StateActionEncoder,
    static_dim,
)
from .model.encoding.role_encoder import (
    REL_NEXT_OPP,
    REL_PARTNER,
    REL_PREV_OPP,
    REL_SELF,
    RoleAwareStateActionEncoder,
    absolute_player,
    relative_role,
)
from .model.q_network import DartQNet, DartQNetConfig, GuanZeroQNet, init_guanzero_nets
from .data.returns import compute_mc_returns

__all__ = [
    # config / production DART
    "NUM_PLAYERS",
    "PARTNER_OFFSET",
    "OTHER_PLAYER_OFFSETS",
    "TrainConfig",
    "EpsilonConfig",
    "MODEL_TYPE_DART",
    "dart_qnet_config",
    "CARD_ID_DIM",
    "HISTORY_LEN",
    "card_to_id",
    "id_to_card",
    "RoleAwareStateActionEncoder",
    "relative_role",
    "absolute_player",
    "REL_SELF",
    "REL_NEXT_OPP",
    "REL_PARTNER",
    "REL_PREV_OPP",
    "DartQNet",
    "DartQNetConfig",
    "RoleAwareReplayBuffer",
    "collate_grouped_encoded",
    "collate_role_encoded",
    "compute_mc_returns",
    "LaneConfig",
    "SeatPolicy",
    "all_latest_seats",
    "play_episode",
    "play_episodes_batched",
    # GuanZero comparison baseline
    "MODEL_TYPE_GUANZERO",
    "QNetConfig",
    "StateActionEncoder",
    "ENCODE_CHANNEL_SHAPES",
    "ENCODE_CHANNEL_KEYS",
    "ENCODE_STATE_KEYS",
    "ENCODE_ACTION_KEYS",
    "static_dim",
    "GuanZeroQNet",
    "init_guanzero_nets",
    "ReplayBuffer",
    "collate_base_encoded",
]
