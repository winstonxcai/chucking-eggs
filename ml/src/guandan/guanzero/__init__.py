"""GuanZero — Deep Monte Carlo training for Guan Dan.

Replicates *GuanZero: Mastering the Game of Guandan with Deep RL and
Behavior Regulating* (arXiv:2402.13582). See ``config.TrainConfig`` for
the hyperparameter surface and ``train.train`` for the entry point.
"""

from .actor import play_episode
from .buffer import ReplayBuffer, collate_encoded
from .config import EpsilonConfig, InferenceConfig, QNetConfig, TrainConfig
from .encoder import (
    CARD_ID_DIM,
    ENCODE_CHANNEL_KEYS,
    ENCODE_CHANNEL_SHAPES,
    HISTORY_LEN,
    StateActionEncoder,
    card_to_id,
    id_to_card,
    static_dim,
)
from .q_network import GuanZeroQNet, init_seat_nets
from .returns import compute_mc_returns

__all__ = [
    # config
    "TrainConfig",
    "QNetConfig",
    "EpsilonConfig",
    "InferenceConfig",
    # encoder
    "StateActionEncoder",
    "ENCODE_CHANNEL_SHAPES",
    "ENCODE_CHANNEL_KEYS",
    "CARD_ID_DIM",
    "HISTORY_LEN",
    "card_to_id",
    "id_to_card",
    "static_dim",
    # network
    "GuanZeroQNet",
    "init_seat_nets",
    # data
    "ReplayBuffer",
    "collate_encoded",
    "compute_mc_returns",
    # actor
    "play_episode",
]
