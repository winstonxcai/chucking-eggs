"""GuanZero — Deep Monte Carlo training for Guan Dan.

Replicates *GuanZero: Mastering the Game of Guandan with Deep RL and
Behavior Regulating* (arXiv:2402.13582). See ``config.TrainConfig`` for
the hyperparameter surface and ``train.train`` for the entry point.
"""

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
from .q_network import GuanZeroQNet
from .returns import compute_mc_returns
from .buffer import ReplayBuffer, collate_encoded

__all__ = [
    "CARD_ID_DIM",
    "ENCODE_CHANNEL_KEYS",
    "ENCODE_CHANNEL_SHAPES",
    "HISTORY_LEN",
    "StateActionEncoder",
    "card_to_id",
    "id_to_card",
    "static_dim",
    "GuanZeroQNet",
    "compute_mc_returns",
    "ReplayBuffer",
    "collate_encoded",
]
