"""GuanZero (M0 baseline) — paper-faithful Deep Monte Carlo.

Replicates *GuanZero: Mastering the Game of Guandan with Deep RL and
Behavior Regulating* (arXiv:2402.13582) as a clean baseline. M1–M4
architecture extensions (transformer history, set encoders, role-aware
encoder, belief encoder) will follow as separate PRs that swap individual
encoder channels and the network module.
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
