"""Training subpackage (Direction C): encoding + bare-MLP Q-network only.

Restored from archive selectively. QMIX, GNN, LSTM history, aux heads, replay
buffer, and self-play modules are intentionally NOT restored — they are on the
LOGBOOK failure list (§6 QMIX, §9 GNN, §13 §20 self-play regression).
"""

from .encoding import (
    ACTION_DIM,
    STATE_DIM,
    cards_to_matrix,
    encode_action,
    encode_state,
)
from .q_network import QNetwork, QValueNet, get_device

__all__ = [
    "ACTION_DIM",
    "STATE_DIM",
    "cards_to_matrix",
    "encode_action",
    "encode_state",
    "QNetwork",
    "QValueNet",
    "get_device",
]
