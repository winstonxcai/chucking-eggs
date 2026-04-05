"""GuanZero agent wrapper — plugs into the existing agent interface.

Supports both 4-network checkpoints (one per seat) and legacy single-network
checkpoints (replicated to all 4 seats for backwards compatibility).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..cards import Rank
from ..training.guanzero_encoding import (
    score_all_hybrid,
    STATE_DIM_HYBRID,
    ACTION_DIM_HYBRID,
    D_MOVE_HYBRID,
)


class GuanZeroBot:
    """Wraps 4 QNetworkLSTM models (one per seat, hybrid encoding) as an Agent.

    Checkpoint format: {"nets": [sd0, sd1, sd2, sd3], "level_rank": int}
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
    ) -> None:
        from ..training.q_network import QNetworkLSTM

        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device
        self.level_rank = level_rank

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        def _make_net():
            return QNetworkLSTM(
                d_state=STATE_DIM_HYBRID,
                d_action=ACTION_DIM_HYBRID,
                d_move=D_MOVE_HYBRID,
                lstm_hidden=128, hidden=512, n_layers=3,
            ).to(device)

        if "nets" in ckpt:
            self.nets = [_make_net() for _ in range(4)]
            for i, sd in enumerate(ckpt["nets"]):
                self.nets[i].load_state_dict(sd)
        else:
            net = _make_net()
            net.load_state_dict(ckpt.get("state_dict", ckpt))
            self.nets = [net] * 4

        for net in self.nets:
            net.eval()

    def act(self, env, player: int):
        legal = env.legal_moves()
        if len(legal) == 1:
            return legal[0]
        q = score_all_hybrid(self.nets[player], env, player, legal, self.level_rank, self.device)
        return legal[int(q.argmax().item())]
