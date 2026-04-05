"""GuanZero agent wrapper — plugs into the existing agent interface.

Supports both 4-network checkpoints (one per seat) and legacy single-network
checkpoints (replicated to all 4 seats for backwards compatibility).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..cards import Rank
from ..training.guanzero_encoding import score_all_actions
from ..training.guanzero_network import GuanZeroNetwork


class GuanZeroBot:
    """Wraps 4 trained GuanZeroNetworks (one per seat) as an Agent.

    Usage:
        bot = GuanZeroBot("checkpoints/guanzero_best.pt")
        action = bot.act(env, player)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        level_rank: int = Rank.TWO,
    ) -> None:
        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device
        self.level_rank = level_rank

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "nets" in ckpt:
            # 4-network checkpoint
            self.nets = [GuanZeroNetwork().to(device) for _ in range(4)]
            for i, sd in enumerate(ckpt["nets"]):
                self.nets[i].load_state_dict(sd)
        else:
            # Legacy single-network checkpoint — replicate to all 4 seats
            net = GuanZeroNetwork().to(device)
            net.load_state_dict(ckpt.get("state_dict", ckpt))
            self.nets = [net] * 4

        for net in self.nets:
            net.eval()

    def act(self, env, player: int):
        """Choose the highest-Q legal action using this seat's network."""
        legal = env.legal_moves()
        if len(legal) == 1:
            return legal[0]

        q_values = score_all_actions(
            self.nets[player], env, player, legal, self.level_rank, self.device
        )
        return legal[int(q_values.argmax().item())]
