"""GuanZeroBot — eval adapter that wraps the four trained Q-networks.

Plug into the existing eval harness via:

    from guandan.guanzero.agent import GuanZeroBot
    bot = GuanZeroBot.load("ml/runs/<run>/checkpoints/final.pt")
    bot.act(env, player) -> Combo
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..agents.base import Agent
from ..combos import Combo
from ..game import GuanDanEnv
from .buffer import collate_encoded
from .encoder import StateActionEncoder
from .legal_utils import dedup_strategic
from .q_network import init_seat_nets


class GuanZeroBot(Agent):
    def __init__(
        self,
        q_nets: dict[int, torch.nn.Module],
        encoder: StateActionEncoder,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.q_nets = {p: q_nets[p].to(self.device).eval() for p in range(4)}
        self.encoder = encoder

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "GuanZeroBot":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        q_nets = init_seat_nets(
            hidden_lstm=cfg["hidden_lstm"],
            hidden_mlp=cfg["hidden_mlp"],
            n_mlp_layers=cfg["n_mlp_layers"],
            dropout=cfg.get("dropout", 0.0),
            use_oracle_others_hand=cfg.get("use_oracle_others_hand", True),
        )
        for p in range(4):
            q_nets[p].load_state_dict(ckpt["q_nets"][p])
        encoder = StateActionEncoder(
            use_oracle_others_hand=cfg.get("use_oracle_others_hand", True)
        )
        return cls(q_nets=q_nets, encoder=encoder, device=device)

    @torch.no_grad()
    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = dedup_strategic(env.legal_moves(player))
        encoded = self.encoder.encode_all(env, player, legal)
        batch = collate_encoded(encoded, device=self.device)
        q = self.q_nets[player](batch)
        return legal[int(q.argmax().item())]


__all__ = ["GuanZeroBot"]
