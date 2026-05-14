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
from .data.buffer import collate_base_encoded, collate_role_encoded
from .checkpoint import migrate_state_dict
from .config import TrainConfig, shared_head_qnet_config, shared_trick_head_qnet_config
from .encoding.base_encoder import StateActionEncoder
from .encoding.role_encoder import RoleAwareStateActionEncoder
from .utils.legal_utils import dedup_strategic
from .q_network import SharedHeadQNet, SharedTrickHeadQNet, init_seat_nets


class GuanZeroBot(Agent):
    def __init__(
        self,
        q_nets: dict[int, torch.nn.Module] | SharedHeadQNet | SharedTrickHeadQNet,
        encoder: StateActionEncoder | RoleAwareStateActionEncoder,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if isinstance(q_nets, (SharedHeadQNet, SharedTrickHeadQNet)):
            self.q_nets: dict[int, torch.nn.Module] | SharedHeadQNet | SharedTrickHeadQNet = (
                q_nets.to(self.device).eval()
            )
        else:
            self.q_nets = {p: q_nets[p].to(self.device).eval() for p in range(4)}
        self.encoder = encoder

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "GuanZeroBot":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = TrainConfig.from_flat_dict(ckpt["config"])
        if cfg.model_type == "shared_trick_heads":
            q_net = SharedTrickHeadQNet(shared_trick_head_qnet_config(cfg))
            q_net.load_state_dict(ckpt["q_net"])
            encoder = RoleAwareStateActionEncoder(
                is_partner_visible=cfg.qnet.is_partner_visible,
                head_scheme="trick_relative",
            )
            return cls(q_nets=q_net, encoder=encoder, device=device)
        if cfg.model_type == "shared_heads" or "q_net" in ckpt:
            q_net = SharedHeadQNet(shared_head_qnet_config(cfg))
            q_net.load_state_dict(ckpt["q_net"])
            encoder = RoleAwareStateActionEncoder(
                is_partner_visible=cfg.qnet.is_partner_visible,
            )
            return cls(q_nets=q_net, encoder=encoder, device=device)
        q_nets = init_seat_nets(cfg.qnet)
        for p in range(4):
            q_nets[p].load_state_dict(migrate_state_dict(ckpt["q_nets"][p]))
        encoder = StateActionEncoder(is_partner_visible=cfg.qnet.is_partner_visible)
        return cls(q_nets=q_nets, encoder=encoder, device=device)

    @torch.no_grad()
    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = dedup_strategic(env.legal_moves(player))
        encoded = self.encoder.encode_all(env, player, legal)
        if isinstance(self.q_nets, (SharedHeadQNet, SharedTrickHeadQNet)):
            state_batch, action_batch, repeats = collate_role_encoded(
                [encoded],
                device=self.device,
            )
            q = self.q_nets.forward_grouped(state_batch, action_batch, repeats)
            return legal[int(q.argmax().item())]
        state_batch, action_batch, repeats = collate_base_encoded(
            [encoded],
            device=self.device,
        )
        assert isinstance(self.q_nets, dict)
        q = self.q_nets[player].forward_grouped(state_batch, action_batch, repeats)
        return legal[int(q.argmax().item())]


__all__ = ["GuanZeroBot"]
