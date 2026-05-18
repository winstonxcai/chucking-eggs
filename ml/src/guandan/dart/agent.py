"""DartBot — eval adapter for DART and GuanZero comparison checkpoints.

Plug into the existing eval harness via:

    from guandan.dart.agent import DartBot
    bot = DartBot.load("ml/runs/<run>/checkpoints/final.pt")
    bot.act(env, player) -> Combo
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..agents.base import Agent
from ..combos import Combo
from ..game import GuanDanEnv
from .data.buffer import collate_base_encoded, collate_role_encoded
from .config import MODEL_TYPE_DART, TrainConfig, dart_qnet_config
from .model.encoding.base_encoder import StateActionEncoder
from .model.encoding.role_encoder import RoleAwareStateActionEncoder
from .utils.legal_utils import select_legal
from .model.q_network import DartQNet, init_guanzero_nets


class DartBot(Agent):
    label = "DART"
    description = "Trained DART or GuanZero comparison checkpoint."
    source = "In-house"
    color = "#111111"

    def __init__(
        self,
        q_nets: dict[int, torch.nn.Module] | DartQNet,
        encoder: StateActionEncoder | RoleAwareStateActionEncoder,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if isinstance(q_nets, DartQNet):
            self.q_nets: dict[int, torch.nn.Module] | DartQNet = (
                q_nets.to(self.device).eval()
            )
        else:
            self.q_nets = {p: q_nets[p].to(self.device).eval() for p in range(4)}
        self.encoder = encoder

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "DartBot":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = TrainConfig.from_flat_dict(ckpt["config"])
        if cfg.model_type == MODEL_TYPE_DART:
            q_net = DartQNet(dart_qnet_config(cfg))
            q_net.load_state_dict(ckpt["q_net"])
            encoder = RoleAwareStateActionEncoder(
                is_partner_visible=cfg.qnet.is_partner_visible,
            )
            return cls(q_nets=q_net, encoder=encoder, device=device)
        q_nets = init_guanzero_nets(cfg.qnet)
        for p in range(4):
            q_nets[p].load_state_dict(ckpt["q_nets"][p])
        encoder = StateActionEncoder(is_partner_visible=cfg.qnet.is_partner_visible)
        return cls(q_nets=q_nets, encoder=encoder, device=device)

    @torch.inference_mode()
    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = select_legal(env, player)
        encoded = self.encoder.encode_all(env, player, legal)
        if isinstance(self.q_nets, DartQNet):
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


__all__ = ["DartBot"]
