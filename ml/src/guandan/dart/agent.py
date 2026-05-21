"""DartBot — eval adapter for DART and GuanZero comparison checkpoints.

Plug into the existing eval harness via:

    from guandan.dart.agent import DartBot
    bot = DartBot.load("ml/runs/<run>/checkpoints/final.pt")
    bot.act(env, player) -> Combo
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..agents.base import Agent
from ..combos import Combo
from ..game import GuanDanEnv
from .config import (
    _REMOVED_OPPONENT_KEYS,
    MODEL_TYPE_DART,
    TrainConfig,
    dart_qnet_config,
)
from .constants import NUM_PLAYERS
from .data.buffer import collate_base_encoded, collate_role_encoded
from .model.encoding.base_encoder import StateActionEncoder
from .model.encoding.role_encoder import RoleAwareStateActionEncoder
from .model.q_network import DartQNet, init_guanzero_nets
from .utils.legal_utils import select_legal

logger = logging.getLogger(__name__)


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
            self.q_nets = {
                p: q_nets[p].to(self.device).eval()
                for p in range(NUM_PLAYERS)
            }
        self.encoder = encoder

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> DartBot:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        # Backward compatibility for historical research checkpoints. Runtime
        # YAML still rejects these keys; old checkpoint configs are read-only
        # provenance and should not block evaluation.
        removed = sorted(set(cfg_dict) & _REMOVED_OPPONENT_KEYS)
        if removed:
            logger.warning(
                "checkpoint %s uses removed opponent config key(s) %s; "
                "dropping them for evaluation compatibility",
                path, removed,
            )
        for key in _REMOVED_OPPONENT_KEYS:
            cfg_dict.pop(key, None)
        migrated: list[str] = []
        for old_key, new_key in {
            "shared_head_role_d_model": "dart_role_d_model",
            "shared_head_history_hidden": "dart_history_hidden",
            "shared_head_global_hidden": "dart_global_hidden",
            "shared_head_action_hidden": "dart_action_hidden",
            "shared_head_trunk_hidden": "dart_trunk_hidden",
            "shared_head_trunk_layers": "dart_trunk_layers",
        }.items():
            if old_key in cfg_dict and new_key not in cfg_dict:
                cfg_dict[new_key] = cfg_dict[old_key]
                migrated.append(f"{old_key}->{new_key}")
            cfg_dict.pop(old_key, None)
        if migrated:
            logger.warning(
                "checkpoint %s uses deprecated Dart config key(s): %s",
                path, ", ".join(migrated),
            )
        for stale_key in ("fresh_optimizer", "fresh_replay"):
            if stale_key in cfg_dict:
                logger.warning(
                    "checkpoint %s uses deprecated config key %s; dropping it",
                    path, stale_key,
                )
                cfg_dict.pop(stale_key, None)
        if cfg_dict.get("model_type") == "shared_trick_heads":
            logger.warning(
                "checkpoint %s uses deprecated model_type='shared_trick_heads'; "
                "loading as %s",
                path, MODEL_TYPE_DART,
            )
            cfg_dict["model_type"] = MODEL_TYPE_DART
        if isinstance(cfg_dict.get("epsilon"), dict):
            cfg_dict["epsilon"] = dict(cfg_dict["epsilon"])
            if "frozen" in cfg_dict["epsilon"]:
                logger.warning(
                    "checkpoint %s uses deprecated epsilon.frozen; dropping it",
                    path,
                )
                cfg_dict["epsilon"].pop("frozen", None)
        cfg = TrainConfig.from_flat_dict(cfg_dict)
        if cfg.model_type == MODEL_TYPE_DART:
            q_net = DartQNet(dart_qnet_config(cfg))
            q_net.load_state_dict(ckpt["q_net"])
            encoder = RoleAwareStateActionEncoder(
                is_partner_visible=cfg.qnet.is_partner_visible,
            )
            return cls(q_nets=q_net, encoder=encoder, device=device)
        q_nets = init_guanzero_nets(cfg.qnet)
        for p in range(NUM_PLAYERS):
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
