"""Actor runtime construction and weight synchronization."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping
from pathlib import Path

import torch

from ...config import MODEL_TYPE_DART, dart_qnet_config
from ...constants import NUM_PLAYERS
from ...model.encoding.base_encoder import ENCODE_CHANNEL_KEYS, StateActionEncoder
from ...model.encoding.role_encoder import RoleAwareStateActionEncoder
from ...model.q_network import DartQNet, GuanZeroQNet, init_guanzero_nets

ActorNetBundle = dict[int, GuanZeroQNet] | DartQNet


@dataclasses.dataclass
class ActorRuntime:
    encoder: StateActionEncoder | RoleAwareStateActionEncoder
    q_nets: ActorNetBundle
    q_nets_actor: ActorNetBundle
    dart_path: bool
    use_int8: bool
    channel_keys: tuple[str, ...]
    include_players: bool

    def refresh_actor_nets(self) -> None:
        if self.use_int8:
            self.q_nets_actor = _quantize_for_actor(self.q_nets, self.dart_path)


def build_actor_runtime(cfg) -> ActorRuntime:
    dart_path = cfg.model_type == MODEL_TYPE_DART
    if dart_path:
        encoder = RoleAwareStateActionEncoder(
            is_partner_visible=cfg.qnet.is_partner_visible,
        )
    else:
        encoder = StateActionEncoder(is_partner_visible=cfg.qnet.is_partner_visible)

    if dart_path:
        q_nets = DartQNet(dart_qnet_config(cfg))
        q_nets.eval()
    else:
        q_nets = init_guanzero_nets(cfg.qnet)
        for net in q_nets.values():
            net.eval()

    use_int8 = bool(getattr(cfg, "use_int8_actor", False))
    q_nets_actor = (
        _quantize_for_actor(q_nets, dart_path)
        if use_int8
        else q_nets
    )
    return ActorRuntime(
        encoder=encoder,
        q_nets=q_nets,
        q_nets_actor=q_nets_actor,
        dart_path=dart_path,
        use_int8=use_int8,
        channel_keys=tuple(encoder.channel_keys) if dart_path else tuple(ENCODE_CHANNEL_KEYS),
        include_players=not dart_path,
    )


def sync_actor_weights(
    runtime: ActorRuntime,
    weight_dir: Path,
    *,
    local_version: int,
    local_updates: int,
    sync_threshold: int,
) -> tuple[int, int, int]:
    return maybe_sync_weights(
        runtime.q_nets, weight_dir, local_version, local_updates, sync_threshold,
    )


def maybe_sync_weights(
    q_nets,
    weight_dir: Path,
    local_version: int,
    local_updates: int = 0,
    sync_interval_updates: int = 0,
) -> tuple[int, int, int]:
    """Conditionally load newer actor weights."""
    from ..weights import load_latest_weights, read_latest_metadata

    meta = read_latest_metadata(weight_dir)
    if meta is None:
        return local_version, local_updates, 0
    latest_version, latest_updates = meta

    if latest_version <= local_version:
        return local_version, local_updates, latest_updates

    if local_version >= 0 and sync_interval_updates > 0 \
            and (latest_updates - local_updates) < sync_interval_updates:
        return local_version, local_updates, latest_updates

    snapshot = load_latest_weights(weight_dir)
    if snapshot is None or snapshot.version <= local_version:
        return local_version, local_updates, latest_updates

    if isinstance(q_nets, Mapping):
        for p in range(NUM_PLAYERS):
            q_nets[p].load_state_dict(snapshot.state_dicts[p])
            q_nets[p].eval()
    else:
        q_nets.load_state_dict(snapshot.state_dicts["q_net"])
        q_nets.eval()

    return snapshot.version, snapshot.updates, snapshot.updates


def jitter_sync_threshold(cfg, rng: random.Random) -> int:
    """Return sync_interval_updates plus per-actor uniform jitter."""
    if cfg.sync_jitter_updates <= 0:
        return cfg.sync_interval_updates
    return cfg.sync_interval_updates + rng.randint(
        -cfg.sync_jitter_updates, cfg.sync_jitter_updates,
    )


def _quantize_for_actor(nets, dart_path: bool):
    import torch.ao.quantization as qao

    if dart_path:
        return qao.quantize_dynamic(nets, {torch.nn.Linear}, dtype=torch.qint8)
    return {
        player: qao.quantize_dynamic(nets[player], {torch.nn.Linear}, dtype=torch.qint8)
        for player in range(NUM_PLAYERS)
    }


__all__ = [
    "ActorRuntime",
    "build_actor_runtime",
    "jitter_sync_threshold",
    "maybe_sync_weights",
    "sync_actor_weights",
]
