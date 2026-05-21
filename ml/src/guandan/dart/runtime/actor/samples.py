"""Actor-side sample accumulation and queue message packing."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ....agents import AGENT_REGISTRY
from ...data.returns import TrainSample
from ...data.sample_tags import EPISODE_MODE_VS_HARD_BOT
from ...model.encoding.role_encoder import RoleAwareStateActionEncoder
from .rollout import LaneConfig


_COORD_TARGET_IDS: frozenset[int] = frozenset(
    cls.sample_tag for cls in AGENT_REGISTRY.values() if cls.coord_target
)

_TAG_DTYPES: dict[str, np.dtype] = {
    "phase_self":        np.dtype(np.int8),
    "trick_role":        np.dtype(np.int8),
    "phase_partner":     np.dtype(np.int8),
    "action_type":       np.dtype(np.int8),
    "is_pass":           np.dtype(np.int8),
    "is_bomb":           np.dtype(np.int8),
    "bomb_available":    np.dtype(np.int8),
    "num_legal_actions": np.dtype(np.int16),
    "q_gap":             np.dtype(np.float32),
    "chosen_by_epsilon": np.dtype(np.int8),
    "episode_mode":      np.dtype(np.int8),
    "opponent_id":       np.dtype(np.int8),
    "latest_team":       np.dtype(np.int8),
    "terminal_reward":   np.dtype(np.float32),
}


@dataclasses.dataclass(frozen=True)
class QueueBatchMeta:
    actor_id: int
    version: int
    local_updates: int
    global_updates: int
    actor_rng_state: dict[str, Any] | None = None


class ActorSampleAccumulator:
    """Accumulate TrainSamples and emit pre-stacked queue messages."""

    def __init__(self, *, channel_keys: tuple[str, ...], include_players: bool) -> None:
        self._channel_keys = channel_keys
        self._include_players = include_players
        self._encoded: list[dict] = []
        self._players: list[int] = []
        self._returns: list[float] = []
        self._buckets: list[int] = []
        self._tags: dict[str, list[Any]] = {name: [] for name in _TAG_DTYPES}

    def __len__(self) -> int:
        return len(self._encoded)

    def append_lane(self, lane: LaneConfig, samples: list[TrainSample]) -> None:
        if not samples:
            return

        is_hard_bot_ep = lane.tags.mode == EPISODE_MODE_VS_HARD_BOT
        target_bot = lane.tags.opponent_id in _COORD_TARGET_IDS
        latest_won = samples[0].mc_return > 0
        is_coord_episode = is_hard_bot_ep and target_bot and not latest_won
        total = len(samples)

        for i, sample in enumerate(samples):
            self._encoded.append(sample.encoded)
            self._players.append(sample.player)
            self._returns.append(sample.mc_return)
            self._tags["phase_self"].append(sample.phase_self)
            self._tags["trick_role"].append(sample.trick_role)
            self._tags["phase_partner"].append(sample.phase_partner)
            self._tags["action_type"].append(sample.action_type)
            self._tags["is_pass"].append(sample.is_pass)
            self._tags["is_bomb"].append(sample.is_bomb)
            self._tags["bomb_available"].append(sample.bomb_available)
            self._tags["num_legal_actions"].append(sample.num_legal_actions)
            self._tags["q_gap"].append(sample.q_gap)
            self._tags["chosen_by_epsilon"].append(sample.chosen_by_epsilon)
            self._tags["episode_mode"].append(sample.episode_mode)
            self._tags["opponent_id"].append(sample.opponent_id)
            self._tags["latest_team"].append(sample.latest_team)
            self._tags["terminal_reward"].append(sample.terminal_reward)
            self._buckets.append(_sample_bucket(sample, i, total, is_hard_bot_ep, is_coord_episode))

    def pop_message(self, batch_size: int, meta: QueueBatchMeta) -> dict | None:
        if len(self) < batch_size:
            return None

        src = self._encoded[:batch_size]
        stacked = {
            key: np.stack([encoded[key] for encoded in src], axis=0)
            for key in self._channel_keys
        }
        msg = {
            "actor_id":       meta.actor_id,
            "version":        meta.version,
            "local_updates":  meta.local_updates,
            "global_updates": meta.global_updates,
            "stacked":        stacked,
            "returns":        np.asarray(self._returns[:batch_size], dtype=np.float32),
            "buckets":        np.asarray(self._buckets[:batch_size], dtype=np.int8),
        }
        if meta.actor_rng_state is not None:
            msg["actor_rng_state"] = meta.actor_rng_state
        for name, dtype in _TAG_DTYPES.items():
            msg[name] = np.asarray(self._tags[name][:batch_size], dtype=dtype)
        if self._include_players:
            msg["players"] = np.asarray(self._players[:batch_size], dtype=np.int8)

        self._drop_first(batch_size)
        return msg

    def _drop_first(self, n: int) -> None:
        del self._encoded[:n]
        del self._players[:n]
        del self._returns[:n]
        del self._buckets[:n]
        for values in self._tags.values():
            del values[:n]


def _sample_bucket(
    sample: TrainSample,
    sample_idx: int,
    total_samples: int,
    is_hard_bot_ep: bool,
    is_coord_episode: bool,
) -> int:
    if not is_hard_bot_ep:
        return 0
    if not is_coord_episode:
        return 1

    self_cards, partner_cards, next_opp_cards, prev_opp_cards = (
        RoleAwareStateActionEncoder.decode_card_counts(sample.encoded)
    )
    min_opp_cards = min(next_opp_cards, prev_opp_cards)
    is_final_third = sample_idx >= total_samples * 2 // 3
    is_coord = (
        self_cards <= 5
        or partner_cards <= 5
        or min_opp_cards <= 5
        or is_final_third
    )
    return 2 if is_coord else 1


__all__ = ["ActorSampleAccumulator", "QueueBatchMeta"]
