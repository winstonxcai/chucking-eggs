"""Actor-side rollout, policies, opponent sampling, and runtime setup."""

from .opponents import OpponentPools
from .rollout import (
    LaneConfig,
    SeatPolicy,
    all_latest_seats,
    argmax_q,
    argmax_q_batched,
    play_episode,
    play_episodes_batched,
    select_legal,
)
from .runtime import (
    ActorRuntime,
    build_actor_runtime,
    jitter_sync_threshold,
    maybe_sync_weights,
    sync_actor_weights,
)
from .samples import ActorSampleAccumulator, QueueBatchMeta

__all__ = [
    "ActorRuntime",
    "ActorSampleAccumulator",
    "LaneConfig",
    "OpponentPools",
    "QueueBatchMeta",
    "SeatPolicy",
    "all_latest_seats",
    "argmax_q",
    "argmax_q_batched",
    "build_actor_runtime",
    "jitter_sync_threshold",
    "maybe_sync_weights",
    "play_episode",
    "play_episodes_batched",
    "select_legal",
    "sync_actor_weights",
]
