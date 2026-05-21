"""Named schema for DART learner loss-bucket metrics.

The learner emits diagnostic JSON metrics every update. This module is the
contract for those keys: it names each axis, defines the integer values stored
in replay tags, and generates the stable metric prefixes used by
``loss_buckets.py`` and ``learner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from ...data.sample_tags import ACTION_CLASS_NAMES, OPP_GRID_TOP, OPPONENT_BY_NAME


class Phase(IntEnum):
    OPENING = 0
    MIDGAME = 1
    ENDGAME = 2


class PartnerPhase(IntEnum):
    OPENING = 0
    MIDGAME = 1
    ENDGAME = 2
    OUT = 3


class TrickRole(IntEnum):
    LEADING = 0
    FOLLOWING_PARTNER_ALIVE = 1
    FOLLOWING_PARTNER_OUT = 2


class EpisodeMode(IntEnum):
    SELF_PLAY = 0
    VS_CHECKPOINT = 1
    VS_HARD_BOT = 2


class Binary(IntEnum):
    NO = 0
    YES = 1


class LatestTeam(IntEnum):
    EVEN_SEATS = 0
    ODD_SEATS = 1


class KBucket(IntEnum):
    FORCED = 0
    NARROW = 1
    MODERATE = 2
    WIDE = 3


class QGapBucket(IntEnum):
    UNDEFINED = 0
    PIVOTAL = 1
    MODERATE = 2
    CONFIDENT = 3


class RewardBucket(IntEnum):
    BIG_LOSS = 0
    SMALL_LOSS = 1
    SMALL_WIN = 2
    BIG_WIN = 3


@dataclass(frozen=True)
class AxisSpec:
    name: str
    labels: tuple[str, ...]

    @classmethod
    def from_enum(cls, name: str, enum_cls: type[IntEnum]) -> AxisSpec:
        values = sorted(enum_cls, key=int)
        return cls(name=name, labels=tuple(v.name.lower() for v in values))


@dataclass(frozen=True)
class GridSpec:
    name: str
    axis_a: AxisSpec
    axis_b: AxisSpec

    @property
    def key_prefix(self) -> str:
        return f"{self.name}_"

    def key(self, a: int, b: int, stat: str) -> str:
        return f"{self.name}_{self.axis_a.labels[a]}__{self.axis_b.labels[b]}_{stat}"


@dataclass(frozen=True)
class MarginalSpec:
    name: str
    axis: AxisSpec

    @property
    def key_prefix(self) -> str:
        return f"{self.name}_"

    def key(self, level: int, stat: str) -> str:
        return f"{self.name}_{self.axis.labels[level]}_{stat}"


def _opponent_grid_labels() -> tuple[str, ...]:
    name_by_id = {value: name for name, value in OPPONENT_BY_NAME.items()}
    labels: list[str] = []
    for opponent_id in OPP_GRID_TOP:
        if opponent_id == 0:
            labels.append("self_play")
        else:
            labels.append(name_by_id.get(opponent_id, f"opponent_{opponent_id}"))
    return tuple(labels)


PHASE_AXIS = AxisSpec.from_enum("phase", Phase)
PARTNER_PHASE_AXIS = AxisSpec.from_enum("partner_phase", PartnerPhase)
TRICK_ROLE_AXIS = AxisSpec.from_enum("trick_role", TrickRole)
EPISODE_MODE_AXIS = AxisSpec.from_enum("episode_mode", EpisodeMode)
OPPONENT_AXIS = AxisSpec("opponent", _opponent_grid_labels())
ACTION_CLASS_AXIS = AxisSpec(
    "action_class",
    tuple(ACTION_CLASS_NAMES[i] for i in sorted(ACTION_CLASS_NAMES)),
)
BINARY_AXIS = AxisSpec.from_enum("binary", Binary)
LATEST_TEAM_AXIS = AxisSpec.from_enum("latest_team", LatestTeam)
K_BUCKET_AXIS = AxisSpec.from_enum("k_bucket", KBucket)
Q_GAP_BUCKET_AXIS = AxisSpec.from_enum("q_gap_bucket", QGapBucket)
REWARD_BUCKET_AXIS = AxisSpec.from_enum("reward_bucket", RewardBucket)


LOSS_BUCKET_GRIDS = (
    GridSpec("phase_role", PHASE_AXIS, TRICK_ROLE_AXIS),
    GridSpec("phase_pair", PHASE_AXIS, PARTNER_PHASE_AXIS),
    GridSpec("source_phase", EPISODE_MODE_AXIS, PHASE_AXIS),
    GridSpec("opp_phase", OPPONENT_AXIS, PHASE_AXIS),
    GridSpec("action_phase", ACTION_CLASS_AXIS, PHASE_AXIS),
)

LOSS_BUCKET_MARGINALS = (
    MarginalSpec("epsilon", BINARY_AXIS),
    MarginalSpec("is_pass", BINARY_AXIS),
    MarginalSpec("is_bomb", BINARY_AXIS),
    MarginalSpec("k_bucket", K_BUCKET_AXIS),
    MarginalSpec("q_gap", Q_GAP_BUCKET_AXIS),
    MarginalSpec("team", LATEST_TEAM_AXIS),
    MarginalSpec("reward", REWARD_BUCKET_AXIS),
)

LOSS_BUCKET_SPECS = (*LOSS_BUCKET_GRIDS, *LOSS_BUCKET_MARGINALS)
PHASE_KEY_PREFIXES = tuple(spec.key_prefix for spec in LOSS_BUCKET_SPECS)

LOSS_BUCKET_SCHEMA = {
    spec.name: {
        "kind": "grid",
        "axis_a": spec.axis_a.labels,
        "axis_b": spec.axis_b.labels,
        "key_format": f"{spec.name}_<axis_a_label>__<axis_b_label>_<stat>",
        "stats": ("n", "frac", "loss"),
    }
    for spec in LOSS_BUCKET_GRIDS
} | {
    spec.name: {
        "kind": "marginal",
        "axis": spec.axis.labels,
        "key_format": f"{spec.name}_<axis_label>_<stat>",
        "stats": ("n", "frac", "loss"),
    }
    for spec in LOSS_BUCKET_MARGINALS
}


__all__ = [
    "AxisSpec",
    "Binary",
    "EpisodeMode",
    "GridSpec",
    "KBucket",
    "LOSS_BUCKET_GRIDS",
    "LOSS_BUCKET_MARGINALS",
    "LOSS_BUCKET_SCHEMA",
    "LOSS_BUCKET_SPECS",
    "LatestTeam",
    "MarginalSpec",
    "PHASE_KEY_PREFIXES",
    "PartnerPhase",
    "Phase",
    "QGapBucket",
    "RewardBucket",
    "TrickRole",
]
