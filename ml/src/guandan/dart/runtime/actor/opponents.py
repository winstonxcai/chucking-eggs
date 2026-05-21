"""Opponent pool loading and per-lane policy sampling."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

from ...config import MODEL_TYPE_DART, EpisodeMixConfig
from ...data.returns import EpisodeTags
from ...data.sample_tags import (
    EPISODE_MODE_SELF_PLAY,
    EPISODE_MODE_VS_CHECKPOINT,
    EPISODE_MODE_VS_HARD_BOT,
    OPPONENT_BY_NAME,
    OPPONENT_CHECKPOINT_BASE,
    OPPONENT_NONE,
)
from .rollout import SeatPolicies, SeatPolicy, all_latest_seats


@dataclasses.dataclass
class OpponentPools:
    """Frozen checkpoints, hard bots, sampling, and per-actor counters."""

    frozen_nets: tuple
    frozen_checkpoint_paths: tuple[str, ...]
    frozen_epsilon: float
    hard_bots_pool: tuple[tuple[str, object], ...]
    hard_bots_by_name: dict[str, object]
    hard_bot_weights: tuple[float, ...]
    pair_names: tuple[str, ...]
    pair_weights: tuple[float, ...]
    episode_mix: EpisodeMixConfig
    latest_learner_team_odd_probability: float
    mode_counts: dict[str, int]
    team_counts: dict[str, int]
    frozen_pick_counts: list[int]
    hard_bot_pick_counts: list[int]
    pair_pick_counts: dict[str, int]

    @classmethod
    def from_config(cls, cfg, logger) -> OpponentPools:
        opponents = cfg.opponents
        dart_path = cfg.model_type == MODEL_TYPE_DART

        frozen_nets: list = []
        checkpoint_paths = tuple(opponents.frozen_pool.checkpoints)
        if dart_path and checkpoint_paths and opponents.episode_mix.frozen_pool > 0.0:
            from ...config import dart_qnet_config
            from ...model.checkpoint import load_frozen_dart_qnet

            qnet_cfg = dart_qnet_config(cfg)
            for ckpt_path in checkpoint_paths:
                frozen_nets.append(load_frozen_dart_qnet(ckpt_path, qnet_cfg, device="cpu"))
            logger.info("preloaded %d frozen opponents", len(frozen_nets))

        hard_bots_pool: list[tuple[str, object]] = []
        hard_bots_by_name: dict[str, object] = {}
        hard_bot_weights: tuple[float, ...] = ()
        pair_names: tuple[str, ...] = ()
        pair_weights: tuple[float, ...] = ()
        hard_bot_cfg = opponents.hard_bot
        sampling = hard_bot_cfg.sampling
        if dart_path and hard_bot_cfg.bots and opponents.episode_mix.hard_bot > 0.0:
            from ....agents import make_agent

            hard_bots_pool = [(name, make_agent(name)) for name in hard_bot_cfg.bots]
            hard_bots_by_name = {name: agent for name, agent in hard_bots_pool}
            if sampling.type == "pair_weighted":
                pair_names = tuple(sampling.weights.keys())
                pair_weights = tuple(sampling.weights.values())
            elif sampling.type == "weighted":
                hard_bot_weights = tuple(
                    sampling.weights.get(name, 0.0)
                    for name, _ in hard_bots_pool
                )

            logger.info(
                "loaded %d hard-bot opponents: %s (%s)",
                len(hard_bots_pool),
                [n for n, _ in hard_bots_pool],
                sampling.type,
            )

        return cls(
            frozen_nets=tuple(frozen_nets),
            frozen_checkpoint_paths=checkpoint_paths,
            frozen_epsilon=opponents.frozen_pool.epsilon,
            hard_bots_pool=tuple(hard_bots_pool),
            hard_bots_by_name=hard_bots_by_name,
            hard_bot_weights=hard_bot_weights,
            pair_names=pair_names,
            pair_weights=pair_weights,
            episode_mix=opponents.episode_mix,
            latest_learner_team_odd_probability=(
                opponents.latest_learner_team_odd_probability
            ),
            mode_counts={"self_play": 0, "vs_frozen": 0, "vs_hard_bot": 0},
            team_counts={"latest_even": 0, "latest_odd": 0},
            frozen_pick_counts=[0] * len(frozen_nets),
            hard_bot_pick_counts=[0] * len(hard_bots_pool),
            pair_pick_counts={k: 0 for k in pair_names},
        )

    def sample_lane(self, rng: random.Random, eps: float) -> tuple[SeatPolicies, EpisodeTags]:
        u = rng.random()
        mix = self.episode_mix
        if u < mix.self_play:
            return self._self_play(eps)
        u -= mix.self_play
        if u < mix.frozen_pool:
            return self._frozen_or_self_play(rng, eps)
        u -= mix.frozen_pool
        if u < mix.hard_bot:
            return self._hard_bot_or_self_play(rng, eps)
        return self._self_play(eps)

    def _self_play(self, eps: float) -> tuple[SeatPolicies, EpisodeTags]:
        self.mode_counts["self_play"] += 1
        return all_latest_seats(eps), EpisodeTags(mode=EPISODE_MODE_SELF_PLAY)

    def _frozen_or_self_play(
        self,
        rng: random.Random,
        eps: float,
    ) -> tuple[SeatPolicies, EpisodeTags]:
        if not self.frozen_nets:
            return self._self_play(eps)

        pick = rng.randrange(len(self.frozen_nets))
        frozen_net = self.frozen_nets[pick]
        self.frozen_pick_counts[pick] += 1
        if rng.random() < self.latest_learner_team_odd_probability:
            frozen_seats = (0, 2)
            latest_team = 1
            self.team_counts["latest_odd"] += 1
        else:
            frozen_seats = (1, 3)
            latest_team = 0
            self.team_counts["latest_even"] += 1

        seats = list(all_latest_seats(eps))
        for seat in frozen_seats:
            seats[seat] = SeatPolicy.frozen(frozen_net, self.frozen_epsilon)
        self.mode_counts["vs_frozen"] += 1
        return (
            tuple(seats),  # type: ignore[return-value]
            EpisodeTags(
                mode=EPISODE_MODE_VS_CHECKPOINT,
                opponent_id=OPPONENT_CHECKPOINT_BASE + pick,
                latest_team=latest_team,
            ),
        )

    def _hard_bot_or_self_play(
        self,
        rng: random.Random,
        eps: float,
    ) -> tuple[SeatPolicies, EpisodeTags]:
        if not self.hard_bots_pool:
            return self._self_play(eps)

        if rng.random() < self.latest_learner_team_odd_probability:
            seat_a, seat_b = 0, 2
            latest_team = 1
            self.team_counts["latest_odd"] += 1
        else:
            seat_a, seat_b = 1, 3
            latest_team = 0
            self.team_counts["latest_even"] += 1

        if self.pair_names:
            pair_key = rng.choices(self.pair_names, weights=self.pair_weights, k=1)[0]
            self.pair_pick_counts[pair_key] += 1
            name_a, name_b = pair_key.split("_")
            bot_a = self.hard_bots_by_name[name_a]
            bot_b = self.hard_bots_by_name[name_b]
            if rng.random() < 0.5:
                hard_bots = {seat_a: bot_a, seat_b: bot_b}
            else:
                hard_bots = {seat_a: bot_b, seat_b: bot_a}
            active_name = "yaoji" if "yaoji" in pair_key else name_a
        else:
            if self.hard_bot_weights:
                pick = rng.choices(
                    range(len(self.hard_bots_pool)),
                    weights=self.hard_bot_weights,
                    k=1,
                )[0]
            else:
                pick = rng.randrange(len(self.hard_bots_pool))
            active_name, bot = self.hard_bots_pool[pick]
            hard_bots = {seat_a: bot, seat_b: bot}
            self.hard_bot_pick_counts[pick] += 1

        seats = list(all_latest_seats(eps))
        seats[seat_a] = SeatPolicy.hard_bot(hard_bots[seat_a])
        seats[seat_b] = SeatPolicy.hard_bot(hard_bots[seat_b])
        self.mode_counts["vs_hard_bot"] += 1
        return (
            tuple(seats),  # type: ignore[return-value]
            EpisodeTags(
                mode=EPISODE_MODE_VS_HARD_BOT,
                opponent_id=OPPONENT_BY_NAME.get(active_name, OPPONENT_NONE),
                latest_team=latest_team,
            ),
        )

    def log_summary(self, logger) -> None:
        logger.info("episode modes: %s", self.mode_counts)
        if self.frozen_nets:
            pool_str = ", ".join(
                f"{Path(p).stem}={c}"
                for p, c in zip(self.frozen_checkpoint_paths, self.frozen_pick_counts, strict=False)
            )
            logger.info("frozen picks: %s", pool_str)
            logger.info("team assignment: %s", self.team_counts)
        if self.hard_bots_pool:
            if self.pair_names:
                pool_str = ", ".join(f"{k}={c}" for k, c in self.pair_pick_counts.items())
                logger.info("hard-bot pair picks: %s", pool_str)
            else:
                pool_str = ", ".join(
                    f"{n}={c}"
                    for (n, _), c in zip(self.hard_bots_pool, self.hard_bot_pick_counts, strict=False)
                )
                logger.info("hard-bot picks: %s", pool_str)
            logger.info("team assignment: %s", self.team_counts)


__all__ = ["OpponentPools"]
