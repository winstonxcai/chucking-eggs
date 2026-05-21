"""Episode rollout helpers for Dart actors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
from typing import Literal

import torch

from ...constants import NUM_PLAYERS, PARTNER_OFFSET
from ....cards import ComboType
from ....game import GuanDanEnv
from ...data.buffer import collate_base_encoded, collate_role_encoded
from ...data.returns import EpisodeTags, TrainSample, compute_mc_returns
from ...data.sample_tags import phase_bucket, phase_with_out, trick_role
from ...config import MODEL_TYPE_DART, MODEL_TYPE_GUANZERO
from ...model.encoding.base_encoder import StateActionEncoder
from ...model.encoding.role_encoder import RoleAwareStateActionEncoder
from ...model.q_network import DartQNet, GuanZeroQNet
from ...utils.legal_utils import select_legal
from ...utils.profiler import PhaseProfiler, k_bucket_label


QNetLike = GuanZeroQNet | DartQNet
QNetBundle = Mapping[int, GuanZeroQNet] | DartQNet
SeatKind = Literal["latest", "frozen", "hard_bot"]


@dataclass(frozen=True)
class SeatPolicy:
    """Action policy for one absolute seat in an episode lane."""

    kind: SeatKind
    epsilon: float = 0.0
    frozen_net: DartQNet | None = None
    hard_bot_agent: object | None = None

    def __post_init__(self) -> None:
        if self.kind == "latest":
            if self.frozen_net is not None or self.hard_bot_agent is not None:
                raise ValueError("latest SeatPolicy cannot carry frozen_net or hard_bot_agent")
        elif self.kind == "frozen":
            if self.frozen_net is None or self.hard_bot_agent is not None:
                raise ValueError("frozen SeatPolicy requires frozen_net only")
        elif self.kind == "hard_bot":
            if self.hard_bot_agent is None or self.frozen_net is not None:
                raise ValueError("hard_bot SeatPolicy requires hard_bot_agent only")
        else:
            raise ValueError(f"unknown SeatPolicy kind: {self.kind!r}")

    @classmethod
    def latest(cls, epsilon: float) -> "SeatPolicy":
        return cls("latest", epsilon=epsilon)

    @classmethod
    def frozen(
        cls,
        net: DartQNet,
        epsilon: float,
    ) -> "SeatPolicy":
        return cls("frozen", epsilon=epsilon, frozen_net=net)

    @classmethod
    def hard_bot(cls, agent: object) -> "SeatPolicy":
        return cls("hard_bot", hard_bot_agent=agent)


SeatPolicies = tuple[SeatPolicy, SeatPolicy, SeatPolicy, SeatPolicy]


def all_latest_seats(epsilon: float) -> SeatPolicies:
    """Return four latest-policy seats with shared epsilon."""
    p = SeatPolicy.latest(epsilon)
    return tuple(p for _ in range(NUM_PLAYERS))  # type: ignore[return-value]


@dataclass(frozen=True)
class LaneConfig:
    seed: int
    seats: SeatPolicies
    tags: EpisodeTags = EpisodeTags()

    def __post_init__(self) -> None:
        if len(self.seats) != NUM_PLAYERS:
            raise ValueError(
                f"LaneConfig.seats must contain {NUM_PLAYERS} policies; "
                f"got {len(self.seats)}"
            )
        if not any(sp.kind == "latest" for sp in self.seats):
            raise ValueError("LaneConfig must contain at least one latest seat")


def _model_schema(net) -> str:
    if isinstance(net, DartQNet):
        return MODEL_TYPE_DART
    return MODEL_TYPE_GUANZERO


def _encoder_schema(encoder: StateActionEncoder | RoleAwareStateActionEncoder) -> str:
    if isinstance(encoder, RoleAwareStateActionEncoder):
        return MODEL_TYPE_DART
    return MODEL_TYPE_GUANZERO


def _latest_schema(q_nets: QNetBundle) -> str:
    if isinstance(q_nets, Mapping):
        return MODEL_TYPE_GUANZERO
    return _model_schema(q_nets)


def _validate_lanes(
    lanes: list[LaneConfig],
    q_nets: QNetBundle,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
) -> None:
    if not lanes:
        raise ValueError("play_episodes_batched: lanes must be non-empty")

    enc_schema = _encoder_schema(encoder)
    flat = [sp for lane in lanes for sp in lane.seats]
    needs_latest = any(sp.kind == "latest" for sp in flat)
    has_frozen = any(sp.kind == "frozen" for sp in flat)

    if needs_latest:
        latest_schema = _latest_schema(q_nets)
        if latest_schema != enc_schema:
            raise ValueError(
                f"latest q_nets schema {latest_schema!r} != encoder schema {enc_schema!r}"
            )

    if has_frozen:
        if enc_schema != MODEL_TYPE_DART:
            raise ValueError("frozen seats require the Dart role-aware encoder")
        for sp in flat:
            if sp.kind == "frozen":
                frozen_schema = _model_schema(sp.frozen_net)
                if frozen_schema != enc_schema:
                    raise ValueError(
                        f"frozen_net schema {frozen_schema!r} != encoder schema {enc_schema!r}"
                    )

def _resolve_acting_net(policy: SeatPolicy, q_nets: QNetBundle, player: int) -> QNetLike:
    if policy.kind == "frozen":
        assert policy.frozen_net is not None
        return policy.frozen_net
    if isinstance(q_nets, DartQNet):
        return q_nets
    assert q_nets is not None
    return q_nets[player]


def argmax_q(
    net: QNetLike,
    encoded_list: list[dict],
    device: torch.device,
    *,
    role_encoded: bool = False,
    profiler: PhaseProfiler | None = None,
    bucket: str = "",
) -> tuple[int, float]:
    """Greedy argmax over candidate actions; returns ``(idx, q_gap)``."""
    return argmax_q_batched(
        net,
        [encoded_list],
        device,
        role_encoded=role_encoded,
        profiler=profiler,
        bucket=bucket,
    )[0]


def argmax_q_batched(
    net: QNetLike,
    encoded_groups: list[list[dict]],
    device: torch.device,
    *,
    role_encoded: bool = False,
    profiler: PhaseProfiler | None = None,
    bucket: str = "",
) -> list[tuple[int, float]]:
    """Greedy argmax for several independent decisions in one grouped forward."""
    if not encoded_groups:
        return []
    collate = collate_role_encoded if role_encoded else collate_base_encoded
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)
    bucket_suffix = f"_{bucket}" if bucket else ""

    with prof.time("q_collate"):
        state_batch, action_batch, repeats = collate(encoded_groups, device=device)
    prof.add_count("q_forward_groups", len(encoded_groups))
    prof.add_count("q_forward_action_rows", sum(len(group) for group in encoded_groups))
    with prof.time(f"q_net_forward{bucket_suffix}"):
        with torch.inference_mode():
            q_vals = net.forward_grouped(state_batch, action_batch, repeats)
    with prof.time("q_argmax_item"):
        q_flat = q_vals.view(-1)
        out: list[tuple[int, float]] = []
        offset = 0
        for group in encoded_groups:
            k = len(group)
            segment = q_flat.narrow(0, offset, k)
            if k >= 2:
                top2 = torch.topk(segment, 2, largest=True).values
                q_gap = float(top2[0].item() - top2[1].item())
            else:
                q_gap = float("nan")
            out.append((int(segment.argmax().item()), q_gap))
            offset += k
        return out


def _trajectory_step(
    *,
    env: GuanDanEnv,
    player: int,
    partner: int,
    encoded: dict,
    legal: list,
    idx: int,
    k: int,
    q_gap: float,
    chosen_by_epsilon: int,
) -> dict:
    action_type = int(legal[idx].type)
    return {
        "player":             player,
        "encoded":            encoded,
        "phase_self":         phase_bucket(len(env.hands[player])),
        "trick_role":         trick_role(env, partner),
        "phase_partner":      phase_with_out(env, partner),
        "action_type":        action_type,
        "is_pass":            int(action_type == 0),
        "is_bomb":            int(action_type >= int(ComboType.BOMB_4)),
        "bomb_available":     int(any(m.type >= ComboType.BOMB_4 for m in legal)),
        "num_legal_actions":  k,
        "q_gap":              q_gap,
        "chosen_by_epsilon":  chosen_by_epsilon,
    }


def play_episode(
    q_nets: QNetBundle,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    *,
    seats: SeatPolicies,
    seed: int,
    tags: EpisodeTags = EpisodeTags(),
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    profiler: PhaseProfiler | None = None,
    rng: random.Random | None = None,
    record_forced_k1_samples: bool = True,
) -> list[TrainSample]:
    """Roll one episode. Thin wrapper over ``play_episodes_batched``."""
    return play_episodes_batched(
        q_nets=q_nets,
        encoder=encoder,
        lanes=[LaneConfig(seed=seed, seats=seats, tags=tags)],
        device=device,
        gamma=gamma,
        profiler=profiler,
        rng=rng,
        record_forced_k1_samples=record_forced_k1_samples,
    )[0]


def play_episodes_batched(
    q_nets: QNetBundle,
    encoder: StateActionEncoder | RoleAwareStateActionEncoder,
    *,
    lanes: list[LaneConfig],
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
    profiler: PhaseProfiler | None = None,
    rng: random.Random | None = None,
    stop_event: object | None = None,
    record_forced_k1_samples: bool = True,
) -> list[list[TrainSample]]:
    """Roll episode lanes, batching greedy Q-forwards by acting network."""
    _validate_lanes(lanes, q_nets, encoder)
    device = torch.device(device)
    prof = profiler if profiler is not None else PhaseProfiler(enabled=False)
    policy_rng = rng if rng is not None else random

    envs: list[GuanDanEnv] = []
    trajectories: list[list[dict]] = []
    active = [True] * len(lanes)
    for lane in lanes:
        env = GuanDanEnv(seed=lane.seed)
        envs.append(env)
        trajectories.append([])

    while any(active):
        if stop_event is not None and stop_event.is_set():
            return [[] for _ in lanes]

        pending: list[dict] = []
        for lane_idx in range(len(lanes)):
            if not active[lane_idx]:
                continue
            env = envs[lane_idx]
            while active[lane_idx]:
                if stop_event is not None and stop_event.is_set():
                    return [[] for _ in lanes]
                if env.done:
                    active[lane_idx] = False
                    break

                player = env.current_player
                policy = lanes[lane_idx].seats[player]
                if policy.kind == "hard_bot":
                    assert policy.hard_bot_agent is not None
                    with prof.time("hard_bot_act"):
                        action = policy.hard_bot_agent.act(env, player)
                    with prof.time("env_step"):
                        env.step(action)
                    if env.done:
                        active[lane_idx] = False
                    continue

                with prof.time("legal_actions"):
                    legal = select_legal(env, player)
                k = len(legal)
                bucket = k_bucket_label(k)
                prof.add_count("num_decisions", 1)
                prof.add_count("num_legal_actions", k)
                prof.add_count(f"decisions_{bucket}", 1)

                is_latest = policy.kind == "latest"
                partner = (player + PARTNER_OFFSET) % NUM_PLAYERS

                if k == 1:
                    prof.add_count("shortcut_K1", 1)
                    if is_latest and record_forced_k1_samples:
                        with prof.time("encode_selected"):
                            encoded = encoder.encode_one(env, player, legal[0], legal)
                        trajectories[lane_idx].append(_trajectory_step(
                            env=env,
                            player=player,
                            partner=partner,
                            encoded=encoded,
                            legal=legal,
                            idx=0,
                            k=k,
                            q_gap=float("nan"),
                            chosen_by_epsilon=0,
                        ))
                    with prof.time("env_step"):
                        env.step(legal[0])
                    if env.done:
                        active[lane_idx] = False
                    continue

                if policy_rng.random() < policy.epsilon:
                    idx = policy_rng.randrange(k)
                    prof.add_count("epsilon_random", 1)
                    if is_latest:
                        with prof.time("encode_selected"):
                            encoded = encoder.encode_one(env, player, legal[idx], legal)
                        trajectories[lane_idx].append(_trajectory_step(
                            env=env,
                            player=player,
                            partner=partner,
                            encoded=encoded,
                            legal=legal,
                            idx=idx,
                            k=k,
                            q_gap=float("nan"),
                            chosen_by_epsilon=1,
                        ))
                    with prof.time("env_step"):
                        env.step(legal[idx])
                    if env.done:
                        active[lane_idx] = False
                    continue

                with prof.time("encode_all"):
                    encoded_list = encoder.encode_all(env, player, legal)
                pending.append({
                    "lane": lane_idx,
                    "player": player,
                    "partner": partner,
                    "policy": policy,
                    "is_latest": is_latest,
                    "legal": legal,
                    "encoded_list": encoded_list,
                    "k": k,
                    "bucket": bucket,
                })
                break

        if not pending:
            continue

        choices_by_idx: dict[int, tuple[int, float]] = {}
        groups_by_net: dict[int, dict] = {}
        for i, item in enumerate(pending):
            net = _resolve_acting_net(item["policy"], q_nets, item["player"])
            group = groups_by_net.setdefault(id(net), {"net": net, "indices": []})
            group["indices"].append(i)

        for group in groups_by_net.values():
            indices = group["indices"]
            net = group["net"]
            encoded_groups = [pending[i]["encoded_list"] for i in indices]
            choices = argmax_q_batched(
                net,
                encoded_groups,
                device,
                role_encoded=isinstance(net, DartQNet),
                profiler=prof,
                bucket="batched",
            )
            for pending_idx, choice in zip(indices, choices):
                choices_by_idx[pending_idx] = choice

        for i, item in enumerate(pending):
            idx, q_gap = choices_by_idx[i]
            lane_idx = item["lane"]
            env = envs[lane_idx]
            if item["is_latest"]:
                trajectories[lane_idx].append(_trajectory_step(
                    env=env,
                    player=item["player"],
                    partner=item["partner"],
                    encoded=item["encoded_list"][idx],
                    legal=item["legal"],
                    idx=idx,
                    k=item["k"],
                    q_gap=q_gap,
                    chosen_by_epsilon=0,
                ))
            with prof.time("env_step"):
                env.step(item["legal"][idx])
            if env.done:
                active[lane_idx] = False

    out: list[list[TrainSample]] = []
    for env, trajectory, lane in zip(envs, trajectories, lanes):
        rewards = env.get_rewards()
        with prof.time("mc_returns"):
            out.append(compute_mc_returns(
                trajectory,
                rewards,
                gamma=gamma,
                tags=lane.tags,
            ))
    return out


__all__ = [
    "LaneConfig",
    "SeatPolicy",
    "SeatPolicies",
    "all_latest_seats",
    "play_episode",
    "play_episodes_batched",
    "select_legal",
    "argmax_q",
    "argmax_q_batched",
]
