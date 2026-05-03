"""Self-play actor: rolls a single episode and returns MC samples.

Single-process for M0 — distributed actors (actor_loop / maybe_sync_weights)
implement the faithful persistent actor-learner DMC paper architecture.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..combos import Combo
from ..game import GuanDanEnv
from ..pvguan.legal_utils import dedup_strategic
from ..pvguan.rollout import _cap_legal
from .buffer import collate_encoded
from .encoder import StateActionEncoder
from .q_network import GuanZeroQNet
from .returns import TrainSample, compute_mc_returns

# Channel keys in the encoded sample dict — order matches encoder.encode_all().
_KEYS = (
    "own_hand", "others_hand", "recent_action_each_player",
    "played_cards_others", "remaining_counts_others",
    "level", "history", "behavior", "candidate_action",
)


def _select_legal(env: GuanDanEnv, player: int, max_legal: int) -> list[Combo]:
    legal = env.legal_moves(player)
    legal = dedup_strategic(legal)
    if max_legal and len(legal) > max_legal:
        keep = _cap_legal(env, player, legal, max_legal)
        legal = [legal[i] for i in keep]
    return legal


@torch.no_grad()
def _argmax_q(
    net: GuanZeroQNet,
    encoded_list: list[dict],
    device: torch.device,
) -> int:
    batch = collate_encoded(encoded_list, device=device)
    q = net(batch)
    return int(q.argmax().item())


def play_episode(
    q_nets: Mapping[int, GuanZeroQNet],
    encoder: StateActionEncoder,
    epsilon: float,
    max_legal_actions: int = 128,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
) -> list[TrainSample]:
    """Roll one self-play episode, return per-step MC training samples."""
    device = torch.device(device)
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []

    while not env.done:
        p = env.current_player
        legal = _select_legal(env, p, max_legal_actions)
        if not legal:
            # Defensive: an active player should always have at least PASS
            # as a response. If we somehow get here, end the episode.
            break

        encoded_list = encoder.encode_all(env, p, legal)

        if random.random() < epsilon:
            idx = random.randrange(len(legal))
        else:
            net = q_nets[p]
            idx = _argmax_q(net, encoded_list, device)

        trajectory.append({"player": p, "encoded": encoded_list[idx]})
        env.step(legal[idx])

    rewards = env.get_rewards() if env.done else {p: 0.0 for p in range(4)}
    return compute_mc_returns(trajectory, rewards, gamma=gamma)


# ─── Distributed actor helpers ───────────────────────────


def maybe_sync_weights(
    q_nets: dict,
    weight_dir: Path,
    local_version: int,
) -> int:
    """Non-blocking weight sync. Loads newer weights if available, returns new version."""
    from .learner import load_latest_weights

    new_ver, state_dicts = load_latest_weights(weight_dir)
    if new_ver is None or new_ver <= local_version:
        return local_version
    for p in range(4):
        q_nets[p].load_state_dict(state_dicts[p])
        q_nets[p].eval()
    return new_ver


def actor_loop(
    actor_id:     int,
    cfg_dict:     dict,
    sample_queue,          # multiprocessing.Queue
    stop_event,            # multiprocessing.Event
    weight_dir:   Path,
) -> None:
    """Persistent actor process for faithful actor-learner DMC.

    Runs self-play continuously, serializes samples to the shared queue, and
    periodically syncs local Q-net copies from the learner's published weights.
    Top-level module function — must be picklable for 'spawn' start method.
    """
    # Lazy import — runs in a spawned child; full package re-imported from scratch
    from .train import TrainConfig, _epsilon
    from .q_network import init_position_nets

    torch.set_num_threads(1)

    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if k in {f.name for f in dataclasses.fields(TrainConfig)}})

    encoder  = StateActionEncoder(use_oracle_others_hand=cfg.use_oracle_others_hand)
    q_nets   = init_position_nets(
        hidden_lstm=cfg.hidden_lstm,
        hidden_mlp=cfg.hidden_mlp,
        n_mlp_layers=cfg.n_mlp_layers,
        dropout=cfg.dropout,
        use_oracle_others_hand=cfg.use_oracle_others_hand,
    )
    for net in q_nets.values():
        net.eval()

    weight_dir    = Path(weight_dir)
    local_version = -1
    episode_count = 0
    # Pre-stacked accumulator: keep raw encoded dicts and stack at push-time.
    # One pickle of 9 contiguous arrays is ~6× faster to unpickle than 512 dicts
    # of 9 small arrays each (measured: 3.81 ms → 0.63 ms per push).
    buf_dicts:   list[dict]  = []
    buf_players: list[int]   = []
    buf_returns: list[float] = []
    rng = random.Random(cfg.seed + actor_id * 10_000)

    while not stop_event.is_set():
        # Periodic weight sync — before every new episode batch
        if episode_count % cfg.sync_interval_episodes == 0:
            local_version = maybe_sync_weights(q_nets, weight_dir, local_version)

        eps  = _epsilon(episode_count + actor_id, cfg)
        seed = rng.randint(0, 10_000_000)

        samples = play_episode(
            q_nets=q_nets,
            encoder=encoder,
            epsilon=eps,
            max_legal_actions=cfg.max_legal_actions,
            seed=seed,
            device="cpu",
            gamma=cfg.gamma,
        )
        episode_count += 1

        for s in samples:
            buf_dicts.append(s.encoded)
            buf_players.append(s.player)
            buf_returns.append(s.mc_return)

        if len(buf_dicts) >= cfg.actor_push_batch_size:
            stacked = {k: np.stack([d[k] for d in buf_dicts], axis=0) for k in _KEYS}
            msg = {
                "actor_id": actor_id,
                "version":  local_version,
                "stacked":  stacked,
                "players":  np.asarray(buf_players, dtype=np.int8),
                "returns":  np.asarray(buf_returns, dtype=np.float32),
            }
            try:
                sample_queue.put(msg, timeout=5)
            except Exception:
                pass   # queue full or closed — drop and continue
            buf_dicts.clear()
            buf_players.clear()
            buf_returns.clear()


__all__ = ["play_episode", "maybe_sync_weights", "actor_loop"]
