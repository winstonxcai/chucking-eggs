"""Coarse rollout profiler for GuanZero actor pipeline.

Re-implements ``play_episode`` inline with per-phase timers so we can answer:
where does actor wall time actually go? Output is a single table:

    section                         sec        %      count    ms/call

Phases timed:
  legal_actions   — env.legal_moves + dedup_strategic + _cap_legal
  encode_state    — encoder._encode_state (7 shared channels)
  encode_actions  — per-action behavior + candidate one-hot
  collate         — np.stack + torch.from_numpy
  q_forward       — Q-net forward + argmax (no_grad)
  env_step        — env.step
  mc_returns      — compute_mc_returns (per-episode, not per-decision)

Usage:
    PYTHONPATH=ml/src python ml/scripts/util/profile_guanzero_rollout.py \
        --episodes 200 --device cpu --epsilon 0.0

    # With trained weights for realistic Q-forward distribution:
    PYTHONPATH=ml/src python ml/scripts/util/profile_guanzero_rollout.py \
        --episodes 200 --device cpu --checkpoint <path>

Prints rollup, plus rates (episodes/sec, decisions/sec, avg_legal/decision).
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from guandan.game import GuanDanEnv
from guandan.guanzero.buffer import collate_encoded
from guandan.guanzero.encoder import StateActionEncoder
from guandan.guanzero.q_network import init_position_nets
from guandan.guanzero.returns import compute_mc_returns
from guandan.guanzero.encoder import _multi_hot
from guandan.azguan.behavior_flags import compute_behavior_flags
from guandan.pvguan.legal_utils import dedup_strategic
from guandan.pvguan.rollout import _cap_legal


class Profiler:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def add_count(self, name: str, value: int) -> None:
        if self.enabled:
            self.counts[name] += value

    def report(self, episodes: int, total_wall: float) -> str:
        total_timed = sum(self.times.values())
        n_dec   = self.counts.get("num_decisions", 1) or 1
        n_legal = self.counts.get("num_legal_actions", 0)

        lines = []
        lines.append("")
        lines.append(f"Rollout profile  |  {episodes} episodes  |  {total_wall:.2f}s wall")
        lines.append(f"{'section':18s} {'sec':>9s} {'%':>7s} {'count':>10s} {'ms/call':>10s}")
        lines.append("-" * 60)
        for k, v in sorted(self.times.items(), key=lambda x: -x[1]):
            c = max(1, self.counts.get(k, 1))
            pct = 100 * v / total_timed if total_timed > 0 else 0
            lines.append(f"{k:18s} {v:9.3f} {pct:7.1f} {c:10d} {1000*v/c:10.3f}")
        lines.append("-" * 60)
        lines.append(f"{'TOTAL TIMED':18s} {total_timed:9.3f}")
        lines.append("")
        lines.append("Derived:")
        lines.append(f"  episodes/sec            : {episodes / total_wall:.2f}")
        lines.append(f"  decisions/sec           : {n_dec / total_wall:.1f}")
        lines.append(f"  decisions/episode (avg) : {n_dec / episodes:.1f}")
        lines.append(f"  legal_actions/decision  : {n_legal / n_dec:.1f}")
        for ph in ("legal_actions", "encode_state", "encode_actions", "collate",
                   "q_forward", "env_step"):
            if ph in self.times:
                lines.append(f"  {ph:22s}: {1000*self.times[ph]/n_dec:.3f} ms/decision")
        return "\n".join(lines)


def _select_legal(env: GuanDanEnv, player: int, max_legal: int):
    legal = env.legal_moves(player)
    legal = dedup_strategic(legal)
    if max_legal and len(legal) > max_legal:
        keep = _cap_legal(env, player, legal, max_legal)
        legal = [legal[i] for i in keep]
    return legal


def play_episode_profiled(
    q_nets,
    encoder: StateActionEncoder,
    epsilon: float,
    prof: Profiler,
    max_legal: int = 128,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    gamma: float = 1.0,
):
    device = torch.device(device)
    env = GuanDanEnv()
    env.reset(seed=seed)

    trajectory: list[dict] = []

    while not env.done:
        p = env.current_player

        with prof.time("legal_actions"):
            legal = _select_legal(env, p, max_legal)
        if not legal:
            break
        prof.add_count("num_decisions", 1)
        prof.add_count("num_legal_actions", len(legal))

        with prof.time("encode_state"):
            shared = encoder._encode_state(env, p)

        with prof.time("encode_actions"):
            encoded_list = []
            for action in legal:
                enc = dict(shared)
                enc["behavior"] = compute_behavior_flags(env, p, action, legal)
                enc["candidate_action"] = _multi_hot(action.cards)
                encoded_list.append(enc)

        if random.random() < epsilon:
            idx = random.randrange(len(legal))
        else:
            with prof.time("collate"):
                batch = collate_encoded(encoded_list, device=device)
            with prof.time("q_forward"):
                with torch.no_grad():
                    q = q_nets[p](batch)
                    idx = int(q.argmax().item())

        trajectory.append({"player": p, "encoded": encoded_list[idx]})

        with prof.time("env_step"):
            env.step(legal[idx])

    rewards = env.get_rewards() if env.done else {p: 0.0 for p in range(4)}
    with prof.time("mc_returns"):
        samples = compute_mc_returns(trajectory, rewards, gamma=gamma)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Force-explore fraction. 0.0 = always Q-forward (worst case).")
    ap.add_argument("--max-legal", type=int, default=128)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Optional .pt to warm-start nets so Q-forward distribution is realistic.")
    ap.add_argument("--hidden-lstm", type=int, default=256)
    ap.add_argument("--hidden-mlp", type=int, default=1024)
    ap.add_argument("--n-mlp-layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="torch.set_num_threads. Default 1 to mirror actor processes.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the 5-episode warmup (e.g. when py-spy attaching).")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    encoder = StateActionEncoder(use_oracle_others_hand=True)
    q_nets = init_position_nets(
        hidden_lstm=args.hidden_lstm,
        hidden_mlp=args.hidden_mlp,
        n_mlp_layers=args.n_mlp_layers,
        dropout=0.0,
        use_oracle_others_hand=True,
    )
    for net in q_nets.values():
        net.eval()
        net.to(args.device)

    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        sd = ckpt.get("q_nets", ckpt.get("state_dicts", ckpt))
        for p in range(4):
            q_nets[p].load_state_dict(sd[p])
        print(f"Loaded weights from {args.checkpoint}")

    rng = random.Random(args.seed)

    if not args.no_warmup:
        warmup_prof = Profiler(enabled=False)
        for _ in range(5):
            play_episode_profiled(
                q_nets, encoder, args.epsilon, warmup_prof,
                max_legal=args.max_legal,
                seed=rng.randint(0, 10_000_000),
                device=args.device,
            )

    prof = Profiler(enabled=True)
    t_start = time.perf_counter()
    for _ in range(args.episodes):
        play_episode_profiled(
            q_nets, encoder, args.epsilon, prof,
            max_legal=args.max_legal,
            seed=rng.randint(0, 10_000_000),
            device=args.device,
        )
    total_wall = time.perf_counter() - t_start

    print(prof.report(episodes=args.episodes, total_wall=total_wall))
    print()
    print(f"Config: device={args.device} epsilon={args.epsilon} max_legal={args.max_legal} "
          f"threads={args.torch_threads} hidden_lstm={args.hidden_lstm} "
          f"hidden_mlp={args.hidden_mlp} n_mlp_layers={args.n_mlp_layers}")


if __name__ == "__main__":
    main()
