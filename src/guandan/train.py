"""DMC training loop for Guan Dan.

Usage:
    python -m guandan.train --episodes 1000 --eval-interval 200
"""

from __future__ import annotations

import argparse
import copy
import random
import time

import numpy as np
import torch

from .cards import Rank
from .encoding import encode_action, encode_state
from .game import GuanDanEnv
from .heuristic import heuristic_play
from .q_network import QNetwork, get_device
from .replay import ReplayBuffer


def play_episode(
    env: GuanDanEnv,
    q_net: QNetwork,
    epsilon: float,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one full game, collect transitions, assign terminal rewards."""
    env.reset()
    transitions: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        p: [] for p in range(4)
    }

    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        state_enc = encode_state(env, player)
        action_encs = np.array([encode_action(m) for m in legal])

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(
                    len(legal), -1
                )
                a = torch.tensor(action_encs, device=device)
                idx = q_net(s, a).argmax().item()

        transitions[player].append((state_enc, action_encs[idx]))
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans: list[tuple[np.ndarray, np.ndarray, float]] = []
    for player, tlist in transitions.items():
        g = rewards[player]
        for s, a in tlist:
            all_trans.append((s, a, g))
    return all_trans


def evaluate(
    q_net: QNetwork,
    device: torch.device,
    n_games: int = 100,
    opponent: str = "random",
) -> float:
    """Evaluate Q-agent (all seats) vs baseline. Returns team {0,2} win rate."""
    env = GuanDanEnv()
    wins = 0
    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            legal = env.legal_moves()

            if opponent == "heuristic" and player in (1, 3):
                move = heuristic_play(legal, env.is_leading())
            elif opponent == "random" and player in (1, 3):
                move = random.choice(legal)
            else:
                # Q-agent plays for team {0, 2}
                state_enc = encode_state(env, player)
                action_encs = np.array([encode_action(m) for m in legal])
                with torch.no_grad():
                    s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(
                        len(legal), -1
                    )
                    a = torch.tensor(action_encs, device=device)
                    idx = q_net(s, a).argmax().item()
                move = legal[idx]

            env.step(move)

        first = env.finish_order[0]
        if first in (0, 2):
            wins += 1

    return wins / n_games


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    q_net = QNetwork().to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    env = GuanDanEnv(level_rank=Rank.TWO)

    eps_start, eps_end = 0.3, 0.02
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        # Epsilon decay
        frac = min(1.0, ep / (args.episodes * 0.8))
        epsilon = eps_start + (eps_end - eps_start) * frac

        # Play episode and store transitions
        trans = play_episode(env, q_net, epsilon, device)
        for s, a, r in trans:
            buffer.push(s, a, r)

        # Train if enough data
        if len(buffer) >= args.batch_size:
            states, actions, rewards = buffer.sample(args.batch_size, device)
            q_values = q_net(states, actions)
            loss = torch.nn.functional.mse_loss(q_values, rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sync target network
        if ep % args.target_sync == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Evaluate and log
        if ep % args.eval_interval == 0:
            wr_random = evaluate(q_net, device, n_games=args.eval_games, opponent="random")
            elapsed = time.time() - t0
            print(
                f"Ep {ep:>6d} | ε={epsilon:.3f} | buf={len(buffer):>6d} | "
                f"WR vs random: {wr_random:.1%} | {elapsed:.0f}s"
            )

        # Save checkpoint
        if ep % args.save_interval == 0:
            path = f"checkpoint_ep{ep}.pt"
            torch.save(
                {
                    "episode": ep,
                    "model_state_dict": q_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                },
                path,
            )
            print(f"  Saved {path}")

    # Final save
    torch.save(
        {
            "episode": args.episodes,
            "model_state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_final.pt",
    )
    print("Training complete. Saved model_final.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Guan Dan DMC agent")
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--target-sync", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
