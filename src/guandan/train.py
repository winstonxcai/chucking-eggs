"""DMC training loop for Guan Dan — LSTM + lead/follow split.

Usage:
    python -m guandan.train --episodes 30000 --eval-interval 500
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from .agents import make_agent
from .cards import Rank
from .encoding import (
    ACTION_DIM,
    D_MOVE,
    MAX_HISTORY,
    STATE_DIM,
    encode_action,
    encode_history,
    encode_state,
)
from .game import GuanDanEnv
from .q_network import QNetworkLSTM, get_device
from .replay import ReplayBuffer


def play_episode(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """Play one full game, collect transitions, assign terminal rewards.

    Returns list of (state, action, history, hist_len, reward) tuples.
    """
    env.reset()
    transitions: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {
        p: [] for p in range(4)
    }

    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        is_leading = env.current_trick is None

        state_enc = encode_state(env, player)
        hand = env.hands[player]
        action_encs = np.array(
            [encode_action(m, hand, env.level_rank) for m in legal]
        )
        history, hist_len = encode_history(env, player, env.level_rank)

        q_net = q_lead if is_leading else q_follow

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = (
                    torch.tensor(state_enc, device=device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
                a = torch.tensor(action_encs, device=device)
                h = (
                    torch.tensor(history, device=device)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                hl = torch.tensor(
                    [hist_len], dtype=torch.long, device=device
                ).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()

        transitions[player].append((state_enc, action_encs[idx], history, hist_len))
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        G = rewards[player]
        for s, a, h, hl in tlist:
            all_trans.append((s, a, h, hl, G))
    return all_trans


def train_step(
    q_net: QNetworkLSTM,
    buf: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float | None:
    """One gradient step on a single network/buffer pair."""
    if len(buf) < batch_size:
        return None

    batch = buf.sample(batch_size, device)
    q_pred = q_net(
        batch["state"], batch["action"], batch["history"], batch["hist_len"]
    )
    loss = torch.nn.functional.mse_loss(q_pred, batch["return"])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def evaluate(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    n_games: int = 200,
    opponent: str = "random",
) -> dict:
    """Evaluate Q-agent (team {0,2}) vs opponent (team {1,3}).

    Returns dict with winrate, avg_reward, finish_breakdown, n_games.
    """
    env = GuanDanEnv()
    opp_agent = make_agent(opponent, env.level_rank)

    wins = 0
    total_reward = 0.0
    finish_12 = 0
    finish_13 = 0
    finish_14 = 0

    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player

            if player in (1, 3):
                move = opp_agent.act(env, player)
            else:
                legal = env.legal_moves()
                is_leading = env.current_trick is None
                q_net = q_lead if is_leading else q_follow

                state_enc = encode_state(env, player)
                hand = env.hands[player]
                action_encs = np.array(
                    [encode_action(m, hand, env.level_rank) for m in legal]
                )
                history, hist_len = encode_history(env, player, env.level_rank)

                with torch.no_grad():
                    B = len(legal)
                    s = (
                        torch.tensor(state_enc, device=device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                    a = torch.tensor(action_encs, device=device)
                    h = (
                        torch.tensor(history, device=device)
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )
                    hl = torch.tensor(
                        [hist_len], dtype=torch.long, device=device
                    ).expand(B)
                    idx = q_net(s, a, h, hl).argmax().item()
                move = legal[idx]

            env.step(move)

        rewards = env.get_rewards()
        team_reward = rewards[0] + rewards[2]
        if team_reward > 0:
            wins += 1
        total_reward += team_reward

        # Track finish type
        fo = env.finish_order
        team_set = {0, 2}
        if fo[0] in team_set and fo[1] in team_set:
            finish_12 += 1
        elif fo[0] in team_set and fo[2] in team_set:
            finish_13 += 1
        elif fo[0] in team_set:
            finish_14 += 1

    return {
        "winrate": wins / n_games,
        "avg_reward": total_reward / n_games,
        "finish_12": finish_12,
        "finish_13": finish_13,
        "finish_14": finish_14,
        "n_games": n_games,
    }


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    q_lead = QNetworkLSTM(
        lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden
    ).to(device)
    q_follow = QNetworkLSTM(
        lstm_hidden=args.lstm_hidden, hidden=args.mlp_hidden
    ).to(device)

    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    # Single shared buffer — avoids lead starvation from split buffers
    buffer = ReplayBuffer(capacity=args.buffer_size)

    env = GuanDanEnv(level_rank=Rank.TWO)

    eps_start, eps_end = 0.25, 0.02
    eps_decay_episodes = int(args.episodes * 0.67)
    best_heuristic_wr = 0.0
    evals_without_improvement = 0
    t0 = time.time()

    n_lead_params = sum(p.numel() for p in q_lead.parameters())
    n_follow_params = sum(p.numel() for p in q_follow.parameters())
    print(f"Lead params: {n_lead_params:,}  Follow params: {n_follow_params:,}")

    for ep in range(1, args.episodes + 1):
        frac = min(1.0, ep / eps_decay_episodes)
        epsilon = eps_start + (eps_end - eps_start) * frac

        trans = play_episode(env, q_lead, q_follow, epsilon, device)
        for s, a, h, hl, G in trans:
            buffer.push(s, a, h, hl, G)

        # Gradient steps — both networks train on shared buffer
        loss_lead = None
        loss_follow = None
        for _ in range(args.train_steps):
            ll = train_step(q_lead, buffer, opt_lead, args.batch_size, device)
            lf = train_step(q_follow, buffer, opt_follow, args.batch_size, device)
            if ll is not None:
                loss_lead = ll
            if lf is not None:
                loss_follow = lf

        # Evaluate and log
        if ep % args.eval_interval == 0:
            result = evaluate(
                q_lead, q_follow, device,
                n_games=args.eval_games, opponent=args.eval_opponent,
            )
            elapsed = time.time() - t0
            wr = result["winrate"]

            if wr > best_heuristic_wr:
                best_heuristic_wr = wr
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            ll_str = f"{loss_lead:.4f}" if loss_lead is not None else "n/a"
            lf_str = f"{loss_follow:.4f}" if loss_follow is not None else "n/a"

            print(
                f"Ep {ep:>6d} | ε={epsilon:.3f} | "
                f"L_lead={ll_str} L_follow={lf_str} | "
                f"buf={len(buffer):>6d} | "
                f"vs {args.eval_opponent}: {wr:.1%} "
                f"(1-2:{result['finish_12']} 1-3:{result['finish_13']} 1-4:{result['finish_14']}) "
                f"(best={best_heuristic_wr:.1%}, "
                f"pat={evals_without_improvement}/{args.patience}) | "
                f"{elapsed:.0f}s"
            )

            if evals_without_improvement >= args.patience:
                print(
                    f"Early stopping: no improvement in vs {args.eval_opponent} WR "
                    f"for {args.patience} evals"
                )
                break

        # Save checkpoint
        if ep % args.save_interval == 0:
            path = f"checkpoint_ep{ep}.pt"
            torch.save(
                {
                    "episode": ep,
                    "lead_state_dict": q_lead.state_dict(),
                    "follow_state_dict": q_follow.state_dict(),
                    "opt_lead_state_dict": opt_lead.state_dict(),
                    "opt_follow_state_dict": opt_follow.state_dict(),
                    "epsilon": epsilon,
                },
                path,
            )
            print(f"  Saved {path}")

    # Final save
    torch.save(
        {
            "episode": args.episodes,
            "lead_state_dict": q_lead.state_dict(),
            "follow_state_dict": q_follow.state_dict(),
            "opt_lead_state_dict": opt_lead.state_dict(),
            "opt_follow_state_dict": opt_follow.state_dict(),
        },
        "model_final.pt",
    )
    print("Training complete. Saved model_final.pt")

    # Quick validation verdict
    if getattr(args, "quick", False):
        final_result = evaluate(
            q_lead, q_follow, device, n_games=300, opponent="heuristic"
        )
        wr = final_result["winrate"]
        print(f"\n=== QUICK VALIDATION RESULT ===")
        print(f"WR vs Heuristic @ {args.episodes} episodes: {wr:.1%}")
        if wr >= 0.55:
            print("✓ LSTM is learning. Proceed with full overnight run:")
            print("  PYTHONPATH=src python -m guandan.train --episodes 30000")
        elif wr >= 0.48:
            print("~ Marginal improvement. Consider running to 15K before deciding.")
            print("  PYTHONPATH=src python -m guandan.train --episodes 15000")
        else:
            print("✗ Not improving. Check debugging checklist before continuing:")
            print("  1. python scripts/test_mps_lstm.py")
            print("  2. Verify move_history is populated: print(len(env.move_history))")
            print("  3. Check LSTM gradients are non-zero")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Guan Dan DMC agent")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-games", type=int, default=300)
    parser.add_argument(
        "--eval-opponent",
        type=str,
        default="heuristic",
        choices=["random", "greedy", "heuristic", "strategic"],
        help="Opponent for mid-training eval (default: heuristic). "
             "Full ladder eval: use scripts/ladder.py after training.",
    )
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stop after N evals with no improvement",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode: cap at 8000 episodes, eval every 2000, "
             "print go/no-go verdict at end. ~4-5 hours on M1 Pro.",
    )
    args = parser.parse_args()

    if args.quick:
        args.episodes = 8000
        args.eval_interval = 2000

    train(args)


if __name__ == "__main__":
    main()
