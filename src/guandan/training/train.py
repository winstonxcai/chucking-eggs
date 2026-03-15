"""DMC training loop for Guan Dan — LSTM + lead/follow split.

Usage:
    python -m guandan.training.train --episodes 30000 --eval-interval 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..agents import make_agent
from ..cards import Rank
from .encoding import (
    ACTION_DIM,
    D_MOVE,
    MAX_HISTORY,
    STATE_DIM,
    encode_action,
    encode_history,
    encode_state,
)
from ..game import GuanDanEnv
from .q_network import QNetworkLSTM, get_device
from .replay import ReplayBuffer

# (opponent_name, promotion_threshold, consecutive_evals_needed)
CURRICULUM = [
    ("random", 0.65, 2),
    ("greedy", 0.60, 2),
    ("heuristic", None, None),
]


class _TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_path: Path):
        self._file = open(log_path, "a")
        self._stdout = sys.stdout

    def write(self, msg: str) -> int:
        self._stdout.write(msg)
        self._file.write(msg)
        self._file.flush()
        return len(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        sys.stdout = self._stdout


def _setup_run_dir(args: argparse.Namespace) -> Path:
    """Create runs/<run_name>/ directory, write config.json, set up tee logging."""
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config = {k: v for k, v in vars(args).items()}
    config["start_time"] = datetime.now().isoformat()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Tee stdout to train.log
    sys.stdout = _TeeLogger(run_dir / "train.log")

    return run_dir


def _append_metrics(run_dir: Path, entry: dict) -> None:
    """Append a metrics entry to metrics.json (one JSON object per line)."""
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def play_episode(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
    opponent=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """Play one full game, collect transitions, assign terminal rewards.

    RL agent plays seats {0,2}. If opponent is provided, it plays seats {1,3};
    otherwise all 4 seats use the RL agent (self-play).

    Returns list of (state, action, history, hist_len, reward) tuples.
    """
    env.reset()
    transitions: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {
        p: [] for p in range(4)
    }

    while not env.done:
        player = env.current_player

        # Opponent seats
        if opponent is not None and player in (1, 3):
            env.step(opponent.act(env, player))
            continue

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
        mc_return = rewards[player]
        for s, a, h, hl in tlist:
            all_trans.append((s, a, h, hl, mc_return))
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
    run_dir = _setup_run_dir(args)

    device = get_device()
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")

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

    eps_start, eps_end = 0.30, 0.05
    eps_decay_episodes = int(args.episodes * 0.85)
    best_wr = 0.0
    evals_without_improvement = 0
    t0 = time.time()

    # Curriculum state
    if args.no_curriculum:
        curriculum_stage = len(CURRICULUM) - 1  # jump to final
        for i, (name, _, _) in enumerate(CURRICULUM):
            if name == args.eval_opponent:
                curriculum_stage = i
                break
    else:
        curriculum_stage = 0
    consecutive_above = 0
    current_opponent_name = CURRICULUM[curriculum_stage][0]
    train_opponent = make_agent(current_opponent_name, env.level_rank)

    n_lead_params = sum(p.numel() for p in q_lead.parameters())
    n_follow_params = sum(p.numel() for p in q_follow.parameters())
    print(f"Lead params: {n_lead_params:,}  Follow params: {n_follow_params:,}")
    print(f"Curriculum: {' → '.join(name for name, _, _ in CURRICULUM)}")
    print(f"Starting vs: {current_opponent_name}")

    for ep in range(1, args.episodes + 1):
        frac = min(1.0, ep / eps_decay_episodes)
        epsilon = eps_start + (eps_end - eps_start) * frac

        trans = play_episode(
            env, q_lead, q_follow, epsilon, device, opponent=train_opponent
        )
        for s, a, h, hl, mc_return in trans:
            buffer.push(s, a, h, hl, mc_return)

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
                n_games=args.eval_games, opponent=current_opponent_name,
            )
            elapsed = time.time() - t0
            wr = result["winrate"]

            if wr > best_wr:
                best_wr = wr
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            ll_str = f"{loss_lead:.4f}" if loss_lead is not None else "n/a"
            lf_str = f"{loss_follow:.4f}" if loss_follow is not None else "n/a"

            print(
                f"Ep {ep:>6d} | ε={epsilon:.3f} | "
                f"L_lead={ll_str} L_follow={lf_str} | "
                f"buf={len(buffer):>6d} | "
                f"vs {current_opponent_name}: {wr:.1%} "
                f"(1-2:{result['finish_12']} 1-3:{result['finish_13']} 1-4:{result['finish_14']}) "
                f"(best={best_wr:.1%}, "
                f"pat={evals_without_improvement}/{args.patience}) | "
                f"{elapsed:.0f}s"
            )

            _append_metrics(run_dir, {
                "episode": ep,
                "epsilon": round(epsilon, 4),
                "loss_lead": loss_lead,
                "loss_follow": loss_follow,
                "buffer_size": len(buffer),
                "curriculum_stage": curriculum_stage,
                "opponent": current_opponent_name,
                "winrate": result["winrate"],
                "avg_reward": result["avg_reward"],
                "finish_12": result["finish_12"],
                "finish_13": result["finish_13"],
                "finish_14": result["finish_14"],
                "best_winrate": best_wr,
                "elapsed_s": round(elapsed, 1),
            })

            # Curriculum promotion check
            _, threshold, required = CURRICULUM[curriculum_stage]
            if threshold is not None and wr >= threshold:
                consecutive_above += 1
                if consecutive_above >= required:
                    curriculum_stage += 1
                    current_opponent_name = CURRICULUM[curriculum_stage][0]
                    train_opponent = make_agent(
                        current_opponent_name, env.level_rank
                    )
                    consecutive_above = 0
                    best_wr = 0.0
                    evals_without_improvement = 0
                    print(
                        f"\n{'='*60}\n"
                        f"  PROMOTED to stage {curriculum_stage}: "
                        f"vs {current_opponent_name}\n"
                        f"{'='*60}\n"
                    )
            else:
                consecutive_above = 0

            if evals_without_improvement >= args.patience:
                print(
                    f"Early stopping: no improvement in vs {current_opponent_name} WR "
                    f"for {args.patience} evals"
                )
                break

        # Save checkpoint
        if ep % args.save_interval == 0:
            path = run_dir / f"checkpoint_ep{ep}.pt"
            torch.save(
                {
                    "episode": ep,
                    "curriculum_stage": curriculum_stage,
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
    final_path = run_dir / "model_final.pt"
    torch.save(
        {
            "episode": args.episodes,
            "curriculum_stage": curriculum_stage,
            "lead_state_dict": q_lead.state_dict(),
            "follow_state_dict": q_follow.state_dict(),
            "opt_lead_state_dict": opt_lead.state_dict(),
            "opt_follow_state_dict": opt_follow.state_dict(),
        },
        final_path,
    )
    print(f"Training complete. Saved {final_path}")
    print(f"Final curriculum stage: {curriculum_stage} ({current_opponent_name})")

    # Quick validation verdict
    if getattr(args, "quick", False):
        final_result = evaluate(
            q_lead, q_follow, device, n_games=200, opponent="heuristic"
        )
        wr = final_result["winrate"]
        _append_metrics(run_dir, {
            "episode": args.episodes,
            "phase": "quick_validation",
            "curriculum_stage": curriculum_stage,
            "opponent": "heuristic",
            "winrate": wr,
            "avg_reward": final_result["avg_reward"],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"\n=== QUICK VALIDATION RESULT ===")
        print(f"WR vs Heuristic @ {args.episodes} episodes: {wr:.1%}")
        print(f"Curriculum reached: {current_opponent_name} (stage {curriculum_stage})")
        if wr >= 0.55:
            print("✓ LSTM is learning. Proceed with full overnight run:")
            print("  PYTHONPATH=src python -m guandan.train --episodes 30000")
        elif wr >= 0.48:
            print("~ Marginal improvement. Consider running to 15K before deciding.")
            print("  PYTHONPATH=src python -m guandan.training.train --episodes 15000")
        else:
            print("✗ Not improving. Check debugging checklist before continuing:")
            print("  1. python scripts/diagnose_device.py")
            print("  2. Verify move_history is populated: print(len(env.move_history))")
            print("  3. Check LSTM gradients are non-zero")

    # Restore stdout
    if isinstance(sys.stdout, _TeeLogger):
        sys.stdout.close()


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
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum: train directly vs --eval-opponent.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated timestamp)",
    )
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.quick:
        args.episodes = 8000
        args.eval_interval = 1000
        args.eval_games = 100

    train(args)


if __name__ == "__main__":
    main()
