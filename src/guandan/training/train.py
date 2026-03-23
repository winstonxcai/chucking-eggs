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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..agents import make_agent
from ..agents.heuristic_bot import HeuristicBot
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
    team_spirit: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """Play one full game, collect transitions, assign terminal rewards.

    RL agent plays seats {0,2}. If opponent is provided, it plays seats {1,3};
    otherwise all 4 seats use the RL agent (self-play).

    team_spirit: mix partner reward into mc_return (0=individual, 1=fully shared).
        mc_return = (1-ts)*rewards[player] + ts*rewards[partner]

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
    partners = {0: 2, 1: 3, 2: 0, 3: 1}
    all_trans = []
    for player, tlist in transitions.items():
        r_self = rewards[player]
        r_partner = rewards[partners[player]]
        mc_return = (1 - team_spirit) * r_self + team_spirit * r_partner
        for s, a, h, hl in tlist:
            all_trans.append((s, a, h, hl, mc_return))
    return all_trans


def _episode_worker(
    lead_sd: dict,
    follow_sd: dict,
    lstm_hidden: int,
    mlp_hidden: int,
    epsilon: float,
    opponent_name: str | None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]:
    """Run one episode in a subprocess with CPU inference."""
    device = torch.device("cpu")
    q_lead = QNetworkLSTM(lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=lstm_hidden, hidden=mlp_hidden).to(device)
    q_lead.load_state_dict(lead_sd)
    q_follow.load_state_dict(follow_sd)
    q_lead.eval()
    q_follow.eval()
    env = GuanDanEnv(level_rank=Rank.TWO)
    opp = make_agent(opponent_name, env.level_rank) if opponent_name else None
    return play_episode(env, q_lead, q_follow, epsilon, device, opponent=opp)


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


def pretrain_from_heuristic(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    opt_lead: torch.optim.Optimizer,
    opt_follow: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    device: torch.device,
    n_games: int = 5000,
    batch_size: int = 1024,
    train_steps_per_game: int = 2,
    eval_interval: int = 500,
    eval_games: int = 200,
    run_dir: Path | None = None,
) -> int:
    """Pre-fill buffer with heuristic self-play, train Q-networks to imitate.

    Returns the number of transitions pushed to the buffer.
    """
    level_rank = Rank.TWO
    heuristic = HeuristicBot(level_rank)
    env = GuanDanEnv(level_rank=level_rank)
    total_trans = 0

    print(f"\n{'='*60}")
    print(f"  PHASE 1: Heuristic Imitation ({n_games} games)")
    print(f"{'='*60}\n")

    pbar = tqdm(range(1, n_games + 1), desc="pretrain", unit="game",
                file=sys.stderr, dynamic_ncols=True)
    for game in pbar:
        env.reset()
        transitions: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {
            p: [] for p in range(4)
        }

        while not env.done:
            player = env.current_player
            action = heuristic.act(env, player)

            state_enc = encode_state(env, player)
            action_enc = encode_action(action, env.hands[player], level_rank)
            history, hist_len = encode_history(env, player, level_rank)

            transitions[player].append((state_enc, action_enc, history, hist_len))
            env.step(action)

        rewards = env.get_rewards()
        for player, tlist in transitions.items():
            mc_return = rewards[player]
            for s, a, h, hl in tlist:
                buffer.push(s, a, h, hl, mc_return)
                total_trans += 1

        # Train on buffer
        loss_lead = loss_follow = None
        for _ in range(train_steps_per_game):
            ll = train_step(q_lead, buffer, opt_lead, batch_size, device)
            lf = train_step(q_follow, buffer, opt_follow, batch_size, device)
            if ll is not None:
                loss_lead = ll
            if lf is not None:
                loss_follow = lf

        if loss_lead is not None:
            pbar.set_postfix(buf=len(buffer), loss=f"{(loss_lead + loss_follow) / 2:.4f}")

        # Periodic eval
        if game % eval_interval == 0:
            result = evaluate(q_lead, q_follow, device, n_games=eval_games,
                              opponent="heuristic")
            tqdm.write(
                f"Pretrain {game}/{n_games} | buf={len(buffer):,} | "
                f"vs heuristic: {result['winrate']:.1%} "
                f"(1-2:{result['finish_12']} 1-3:{result['finish_13']} "
                f"1-4:{result['finish_14']})"
            )
            if run_dir is not None:
                _append_metrics(run_dir, {
                    "phase": "pretrain",
                    "game": game,
                    "buffer_size": len(buffer),
                    "loss_lead": loss_lead,
                    "loss_follow": loss_follow,
                    "winrate": result["winrate"],
                    "avg_reward": result["avg_reward"],
                    "finish_12": result["finish_12"],
                    "finish_13": result["finish_13"],
                    "finish_14": result["finish_14"],
                })

    pbar.close()
    print(f"Pretrain complete. Buffer: {len(buffer):,} transitions "
          f"({total_trans:,} from heuristic games)")
    return total_trans


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
    finish_12 = finish_13 = finish_14 = 0
    finish_23 = finish_24 = finish_34 = 0

    for _ in tqdm(range(n_games), desc=f"eval vs {opponent}", unit="game",
                  leave=False, file=sys.stderr, dynamic_ncols=True):
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

        # Track finish type — team positions in finish_order
        fo = env.finish_order
        team_pos = sorted(fo.index(p) for p in (0, 2))  # e.g. [0, 1] = 1-2 finish
        key = (team_pos[0] + 1, team_pos[1] + 1)  # 1-indexed
        if key == (1, 2):
            finish_12 += 1
        elif key == (1, 3):
            finish_13 += 1
        elif key == (1, 4):
            finish_14 += 1
        elif key == (2, 3):
            finish_23 += 1
        elif key == (2, 4):
            finish_24 += 1
        elif key == (3, 4):
            finish_34 += 1

    return {
        "winrate": wins / n_games,
        "avg_reward": total_reward / n_games,
        "finish_12": finish_12,
        "finish_13": finish_13,
        "finish_14": finish_14,
        "finish_23": finish_23,
        "finish_24": finish_24,
        "finish_34": finish_34,
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

    t0 = time.time()

    n_lead_params = sum(p.numel() for p in q_lead.parameters())
    n_follow_params = sum(p.numel() for p in q_follow.parameters())
    print(f"Lead params: {n_lead_params:,}  Follow params: {n_follow_params:,}")
    print(f"Training: heuristic imitation → self-play")
    print(f"Eval opponent: heuristic")

    # ── Phase 1: Heuristic Imitation ──────────────────────────
    pretrain_trans = 0
    if args.pretrain_games > 0:
        pretrain_trans = pretrain_from_heuristic(
            q_lead, q_follow, opt_lead, opt_follow, buffer, device,
            n_games=args.pretrain_games, batch_size=args.batch_size,
            eval_games=args.eval_games, run_dir=run_dir,
        )
        # Save pretrain checkpoint
        pretrain_path = run_dir / "checkpoint_pretrain.pt"
        torch.save({
            "phase": "pretrain",
            "pretrain_games": args.pretrain_games,
            "lead_state_dict": q_lead.state_dict(),
            "follow_state_dict": q_follow.state_dict(),
            "opt_lead_state_dict": opt_lead.state_dict(),
            "opt_follow_state_dict": opt_follow.state_dict(),
        }, pretrain_path)
        print(f"  Saved {pretrain_path}")

    # ── Phase 2: Self-Play ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Self-Play ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"Buffer: {len(buffer):,} transitions "
          f"({pretrain_trans:,} from pretrain, will be gradually replaced)\n")

    eps_start, eps_end = 0.20, 0.05  # Lower start since network is warm
    eps_decay_episodes = int(args.episodes * 0.85)
    best_wr = 0.0
    evals_without_improvement = 0

    # Parallel episode collection setup
    n_workers = args.n_workers
    use_parallel = n_workers > 1
    pool = ProcessPoolExecutor(max_workers=n_workers) if use_parallel else None

    def _sync_weights():
        return (
            {k: v.cpu() for k, v in q_lead.state_dict().items()},
            {k: v.cpu() for k, v in q_follow.state_dict().items()},
        )

    if use_parallel:
        lead_sd, follow_sd = _sync_weights()

    ep = 0
    last_eval_ep = 0
    last_save_ep = 0
    pbar = tqdm(
        total=args.episodes, unit="ep", file=sys.stderr,
        dynamic_ncols=True, desc="self-play",
    )
    while ep < args.episodes:
        batch_size_ep = min(n_workers, args.episodes - ep)
        frac = min(1.0, (ep + 1) / eps_decay_episodes)
        epsilon = eps_start + (eps_end - eps_start) * frac

        # Self-play: opponent=None, all 4 seats use Q-network
        if use_parallel:
            futures = [
                pool.submit(
                    _episode_worker, lead_sd, follow_sd,
                    args.lstm_hidden, args.mlp_hidden,
                    epsilon, None,  # self-play
                )
                for _ in range(batch_size_ep)
            ]
            for f in futures:
                for s, a, h, hl, mc_return in f.result():
                    buffer.push(s, a, h, hl, mc_return)
        else:
            trans = play_episode(
                env, q_lead, q_follow, epsilon, device, opponent=None
            )
            for s, a, h, hl, mc_return in trans:
                buffer.push(s, a, h, hl, mc_return)

        # Gradient steps — scale with batch size
        loss_lead = None
        loss_follow = None
        for _ in range(args.train_steps * batch_size_ep):
            ll = train_step(q_lead, buffer, opt_lead, args.batch_size, device)
            lf = train_step(q_follow, buffer, opt_follow, args.batch_size, device)
            if ll is not None:
                loss_lead = ll
            if lf is not None:
                loss_follow = lf

        # Sync weights to workers after training steps
        if use_parallel:
            lead_sd, follow_sd = _sync_weights()

        ep += batch_size_ep
        pbar.update(batch_size_ep)
        if loss_lead is not None:
            pbar.set_postfix(
                ε=f"{epsilon:.3f}",
                loss=f"{(loss_lead + loss_follow) / 2:.4f}" if loss_follow is not None else f"{loss_lead:.4f}",
                wr=f"{best_wr:.1%}",
                buf=len(buffer),
            )

        # Evaluate and log
        if ep - last_eval_ep >= args.eval_interval:
            last_eval_ep = ep
            result = evaluate(
                q_lead, q_follow, device,
                n_games=args.eval_games, opponent="heuristic",
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

            tqdm.write(
                f"Ep {ep:>6d} | ε={epsilon:.3f} | "
                f"L_lead={ll_str} L_follow={lf_str} | "
                f"buf={len(buffer):>6d} | "
                f"vs heuristic: {wr:.1%} "
                f"(1-2:{result['finish_12']} 1-3:{result['finish_13']} 1-4:{result['finish_14']}) "
                f"(best={best_wr:.1%}, "
                f"pat={evals_without_improvement}/{args.patience}) | "
                f"{elapsed:.0f}s"
            )

            _append_metrics(run_dir, {
                "phase": "self-play",
                "episode": ep,
                "epsilon": round(epsilon, 4),
                "loss_lead": loss_lead,
                "loss_follow": loss_follow,
                "buffer_size": len(buffer),
                "winrate": result["winrate"],
                "avg_reward": result["avg_reward"],
                "finish_12": result["finish_12"],
                "finish_13": result["finish_13"],
                "finish_14": result["finish_14"],
                "best_winrate": best_wr,
                "elapsed_s": round(elapsed, 1),
            })

            if evals_without_improvement >= args.patience:
                tqdm.write(
                    f"Early stopping: no improvement in WR vs heuristic "
                    f"for {args.patience} evals"
                )
                break

        # Save checkpoint
        if ep - last_save_ep >= args.save_interval:
            last_save_ep = ep
            path = run_dir / f"checkpoint_ep{ep}.pt"
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
            tqdm.write(f"  Saved {path}")

    pbar.close()
    if pool is not None:
        pool.shutdown(wait=False)

    # Final save
    final_path = run_dir / "model_final.pt"
    torch.save(
        {
            "episode": args.episodes,
            "lead_state_dict": q_lead.state_dict(),
            "follow_state_dict": q_follow.state_dict(),
            "opt_lead_state_dict": opt_lead.state_dict(),
            "opt_follow_state_dict": opt_follow.state_dict(),
        },
        final_path,
    )
    print(f"Training complete. Saved {final_path}")

    # Quick validation verdict
    if getattr(args, "quick", False):
        final_result = evaluate(
            q_lead, q_follow, device, n_games=200, opponent="heuristic"
        )
        wr = final_result["winrate"]
        _append_metrics(run_dir, {
            "episode": args.episodes,
            "phase": "quick_validation",
            "opponent": "heuristic",
            "winrate": wr,
            "avg_reward": final_result["avg_reward"],
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"\n=== QUICK VALIDATION RESULT ===")
        print(f"WR vs Heuristic @ {args.episodes} episodes: {wr:.1%}")
        if wr >= 0.55:
            print("✓ LSTM is learning. Proceed with full overnight run:")
            print("  PYTHONPATH=src python -m guandan.training.train --episodes 50000")
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
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument(
        "--pretrain-games",
        type=int,
        default=5000,
        help="Number of heuristic self-play games for imitation pre-training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early stop after N evals with no improvement",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode: cap at 8000 episodes, eval every 2000, "
             "print go/no-go verdict at end. ~4-5 hours on M1 Pro.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel episode workers (1 = sequential, default: 4).",
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
        args.pretrain_games = 500
        args.episodes = 4000
        args.eval_interval = 1000
        args.eval_games = 100

    train(args)


if __name__ == "__main__":
    main()
