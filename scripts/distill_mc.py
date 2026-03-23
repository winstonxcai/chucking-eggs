#!/usr/bin/env python
"""Distill MC teacher data into Q-networks via MSE regression on candidate scores.

Loads pre-generated MC decisions (from generate_mc_data.py), fine-tunes
existing checkpoint weights by regressing Q-values toward MC's rollout scores.
MSE preserves value semantics; cross-entropy destroys them (Q-values become logits).

Usage:
  # Smoke test (1 epoch on tiny data)
  PYTHONPATH=src python scripts/distill_mc.py \\
      --data /tmp/mc_test.pt \\
      --resume checkpoints/newrewards_ts05.pt \\
      --epochs 1 --run-name mc_smoke

  # Full run
  PYTHONPATH=src python scripts/distill_mc.py \\
      --data data/mc_vs_strategic_1000.pt \\
      --resume checkpoints/newrewards_ts05.pt \\
      --epochs 5 --lr 3e-5 --run-name mc_distill
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from guandan.agents import HeuristicBot, RLAgentLSTM, StrategicBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.encoding import ACTION_DIM, D_MOVE, MAX_HISTORY
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.supervised import _log_epoch


def _setup_logging(log_path: Path) -> None:
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler()],
    )


log = logging.getLogger(__name__)


def _encode_batch(decisions, q_net, device):
    """Shared encoding: states, histories, actions → Q-value matrix + mask."""
    N = len(decisions)
    B_sizes = [len(d["action_encs"]) for d in decisions]
    max_B = max(B_sizes)

    action_pad = torch.zeros(N, max_B, ACTION_DIM, dtype=torch.float32)
    score_pad  = torch.zeros(N, max_B, dtype=torch.float32)
    mask       = torch.zeros(N, max_B, dtype=torch.bool)

    for i, d in enumerate(decisions):
        Bi = B_sizes[i]
        action_pad[i, :Bi] = torch.tensor(d["action_encs"], dtype=torch.float32)
        score_pad[i, :Bi]  = torch.tensor(d["scores"], dtype=torch.float32)
        mask[i, :Bi] = True

    states = torch.stack([torch.tensor(d["state"], dtype=torch.float32) for d in decisions])
    hists  = torch.zeros(N, MAX_HISTORY, D_MOVE, dtype=torch.float32)
    hlens  = torch.tensor([d["hist_len"] for d in decisions], dtype=torch.long)
    for i, d in enumerate(decisions):
        h = d["history"]
        T = min(len(h), MAX_HISTORY)
        hists[i, :T] = torch.tensor(h[:T], dtype=torch.float32)

    states     = states.to(device)
    hists      = hists.to(device)
    action_pad = action_pad.to(device)
    score_pad  = score_pad.to(device)
    mask       = mask.to(device)

    hist_emb = q_net.encode_history(hists, hlens)
    states_exp  = states.unsqueeze(1).expand(N, max_B, -1).reshape(N * max_B, -1)
    actions_flat = action_pad.reshape(N * max_B, -1)
    hist_exp    = hist_emb.unsqueeze(1).expand(N, max_B, -1).reshape(N * max_B, -1)

    q_flat   = q_net.forward_from_embedding(states_exp, actions_flat, hist_exp)
    q_matrix = q_flat.reshape(N, max_B)

    return q_matrix, score_pad, mask, N


def _train_batch_mse(decisions, q_net, optimizer, device, stats, role):
    """MSE regression: Q-values → MC rollout scores for each candidate."""
    q_matrix, score_pad, mask, N = _encode_batch(decisions, q_net, device)

    loss = F.mse_loss(q_matrix[mask], score_pad[mask])
    loss.backward()

    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    q_masked     = q_matrix.masked_fill(~mask, float("-inf"))
    score_masked = score_pad.masked_fill(~mask, float("-inf"))
    pred   = q_masked.argmax(dim=-1)
    target = score_masked.argmax(dim=-1)
    correct = (pred == target).sum().item()

    stats[f"{role}_losses"].append(loss.item())
    stats[f"{role}_correct"] += correct
    stats[f"{role}_total"]   += N


def _train_batch_rank(decisions, q_net, optimizer, device, stats, role, margin=0.5):
    """Margin ranking loss: push Q(best) > Q(others) by margin.

    Only teaches ordering — does not touch the absolute Q-value scale.
    Much safer than MSE when MC scores are noisy.
    """
    q_matrix, score_pad, mask, N = _encode_batch(decisions, q_net, device)

    # For each decision, find the MC-best action and form pairs
    score_masked = score_pad.masked_fill(~mask, float("-inf"))
    best_idx = score_masked.argmax(dim=-1)  # [N]

    # Gather Q-values for best action
    q_best = q_matrix.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # [N]

    # Pair with each non-best valid action
    pair_losses = []
    for i in range(q_matrix.shape[1]):
        col_mask = mask[:, i] & (torch.arange(q_matrix.shape[1], device=device)[i] != best_idx)
        if col_mask.sum() == 0:
            continue
        q_other = q_matrix[:, i]
        # MarginRankingLoss: loss = max(0, -y*(x1-x2) + margin) where y=+1
        pair_loss = F.relu(margin - (q_best - q_other))
        pair_losses.append((pair_loss * col_mask.float()).sum())

    if not pair_losses:
        return

    loss = sum(pair_losses) / mask.sum()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    q_masked = q_matrix.masked_fill(~mask, float("-inf"))
    pred   = q_masked.argmax(dim=-1)
    correct = (pred == best_idx).sum().item()

    stats[f"{role}_losses"].append(loss.item())
    stats[f"{role}_correct"] += correct
    stats[f"{role}_total"]   += N


def _run_eval(q_lead, q_follow, device, level_rank, n_games: int = 300) -> None:
    """Ladder eval vs heuristic and strategic."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)
    opponents = [
        ("Heuristic", HeuristicBot(level_rank), n_games),
        ("Strategic", StrategicBot(level_rank), n_games),
    ]

    log.info("%-12s | %5s | %6s | Net lvl/g", "Opponent", "Games", "WR")
    log.info("-" * 44)

    for name, opp, ng in opponents:
        wins = 0
        net_levels = 0.0
        env = GuanDanEnv(level_rank)
        for _ in range(ng):
            env.reset()
            while not env.done:
                p = env.current_player
                env.step(rl.act(env, p) if p in (0, 2) else opp.act(env, p))
            r = env.get_rewards()
            if r[0] + r[2] > 0:
                wins += 1
            net_levels += r[0]
        log.info("%-12s | %5d | %5.1f%% | %+.2f",
                 name, ng, 100 * wins / ng, net_levels / ng)


def main() -> None:
    parser = argparse.ArgumentParser(description="MC distillation (MSE regression) from saved data")
    parser.add_argument("--data",           type=str, required=True,
                        help="Path to decisions .pt file from generate_mc_data.py")
    parser.add_argument("--resume",         type=str, required=True,
                        help="Checkpoint to fine-tune (lead/follow keys)")
    parser.add_argument("--epochs",         type=int, default=5)
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--run-name",       type=str, default="mc_distill")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-eval",        action="store_true",
                        help="Skip post-training eval (faster smoke tests)")
    parser.add_argument("--loss",           type=str, default="rank",
                        choices=["mse", "rank"],
                        help="Loss function: mse (regress Q→MC scores) or rank (margin ranking)")
    parser.add_argument("--min-gap",        type=float, default=0.0,
                        help="Filter: only keep decisions where best-2nd MC score gap > this")
    parser.add_argument("--margin",         type=float, default=0.5,
                        help="Margin for ranking loss (default 0.5)")
    args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = get_device()
    level_rank = Rank.TWO
    t0 = time.time()

    # Load data
    log.info("Loading data: %s", args.data)
    decisions: list[dict] = torch.load(args.data, weights_only=False)
    lead_decisions   = [d for d in decisions if d["is_leading"]]
    follow_decisions = [d for d in decisions if not d["is_leading"]]
    log.info(
        "Loaded %d decisions (%d lead, %d follow)",
        len(decisions), len(lead_decisions), len(follow_decisions),
    )

    # Filter by MC score gap (remove noisy decisions where MC can't distinguish)
    if args.min_gap > 0:
        def _has_gap(d, min_gap):
            s = sorted(d["scores"], reverse=True)
            return len(s) >= 2 and (s[0] - s[1]) > min_gap

        n_lead_before, n_follow_before = len(lead_decisions), len(follow_decisions)
        lead_decisions   = [d for d in lead_decisions if _has_gap(d, args.min_gap)]
        follow_decisions = [d for d in follow_decisions if _has_gap(d, args.min_gap)]
        log.info(
            "Filtered (min_gap=%.2f): %d→%d lead, %d→%d follow",
            args.min_gap,
            n_lead_before, len(lead_decisions),
            n_follow_before, len(follow_decisions),
        )

    if not decisions:
        log.error("No decisions found in data file. Exiting.")
        return

    # Verify format: new data has "scores", old has "expert_idx"
    if "expert_idx" in decisions[0] and "scores" not in decisions[0]:
        log.error(
            "Data file uses old format (expert_idx). "
            "Re-generate with the updated generate_mc_data.py to get 'scores' format."
        )
        return

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.resume)
    q_lead   = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    q_lead.load_state_dict(ckpt["lead"])
    q_follow.load_state_dict(ckpt["follow"])

    opt_lead   = torch.optim.Adam(q_lead.parameters(),   lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Network: %s params/head  Device: %s  LR: %g  Batch: %d  Epochs: %d",
             f"{n_params:,}", device, args.lr, args.batch_size, args.epochs)
    train_fn = _train_batch_rank if args.loss == "rank" else _train_batch_mse
    loss_desc = {
        "mse": "MSE(Q-values, MC scores)",
        "rank": f"MarginRanking(margin={args.margin})",
    }[args.loss]
    log.info("Loss: %s  min_gap: %g", loss_desc, args.min_gap)

    # Training loop
    for epoch in range(args.epochs):
        stats = {
            "lead_losses": [],   "follow_losses": [],
            "lead_correct": 0,   "lead_total": 0,
            "follow_correct": 0, "follow_total": 0,
        }
        opt_lead.zero_grad()
        opt_follow.zero_grad()

        random.shuffle(lead_decisions)
        random.shuffle(follow_decisions)

        # Lead network
        with tqdm(total=len(lead_decisions), desc=f"Epoch {epoch+1}/{args.epochs} [lead]",
                  unit="dec", leave=False) as pbar:
            for i in range(0, len(lead_decisions), args.batch_size):
                batch = lead_decisions[i : i + args.batch_size]
                if batch:
                    train_fn(batch, q_lead, opt_lead, device, stats, "lead",
                            **({"margin": args.margin} if args.loss == "rank" else {}))
                pbar.update(len(batch))

        # Follow network
        with tqdm(total=len(follow_decisions), desc=f"Epoch {epoch+1}/{args.epochs} [follow]",
                  unit="dec", leave=False) as pbar:
            for i in range(0, len(follow_decisions), args.batch_size):
                batch = follow_decisions[i : i + args.batch_size]
                if batch:
                    train_fn(batch, q_follow, opt_follow, device, stats, "follow",
                            **({"margin": args.margin} if args.loss == "rank" else {}))
                pbar.update(len(batch))

        _log_epoch(epoch, args.epochs, stats)

    # Save
    ckpt_path = os.path.join(args.checkpoint_dir, "mc_distilled.pt")
    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict()}, ckpt_path)
    log.info("Saved → %s", ckpt_path)

    # Eval
    if not args.no_eval:
        log.info("=" * 60)
        log.info("POST-DISTILLATION EVAL")
        log.info("=" * 60)
        _run_eval(q_lead, q_follow, device, level_rank)

    log.info("Done. Total: %.0fs (%.1fmin)", time.time() - t0, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
