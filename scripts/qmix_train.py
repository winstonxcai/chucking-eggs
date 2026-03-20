#!/usr/bin/env python
"""QMIX training: Phase A (detached mixer) and Phase B (end-to-end).

Usage:
  # Phase A only (fill buffer + train mixer, Q-nets frozen):
  PYTHONPATH=src python scripts/qmix_train.py \\
      --resume checkpoints/selfplay_best.pt \\
      --phase A --episodes 10000 --run-name qmix_phaseA

  # Phase B only (end-to-end, requires --mixer-resume):
  PYTHONPATH=src python scripts/qmix_train.py \\
      --resume checkpoints/selfplay_best.pt \\
      --phase B --episodes 15000 --mixer-resume checkpoints/mixer_detached.pt \\
      --run-name qmix_phaseB

  # Both phases sequentially:
  PYTHONPATH=src python scripts/qmix_train.py \\
      --resume checkpoints/selfplay_best.pt \\
      --phase AB --episodes 15000 --run-name qmix_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from guandan.agents import GreedyBot, HeuristicBot, RandomBot, RLAgentLSTM, StrategicBot
from guandan.cards import Rank
from guandan.game import GuanDanEnv
from guandan.training.mixing import TeamMixer
from guandan.training.q_network import QNetworkLSTM, get_device
from guandan.training.qmix import fill_buffers, train_mixer_step, train_qmix_e2e_step, verify_e2e_gradients
from guandan.training.qmix_buffer import QMIXBuffer

log = logging.getLogger(__name__)


def _setup_logging(log_path: Path) -> None:
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )


def _run_eval(q_lead, q_follow, device, level_rank, n_games: int = 100) -> dict:
    """Ladder eval against all opponents."""
    rl = RLAgentLSTM(q_lead, q_follow, device, level_rank)
    opponents = [
        ("Random",    RandomBot(),              min(n_games, 200)),
        ("Greedy",    GreedyBot(level_rank),    min(n_games, 200)),
        ("Heuristic", HeuristicBot(level_rank), n_games),
        ("Strategic", StrategicBot(level_rank), n_games),
    ]

    log.info("%-12s | %5s | %6s | %5s %5s %5s", "Opponent", "Games", "WR", "1-2", "1-3", "1-4")
    log.info("-" * 52)
    results = {}
    for name, opp, ng in opponents:
        wins = f12 = f13 = f14 = 0
        env = GuanDanEnv(level_rank)
        for _ in range(ng):
            env.reset()
            while not env.done:
                p = env.current_player
                env.step(rl.act(env, p) if p in (0, 2) else opp.act(env, p))
            rewards = env.get_rewards()
            if rewards[0] + rewards[2] > 0:
                wins += 1
            order = env.finish_order
            pos = sorted(order.index(p) + 1 for p in (0, 2) if p in order)
            if pos == [1, 2]:
                f12 += 1
            elif 1 in pos and 3 in pos:
                f13 += 1
            elif 1 in pos and 4 in pos:
                f14 += 1
        wr = wins / ng
        log.info("%-12s | %5d | %5.1f%% | %5d %5d %5d", name, ng, 100 * wr, f12, f13, f14)
        results[name.lower()] = wr
    return results


def phase_a(q_lead, q_follow, mixer, opt_mixer, device, level_rank, args) -> str:
    """Phase A: freeze Q-nets, train mixer (Q-values re-computed with no_grad)."""
    log.info("=" * 60)
    log.info("PHASE A — Detached mixer training")
    log.info("=" * 60)

    qmix_buf = QMIXBuffer(capacity=50_000)
    env = GuanDanEnv(level_rank)

    q_lead.eval()
    q_follow.eval()
    for p in list(q_lead.parameters()) + list(q_follow.parameters()):
        p.requires_grad_(False)

    # Fill buffer
    fill_ep = max(args.episodes // 2, 5000)
    log.info("Filling QMIX buffer: %d episodes...", fill_ep)
    n_tricks = fill_buffers(
        env, q_lead, q_follow, args.epsilon_start, device, level_rank,
        qmix_buf, fill_ep,
    )
    log.info("Buffer: %d tricks from %d episodes", n_tricks, fill_ep)

    # Train mixer
    train_ep = args.episodes - fill_ep
    train_steps = max(train_ep * 4, 1000)
    log.info("Training mixer: %d steps...", train_steps)
    losses = []
    for step in tqdm(range(train_steps), desc="Mixer training", unit="step"):
        if len(qmix_buf) < args.batch_size:
            continue
        loss = train_mixer_step(mixer, q_lead, q_follow, qmix_buf, opt_mixer, args.batch_size, device)
        losses.append(loss)
        if (step + 1) % 500 == 0:
            log.info("Step %d | mixer_loss=%.4f", step + 1, sum(losses[-500:]) / 500)

    # Re-enable Q-net gradients for Phase B
    for p in list(q_lead.parameters()) + list(q_follow.parameters()):
        p.requires_grad_(True)

    ckpt_path = os.path.join(args.checkpoint_dir, "mixer_detached.pt")
    torch.save({
        "mixer": mixer.state_dict(),
        "lead": q_lead.state_dict(),
        "follow": q_follow.state_dict(),
    }, ckpt_path)
    log.info("Phase A done. Saved mixer → %s", ckpt_path)

    # Eval: confirm WR unchanged
    log.info("Phase A eval (WR should be ≈ baseline):")
    _run_eval(q_lead, q_follow, device, level_rank, args.eval_games)

    return ckpt_path


def phase_b(q_lead, q_follow, mixer, opt_e2e, device, level_rank, args) -> None:
    """Phase B: true end-to-end QMIX — mixer loss back-props to Q-net weights."""
    log.info("=" * 60)
    log.info("PHASE B — End-to-end QMIX training")
    log.info("=" * 60)

    qmix_buf = QMIXBuffer(capacity=50_000)
    env = GuanDanEnv(level_rank)

    # Pre-fill buffer before verifying gradients
    log.info("Pre-filling buffer for gradient verification...")
    q_lead.eval()
    q_follow.eval()
    fill_buffers(env, q_lead, q_follow, args.epsilon_start, device, level_rank, qmix_buf, 64)

    # Gate: verify gradient tape is connected before training
    log.info("Verifying e2e gradient flow...")
    q_lead.train()
    q_follow.train()
    verify_e2e_gradients(mixer, q_lead, q_follow, qmix_buf, device, args.batch_size)

    best_wr = 0.0
    episodes_done = 0
    train_every = 64

    with tqdm(total=args.episodes, desc="QMIX e2e", unit="ep") as pbar:
        while episodes_done < args.episodes:
            eps = max(
                args.epsilon_end,
                args.epsilon_start - (args.epsilon_start - args.epsilon_end)
                * episodes_done / max(args.episodes * args.epsilon_decay_frac, 1),
            )
            batch = min(train_every, args.episodes - episodes_done)

            q_lead.eval()
            q_follow.eval()
            fill_buffers(env, q_lead, q_follow, eps, device, level_rank, qmix_buf, batch)
            episodes_done += batch
            pbar.update(batch)
            pbar.set_postfix(qmix=f"{len(qmix_buf):,}", eps=f"{eps:.3f}")

            if len(qmix_buf) >= args.batch_size:
                q_lead.train()
                q_follow.train()
                last_losses: dict = {}
                for _ in range(args.train_steps):
                    last_losses = train_qmix_e2e_step(
                        mixer, q_lead, q_follow,
                        qmix_buf, opt_e2e,
                        args.batch_size, device,
                        loss_weight=args.qmix_loss_weight,
                    )

                if episodes_done % 500 == 0 and last_losses:
                    q_gn = last_losses.get("q_lead_grad", 0.0)
                    m_gn = last_losses.get("mixer_grad", 0.0)
                    ratio = m_gn / max(q_gn, 1e-10)
                    log.info(
                        "ep %d | qmix_loss=%.1f | grad norms — Q: %.6f  Mixer: %.4f  Ratio: %.0f×",
                        episodes_done, last_losses.get("qmix", 0.0),
                        q_gn, m_gn, ratio,
                    )

            if episodes_done > 0 and episodes_done % args.eval_interval == 0:
                log.info("=" * 60)
                log.info("EVAL @ ep %d/%d | ε=%.3f", episodes_done, args.episodes, eps)
                log.info("=" * 60)
                q_lead.eval()
                q_follow.eval()
                results = _run_eval(q_lead, q_follow, device, level_rank, args.eval_games)
                wr_h = results.get("heuristic", 0)
                if wr_h > best_wr:
                    best_wr = wr_h
                    path = os.path.join(args.checkpoint_dir, "qmix_best.pt")
                    torch.save({
                        "lead": q_lead.state_dict(),
                        "follow": q_follow.state_dict(),
                        "mixer": mixer.state_dict(),
                        "episode": episodes_done,
                        "wr_heuristic": wr_h,
                    }, path)
                    log.info("★ New best: %.1f%% vs Heuristic → %s", wr_h * 100, path)

    log.info("Phase B done. Best WR vs Heuristic: %.1f%%", best_wr * 100)
    path = os.path.join(args.checkpoint_dir, "qmix_final.pt")
    torch.save({
        "lead": q_lead.state_dict(),
        "follow": q_follow.state_dict(),
        "mixer": mixer.state_dict(),
    }, path)
    log.info("Saved final → %s", path)

    log.info("=" * 60)
    log.info("FINAL EVAL")
    log.info("=" * 60)
    q_lead.eval()
    q_follow.eval()
    _run_eval(q_lead, q_follow, device, level_rank, args.eval_games)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", required=True)
        parser.add_argument("--phase", choices=["A", "B", "AB"], default="AB")
        parser.add_argument("--episodes", type=int, default=15000)
        parser.add_argument("--mixer-resume", type=str, default="")
        parser.add_argument("--batch-size", type=int, default=512)
        parser.add_argument("--train-steps", type=int, default=4)
        parser.add_argument("--lr-mixer", type=float, default=1e-3)
        parser.add_argument("--lr-q", type=float, default=1e-6)
        parser.add_argument("--qmix-loss-weight", type=float, default=0.001,
                            help="Scale factor for QMIX loss (tune: 0.0001→0.001→0.01)")
        parser.add_argument("--eval-interval", type=int, default=2000)
        parser.add_argument("--eval-games", type=int, default=100)
        parser.add_argument("--epsilon-start", type=float, default=0.10)
        parser.add_argument("--epsilon-end", type=float, default=0.01)
        parser.add_argument("--epsilon-decay-frac", type=float, default=0.80)
        parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
        parser.add_argument("--run-name", type=str, default="qmix")
        args = parser.parse_args()

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir / "train.log")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = get_device()
    level_rank = Rank.TWO
    t0 = time.time()

    log.info("Device: %s | Phase: %s | Episodes: %d", device, args.phase, args.episodes)

    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=True)
    q_lead.load_state_dict(ckpt["lead"])
    q_follow.load_state_dict(ckpt["follow"])
    log.info("Loaded Q-networks from %s", args.resume)

    mixer = TeamMixer().to(device)
    opt_mixer = torch.optim.Adam(mixer.parameters(), lr=args.lr_mixer)

    if args.mixer_resume:
        mc = torch.load(args.mixer_resume, map_location=device, weights_only=True)
        mixer.load_state_dict(mc["mixer"])
        log.info("Loaded mixer from %s", args.mixer_resume)

    # Phase B: single param-group optimizer — different LRs per component
    opt_e2e = torch.optim.Adam([
        {"params": mixer.parameters(),    "lr": args.lr_mixer},
        {"params": q_lead.parameters(),   "lr": args.lr_q},
        {"params": q_follow.parameters(), "lr": args.lr_q},
    ])

    # Save config
    cfg = vars(args)
    cfg["device"] = str(device)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    if args.phase in ("A", "AB"):
        phase_a(q_lead, q_follow, mixer, opt_mixer, device, level_rank, args)

    if args.phase in ("B", "AB"):
        # For AB: load fresh mixer after phase A
        if args.phase == "AB":
            mc = torch.load(
                os.path.join(args.checkpoint_dir, "mixer_detached.pt"),
                map_location=device, weights_only=True,
            )
            mixer.load_state_dict(mc["mixer"])
            # Rebuild opt_e2e so Phase A's state doesn't carry over
            opt_e2e = torch.optim.Adam([
                {"params": mixer.parameters(),    "lr": args.lr_mixer},
                {"params": q_lead.parameters(),   "lr": args.lr_q},
                {"params": q_follow.parameters(), "lr": args.lr_q},
            ])
        phase_b(q_lead, q_follow, mixer, opt_e2e, device, level_rank, args)

    log.info("Total time: %.0fs (%.1fh)", time.time() - t0, (time.time() - t0) / 3600)


if __name__ == "__main__":
    main()
