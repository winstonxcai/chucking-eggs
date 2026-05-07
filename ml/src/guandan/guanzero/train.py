"""GuanZero M0 training orchestrator.

Single-process: alternate between rolling self-play episodes and taking
gradient steps. Configurable via YAML or CLI flags. Writes
``ml/runs/<run_name>/{config.json, metrics.jsonl, checkpoints/}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .actor import play_episode
from .buffer import ReplayBuffer
from .checkpoint import save_checkpoint
from .config import TrainConfig
from .encoder import StateActionEncoder
from .learner import Learner
from .logging_setup import setup_run_logging
from .q_network import init_seat_nets
from .schedules import epsilon_linear




def _count_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _log_header(logger: logging.Logger, cfg: TrainConfig,
                q_nets: dict, log_path: Path) -> None:
    sep = "=" * 68
    params_per = {p: _count_params(q_nets[p]) for p in range(4)}
    total = sum(params_per.values())

    # File: full config table
    logger.debug(sep)
    logger.debug("  GuanZero M0 — Deep Monte Carlo training")
    logger.debug("  Log → %s", log_path)
    logger.debug(sep)
    logger.debug("CONFIG")
    for field in dataclasses.fields(cfg):
        logger.debug("  %-32s %s", field.name, getattr(cfg, field.name))
    logger.debug(sep)
    logger.debug("  Network: LSTM(108→%d) + MLP(%d layers, %d hidden) | %s params/seat | %s total",
                 cfg.qnet.hidden_lstm, cfg.qnet.n_mlp_layers, cfg.qnet.hidden_mlp,
                 f"{params_per[0]:,}", f"{total:,}")
    logger.debug(sep)

    # Stdout: compact banner (written before tqdm bar appears)
    tqdm.write(sep)
    tqdm.write(f"  GuanZero M0  |  episodes  |  "
               f"LSTM {cfg.qnet.hidden_lstm}→MLP {cfg.qnet.n_mlp_layers}×{cfg.qnet.hidden_mlp}  |  "
               f"{params_per[0]:,} params/seat")
    tqdm.write(f"  Log → {log_path}")
    tqdm.write(sep)


def _fmt_losses(losses: dict[int, float]) -> str:
    if not losses:
        return "—"
    parts = [f"p{p}={v:.4f}" for p, v in sorted(losses.items())]
    avg = sum(losses.values()) / len(losses)
    return " ".join(parts) + f"  avg={avg:.4f}"


def _fmt_eta(elapsed_s: float, ep: int, total: int) -> str:
    if ep == 0:
        return "?"
    rate = ep / elapsed_s
    remaining_s = (total - ep) / rate
    h = int(remaining_s // 3600)
    m = int((remaining_s % 3600) // 60)
    s = int(remaining_s % 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"



def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path(cfg.resolved_run_dir)
    logger, log_path = setup_run_logging(run_dir)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    metrics_path = run_dir / "metrics.jsonl"

    encoder = StateActionEncoder(use_oracle_others_hand=cfg.qnet.use_oracle_others_hand)
    q_nets = init_seat_nets(cfg.qnet)
    learner = Learner(q_nets=q_nets, lr=cfg.lr, device=cfg.device,
                      use_bf16=cfg.use_bf16_learner)
    buffer = ReplayBuffer(capacity_per_player=cfg.buffer_capacity_per_player)

    _log_header(logger, cfg, q_nets, log_path)
    t0 = time.time()
    last_log_t = t0
    last_log_ep = 0
    last_losses: dict[int, float] = {}

    bar = tqdm(
        range(1, cfg.episodes + 1),
        desc="training",
        unit="ep",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    for ep in bar:
        eps = epsilon_linear(ep, cfg.epsilon)
        for net in q_nets.values():
            net.eval()
        samples = play_episode(
            q_nets=q_nets,
            encoder=encoder,
            epsilon=eps,
            seed=cfg.seed + ep,
            device=cfg.device,
            gamma=cfg.gamma,
        )
        buffer.push(samples)

        if ep % cfg.learn_every_episodes == 0:
            for net in q_nets.values():
                net.train()
            last_losses = learner.update(
                buffer=buffer,
                batch_size=cfg.batch_size,
                min_buffer_size=cfg.buffer_min_size,
            )
            # Keep bar postfix current after every learner step
            if last_losses:
                avg_loss = sum(last_losses.values()) / len(last_losses)
                bar.set_postfix(
                    ε=f"{eps:.3f}",
                    buf=buffer.total_size(),
                    loss=f"{avg_loss:.4f}",
                    refresh=False,
                )

        if ep % cfg.log_every_episodes == 0:
            now = time.time()
            elapsed_total = now - t0
            interval = now - last_log_t
            rate = (ep - last_log_ep) / max(interval, 1e-6)
            eta = _fmt_eta(elapsed_total, ep, cfg.episodes)
            buf_pp = {p: buffer.size(p) for p in range(4)}

            # Structured row to metrics.jsonl
            row = {
                "episode": ep,
                "epsilon": round(eps, 4),
                "buffer_total": buffer.total_size(),
                "buffer_per_player": buf_pp,
                "loss": {str(p): round(v, 6) for p, v in last_losses.items()},
                "eps_per_sec": round(rate, 2),
                "elapsed_s": round(elapsed_total, 1),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

            # File: full structured line
            buf_str = "/".join(str(buf_pp[p]) for p in range(4))
            logger.info(
                "ep=%d/%d  ε=%.3f  buf=%d (%s)  loss: %s  rate=%.1f ep/s  ETA=%s",
                ep, cfg.episodes, eps, buffer.total_size(), buf_str,
                _fmt_losses(last_losses), rate, eta,
            )

            # Stdout: tqdm.write so the bar doesn't get stomped
            tqdm.write(
                f"  ep {ep:>{len(str(cfg.episodes))}}/{cfg.episodes} "
                f"| ε={eps:.3f} | buf={buffer.total_size()} ({buf_str}) "
                f"| loss: {_fmt_losses(last_losses)} "
                f"| {rate:.1f} ep/s | ETA {eta}"
            )

            last_log_t = now
            last_log_ep = ep

        if ep % cfg.checkpoint_every_episodes == 0 or ep == cfg.episodes:
            ckpt = run_dir / "checkpoints" / f"ep_{ep:07d}.pt"
            save_checkpoint(ckpt, q_nets, cfg, ep)
            logger.debug("checkpoint saved → %s", ckpt)

    bar.close()
    final_ckpt = run_dir / "checkpoints" / "final.pt"
    save_checkpoint(final_ckpt, q_nets, cfg, cfg.episodes)
    elapsed = time.time() - t0
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    sep = "=" * 68
    summary = (f"  Done: {cfg.episodes} episodes in "
               f"{h}h{m:02d}m{s:02d}s  "
               f"({cfg.episodes / max(elapsed, 1e-6):.1f} ep/s avg)  "
               f"→ {final_ckpt}")
    logger.info(sep)
    logger.info(summary)
    logger.info(sep)
    tqdm.write(sep)
    tqdm.write(summary)
    tqdm.write(sep)


# ─── CLI ─────────────────────────────────────────────────


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config; CLI flags override.")
    p.add_argument("--episodes", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--run-dir", type=str)
    p.add_argument("--quick", action="store_true",
                   help="Smoke run: 100 episodes, tiny network.")
    args = p.parse_args()

    cfg_dict: dict[str, Any] = {}
    if args.config:
        import yaml
        cfg_dict.update(yaml.safe_load(Path(args.config).read_text()) or {})
    if args.episodes is not None: cfg_dict["episodes"] = args.episodes
    if args.seed is not None: cfg_dict["seed"] = args.seed
    if args.device is not None: cfg_dict["device"] = args.device
    if args.run_dir is not None: cfg_dict["run_dir"] = args.run_dir
    if args.quick:
        cfg_dict.update({
            "episodes": 100,
            "hidden_lstm": 64,
            "hidden_mlp": 128,
            "n_mlp_layers": 3,
            "log_every_episodes": 10,
            "checkpoint_every_episodes": 50,
            "buffer_min_size": 50,
            "learn_every_episodes": 2,
        })

    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
    return TrainConfig(**cfg_dict)


def main() -> None:
    cfg = _parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
