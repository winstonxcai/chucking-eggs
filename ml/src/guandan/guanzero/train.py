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
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .actor import play_episode
from .buffer import ReplayBuffer
from .encoder import StateActionEncoder
from .learner import Learner
from .q_network import init_position_nets


@dataclasses.dataclass
class TrainConfig:
    seed: int = 0
    episodes: int = 30_000

    gamma: float = 1.0
    batch_size: int = 512
    lr: float = 1e-4

    epsilon_start: float = 0.1
    epsilon_final: float = 0.01
    epsilon_decay_episodes: int = 15_000

    learn_every_episodes: int = 4
    log_every_episodes: int = 100
    checkpoint_every_episodes: int = 5_000

    buffer_capacity_per_player: int = 50_000
    buffer_min_size: int = 1_000

    hidden_lstm: int = 256
    hidden_mlp: int = 1024
    n_mlp_layers: int = 6
    dropout: float = 0.0

    use_oracle_others_hand: bool = True
    history_window: int = 20  # informational; encoder uses HISTORY_LEN
    max_legal_actions: int = 128

    device: str = "cpu"
    run_dir: str = ""   # empty → auto-generate timestamped name via resolved_run_dir

    # ── Distributed actor-learner fields ──────────────────
    n_actors: int = 1
    sync_interval_episodes: int = 20
    actor_push_batch_size: int = 512
    sample_queue_maxsize: int = 64
    max_drain_batches_per_loop: int = 32
    publish_interval_updates: int = 100
    checkpoint_every_updates: int = 5_000
    total_updates_target: int = 0    # 0 = run until stopped; >0 = stop here
    log_every_updates: int = 200
    updates_per_learner_step: int = 1

    # ── A10G / CUDA throughput knobs ──────────────────────
    use_bf16_learner: bool = False     # BF16 autocast in Learner.update (cuda only)
    compile_mode: str = "default"      # passes to torch.compile(mode=...)
    compile_actor: bool = False        # torch.compile actor q-nets (LOGBOOK §50: regresses paper-spec)
    env_lanes_per_actor: int = 1       # >1 enables VectorizedRollout for batched per-seat fwd

    # ── Shared GPU inference server (Phase 4+) ────────────
    # When true, actors send inference requests to a shared GPU server instead
    # of running local CPU q-nets. Server lives in its own subprocess on the
    # same GPU as the learner.
    use_inference_server:               bool  = False
    inference_device:                   str   = "cuda"   # "cpu" for M1 dev/test
    inference_batch_max_requests:       int   = 32
    inference_batch_max_action_rows:    int   = 4096
    inference_batch_timeout_ms:         float = 5.0      # bench §45 sweet spot
    inference_n_slots:                  int   = 512
    inference_max_actions:              int   = 320      # match max_legal_actions
    inference_timeout_s:                float = 60.0     # actor-side wait timeout
    inference_weight_refresh_s:         float = 5.0      # disk-based refresh interval

    # ── Replay-ratio controller (Phase 5) ─────────────────
    target_replay_ratio:                float = 0.0      # 0 = disabled
    max_replay_ratio:                   float = 4.0
    max_throttle_sleep_s:               float = 0.05

    @property
    def resolved_run_dir(self) -> str:
        if self.run_dir:
            return self.run_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"ml/runs/guanzero_m0_{ts}"


def _epsilon(ep: int, cfg: TrainConfig) -> float:
    if cfg.epsilon_decay_episodes <= 0:
        return cfg.epsilon_final
    frac = min(1.0, ep / cfg.epsilon_decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_final - cfg.epsilon_start)


def _setup_logging(run_dir: Path) -> tuple[logging.Logger, Path]:
    """Configure logging.

    File handler — DEBUG level, full timestamp + level prefix.
                   Gets everything: config block, model summary, per-interval rows.
    No stream handler — stdout is owned by the tqdm progress bar.
                        Use ``tqdm.write()`` for any messages alongside the bar.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    logger = logging.getLogger("guanzero")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if os.environ.get("GUANZERO_STREAM_LOGS") == "1":
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger, log_path


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
                 cfg.hidden_lstm, cfg.n_mlp_layers, cfg.hidden_mlp,
                 f"{params_per[0]:,}", f"{total:,}")
    logger.debug(sep)

    # Stdout: compact banner (written before tqdm bar appears)
    tqdm.write(sep)
    tqdm.write(f"  GuanZero M0  |  {cfg.episodes} episodes  |  "
               f"LSTM {cfg.hidden_lstm}→MLP {cfg.n_mlp_layers}×{cfg.hidden_mlp}  |  "
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


def _save_checkpoint(
    path: Path,
    q_nets: dict[int, Any],
    cfg: TrainConfig,
    episode: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _unwrap = lambda net: getattr(net, "_orig_mod", net)
    torch.save(
        {
            "episode": episode,
            "config": dataclasses.asdict(cfg),
            "q_nets": {p: _unwrap(q_nets[p]).state_dict() for p in range(4)},
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path(cfg.resolved_run_dir)
    logger, log_path = _setup_logging(run_dir)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    metrics_path = run_dir / "metrics.jsonl"

    encoder = StateActionEncoder(use_oracle_others_hand=cfg.use_oracle_others_hand)
    q_nets = init_position_nets(
        hidden_lstm=cfg.hidden_lstm,
        hidden_mlp=cfg.hidden_mlp,
        n_mlp_layers=cfg.n_mlp_layers,
        dropout=cfg.dropout,
        use_oracle_others_hand=cfg.use_oracle_others_hand,
    )
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
        eps = _epsilon(ep, cfg)
        for net in q_nets.values():
            net.eval()
        samples = play_episode(
            q_nets=q_nets,
            encoder=encoder,
            epsilon=eps,
            max_legal_actions=cfg.max_legal_actions,
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
            _save_checkpoint(ckpt, q_nets, cfg, ep)
            logger.debug("checkpoint saved → %s", ckpt)

    bar.close()
    final_ckpt = run_dir / "checkpoints" / "final.pt"
    _save_checkpoint(final_ckpt, q_nets, cfg, cfg.episodes)
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
