"""Stage 0: bot-agnostic distillation into the pvguan architecture.

Collects supervisor decisions via partner-visible self-play, trains the
ActorCriticNet actor head on masked-softmax cross-entropy, then validates
against acceptance gates before saving the checkpoint.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/distill_pvguan.py \\
        --supervisor oracle --supervisor-search on \\
        --n-decisions 500000 \\
        --output ml/checkpoints/pvguan_distilled_oracle.pt

    # smoke test (~5 min on M1)
    PYTHONPATH=ml/src python ml/scripts/train/distill_pvguan.py \\
        --supervisor jidan --supervisor-search off \\
        --n-decisions 5000 --epochs 2 --workers 2 \\
        --output ml/checkpoints/pvguan_distilled_smoke.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from guandan.agents import JidanBot
from guandan.agents.base import Agent
from guandan.agents.greedy_bot import GreedyBot
from guandan.agents.heuristic_bot import HeuristicBot
from guandan.agents.partner_oracle_bot import _reflect_env, PartnerOracleBot
from guandan.agents.strategic_bot import StrategicBot
from guandan.cards import Rank
from guandan.combos import Combo
from guandan.game import GuanDanEnv
from guandan.pvguan.actor_critic import ActorCriticNet, get_device
from guandan.pvguan.config import DistillConfig
from guandan.pvguan.plot_run import plot_distill
from guandan.pvguan.encoders import (
    ACTION_DIM,
    ACTOR_DIM,
    encode_action,
    encode_actor_pair_features,
)

MAX_LEGAL_PER_SAMPLE = 64

log = logging.getLogger("distill_pvguan")


# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logger(output_path: Path) -> None:
    log_path = output_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("distill_pvguan")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # File: full timestamps + level
    fmt_file = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console: clean message only (tqdm handles its own output)
    fmt_con = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_con)

    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False
    log.debug(f"Log file: {log_path}")


# ─── Partner-Visible Info State ───────────────────────────────────────────────

@dataclass(frozen=True)
class PartnerVisibleInfoState:
    """Supervisor input: own+partner hands + public info. No opponent hands."""
    own_hand:       frozenset
    partner_hand:   frozenset
    public_history: tuple          # (actor, combo) tuples
    hand_counts:    tuple          # 4 ints
    is_out:         tuple          # 4 bools
    finish_order:   tuple          # seats in finish order
    level_rank:     int
    current_trick:  Any            # Combo | None
    legal_moves:    tuple          # Combo objects

    @classmethod
    def from_env(cls, env: GuanDanEnv, player: int) -> "PartnerVisibleInfoState":
        partner = (player + 2) % 4
        return cls(
            own_hand=frozenset(env.hands[player]),
            partner_hand=frozenset(env.hands[partner]),
            public_history=tuple(env.move_history[-30:]),
            hand_counts=tuple(len(env.hands[p]) for p in range(4)),
            is_out=tuple(env.is_out),
            finish_order=tuple(env.finish_order),
            level_rank=env.level_rank,
            current_trick=env.current_trick,
            legal_moves=tuple(env.legal_moves(player)),
        )


# ─── Supervisor registry ──────────────────────────────────────────────────────

def _build_supervisor(name: str, use_search: bool, n_det: int, checkpoint: str | None) -> Agent:
    lr = Rank.TWO
    if name == "oracle":
        ckpt = checkpoint or "ml/checkpoints/jidan_policy.pt"
        return PartnerOracleBot(ckpt, level_rank=lr, use_search=use_search, n_det=n_det)
    if name == "jidan":
        return JidanBot(level_rank=lr)
    if name == "greedy":
        return GreedyBot()
    if name == "heuristic":
        return HeuristicBot()
    if name == "strategic":
        return StrategicBot()
    try:
        from guandan.agents.yaoji_bot import YaojiBot
        if name == "yaoji":
            return YaojiBot(level_rank=lr)
    except ImportError:
        pass
    raise ValueError(f"Unknown supervisor: {name!r}")


def _combo_key(combo: Combo) -> tuple:
    return (int(combo.type), tuple(sorted((c.rank, c.suit, c.deck) for c in combo.cards)))


# ─── Data collection worker ───────────────────────────────────────────────────

def _worker(args: tuple) -> list[dict]:
    n_dec, level_range, seed, sup_name, use_search, n_det, ckpt_path = args
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

    from guandan.agents import JidanBot
    from guandan.agents.partner_oracle_bot import PartnerOracleBot, _reflect_env
    from guandan.cards import Rank
    from guandan.combos import Combo
    from guandan.game import GuanDanEnv
    from guandan.pvguan.encoders import encode_action, encode_actor_pair_features

    random.seed(seed)
    np.random.seed(seed)

    supervisor = _build_supervisor(sup_name, use_search, n_det, ckpt_path)
    env = GuanDanEnv(level_rank=Rank.TWO)
    samples: list[dict] = []

    while len(samples) < n_dec:
        level_rank = random.randint(*level_range)
        env.level_rank = level_rank
        env.reset()

        while not env.done:
            player = env.current_player
            needs_reflect = player in (1, 3)
            canonical_player = player ^ 1 if needs_reflect else player
            enc_env = _reflect_env(env) if needs_reflect else env

            legal = enc_env.legal_moves(canonical_player)
            if len(legal) <= 1:
                pick = supervisor.act(enc_env, canonical_player)
                env.step(pick)
                continue

            sup_pick = supervisor.act(enc_env, canonical_player)
            pick_key = _combo_key(sup_pick)

            pick_idx_full = next(
                (i for i, a in enumerate(legal) if _combo_key(a) == pick_key), 0
            )

            if len(legal) > MAX_LEGAL_PER_SAMPLE:
                scored = sorted(range(len(legal)),
                                key=lambda i: sum(c.rank for c in legal[i].cards))
                keep_idx = scored[:MAX_LEGAL_PER_SAMPLE]
                if pick_idx_full not in keep_idx:
                    keep_idx[-1] = pick_idx_full
                keep_idx = sorted(keep_idx)
                capped_legal = [legal[i] for i in keep_idx]
                pick_idx = keep_idx.index(pick_idx_full)
            else:
                capped_legal = legal
                pick_idx = pick_idx_full

            hand = list(enc_env.hands[canonical_player])
            sa = np.stack([
                encode_actor_pair_features(enc_env, canonical_player, a, capped_legal)
                for a in capped_legal
            ]).astype(np.float32)
            ac = np.stack([
                encode_action(a, hand, enc_env.level_rank)
                for a in capped_legal
            ]).astype(np.float32)

            samples.append({
                "states":     sa,
                "actions":    ac,
                "pick_idx":   pick_idx,
                "n_legal":    len(capped_legal),
                "is_pass":    sup_pick.type.name == "PASS",
                "level_rank": level_rank,
            })
            env.step(sup_pick)

    return samples[:n_dec]


# ─── Collection stats ─────────────────────────────────────────────────────────

def _log_collection_stats(samples: list[dict]) -> None:
    Ks        = np.array([s["n_legal"]    for s in samples])
    is_pass   = np.array([s["is_pass"]    for s in samples], dtype=bool)
    ranks     = np.array([s["level_rank"] for s in samples])

    log.info(f"  K (legal moves/decision):  "
             f"mean={Ks.mean():.1f}  "
             f"p50={np.percentile(Ks, 50):.0f}  "
             f"p95={np.percentile(Ks, 95):.0f}  "
             f"max={Ks.max()}")
    log.info(f"  pass_rate={is_pass.mean():.3f}  "
             f"non_trivial (K>1)={(Ks > 1).mean():.3f}")

    # Level rank distribution (should be roughly uniform over 2..14)
    rank_names  = "2  3  4  5  6  7  8  9  T  J  Q  K  A"
    rank_counts = "  ".join(f"{(ranks == r).sum():4d}" for r in range(2, 15))
    log.info(f"  level_rank dist:  {rank_names}")
    log.info(f"                    {rank_counts}")

    # Memory estimate for 764+198 tensors at float32
    mem_mb = sum(
        s["states"].nbytes + s["actions"].nbytes for s in samples
    ) / 1e6
    log.info(f"  dataset RAM: {mem_mb:.0f} MB  "
             f"({len(samples):,} samples)")


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_data(
    n_decisions: int,
    workers: int,
    sup_name: str,
    use_search: bool,
    n_det: int,
    ckpt_path: str | None,
    seed: int,
    level_range: tuple[int, int] = (2, 14),
) -> list[dict]:
    # Split into workers*4 chunks for a smoother progress bar
    n_chunks  = max(workers * 4, 4)
    per_chunk = math.ceil(n_decisions / n_chunks)

    log.info(f"  data collection:  {workers} workers  "
             f"{n_chunks} chunks × ~{per_chunk:,} decisions  "
             f"(target {n_decisions:,})")

    arg_list = [
        (per_chunk, level_range, seed + w, sup_name, use_search, n_det, ckpt_path)
        for w in range(n_chunks)
    ]

    t0 = time.time()
    with logging_redirect_tqdm(loggers=[log]):
        results = process_map(
            _worker, arg_list,
            max_workers=workers,
            desc="collecting",
            unit="chunk",
            leave=True,
        )

    all_samples = [s for chunk in results for s in chunk][:n_decisions]
    elapsed = time.time() - t0
    log.info(f"  collected {len(all_samples):,} samples in "
             f"{elapsed:.1f}s  ({len(all_samples)/elapsed:.0f}/s)")
    _log_collection_stats(all_samples)
    return all_samples


# ─── Training step ────────────────────────────────────────────────────────────

def _train_step(
    net: ActorCriticNet,
    batch: list[dict],
    device,
    opt=None,
    label_smoothing: float = 0.0,
) -> tuple[float, int, int]:
    B     = len(batch)
    max_K = max(s["states"].shape[0] for s in batch)

    sa      = torch.zeros(B, max_K, ACTOR_DIM,  dtype=torch.float32, device=device)
    ac      = torch.zeros(B, max_K, ACTION_DIM, dtype=torch.float32, device=device)
    mask    = torch.zeros(B, max_K, dtype=torch.bool,  device=device)
    targets = torch.zeros(B,       dtype=torch.long,   device=device)

    for i, s in enumerate(batch):
        K = s["states"].shape[0]
        sa[i, :K]   = torch.from_numpy(s["states"])
        ac[i, :K]   = torch.from_numpy(s["actions"])
        mask[i, :K] = True
        targets[i]  = s["pick_idx"]

    sa_flat     = sa.view(B * max_K, ACTOR_DIM)
    ac_flat     = ac.view(B * max_K, ACTION_DIM)
    logits_flat = net.actor_head(torch.cat([sa_flat, ac_flat], dim=-1))
    logits      = logits_flat.view(B, max_K).masked_fill(~mask, -1e9)

    loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    correct = (logits.argmax(dim=-1) == targets).sum().item()
    return loss.item(), correct, B


# ─── Main training loop ───────────────────────────────────────────────────────

def train_distillation(
    samples: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    label_smoothing: float = 0.1,
    val_split: float = 0.10,
    patience: int = 3,
    seed: int = 0,
    run_dir: Path | None = None,
) -> tuple[ActorCriticNet, dict]:
    net   = ActorCriticNet(hidden=hidden).to(device)
    opt   = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    n_params = sum(p.numel() for p in net.parameters())
    log.info(f"  net: {n_params:,} params on {device} (hidden={hidden})")
    log.info(f"  optimizer: AdamW  lr={lr}  weight_decay=1e-4  "
             f"label_smoothing={label_smoothing}")

    rng        = random.Random(seed)
    n          = len(samples)
    idxs       = list(range(n))
    rng.shuffle(idxs)
    n_val      = int(n * val_split)
    val_idxs   = idxs[:n_val]
    train_idxs = idxs[n_val:]
    n_batches  = math.ceil(len(train_idxs) / batch_size)
    log.info(f"  split: {len(train_idxs):,} train / {n_val:,} val  "
             f"({n_batches} batches/epoch  bs={batch_size})")

    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None
    stale        = 0
    history: list[dict] = []

    with logging_redirect_tqdm(loggers=[log]):
        epoch_bar = tqdm(range(epochs), desc="training", unit="epoch", leave=True)

        for epoch in epoch_bar:
            t_ep = time.time()

            # ── train ──
            rng.shuffle(train_idxs)
            net.train()
            tr_loss_sum = tr_correct = tr_total = 0
            batch_bar = tqdm(
                range(0, len(train_idxs), batch_size),
                desc=f"  ep {epoch+1:2d}/{epochs}",
                unit="batch",
                leave=False,
                total=n_batches,
            )
            for bs in batch_bar:
                batch = [samples[i] for i in train_idxs[bs: bs + batch_size]]
                l, c, t = _train_step(net, batch, device, opt=opt,
                                      label_smoothing=label_smoothing)
                tr_loss_sum += l * t
                tr_correct  += c
                tr_total    += t
                batch_bar.set_postfix(
                    loss=f"{l:.4f}",
                    acc=f"{tr_correct / max(tr_total, 1):.3f}",
                )
            sched.step()

            # ── val (no label smoothing for clean accuracy) ──
            net.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for bs in range(0, len(val_idxs), batch_size):
                    batch = [samples[i] for i in val_idxs[bs: bs + batch_size]]
                    _, c, t = _train_step(net, batch, device)
                    val_correct += c
                    val_total   += t

            tr_loss  = tr_loss_sum / max(tr_total, 1)
            tr_acc   = tr_correct  / max(tr_total, 1)
            val_acc  = val_correct / max(val_total, 1)
            epoch_s  = time.time() - t_ep
            cur_lr   = sched.get_last_lr()[0]
            is_best  = val_acc > best_val_acc

            row = dict(
                epoch=epoch + 1, tr_loss=tr_loss, tr_acc=tr_acc,
                val_acc=val_acc, lr=cur_lr, wall_s=epoch_s, is_best=is_best,
            )
            history.append(row)

            # Live figure after every epoch
            if run_dir is not None:
                plot_distill(history, run_dir)

            best_tag = "  ← best" if is_best else ""
            log.info(
                f"  epoch {epoch+1:3d}/{epochs}  "
                f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                f"val_acc={val_acc:.4f}  lr={cur_lr:.2e}  "
                f"t={epoch_s:.1f}s{best_tag}"
            )

            epoch_bar.set_postfix(
                val=f"{val_acc:.4f}",
                best=f"{best_val_acc:.4f}",
                lr=f"{cur_lr:.1e}",
            )

            if is_best:
                best_val_acc = val_acc
                best_epoch   = epoch + 1
                best_state   = {k: v.clone() for k, v in net.state_dict().items()}
                stale        = 0
            else:
                stale += 1
                if stale >= patience:
                    log.info(f"  early stop at epoch {epoch+1}  (patience={patience})")
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
        log.info(f"  restored best weights from epoch {best_epoch}")

    # Final val accuracy on best weights
    net.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for bs in range(0, len(val_idxs), batch_size):
            batch = [samples[i] for i in val_idxs[bs: bs + batch_size]]
            _, c, t = _train_step(net, batch, device)
            val_correct += c
            val_total   += t
    final_val_acc = val_correct / max(val_total, 1)

    log.info(
        f"  final val_acc={final_val_acc:.4f}  "
        f"best_val_acc={best_val_acc:.4f}  "
        f"best_epoch={best_epoch}"
    )
    return net, {
        "val_acc":      final_val_acc,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "history":      history,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("distill_pvguan")
    parser.add_argument("--supervisor", default="oracle",
        choices=["oracle", "jidan", "greedy", "heuristic", "strategic", "yaoji"])
    parser.add_argument("--supervisor-search", default="on", choices=["on", "off"])
    parser.add_argument("--supervisor-pimc-n-det", type=int, default=16)
    parser.add_argument("--supervisor-checkpoint", default=None,
        help="Path to supervisor model checkpoint (for oracle)")
    parser.add_argument("--n-decisions", type=int, default=500_000)
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--label-mode", default="hard",
        choices=["hard", "soft", "hybrid"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--output", default="ml/checkpoints/pvguan_distilled.pt")
    args = parser.parse_args()

    cfg         = DistillConfig.from_args(args)
    device      = get_device()
    output_path = Path(cfg.resolved_output)
    run_dir     = Path(cfg.resolved_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log to run_dir/train.log (timestamped dir) AND checkpoint dir
    _setup_logger(run_dir / "train")

    log.info("=" * 64)
    log.info(f"[distill_pvguan]  device={device}")
    log.info(str(cfg))
    log.info("=" * 64)

    # Save config to run dir
    cfg.save(run_dir / "config.json")

    # 1. Collect labels
    t_total = time.time()
    samples = collect_data(
        n_decisions=cfg.n_decisions,
        workers=cfg.workers,
        sup_name=cfg.supervisor,
        use_search=cfg.supervisor_search,
        n_det=cfg.supervisor_pimc_n_det,
        ckpt_path=cfg.supervisor_checkpoint,
        seed=cfg.seed,
    )

    # 2. Train actor head
    log.info("─" * 64)
    log.info("Training ...")
    net, metrics = train_distillation(
        samples=samples,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        hidden=cfg.hidden,
        device=device,
        label_smoothing=cfg.label_smoothing,
        seed=cfg.seed,
        run_dir=run_dir,
    )

    val_acc    = metrics["val_acc"]
    total_time = time.time() - t_total
    log.info("─" * 64)
    log.info(f"[distill_pvguan]  val_acc={val_acc:.4f}  "
             f"best_val_acc={metrics['best_val_acc']:.4f}  "
             f"best_epoch={metrics['best_epoch']}  "
             f"total_time={total_time:.0f}s")

    # 3. Acceptance gates
    if val_acc < 0.90:
        log.warning(f"WARN  val_acc {val_acc:.3f} < 0.90  "
                    f"(target ≥ 0.95 for production)")
    if val_acc < 0.95:
        log.warning(f"WARN  argmax agreement {val_acc:.3f} below 95% gate — "
                    f"review before PPO")
    else:
        log.info("PASS  acceptance gate  val_acc ≥ 0.95")

    # 4. Save checkpoint
    metadata = {
        "actor_dim":             ACTOR_DIM,
        "action_dim":            ACTION_DIM,
        "critic_dim":            875,
        "hidden":                cfg.hidden,
        "supervisor":            cfg.supervisor,
        "supervisor_search":     cfg.supervisor_search,
        "supervisor_pimc_n_det": cfg.supervisor_pimc_n_det,
        "n_samples":             len(samples),
        "val_acc":               val_acc,
        "label_mode":            cfg.label_mode,
        "label_smoothing":       cfg.label_smoothing,
    }

    torch.save({
        "actor_state_dict": net.actor_head.state_dict(),
        "full_state_dict":  net.state_dict(),
        "metadata":         metadata,
    }, output_path)
    log.info(f"[distill_pvguan]  saved → {output_path}")
    log.info(f"  metadata: {metadata}")


if __name__ == "__main__":
    main()
