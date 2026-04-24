#!/usr/bin/env python
"""From-scratch 477-dim training for Tier 1 partner visibility.

Trains QNetworkLSTM(d_state=477) from random initialization with full
curriculum: heuristic pretrain → self-play. Partner hand is a native
input from episode 1 — no fine-tuning, no forgetting.

Usage:
    PYTHONPATH=ml/src python ml/scripts/train/train_scratch_tier1.py \
        --run-name scratch_tier1_v1 --episodes 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


# ─── Persistent worker state ────────────────────────────────────────────
# Workers construct networks once via pool initializer, then reuse them
# across all episodes (only load_state_dict for weight updates).
_WORKER_QLEAD = None
_WORKER_QFOLLOW = None
_WORKER_ENV = None


def _init_worker(lstm_hidden: int, mlp_hidden: int, d_state: int, aux_output_dim: int):
    """Called once per worker process at pool startup."""
    global _WORKER_QLEAD, _WORKER_QFOLLOW, _WORKER_ENV
    import torch as _torch
    from guandan.game import GuanDanEnv
    from guandan.training.q_network import QNetworkLSTM

    device = _torch.device("cpu")
    _WORKER_QLEAD = QNetworkLSTM(
        d_state=d_state, lstm_hidden=lstm_hidden,
        hidden=mlp_hidden, aux_output_dim=aux_output_dim,
    ).to(device)
    _WORKER_QFOLLOW = QNetworkLSTM(
        d_state=d_state, lstm_hidden=lstm_hidden,
        hidden=mlp_hidden, aux_output_dim=aux_output_dim,
    ).to(device)
    _WORKER_QLEAD.eval()
    _WORKER_QFOLLOW.eval()
    _WORKER_ENV = GuanDanEnv()


def _episode_worker(
    lead_sd: dict,
    follow_sd: dict,
    epsilon: float,
    opponent_name: str | None,
) -> list:
    """Run one episode using the persistent worker networks.

    Only does load_state_dict (cheap copy into pre-allocated nets) and
    play_episode — no network construction.
    """
    global _WORKER_QLEAD, _WORKER_QFOLLOW, _WORKER_ENV
    import torch as _torch
    from guandan.agents import make_agent
    from guandan.training.encoding import (
        encode_action,
        encode_history,
        encode_opponent_cards_split_team,
    )
    from guandan.training.guanzero_encoding import compute_behavior_flags
    from guandan.training.visibility.encoding import encode_state_tier1_team

    _WORKER_QLEAD.load_state_dict(lead_sd)
    _WORKER_QFOLLOW.load_state_dict(follow_sd)

    def _opp_adapter(env, player):
        return encode_opponent_cards_split_team(env)

    device = _torch.device("cpu")
    opp = make_agent(opponent_name, _WORKER_ENV.level_rank) if opponent_name else None
    return play_episode(_WORKER_ENV, _WORKER_QLEAD, _WORKER_QFOLLOW, epsilon, device,
                        encode_state_tier1_team, encode_action, encode_history,
                        _opp_adapter, compute_behavior_flags,
                        _WORKER_ENV.level_rank, opponent=opp)


def _setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log", mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
               for h in root.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def play_episode(env, q_lead, q_follow, epsilon, device,
                 encode_fn, encode_action_fn, encode_history_fn,
                 encode_opp_fn, compute_flags_fn, level_rank, opponent=None):
    """Play one episode with per-action behavior flags. Self-play if opponent is None."""
    env.reset()
    our_seats = (0, 1, 2, 3) if opponent is None else (0, 2)
    transitions: dict[int, list] = {p: [] for p in our_seats}

    while not env.done:
        player = env.current_player

        if opponent is not None and player in (1, 3):
            env.step(opponent.act(env, player))
            continue

        legal = env.legal_moves()
        is_leading = env.current_trick is None
        q_net = q_lead if is_leading else q_follow

        base_state = encode_fn(env, player)  # [477]
        hand = env.hands[player]
        action_encs = np.array([encode_action_fn(m, hand, level_rank) for m in legal])
        history, hist_len = encode_history_fn(env, player, level_rank)

        # Per-action states: base_state + behavior flags (9 dims each)
        state_encs = np.array([
            np.concatenate([base_state, compute_flags_fn(env, player, m, legal)])
            for m in legal
        ])  # [B, 486]

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_encs, device=device)  # [B, 486]
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()

        opp_cards = encode_opp_fn(env, player)
        transitions[player].append((state_encs[idx], action_encs[idx], history, hist_len, opp_cards))
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


def play_heuristic_episode(env, encode_fn, encode_action_fn, encode_history_fn,
                           encode_opp_fn, compute_flags_fn, level_rank, heuristic):
    """Collect transitions from heuristic play for imitation pretraining."""
    env.reset()
    transitions: dict[int, list] = {p: [] for p in range(4)}

    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        action = heuristic.act(env, player)

        base_state = encode_fn(env, player)
        flags = compute_flags_fn(env, player, action, legal)
        state_enc = np.concatenate([base_state, flags])  # [486]
        action_enc = encode_action_fn(action, env.hands[player], level_rank)
        history, hist_len = encode_history_fn(env, player, level_rank)
        opp_cards = encode_opp_fn(env, player)
        transitions[player].append((state_enc, action_enc, history, hist_len, opp_cards))
        env.step(action)

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


def train_step(q_net, buf, optimizer, batch_size, device, aux_weight=0.1):
    if len(buf) < batch_size:
        return None
    batch = buf.sample(batch_size, device)
    hist_emb = q_net.encode_history(batch["history"], batch["hist_len"])
    q_pred = q_net.forward_from_embedding(batch["state"], batch["action"], hist_emb)
    q_loss = torch.nn.functional.mse_loss(q_pred, batch["return"])

    hand_pred = q_net.predict_opponent_cards(batch["state"], hist_emb)
    aux_loss = torch.nn.functional.binary_cross_entropy(hand_pred, batch["opponent_cards"])
    total = q_loss + aux_weight * aux_loss

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return q_loss.item(), aux_loss.item()


def evaluate(q_lead, q_follow, device, n_games, opponent_name,
             encode_fn, encode_action_fn, encode_history_fn,
             compute_flags_fn, level_rank):
    from guandan.agents import make_agent
    from guandan.game import GuanDanEnv
    env = GuanDanEnv()
    opp = make_agent(opponent_name, env.level_rank)
    wins = finish_12 = 0

    for _ in tqdm(range(n_games), desc=f"eval vs {opponent_name}", unit="game",
                  leave=False, file=sys.stderr, dynamic_ncols=True):
        env.reset()
        while not env.done:
            player = env.current_player
            if player in (1, 3):
                env.step(opp.act(env, player))
                continue
            legal = env.legal_moves()
            is_leading = env.current_trick is None
            q_net = q_lead if is_leading else q_follow
            base_state = encode_fn(env, player)
            hand = env.hands[player]
            action_encs = np.array([encode_action_fn(m, hand, level_rank) for m in legal])
            state_encs = np.array([
                np.concatenate([base_state, compute_flags_fn(env, player, m, legal)])
                for m in legal
            ])
            history, hist_len = encode_history_fn(env, player, level_rank)
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_encs, device=device)
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                idx = q_net(s, a, h, hl).argmax().item()
            env.step(legal[idx])

        rewards = env.get_rewards()
        if rewards[0] + rewards[2] > 0:
            wins += 1
        fo = env.finish_order
        tp = sorted(fo.index(p) for p in (0, 2))
        if (tp[0] + 1, tp[1] + 1) == (1, 2):
            finish_12 += 1

    return {"winrate": wins / n_games, "finish_12": finish_12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="scratch_tier1_v1")
    parser.add_argument("--pretrain-games", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-end", type=float, default=1e-5,
                        help="Final LR after cosine decay. Set equal to --lr to disable decay.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--buffer-capacity", type=int, default=300000)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=0.10)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--mlp-hidden", type=int, default=1024)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (skip pretrain)")
    parser.add_argument("--opponent-pool", type=str,
                        default="heuristic,strategic,xingdream,noai",
                        help="Comma-separated opponent names to mix into training episodes.")
    parser.add_argument("--self-play-frac", type=float, default=0.3,
                        help="Fraction of episodes that are pure self-play (0.0=all pool, 1.0=all self-play).")
    parser.add_argument("--eval-pool", type=str, default="noai,yaoji,jidan",
                        help="Comma-separated opponents to evaluate against each interval.")
    args = parser.parse_args()

    opponent_pool = [s.strip() for s in args.opponent_pool.split(",") if s.strip()]
    eval_pool = [s.strip() for s in args.eval_pool.split(",") if s.strip()]

    from guandan.agents.heuristic_bot import HeuristicBot
    from guandan.game import GuanDanEnv
    from guandan.training.encoding import (
        OPP_CARDS_DIM_SPLIT,
        encode_action,
        encode_history,
        encode_opponent_cards_split_team,
    )
    from guandan.training.guanzero_encoding import compute_behavior_flags
    from guandan.training.q_network import QNetworkLSTM, get_device
    from guandan.training.replay import ReplayBuffer
    from guandan.training.visibility.encoding import (
        STATE_DIM_TIER1_TEAM,
        encode_state_tier1_team,
    )

    D_STATE = STATE_DIM_TIER1_TEAM + 9  # 480 + 9 behavior flags = 489
    AUX_DIM = OPP_CARDS_DIM_SPLIT       # 120

    # Player-ignoring adapter so the existing encode_opp_fn(env, player) signature
    # keeps working with the team-invariant target.
    def _opp_adapter(env, player):
        return encode_opponent_cards_split_team(env)

    device = get_device()
    run_dir = Path("ml/runs") / args.run_name
    _setup_logging(run_dir)

    log.info("=" * 60)
    log.info("From-Scratch Tier 1 Training — %d-dim team-centric (480 + 9 flags) | aux=%d split",
             D_STATE, AUX_DIM)
    log.info("=" * 60)
    log.info("Device: %s | d_state: %d | workers: %d", device, D_STATE, args.n_workers)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    q_lead = QNetworkLSTM(d_state=D_STATE,
                          lstm_hidden=args.lstm_hidden,
                          hidden=args.mlp_hidden,
                          aux_output_dim=AUX_DIM).to(device)
    q_follow = QNetworkLSTM(d_state=D_STATE,
                            lstm_hidden=args.lstm_hidden,
                            hidden=args.mlp_hidden,
                            aux_output_dim=AUX_DIM).to(device)

    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=args.lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=args.lr)

    buffer = ReplayBuffer(capacity=args.buffer_capacity, d_state=D_STATE,
                          d_opp_cards=AUX_DIM)
    env = GuanDanEnv()
    level_rank = env.level_rank
    log.info("Total params per net: %d", sum(p.numel() for p in q_lead.parameters()))

    use_parallel = args.n_workers > 1
    pool = None
    if use_parallel:
        pool = ProcessPoolExecutor(
            max_workers=args.n_workers,
            initializer=_init_worker,
            initargs=(args.lstm_hidden, args.mlp_hidden, D_STATE, AUX_DIM),
        )
    step = args.n_workers if use_parallel else 1

    def _snapshot_weights():
        """CPU copy of current weights, ready to ship to workers."""
        return (
            {k: v.cpu() for k, v in q_lead.state_dict().items()},
            {k: v.cpu() for k, v in q_follow.state_dict().items()},
        )

    def _submit_batch(epsilon: float):
        """Submit one batch of `step` episodes. Returns list of futures.

        Each episode independently samples: self-play with prob self_play_frac,
        else a random opponent from opponent_pool. This keeps the policy anchored
        to the evaluation distribution while retaining self-improvement signal.
        """
        lead_sd, follow_sd = _snapshot_weights()
        futs = []
        for _ in range(step):
            if random.random() < args.self_play_frac or not opponent_pool:
                opp = None
            else:
                opp = random.choice(opponent_pool)
            futs.append(pool.submit(_episode_worker, lead_sd, follow_sd, epsilon, opp))
        return futs

    def _collect_batch(epsilon: float, n: int) -> list:
        """Synchronous collect (used for pretrain prefill / non-parallel path)."""
        if use_parallel:
            futs = _submit_batch(epsilon)
            return [t for fut in futs for t in fut.result()]
        else:
            from guandan.agents import make_agent
            all_trans = []
            for _ in range(n):
                if random.random() < args.self_play_frac or not opponent_pool:
                    opp_name = None
                else:
                    opp_name = random.choice(opponent_pool)
                opp = make_agent(opp_name, level_rank) if opp_name else None
                all_trans.extend(play_episode(
                    env, q_lead, q_follow, epsilon, device,
                    encode_state_tier1_team, encode_action, encode_history,
                    _opp_adapter, compute_behavior_flags,
                    level_rank, opponent=opp))
            return all_trans

    t0 = time.time()

    # ── Resume or Pretrain ───────────────────────────────────────────────────
    if args.resume:
        log.info("Resuming from %s (skipping pretrain)", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        lead_key = "lead_state_dict" if "lead_state_dict" in ckpt else "lead"
        follow_key = "follow_state_dict" if "follow_state_dict" in ckpt else "follow"
        q_lead.load_state_dict(ckpt[lead_key])
        q_follow.load_state_dict(ckpt[follow_key])
    elif args.pretrain_games > 0:
        log.info("")
        log.info("=" * 60)
        log.info("Phase 1: Heuristic Pretrain (%d games)", args.pretrain_games)
        log.info("=" * 60)
        heuristic = HeuristicBot(level_rank)
        loss_q_sum = loss_aux_sum = 0.0
        loss_count = 0

        pbar = tqdm(range(1, args.pretrain_games + 1), desc="pretrain", file=sys.stderr)
        for game in pbar:
            trans = play_heuristic_episode(env, encode_state_tier1_team, encode_action,
                                          encode_history, _opp_adapter,
                                          compute_behavior_flags, level_rank, heuristic)
            for t in trans:
                buffer.push(*t)

            for _ in range(2):
                r = train_step(q_lead, buffer, opt_lead, args.batch_size, device)
                train_step(q_follow, buffer, opt_follow, args.batch_size, device)
                if r is not None:
                    loss_q_sum += r[0]; loss_aux_sum += r[1]; loss_count += 1

            if game % 500 == 0 and loss_count > 0:
                elapsed = time.time() - t0
                log.info("Pretrain %5d | q=%.4f aux=%.4f | buf=%d | %s",
                         game, loss_q_sum / loss_count, loss_aux_sum / loss_count,
                         len(buffer), _fmt(elapsed))
                loss_q_sum = loss_aux_sum = 0.0; loss_count = 0

            if game % 2000 == 0:
                q_lead.eval(); q_follow.eval()
                res = evaluate(q_lead, q_follow, device, 200, "heuristic",
                               encode_state_tier1_team, encode_action, encode_history,
                               compute_behavior_flags, level_rank)
                log.info("Pretrain EVAL %5d | vs_heuristic=%.1f%%",
                         game, res["winrate"] * 100)

        pbar.close()
        # Save pretrain checkpoint
        pt_path = run_dir / "checkpoint_pretrain.pt"
        torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                    "pretrain_games": args.pretrain_games, "d_state": D_STATE},
                   pt_path)
        log.info("Pretrain complete. Saved %s", pt_path)

    # Baseline eval across eval pool
    q_lead.eval(); q_follow.eval()
    baseline_results = {}
    for opp_name in eval_pool:
        res = evaluate(q_lead, q_follow, device, args.eval_games, opp_name,
                       encode_state_tier1_team, encode_action, encode_history,
                       compute_behavior_flags, level_rank)
        baseline_results[opp_name] = res["winrate"]
    baseline_avg = sum(baseline_results.values()) / len(baseline_results)
    log.info("Baseline | %s | avg=%.1f%%",
             "  ".join(f"vs_{k}={v*100:.1f}%" for k, v in baseline_results.items()),
             baseline_avg * 100)

    # ── Phase 2: Mixed Training ───────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Phase 2: Mixed Training (%d episodes, lr=%.1e)", args.episodes, args.lr)
    log.info("Train pool: %s | self-play: %.0f%%", ", ".join(opponent_pool) if opponent_pool else "none",
             args.self_play_frac * 100)
    log.info("Eval pool:  %s | %d games each", ", ".join(eval_pool), args.eval_games)
    log.info("=" * 60)
    t_start = time.time()
    best_wr = baseline_avg
    loss_q_sum = loss_aux_sum = 0.0
    loss_count = 0

    eps_decay = int(args.episodes * 0.85)

    pbar = tqdm(total=args.episodes, desc="self-play", file=sys.stderr)
    ep = 0

    def _epsilon_at(ep_count: int) -> float:
        frac = min(1.0, ep_count / eps_decay)
        return args.epsilon_start + (args.epsilon_end - args.epsilon_start) * frac

    # Prime the pipeline: submit first collection batch before main loop.
    # Subsequent iterations submit the NEXT batch before training on the
    # CURRENT batch, so collection runs in parallel with gradient steps.
    pending_futures = None
    if use_parallel:
        q_lead.eval(); q_follow.eval()
        pending_futures = _submit_batch(_epsilon_at(ep))

    while ep < args.episodes:
        epsilon = _epsilon_at(ep)

        # Cosine LR decay
        frac = ep / max(1, args.episodes)
        lr = args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * frac))
        for opt in (opt_lead, opt_follow):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # Wait for the current batch to finish collecting
        if use_parallel:
            trans = [t for fut in pending_futures for t in fut.result()]
            # Immediately submit the NEXT batch (runs in background while we train)
            if ep + step < args.episodes:
                q_lead.eval(); q_follow.eval()
                pending_futures = _submit_batch(_epsilon_at(ep + step))
            else:
                pending_futures = None
        else:
            q_lead.eval(); q_follow.eval()
            trans = _collect_batch(epsilon, step)

        for t in trans:
            buffer.push(*t)

        # Train (runs concurrently with next batch collection when parallel)
        q_lead.train(); q_follow.train()
        for _ in range(args.train_steps * step):
            r = train_step(q_lead, buffer, opt_lead, args.batch_size, device,
                           aux_weight=args.aux_weight)
            train_step(q_follow, buffer, opt_follow, args.batch_size, device,
                       aux_weight=args.aux_weight)
            if r is not None:
                loss_q_sum += r[0]; loss_aux_sum += r[1]; loss_count += 1

        ep += step
        pbar.update(step)

        if loss_count > 0:
            elapsed = time.time() - t_start
            pbar.set_postfix_str(
                f"q={loss_q_sum/loss_count:.3f} aux={loss_aux_sum/loss_count:.3f} "
                f"e={epsilon:.3f} {ep/elapsed:.1f}ep/s avg={best_wr:.1%}"
            )

        if ep % 200 < step and loss_count > 0:
            elapsed = time.time() - t_start
            log.info("Ep %6d | q=%.4f aux=%.4f | e=%.3f lr=%.2e | buf=%d | %.1f ep/s | %s",
                     ep, loss_q_sum / loss_count, loss_aux_sum / loss_count,
                     epsilon, lr, len(buffer), ep / elapsed, _fmt(elapsed))
            loss_q_sum = loss_aux_sum = 0.0; loss_count = 0

        # Eval across pool
        if ep % args.eval_interval < step:
            q_lead.eval(); q_follow.eval()
            elapsed = time.time() - t_start
            eval_results = {}
            for opp_name in eval_pool:
                res = evaluate(q_lead, q_follow, device, args.eval_games, opp_name,
                               encode_state_tier1_team, encode_action, encode_history,
                               compute_behavior_flags, level_rank)
                eval_results[opp_name] = res["winrate"]
            avg_wr = sum(eval_results.values()) / len(eval_results)

            log.info("-" * 60)
            log.info("EVAL Ep %6d | %s | avg=%.1f%% | best=%.1f%% | %s",
                     ep,
                     "  ".join(f"vs_{k}={v*100:.1f}%" for k, v in eval_results.items()),
                     avg_wr * 100, best_wr * 100, _fmt(elapsed))
            log.info("-" * 60)

            entry = {"episode": ep, "elapsed_s": elapsed,
                     **{f"wr_{k}": v for k, v in eval_results.items()},
                     "wr_avg": avg_wr}
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

            if avg_wr > best_wr:
                best_wr = avg_wr
                best_path = run_dir / "checkpoint_best.pt"
                torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                            "episode": ep, "wr_avg": avg_wr, "d_state": D_STATE},
                           best_path)
                log.info("  *** New best avg: %.1f%% → %s ***", avg_wr * 100, best_path)

        # Save checkpoint
        if ep % args.save_interval < step:
            path = run_dir / f"checkpoint_ep{ep}.pt"
            torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                        "episode": ep, "d_state": D_STATE}, path)
            log.info("Saved: %s", path)

    pbar.close()
    if pool is not None:
        pool.shutdown(wait=False)

    total = time.time() - t0
    final_path = run_dir / "model_final.pt"
    torch.save({"lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                "episode": ep, "d_state": D_STATE,
                "best_wr_jidan": best_wr}, final_path)
    log.info("=" * 60)
    log.info("Done. Best vs jidan: %.1f%%. Wall time: %s", best_wr * 100, _fmt(total))
    log.info("Final: %s", final_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
