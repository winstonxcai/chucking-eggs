"""Tier 1 training: fine-tune with partner hand visibility.

Two-phase approach to prevent catastrophic forgetting:

Phase 1 (representation): Freeze entire network. Only update the 60 partner
  columns in mlp.0.weight and hand_pred.0.weight via gradient masking.
  Train with aux loss only (q_loss_weight=0). The aux head learns to predict
  opponent cards using partner hand info — a clean, well-defined signal.
  Policy stays identical to base checkpoint.

Phase 2 (fine-tuning): Unfreeze everything. Re-enable q_loss. The partner
  columns now encode meaningful features, so Q-loss gradients carry real
  partner-specific signal instead of just pretrained calibration noise.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

from ...agents import make_agent
from ...game import GuanDanEnv
from ..encoding import (
    ACTION_DIM,
    encode_action,
    encode_history,
    encode_opponent_cards,
)
from ..q_network import QNetworkLSTM, get_device
from ..replay import ReplayBuffer
from .encoding import PARTNER_INSERT_POS, PARTNER_HAND_DIM, STATE_DIM_TIER1, encode_state_tier1


# Gradient mask: which columns of the first layer to update
_PARTNER_COL_START = PARTNER_INSERT_POS  # 60
_PARTNER_COL_END = PARTNER_INSERT_POS + PARTNER_HAND_DIM  # 120


def _mask_gradients(model: QNetworkLSTM) -> None:
    """Zero gradients for all columns except partner hand (60:120) in first layers.

    Called after loss.backward(), before optimizer.step(). Ensures only the 60
    partner columns in mlp.0.weight and hand_pred.0.weight receive updates.
    All other parameters (including biases) stay at their pretrained values.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if name == "mlp.0.weight" or name == "hand_pred.0.weight":
            # Zero everything except partner columns
            param.grad[:, :_PARTNER_COL_START] = 0
            param.grad[:, _PARTNER_COL_END:] = 0
        else:
            # Zero all gradients for other params
            param.grad.zero_()


def train_step_tier1(
    q_net: QNetworkLSTM,
    buf: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    q_loss_weight: float = 1.0,
    aux_weight: float = 0.1,
    partner_only: bool = False,
) -> tuple[float, float] | None:
    """One gradient step with configurable Q-loss and aux loss weights.

    If partner_only=True, applies gradient masking after backward to restrict
    updates to partner columns (60:120) in mlp.0.weight and hand_pred.0.weight.
    """
    if len(buf) < batch_size:
        return None

    batch = buf.sample(batch_size, device)
    hist_emb = q_net.encode_history(batch["history"], batch["hist_len"])

    total = torch.tensor(0.0, device=device)

    # Q-loss
    if q_loss_weight > 0:
        q_pred = q_net.forward_from_embedding(
            batch["state"], batch["action"], hist_emb)
        q_loss = torch.nn.functional.mse_loss(q_pred, batch["return"])
        total = total + q_loss_weight * q_loss
        q_loss_val = q_loss.item()
    else:
        q_loss_val = 0.0

    # Aux loss (opponent card prediction)
    if aux_weight > 0:
        hand_pred = q_net.predict_opponent_cards(batch["state"], hist_emb)
        aux_loss = torch.nn.functional.binary_cross_entropy(
            hand_pred, batch["opponent_cards"]
        )
        total = total + aux_weight * aux_loss
        aux_loss_val = aux_loss.item()
    else:
        aux_loss_val = 0.0

    optimizer.zero_grad()
    total.backward()

    if partner_only:
        _mask_gradients(q_net)

    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

    return q_loss_val, aux_loss_val


def play_episode_tier1(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
    opponent=None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]]:
    """Play one game with partner-visible encoding.

    If opponent is None: symmetric self-play — all 4 seats use Tier 1 encoding.
    If opponent is provided: seats {0,2} Tier 1, seats {1,3} opponent.
    """
    env.reset()
    selfplay = opponent is None
    seats = range(4) if selfplay else (0, 2)
    transitions: dict[int, list[tuple]] = {p: [] for p in seats}

    while not env.done:
        player = env.current_player

        if not selfplay and player in (1, 3):
            env.step(opponent.act(env, player))
            continue

        legal = env.legal_moves()
        is_leading = env.current_trick is None

        state_enc = encode_state_tier1(env, player)
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

        opp_cards = encode_opponent_cards(env, player)
        transitions[player].append(
            (state_enc, action_encs[idx], history, hist_len, opp_cards)
        )
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


def evaluate_tier1(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    n_games: int = 300,
    opponent: str = "jidan",
) -> dict:
    """Evaluate Tier 1 agent (seats {0,2}) vs opponent (seats {1,3})."""
    env = GuanDanEnv()
    opp_agent = make_agent(opponent, env.level_rank)

    wins = 0
    total_reward = 0.0
    finish_12 = finish_13 = finish_14 = 0

    for _ in tqdm(range(n_games), desc=f"eval vs {opponent}", unit="game",
                  leave=False, file=sys.stderr, dynamic_ncols=True):
        env.reset()
        while not env.done:
            player = env.current_player
            if player in (1, 3):
                env.step(opp_agent.act(env, player))
                continue

            legal = env.legal_moves()
            is_leading = env.current_trick is None
            q_net = q_lead if is_leading else q_follow

            state_enc = encode_state_tier1(env, player)
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

            env.step(legal[idx])

        rewards = env.get_rewards()
        team_reward = rewards[0] + rewards[2]
        if team_reward > 0:
            wins += 1
        total_reward += team_reward

        fo = env.finish_order
        team_pos = sorted(fo.index(p) for p in (0, 2))
        key = (team_pos[0] + 1, team_pos[1] + 1)
        if key == (1, 2):
            finish_12 += 1
        elif key == (1, 3):
            finish_13 += 1
        elif key == (1, 4):
            finish_14 += 1

    return {
        "winrate": wins / n_games,
        "avg_reward": total_reward / n_games,
        "finish_12": finish_12,
        "finish_13": finish_13,
        "finish_14": finish_14,
        "n_games": n_games,
    }


def _append_metrics(run_dir: Path, entry: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


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


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _partner_col_norm(model: QNetworkLSTM) -> float:
    """L2 norm of partner columns in mlp.0.weight."""
    return torch.norm(model.mlp[0].weight[:, _PARTNER_COL_START:_PARTNER_COL_END]).item()


def _run_eval(q_lead, q_follow, device, config, total_ep, run_dir, t_start,
              opp_name, best_wr_jidan):
    """Run eval against stage opponent + jidan, log results, save best."""
    t_eval = time.time()
    n_games = config["eval_games"]

    result_stage = evaluate_tier1(q_lead, q_follow, device, n_games=n_games, opponent=opp_name)
    result_jidan = evaluate_tier1(q_lead, q_follow, device, n_games=n_games, opponent="jidan")
    eval_secs = time.time() - t_eval

    pn_lead = _partner_col_norm(q_lead)
    pn_follow = _partner_col_norm(q_follow)
    wr_stage = result_stage["winrate"]
    wr_jidan = result_jidan["winrate"]
    elapsed = time.time() - t_start
    total_all = config["stage1_episodes"] + config["stage2_episodes"]
    remaining_ep = total_all - total_ep
    eta_secs = remaining_ep / (total_ep / elapsed) if total_ep > 0 else 0

    log.info("-" * 60)
    log.info(
        "EVAL Ep %6d | vs_%s=%.1f%% | vs_jidan=%.1f%% | "
        "1-2: %d/%d (%.0f%%) | partner_L2: lead=%.4f follow=%.4f",
        total_ep, opp_name, wr_stage * 100, wr_jidan * 100,
        result_jidan["finish_12"], n_games,
        result_jidan["finish_12"] / n_games * 100,
        pn_lead, pn_follow,
    )
    log.info(
        "     jidan_detail: 1-2=%d 1-3=%d 1-4=%d | "
        "eval took %s | elapsed %s | ETA %s",
        result_jidan["finish_12"], result_jidan["finish_13"],
        result_jidan["finish_14"],
        _fmt_elapsed(eval_secs), _fmt_elapsed(elapsed), _fmt_elapsed(eta_secs),
    )
    log.info("-" * 60)

    _append_metrics(run_dir, {
        "episode": total_ep,
        "timestamp": time.time(),
        "elapsed_s": elapsed,
        f"wr_{opp_name}": wr_stage,
        "wr_jidan": wr_jidan,
        "avg_reward_jidan": result_jidan["avg_reward"],
        "finish_12_jidan": result_jidan["finish_12"],
        "finish_13_jidan": result_jidan["finish_13"],
        "finish_14_jidan": result_jidan["finish_14"],
        "partner_column_l2_lead": pn_lead,
        "partner_column_l2_follow": pn_follow,
    })

    if wr_jidan > best_wr_jidan:
        best_wr_jidan = wr_jidan
        best_path = run_dir / "checkpoint_best.pt"
        torch.save({
            "lead": q_lead.state_dict(),
            "follow": q_follow.state_dict(),
            "episode": total_ep,
            "wr_jidan": wr_jidan,
            "d_state": STATE_DIM_TIER1,
        }, best_path)
        log.info("  *** New best: %.1f%% vs jidan → %s ***", wr_jidan * 100, best_path)

    return best_wr_jidan, wr_stage


def run_training(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Two-phase Tier 1 training: representation → fine-tuning."""
    _setup_logging(run_dir)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Tier 1 Training — Two-Phase Partner Visibility")
    log.info("=" * 60)
    log.info("Device: %s", device)
    log.info("Config: %s", json.dumps(config, indent=2))
    log.info("Run dir: %s", run_dir)

    phase1_episodes = config.get("phase1_episodes", 5000)
    phase1_aux_weight = config.get("phase1_aux_weight", 1.0)
    phase1_lr = config.get("phase1_lr", 1e-4)
    phase2_lr_start = config.get("lr_start", 3e-5)
    phase2_lr_end = config.get("lr_end", 3e-6)

    total_all = config["stage1_episodes"] + config["stage2_episodes"]
    log.info("Phase 1: %d episodes, aux-only (q_loss=0), partner columns only, lr=%.1e",
             phase1_episodes, phase1_lr)
    log.info("Phase 2: %d episodes, full fine-tuning, lr=%.1e→%.1e",
             total_all - phase1_episodes, phase2_lr_start, phase2_lr_end)

    partner_params_per_model = (
        PARTNER_HAND_DIM * q_lead.mlp[0].weight.shape[0] +  # mlp.0
        PARTNER_HAND_DIM * q_lead.hand_pred[0].weight.shape[0]  # hand_pred.0
    )
    total_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Partner columns: %d params (%.1f%% of %d total)",
             partner_params_per_model, partner_params_per_model / total_params * 100,
             total_params)

    buffer = ReplayBuffer(capacity=config["buffer_capacity"], d_state=STATE_DIM_TIER1)
    use_selfplay = config.get("selfplay", True)

    # Phase 1 uses all params in optimizer (gradient masking handles the rest)
    opt_lead = torch.optim.Adam(q_lead.parameters(), lr=phase1_lr)
    opt_follow = torch.optim.Adam(q_follow.parameters(), lr=phase1_lr)

    env = GuanDanEnv()
    t_start = time.time()

    # --- Pre-fill buffer ---
    prefill_eps = config["prefill_episodes"]
    log.info("Pre-filling buffer with %d self-play episodes...", prefill_eps)
    t_prefill = time.time()
    q_lead.eval()
    q_follow.eval()
    prefill_trans = 0
    for _ in tqdm(range(prefill_eps), desc="prefill", file=sys.stderr):
        trans = play_episode_tier1(
            env, q_lead, q_follow,
            epsilon=config["epsilon_start"], device=device,
        )
        prefill_trans += len(trans)
        for s, a, h, hl, r, oc in trans:
            buffer.push(s, a, h, hl, r, oc)
    log.info("Buffer pre-filled: %d transitions (%.1f trans/ep, %s)",
             len(buffer), prefill_trans / prefill_eps,
             _fmt_elapsed(time.time() - t_prefill))

    # --- Training loop ---
    stages = [
        (config["stage1_opponent"], config["stage1_episodes"], config.get("stage1_gate")),
        (config["stage2_opponent"], config["stage2_episodes"], None),
    ]

    total_ep = 0
    best_wr_jidan = 0.0
    in_phase1 = True
    loss_q_sum = loss_aux_sum = 0.0
    loss_count = 0

    for stage_idx, (opp_name, stage_episodes, gate) in enumerate(stages, 1):
        log.info("")
        log.info("=" * 60)
        log.info("Stage %d: vs %s (%d episodes, gate=%s)",
                 stage_idx, opp_name, stage_episodes,
                 f"{gate:.0%}" if gate else "none")
        log.info("=" * 60)
        opp = None if use_selfplay else make_agent(opp_name, env.level_rank)
        t_stage = time.time()

        eps_start = config["epsilon_start"]
        eps_end = config["epsilon_end"]

        pbar = tqdm(range(stage_episodes), desc=f"stage{stage_idx}", file=sys.stderr)
        for ep_in_stage in pbar:
            total_ep += 1
            frac = min(1.0, total_ep / (0.85 * total_all))
            epsilon = eps_start + (eps_end - eps_start) * frac

            # --- Phase transition ---
            if in_phase1 and total_ep > phase1_episodes:
                in_phase1 = False
                opt_lead = torch.optim.Adam(q_lead.parameters(), lr=phase2_lr_start)
                opt_follow = torch.optim.Adam(q_follow.parameters(), lr=phase2_lr_start)
                log.info("")
                log.info("*" * 60)
                log.info("*** PHASE 2: Full fine-tuning (ep %d) ***", total_ep)
                log.info("*" * 60)
                log.info("  Partner L2 at transition: lead=%.4f follow=%.4f",
                         _partner_col_norm(q_lead), _partner_col_norm(q_follow))

            # LR schedule
            if in_phase1:
                lr = phase1_lr
            else:
                phase2_ep = total_ep - phase1_episodes
                phase2_total = total_all - phase1_episodes
                lr = phase2_lr_end + 0.5 * (phase2_lr_start - phase2_lr_end) * (
                    1 + np.cos(np.pi * phase2_ep / phase2_total)
                )
            for opt in (opt_lead, opt_follow):
                for pg in opt.param_groups:
                    pg["lr"] = lr

            # Collect episode
            q_lead.eval()
            q_follow.eval()
            trans = play_episode_tier1(env, q_lead, q_follow, epsilon, device, opponent=opp)
            for s, a, h, hl, r, oc in trans:
                buffer.push(s, a, h, hl, r, oc)

            # Train steps
            q_lead.train()
            q_follow.train()
            bs = config["batch_size"]
            q_w = 0.0 if in_phase1 else 1.0
            aux_w = phase1_aux_weight if in_phase1 else config["aux_weight"]

            for _ in range(config["train_steps_per_episode"]):
                res_lead = train_step_tier1(
                    q_lead, buffer, opt_lead, bs, device,
                    q_loss_weight=q_w, aux_weight=aux_w, partner_only=in_phase1,
                )
                train_step_tier1(
                    q_follow, buffer, opt_follow, bs, device,
                    q_loss_weight=q_w, aux_weight=aux_w, partner_only=in_phase1,
                )
                if res_lead is not None:
                    loss_q_sum += res_lead[0]
                    loss_aux_sum += res_lead[1]
                    loss_count += 1

            # tqdm postfix
            if loss_count > 0:
                elapsed = time.time() - t_start
                phase_str = "P1" if in_phase1 else "P2"
                pbar.set_postfix_str(
                    f"{phase_str} q={loss_q_sum/loss_count:.4f} "
                    f"aux={loss_aux_sum/loss_count:.4f} "
                    f"lr={lr:.1e} "
                    f"{total_ep/elapsed:.1f}ep/s"
                )

            # Periodic loss log
            if total_ep % 200 == 0 and loss_count > 0:
                elapsed = time.time() - t_start
                phase_str = "PHASE1" if in_phase1 else "PHASE2"
                log.info(
                    "Ep %6d [%s] | q=%.4f aux=%.4f | "
                    "lr=%.2e e=%.3f | buf=%d | %.1f ep/s | %s",
                    total_ep, phase_str,
                    loss_q_sum / loss_count, loss_aux_sum / loss_count,
                    lr, epsilon, len(buffer),
                    total_ep / elapsed, _fmt_elapsed(elapsed),
                )
                loss_q_sum = loss_aux_sum = 0.0
                loss_count = 0

            # Eval
            eval_interval = config["eval_interval"]
            if (ep_in_stage + 1) % eval_interval == 0 or ep_in_stage == stage_episodes - 1:
                q_lead.eval()
                q_follow.eval()
                best_wr_jidan, wr_stage = _run_eval(
                    q_lead, q_follow, device, config, total_ep,
                    run_dir, t_start, opp_name, best_wr_jidan,
                )
                if gate is not None and wr_stage >= gate:
                    log.info("  Stage gate: %.1f%% >= %.0f%%. Advancing.",
                             wr_stage * 100, gate * 100)
                    pbar.close()
                    break

            # Periodic save
            if total_ep % config["save_interval"] == 0:
                path = run_dir / f"checkpoint_ep{total_ep}.pt"
                torch.save({
                    "lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
                    "episode": total_ep, "d_state": STATE_DIM_TIER1,
                }, path)
                log.info("Saved checkpoint: %s", path)

        else:
            pbar.close()

        log.info("Stage %d complete: %d ep in %s",
                 stage_idx, ep_in_stage + 1, _fmt_elapsed(time.time() - t_stage))

    # Final save
    total_secs = time.time() - t_start
    final_path = run_dir / "model_final.pt"
    torch.save({
        "lead": q_lead.state_dict(), "follow": q_follow.state_dict(),
        "episode": total_ep, "d_state": STATE_DIM_TIER1,
        "best_wr_jidan": best_wr_jidan,
    }, final_path)
    log.info("")
    log.info("=" * 60)
    log.info("Training complete. Best vs jidan: %.1f%%. Wall time: %s",
             best_wr_jidan * 100, _fmt_elapsed(total_secs))
    log.info("Final: %s", final_path)
    log.info("=" * 60)
