"""Tier 1 training: partner visibility via frozen base + partner adapter.

Architecture:
  Q_final(s, a) = Q_base(s, a) + PartnerAdapter(partner_hand, action_enc)

The base Q-network is loaded from the production checkpoint and frozen.
Only the small PartnerAdapter (~90K params) is trained, so catastrophic
forgetting is impossible by construction — the base policy never changes.
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
from ..encoding import ACTION_DIM, encode_action, encode_history
from ..q_network import QNetworkLSTM, get_device
from ..replay import ReplayBuffer
from .adapter import PartnerAdapter
from .encoding import PARTNER_INSERT_POS, PARTNER_HAND_DIM, STATE_DIM_TIER1, encode_state_tier1


def _extract_partner_hand(states: torch.Tensor) -> torch.Tensor:
    """Slice partner hand columns from a batch of states. Shape: (batch, 60)."""
    return states[:, PARTNER_INSERT_POS: PARTNER_INSERT_POS + PARTNER_HAND_DIM]


def train_step_adapter(
    q_net: QNetworkLSTM,
    adapter: PartnerAdapter,
    buf: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    output_reg: float = 2.0,
) -> float | None:
    """One gradient step on the adapter only using residual learning.

    Target: clamp(mc_return - Q_base, -1, 1).

    Clipping removes game-level noise outliers the adapter cannot predict.
    output_reg penalizes large adapter outputs — forces near-zero when there
    is no real partner-action signal, small adjustments when there is.
    """
    if len(buf) < batch_size:
        return None

    batch = buf.sample(batch_size, device)

    with torch.no_grad():
        hist_emb = q_net.encode_history(batch["history"], batch["hist_len"])
        q_base = q_net.forward_from_embedding(batch["state"], batch["action"], hist_emb)
        # Clip residual to remove outlier game noise adapter can't predict
        residual = (batch["return"] - q_base).clamp(-1.0, 1.0)

    partner_hand = _extract_partner_hand(batch["state"])
    q_adj = adapter(partner_hand, batch["action"])

    mse = torch.nn.functional.mse_loss(q_adj, residual)
    # L2 output regularization: pushes adapter toward zero when signal is weak
    output_penalty = (q_adj ** 2).mean()
    loss = mse + output_reg * output_penalty

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
    optimizer.step()

    return mse.item()


def play_episode_tier1(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    adapter_lead: PartnerAdapter,
    adapter_follow: PartnerAdapter,
    epsilon: float,
    device: torch.device,
    opponent=None,
) -> list[tuple]:
    """Play one game. Seats {0,2} use Q_base + adapter; seats {1,3} use opponent."""
    env.reset()
    transitions: dict[int, list[tuple]] = {0: [], 2: []}

    while not env.done:
        player = env.current_player

        if player in (1, 3):
            env.step(opponent.act(env, player))
            continue

        legal = env.legal_moves()
        is_leading = env.current_trick is None
        q_net = q_lead if is_leading else q_follow
        adapter = adapter_lead if is_leading else adapter_follow

        state_enc = encode_state_tier1(env, player)
        hand = env.hands[player]
        action_encs = np.array([encode_action(m, hand, env.level_rank) for m in legal])
        history, hist_len = encode_history(env, player, env.level_rank)

        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                hist_emb = q_net.encode_history(h, hl)
                q_base = q_net.forward_from_embedding(s, a, hist_emb)
                partner_hand = s[:, PARTNER_INSERT_POS: PARTNER_INSERT_POS + PARTNER_HAND_DIM]
                q_total = q_base + adapter(partner_hand, a, hist_emb)
                idx = q_total.argmax().item()

        opp_cards_enc = _dummy_opp_cards(state_enc)
        transitions[player].append((state_enc, action_encs[idx], history, hist_len, opp_cards_enc))
        env.step(legal[idx])

    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        r = rewards[player]
        for s, a, h, hl, oc in tlist:
            all_trans.append((s, a, h, hl, r, oc))
    return all_trans


def _dummy_opp_cards(state_enc: np.ndarray) -> np.ndarray:
    """Placeholder opponent cards (zeros). Aux loss is not used in adapter training."""
    return np.zeros(60, dtype=np.float32)


def evaluate_tier1(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    device: torch.device,
    n_games: int = 300,
    opponent: str = "jidan",
    adapter_lead: PartnerAdapter | None = None,
    adapter_follow: PartnerAdapter | None = None,
) -> dict:
    """Evaluate Tier 1 agent vs opponent. Adapter is optional (for compatibility)."""
    env = GuanDanEnv()
    opp_agent = make_agent(opponent, env.level_rank)

    wins = finish_12 = finish_13 = finish_14 = 0

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
            adapter = (adapter_lead if is_leading else adapter_follow) if adapter_lead else None

            state_enc = encode_state_tier1(env, player)
            hand = env.hands[player]
            action_encs = np.array([encode_action(m, hand, env.level_rank) for m in legal])
            history, hist_len = encode_history(env, player, env.level_rank)

            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_enc, device=device).unsqueeze(0).expand(B, -1)
                a = torch.tensor(action_encs, device=device)
                h = torch.tensor(history, device=device).unsqueeze(0).expand(B, -1, -1)
                hl = torch.tensor([hist_len], dtype=torch.long, device=device).expand(B)
                hist_emb = q_net.encode_history(h, hl)
                q_vals = q_net.forward_from_embedding(s, a, hist_emb)
                if adapter is not None:
                    partner_hand = s[:, PARTNER_INSERT_POS: PARTNER_INSERT_POS + PARTNER_HAND_DIM]
                    q_vals = q_vals + adapter(partner_hand, a, hist_emb)
                idx = q_vals.argmax().item()

            env.step(legal[idx])

        rewards = env.get_rewards()
        if rewards[0] + rewards[2] > 0:
            wins += 1

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


def _adapter_norm(adapter: PartnerAdapter) -> float:
    """L2 norm of the adapter's first-layer weights — proxy for how much it's learned."""
    return torch.norm(adapter.net[0].weight).item()


def _run_eval(q_lead, q_follow, adapter_lead, adapter_follow, device, config,
              total_ep, run_dir, t_start, opp_name, best_wr_jidan):
    t_eval = time.time()
    n_games = config["eval_games"]

    result_stage = evaluate_tier1(q_lead, q_follow, device, n_games=n_games,
                                  opponent=opp_name,
                                  adapter_lead=adapter_lead, adapter_follow=adapter_follow)
    result_jidan = evaluate_tier1(q_lead, q_follow, device, n_games=n_games,
                                  opponent="jidan",
                                  adapter_lead=adapter_lead, adapter_follow=adapter_follow)
    eval_secs = time.time() - t_eval

    an_lead = _adapter_norm(adapter_lead)
    an_follow = _adapter_norm(adapter_follow)
    wr_stage = result_stage["winrate"]
    wr_jidan = result_jidan["winrate"]
    elapsed = time.time() - t_start
    total_all = config["stage1_episodes"] + config["stage2_episodes"]
    eta_secs = (total_all - total_ep) / (total_ep / elapsed) if total_ep > 0 else 0

    log.info("-" * 60)
    log.info(
        "EVAL Ep %6d | vs_%s=%.1f%% | vs_jidan=%.1f%% | "
        "1-2: %d/%d (%.0f%%) | adapter_norm: lead=%.4f follow=%.4f",
        total_ep, opp_name, wr_stage * 100, wr_jidan * 100,
        result_jidan["finish_12"], n_games,
        result_jidan["finish_12"] / n_games * 100,
        an_lead, an_follow,
    )
    log.info(
        "     jidan_detail: 1-2=%d 1-3=%d 1-4=%d | "
        "eval took %s | elapsed %s | ETA %s",
        result_jidan["finish_12"], result_jidan["finish_13"], result_jidan["finish_14"],
        _fmt_elapsed(eval_secs), _fmt_elapsed(elapsed), _fmt_elapsed(eta_secs),
    )
    log.info("-" * 60)

    _append_metrics(run_dir, {
        "episode": total_ep,
        "timestamp": time.time(),
        "elapsed_s": elapsed,
        f"wr_{opp_name}": wr_stage,
        "wr_jidan": wr_jidan,
        "finish_12_jidan": result_jidan["finish_12"],
        "finish_13_jidan": result_jidan["finish_13"],
        "finish_14_jidan": result_jidan["finish_14"],
        "adapter_norm_lead": an_lead,
        "adapter_norm_follow": an_follow,
    })

    if wr_jidan > best_wr_jidan:
        best_wr_jidan = wr_jidan
        best_path = run_dir / "checkpoint_best.pt"
        torch.save({
            "lead": q_lead.state_dict(),
            "follow": q_follow.state_dict(),
            "adapter_lead": adapter_lead.state_dict(),
            "adapter_follow": adapter_follow.state_dict(),
            "episode": total_ep,
            "wr_jidan": wr_jidan,
            "d_state": STATE_DIM_TIER1,
            "adapter_hidden": adapter_lead.net[0].out_features,
        }, best_path)
        log.info("  *** New best: %.1f%% vs jidan → %s ***", wr_jidan * 100, best_path)

    return best_wr_jidan, wr_stage


def run_training(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    adapter_lead: PartnerAdapter,
    adapter_follow: PartnerAdapter,
    config: dict,
    run_dir: Path,
    device: torch.device,
) -> None:
    """Adapter-only Tier 1 training. Base Q-networks must already be frozen."""
    _setup_logging(run_dir)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("=" * 60)
    log.info("Tier 1 Training — Frozen Base + Partner Adapter")
    log.info("=" * 60)
    log.info("Device: %s", device)
    log.info("Config: %s", json.dumps(config, indent=2))
    log.info("Run dir: %s", run_dir)

    adapter_params = sum(p.numel() for p in adapter_lead.parameters())
    base_params = sum(p.numel() for p in q_lead.parameters())
    log.info("Adapter params: %d (%.1f%% of frozen base %d)",
             adapter_params, adapter_params / base_params * 100, base_params)

    buffer = ReplayBuffer(capacity=config["buffer_capacity"], d_state=STATE_DIM_TIER1)

    lr_start = config["lr_start"]
    lr_end = config["lr_end"]
    wd = config.get("weight_decay", 1e-2)
    opt_lead = torch.optim.Adam(adapter_lead.parameters(), lr=lr_start, weight_decay=wd)
    opt_follow = torch.optim.Adam(adapter_follow.parameters(), lr=lr_start, weight_decay=wd)

    env = GuanDanEnv()
    t_start = time.time()
    total_all = config["stage1_episodes"] + config["stage2_episodes"]

    # --- Pre-fill buffer ---
    prefill_eps = config["prefill_episodes"]
    log.info("Pre-filling buffer with %d episodes...", prefill_eps)
    t_prefill = time.time()
    q_lead.eval()
    q_follow.eval()
    adapter_lead.eval()
    adapter_follow.eval()
    opp_prefill = make_agent(config["stage1_opponent"], env.level_rank)
    prefill_trans = 0
    for _ in tqdm(range(prefill_eps), desc="prefill", file=sys.stderr):
        trans = play_episode_tier1(
            env, q_lead, q_follow, adapter_lead, adapter_follow,
            epsilon=config["epsilon_start"], device=device, opponent=opp_prefill,
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
    loss_sum = loss_count = 0

    for stage_idx, (opp_name, stage_episodes, gate) in enumerate(stages, 1):
        log.info("")
        log.info("=" * 60)
        log.info("Stage %d: vs %s (%d episodes, gate=%s)",
                 stage_idx, opp_name, stage_episodes,
                 f"{gate:.0%}" if gate else "none")
        log.info("=" * 60)
        opp = make_agent(opp_name, env.level_rank)
        t_stage = time.time()

        eps_start = config["epsilon_start"]
        eps_end = config["epsilon_end"]

        pbar = tqdm(range(stage_episodes), desc=f"stage{stage_idx}", file=sys.stderr)
        for ep_in_stage in pbar:
            total_ep += 1
            frac = min(1.0, total_ep / (0.85 * total_all))
            epsilon = eps_start + (eps_end - eps_start) * frac

            # Cosine LR decay over full training
            lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + np.cos(np.pi * total_ep / total_all))
            for opt in (opt_lead, opt_follow):
                for pg in opt.param_groups:
                    pg["lr"] = lr

            # Collect episode
            q_lead.eval()
            q_follow.eval()
            adapter_lead.eval()
            adapter_follow.eval()
            trans = play_episode_tier1(
                env, q_lead, q_follow, adapter_lead, adapter_follow,
                epsilon, device, opponent=opp,
            )
            for s, a, h, hl, r, oc in trans:
                buffer.push(s, a, h, hl, r, oc)

            # Train steps
            adapter_lead.train()
            adapter_follow.train()
            bs = config["batch_size"]
            output_reg = config.get("output_reg", 2.0)
            for _ in range(config["train_steps_per_episode"]):
                loss_l = train_step_adapter(q_lead, adapter_lead, buffer, opt_lead, bs, device,
                                            output_reg=output_reg)
                train_step_adapter(q_follow, adapter_follow, buffer, opt_follow, bs, device,
                                   output_reg=output_reg)
                if loss_l is not None:
                    loss_sum += loss_l
                    loss_count += 1

            if loss_count > 0:
                elapsed = time.time() - t_start
                pbar.set_postfix_str(
                    f"loss={loss_sum/loss_count:.4f} lr={lr:.1e} "
                    f"{total_ep/elapsed:.1f}ep/s"
                )

            # Periodic loss log
            if total_ep % 200 == 0 and loss_count > 0:
                elapsed = time.time() - t_start
                log.info(
                    "Ep %6d | loss=%.4f | lr=%.2e e=%.3f | buf=%d | %.1f ep/s | %s",
                    total_ep, loss_sum / loss_count, lr, epsilon, len(buffer),
                    total_ep / elapsed, _fmt_elapsed(elapsed),
                )
                loss_sum = loss_count = 0

            # Eval
            if (ep_in_stage + 1) % config["eval_interval"] == 0 or ep_in_stage == stage_episodes - 1:
                q_lead.eval()
                q_follow.eval()
                adapter_lead.eval()
                adapter_follow.eval()
                best_wr_jidan, wr_stage = _run_eval(
                    q_lead, q_follow, adapter_lead, adapter_follow,
                    device, config, total_ep, run_dir, t_start, opp_name, best_wr_jidan,
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
                    "lead": q_lead.state_dict(),
                    "follow": q_follow.state_dict(),
                    "adapter_lead": adapter_lead.state_dict(),
                    "adapter_follow": adapter_follow.state_dict(),
                    "episode": total_ep,
                    "d_state": STATE_DIM_TIER1,
                    "adapter_hidden": adapter_lead.net[0].out_features,
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
        "lead": q_lead.state_dict(),
        "follow": q_follow.state_dict(),
        "adapter_lead": adapter_lead.state_dict(),
        "adapter_follow": adapter_follow.state_dict(),
        "episode": total_ep,
        "d_state": STATE_DIM_TIER1,
        "best_wr_jidan": best_wr_jidan,
        "adapter_hidden": adapter_lead.net[0].out_features,
    }, final_path)
    log.info("")
    log.info("=" * 60)
    log.info("Training complete. Best vs jidan: %.1f%%. Wall time: %s",
             best_wr_jidan * 100, _fmt_elapsed(total_secs))
    log.info("Final: %s", final_path)
    log.info("=" * 60)
