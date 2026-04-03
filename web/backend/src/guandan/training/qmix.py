"""QMIX training: episode collection and end-to-end gradient training.

Two phases:
  Phase A — Freeze Q-networks, train mixer only (Q-values re-computed with no_grad).
  Phase B — End-to-end: re-run Q-networks WITH gradients so mixer loss back-props
             through Q-values into Q-network weights.

Key fix vs. original broken implementation: Q-values are NOT stored as detached
scalars in the buffer. Instead, raw (state, action, history) inputs are stored
and Q-networks are re-run during training to maintain gradient tape.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..cards import Rank
from ..game import GuanDanEnv
from .encoding import (
    D_MOVE, MAX_HISTORY,
    encode_action, encode_global_state, encode_history, encode_state,
)
from .mixing import TeamMixer, UnrestrictedMixer
from .q_network import QNetworkLSTM
from .qmix_buffer import QMIXBuffer


# ── Trick-level data collection ────────────────────────────────────────────────

class TrickCollector:
    """Accumulates raw Q-net inputs per teammate per trick, emits trick records."""

    def __init__(self, team: tuple[int, int] = (0, 2)):
        self.team = team
        self._transitions: dict[int, dict] = {}
        self._gs: np.ndarray | None = None

    def start_trick(self, env: GuanDanEnv) -> None:
        self._transitions = {}
        self._gs = encode_global_state(env, self.team)

    def record(
        self,
        player: int,
        state: np.ndarray,
        action_enc: np.ndarray,
        history: np.ndarray,
        hist_len: int,
        is_leading: bool,
    ) -> None:
        if player in self.team:
            self._transitions[player] = {
                "state":      state,
                "action":     np.array(action_enc, dtype=np.float32),
                "history":    history,
                "hist_len":   hist_len,
                "is_leading": is_leading,
            }

    def emit(self) -> dict | None:
        """Return trick record, or None if no teammate acted this trick."""
        if not self._transitions or self._gs is None:
            return None
        return {
            "transitions": dict(self._transitions),
            "global_state": self._gs,
        }


def play_episode_qmix(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
    level_rank: int,
    team: tuple[int, int] = (0, 2),
) -> tuple[list[dict], dict[int, float]]:
    """Self-play episode collecting trick-level QMIX data.

    Returns:
        trick_data:  list of trick records (transitions, global_state) with team_return filled in
        rewards:     dict[player → final reward]
    """
    env.reset()
    q_lead.eval()
    q_follow.eval()

    trick_records_raw: list[dict] = []
    collector = TrickCollector(team)
    prev_trick_winner: int | None = None

    with torch.no_grad():
        while not env.done:
            player = env.current_player
            legal = env.legal_moves(player)

            # Track trick boundaries
            if env.trick_winner != prev_trick_winner:
                if prev_trick_winner is not None:
                    rec = collector.emit()
                    if rec is not None:
                        trick_records_raw.append(rec)
                collector.start_trick(env)
                prev_trick_winner = env.trick_winner

            if len(legal) == 1:
                env.step(legal[0])
                continue

            state = encode_state(env, player)
            history, hist_len = encode_history(env, player, level_rank)
            hand = env.hands[player]
            is_leading = env.current_trick is None

            q_net = q_lead if is_leading else q_follow
            action_encs = [encode_action(m, hand, level_rank) for m in legal]
            B = len(legal)

            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
            hist_t = torch.zeros(1, MAX_HISTORY, D_MOVE, dtype=torch.float32, device=device)
            T = min(hist_len, MAX_HISTORY)
            if T > 0:
                hist_t[0, :T] = torch.tensor(history[:T], dtype=torch.float32)
            hist_emb = q_net.encode_history(hist_t, torch.tensor([max(hist_len, 1)]))
            hist_emb_exp = hist_emb.expand(B, -1)
            a_t = torch.tensor(np.array(action_encs), dtype=torch.float32, device=device)
            q_vals = q_net.forward_from_embedding(state_t, a_t, hist_emb_exp)

            idx = random.randrange(B) if random.random() < epsilon else q_vals.argmax().item()

            # Record raw inputs (NOT the Q-value scalar)
            collector.record(
                player, state, action_encs[idx], history, hist_len, is_leading
            )
            env.step(legal[idx])

    # Emit final trick
    rec = collector.emit()
    if rec is not None:
        trick_records_raw.append(rec)

    rewards = env.get_rewards()
    team_return = float(rewards[team[0]] + rewards[team[1]])

    # Fill in team_return for all trick records
    trick_data = []
    for r in trick_records_raw:
        r["team_return"] = team_return
        trick_data.append(r)

    return trick_data, rewards


# ── Training steps ─────────────────────────────────────────────────────────────

def _compute_q_vals(
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    batch: dict,
    device: torch.device,
    no_grad: bool = False,
) -> torch.Tensor:
    """Re-run Q-networks for both slots, return [B, 2] Q-values.

    With no_grad=True (Phase A): Q-nets are frozen, used as encoders.
    With no_grad=False (Phase B): gradients flow back to Q-net weights.
    """
    B = batch["states"].size(0)
    q_vals = torch.zeros(B, 2, device=device)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        for slot in range(2):
            act_mask = batch["active"][:, slot]
            if not act_mask.any():
                continue

            idx = act_mask.nonzero(as_tuple=True)[0]
            s  = batch["states"][idx, slot]
            a  = batch["actions"][idx, slot]
            h  = batch["histories"][idx, slot]
            hl = batch["hist_lens"][idx, slot]
            lead_mask = batch["is_leading"][idx, slot]

            for net, mask in [(q_lead, lead_mask), (q_follow, ~lead_mask)]:
                if not mask.any():
                    continue
                mi = mask.nonzero(as_tuple=True)[0]
                q_vals[idx[mi], slot] = net(s[mi], a[mi], h[mi], hl[mi])

    return q_vals


def train_mixer_step(
    mixer: TeamMixer,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    qmix_buf: QMIXBuffer,
    opt_mixer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> float:
    """Phase A: train mixer on Q-values from frozen Q-networks. Returns loss."""
    batch = qmix_buf.sample(batch_size, device)
    returns = batch["return"].float()

    # Q-nets are frozen (requires_grad=False in Phase A) — no_grad for efficiency
    q_vals = _compute_q_vals(q_lead, q_follow, batch, device, no_grad=True)

    q_team = mixer(q_vals, batch["global_state"])
    loss = nn.functional.mse_loss(q_team, returns)

    opt_mixer.zero_grad()
    loss.backward()
    opt_mixer.step()
    return loss.item()


def train_qmix_e2e_step(
    mixer: TeamMixer,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    qmix_buf: QMIXBuffer,
    opt: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    loss_weight: float = 0.001,
) -> dict[str, float]:
    """Phase B: true end-to-end QMIX — gradient flows from mixer loss to Q-nets.

    Q-networks are re-run WITH gradients so the computational graph connects
    mixer → q_vals → Q-net weights. loss_weight scales the QMIX loss to keep
    Q-net gradient magnitude reasonable relative to the mixer gradient.
    """
    batch = qmix_buf.sample(batch_size, device)
    returns = batch["return"].float()

    # Re-run Q-nets WITH gradients (this is the key fix)
    q_vals = _compute_q_vals(q_lead, q_follow, batch, device, no_grad=False)

    q_team = mixer(q_vals, batch["global_state"])
    qmix_loss = nn.functional.mse_loss(q_team, returns) * loss_weight

    opt.zero_grad()
    qmix_loss.backward()

    q_lead_grad  = nn.utils.clip_grad_norm_(q_lead.parameters(),  max_norm=0.1).item()
    q_follow_grad = nn.utils.clip_grad_norm_(q_follow.parameters(), max_norm=0.1).item()
    mixer_grad   = nn.utils.clip_grad_norm_(mixer.parameters(),   max_norm=1.0).item()
    opt.step()

    return {
        "qmix":          qmix_loss.item() / loss_weight,  # unscaled for readability
        "q_lead_grad":   q_lead_grad,
        "q_follow_grad": q_follow_grad,
        "mixer_grad":    mixer_grad,
    }


def train_wqmix_e2e_step(
    mixer: TeamMixer,
    mixer_star: UnrestrictedMixer,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    qmix_buf: QMIXBuffer,
    opt: torch.optim.Optimizer,
    opt_star: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    loss_weight: float = 0.001,
    alpha: float = 0.1,
) -> dict[str, float]:
    """WQMIX: Optimistically-Weighted end-to-end training.

    Two mixers:
      - mixer (TeamMixer): monotonic Q_tot — deployed at inference
      - mixer_star (UnrestrictedMixer): unrestricted Q* — training-only reference

    OW weighting: w=1 if Q_tot < Q*, w=alpha if Q_tot >= Q*.
    This down-weights overestimated actions, letting the monotonic mixer
    avoid forcing up Q-values for actions that hurt the team.
    """
    batch = qmix_buf.sample(batch_size, device)
    returns = batch["return"].float()
    gs = batch["global_state"]

    # Re-run Q-nets WITH gradients
    q_vals = _compute_q_vals(q_lead, q_follow, batch, device, no_grad=False)

    # --- Step 1: Train Q* (unrestricted mixer) on unweighted MSE ---
    q_star = mixer_star(q_vals.detach(), gs)
    star_loss = nn.functional.mse_loss(q_star, returns)
    opt_star.zero_grad()
    star_loss.backward()
    nn.utils.clip_grad_norm_(mixer_star.parameters(), max_norm=1.0)
    opt_star.step()

    # --- Step 2: Compute OW weights ---
    with torch.no_grad():
        q_tot_detached = mixer(q_vals.detach(), gs)
        q_star_detached = mixer_star(q_vals.detach(), gs)
        # w=1 where Q_tot underestimates (safe to push up), w=alpha where overestimated
        weights = torch.where(q_tot_detached < q_star_detached, 1.0, alpha)

    # --- Step 3: Train Q_tot (monotonic mixer + Q-nets) with weighted MSE ---
    q_team = mixer(q_vals, gs)
    td_error = (q_team - returns) ** 2
    wqmix_loss = (weights * td_error).mean() * loss_weight

    opt.zero_grad()
    wqmix_loss.backward()

    q_lead_grad = nn.utils.clip_grad_norm_(q_lead.parameters(), max_norm=0.1).item()
    q_follow_grad = nn.utils.clip_grad_norm_(q_follow.parameters(), max_norm=0.1).item()
    mixer_grad = nn.utils.clip_grad_norm_(mixer.parameters(), max_norm=1.0).item()
    opt.step()

    return {
        "wqmix": wqmix_loss.item() / loss_weight,
        "star_loss": star_loss.item(),
        "q_lead_grad": q_lead_grad,
        "q_follow_grad": q_follow_grad,
        "mixer_grad": mixer_grad,
        "ow_frac": (weights < 1.0).float().mean().item(),
    }


def verify_e2e_gradients(
    mixer: TeamMixer,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    qmix_buf: QMIXBuffer,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    """Run one e2e training step and assert Q-net gradients are nonzero.

    Call before Phase B training to confirm the gradient tape is connected.
    Raises AssertionError if Q-net gradient is zero (broken e2e).
    """
    opt = torch.optim.Adam([
        {"params": mixer.parameters(),    "lr": 1e-3},
        {"params": q_lead.parameters(),   "lr": 1e-6},
        {"params": q_follow.parameters(), "lr": 1e-6},
    ])
    diag = train_qmix_e2e_step(mixer, q_lead, q_follow, qmix_buf, opt, batch_size, device)
    q_gn = diag["q_lead_grad"]
    m_gn = diag["mixer_grad"]
    ratio = m_gn / max(q_gn, 1e-10)
    print(f"  Gradient verification:")
    print(f"    Q grad norm: {q_gn:.6f}  Mixer grad norm: {m_gn:.4f}  Ratio: {ratio:.0f}×")
    assert q_gn > 0, f"Q-network gradient is zero — e2e gradient tape is broken"
    print(f"  PASS")


# ── Buffer fill helper ──────────────────────────────────────────────────────────

def fill_buffers(
    env: GuanDanEnv,
    q_lead: QNetworkLSTM,
    q_follow: QNetworkLSTM,
    epsilon: float,
    device: torch.device,
    level_rank: int,
    qmix_buf: QMIXBuffer,
    n_episodes: int,
) -> int:
    """Run n_episodes and push trick-level data to qmix_buf. Returns trick count."""
    total_tricks = 0
    for _ in range(n_episodes):
        trick_data, rewards = play_episode_qmix(
            env, q_lead, q_follow, epsilon, device, level_rank,
        )
        team_return = float(rewards[0] + rewards[2])
        for rec in trick_data:
            qmix_buf.push(rec, team_return)
        total_tricks += len(trick_data)
    return total_tricks
