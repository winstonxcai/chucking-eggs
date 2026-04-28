"""Per-player GAE + PPO rollout buffer.

Each decision is stored in the track of the acting player only. GAE
bootstraps from the same player's next decision — never from another
player's intervening turn.

Early-finisher handling: a player's track terminates when they go out.
Their terminal step receives the team reward and bootstraps from V_next=0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch


class Decision(NamedTuple):
    """One decision stored per acting player per step."""
    state_actor: np.ndarray    # [K, ACTOR_DIM]  — K legal candidates
    actions:     np.ndarray    # [K, ACTION_DIM]
    legal_mask:  np.ndarray    # [K] bool
    sampled_idx: int           # which candidate was chosen
    log_prob:    float         # log π(a|s) at sample time
    state_critic: np.ndarray   # [CRITIC_DIM]  — action-independent
    v_old:        float        # V(s) frozen at rollout time


@dataclass
class PlayerTrack:
    """Trajectory for one player within one hand."""
    player: int
    decisions: list[Decision] = field(default_factory=list)
    terminal_reward: float = 0.0
    is_complete: bool = False

    def add(self, decision: Decision) -> None:
        self.decisions.append(decision)

    def finalize(self, reward: float) -> None:
        self.terminal_reward = reward
        self.is_complete = True


@dataclass
class PPOBatch:
    """Flattened batch ready for PPO updates. All tensors are float32."""
    # Per-decision candidate tensors (ragged → padded to max_K)
    state_actor:  torch.Tensor   # [N, max_K, ACTOR_DIM]
    actions:      torch.Tensor   # [N, max_K, ACTION_DIM]
    legal_mask:   torch.Tensor   # [N, max_K] bool
    sampled_idx:  torch.Tensor   # [N] long
    old_log_prob: torch.Tensor   # [N]
    state_critic: torch.Tensor   # [N, CRITIC_DIM]
    v_old:        torch.Tensor   # [N]
    returns:      torch.Tensor   # [N]  — GAE returns (advantages + V_old)
    advantages:   torch.Tensor   # [N]  — normalised (mean=0, std=1)


@dataclass
class RolloutStats:
    """Aggregate rollout statistics — diagnostic only."""
    K_values:      list[int] = field(default_factory=list)   # legal-move count per decision
    pass_count:    int = 0                                   # PASS actions sampled
    non_pass_count:int = 0
    bomb_count:    int = 0                                   # bomb-tier actions sampled
    lead_count:    int = 0                                   # lead actions (no current_trick)
    hand_lengths:  list[int] = field(default_factory=list)   # decisions per hand

    def merge(self, other: "RolloutStats") -> None:
        """Merge another RolloutStats in-place (used to aggregate worker stats)."""
        self.K_values.extend(other.K_values)
        self.pass_count     += other.pass_count
        self.non_pass_count += other.non_pass_count
        self.bomb_count     += other.bomb_count
        self.lead_count     += other.lead_count
        self.hand_lengths.extend(other.hand_lengths)

    def summary(self) -> dict:
        K = np.array(self.K_values, dtype=np.int32) if self.K_values else np.zeros(1)
        total_actions = self.pass_count + self.non_pass_count
        return {
            "mean_K":              float(K.mean()),
            "p95_K":               float(np.percentile(K, 95)),
            "max_K":               int(K.max()),
            "pass_rate":           self.pass_count / max(total_actions, 1),
            "bomb_play_rate":      self.bomb_count / max(total_actions, 1),
            "lead_rate":           self.lead_count / max(total_actions, 1),
            "mean_hand_length":    float(np.mean(self.hand_lengths)) if self.hand_lengths else 0.0,
            "n_hands_iter":        len(self.hand_lengths),
        }


class RolloutBuffer:
    """Collects per-player tracks across multiple hands, then computes GAE."""

    def __init__(self, gamma: float = 1.0, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam
        self._completed_tracks: list[PlayerTrack] = []
        self.stats = RolloutStats()

    def add_track(self, track: PlayerTrack) -> None:
        """Store a completed player track (must have is_complete=True)."""
        assert track.is_complete, "Track must be finalized before adding"
        self._completed_tracks.append(track)

    @classmethod
    def from_buffers(
        cls, bufs: list["RolloutBuffer"], gamma: float = 1.0, lam: float = 0.95
    ) -> "RolloutBuffer":
        """Aggregate per-worker buffers into a single buffer for batch building."""
        merged = cls(gamma=gamma, lam=lam)
        for b in bufs:
            merged._completed_tracks.extend(b._completed_tracks)
            merged.stats.merge(b.stats)
        return merged

    def size(self) -> int:
        """Total decisions stored."""
        return sum(len(t.decisions) for t in self._completed_tracks)

    def clear(self) -> None:
        self._completed_tracks.clear()

    def compute_batch(self, device: torch.device | None = None) -> PPOBatch:
        """Compute GAE advantages and build a PPOBatch.

        V_old and advantages are frozen from rollout-time values.
        """
        if not self._completed_tracks:
            raise RuntimeError("Buffer is empty")

        all_state_actor: list[np.ndarray] = []
        all_actions:     list[np.ndarray] = []
        all_legal_mask:  list[np.ndarray] = []
        all_sampled_idx: list[int] = []
        all_log_prob:    list[float] = []
        all_state_critic: list[np.ndarray] = []
        all_v_old:       list[float] = []
        all_returns:     list[float] = []
        all_advantages:  list[float] = []

        for track in self._completed_tracks:
            T = len(track.decisions)
            if T == 0:
                continue

            v_olds = np.array([d.v_old for d in track.decisions], dtype=np.float64)
            rewards = np.zeros(T, dtype=np.float64)
            rewards[-1] = track.terminal_reward  # only terminal step gets reward

            # Per-player GAE: V_next for terminal step = 0 (player is done)
            advantages = np.zeros(T, dtype=np.float64)
            gae = 0.0
            for t in reversed(range(T)):
                v_next = v_olds[t + 1] if t + 1 < T else 0.0
                delta = rewards[t] + self.gamma * v_next - v_olds[t]
                gae = delta + self.gamma * self.lam * gae
                advantages[t] = gae

            returns = advantages + v_olds

            for t, dec in enumerate(track.decisions):
                all_state_actor.append(dec.state_actor)
                all_actions.append(dec.actions)
                all_legal_mask.append(dec.legal_mask)
                all_sampled_idx.append(dec.sampled_idx)
                all_log_prob.append(dec.log_prob)
                all_state_critic.append(dec.state_critic)
                all_v_old.append(float(v_olds[t]))
                all_returns.append(float(returns[t]))
                all_advantages.append(float(advantages[t]))

        # Normalise advantages
        adv_arr = np.array(all_advantages, dtype=np.float32)
        adv_mean = adv_arr.mean()
        adv_std  = adv_arr.std() + 1e-8
        adv_norm = (adv_arr - adv_mean) / adv_std

        # Pad ragged candidate sets to max_K
        max_K = max(a.shape[0] for a in all_state_actor)
        N = len(all_state_actor)
        ACTOR_DIM  = all_state_actor[0].shape[1]
        ACTION_DIM = all_actions[0].shape[1]
        CRITIC_DIM = all_state_critic[0].shape[0]

        pad_state_actor = np.zeros((N, max_K, ACTOR_DIM),  dtype=np.float32)
        pad_actions     = np.zeros((N, max_K, ACTION_DIM), dtype=np.float32)
        pad_legal_mask  = np.zeros((N, max_K),             dtype=bool)

        for i, (sa, ac, lm) in enumerate(zip(all_state_actor, all_actions, all_legal_mask)):
            K = sa.shape[0]
            pad_state_actor[i, :K] = sa
            pad_actions[i, :K]     = ac
            pad_legal_mask[i, :K]  = lm

        dev = device or torch.device("cpu")

        return PPOBatch(
            state_actor  = torch.tensor(pad_state_actor, dtype=torch.float32, device=dev),
            actions      = torch.tensor(pad_actions,     dtype=torch.float32, device=dev),
            legal_mask   = torch.tensor(pad_legal_mask,  dtype=torch.bool,    device=dev),
            sampled_idx  = torch.tensor(all_sampled_idx, dtype=torch.long,    device=dev),
            old_log_prob = torch.tensor(all_log_prob,    dtype=torch.float32, device=dev),
            state_critic = torch.tensor(np.array(all_state_critic, dtype=np.float32), device=dev),
            v_old        = torch.tensor(all_v_old,       dtype=torch.float32, device=dev),
            returns      = torch.tensor(all_returns,     dtype=torch.float32, device=dev),
            advantages   = torch.tensor(adv_norm,        dtype=torch.float32, device=dev),
        )
