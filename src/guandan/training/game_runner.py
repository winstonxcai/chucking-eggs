"""Batched N-env game runner with LSTM-dedup GPU inference.

Manages N parallel GuanDanEnv instances, batches all pending decisions
into a single GPU forward pass per round. ~100 games/sec on A10G.
"""

from __future__ import annotations

import random
from typing import Generator

import numpy as np
import torch

from ..game import GuanDanEnv
from .encoding import D_MOVE, MAX_HISTORY, encode_action, encode_history, encode_state
from .q_network import QNetworkLSTM


class GameRunner:
    """Manage N parallel games with batched GPU inference."""

    def __init__(
        self,
        n_envs: int,
        q_lead: QNetworkLSTM,
        q_follow: QNetworkLSTM,
        device: torch.device,
        level_rank,
        epsilon: float = 0.1,
        opponent=None,
    ):
        self.n_envs = n_envs
        self.q_lead = q_lead
        self.q_follow = q_follow
        self.device = device
        self.level_rank = level_rank
        self.epsilon = epsilon
        self.opponent = opponent  # None = self-play, else agent for seats 1,3
        self.team_spirit = 0.0  # mix partner reward: 0=individual, 1=fully shared

        self.envs = [GuanDanEnv(level_rank) for _ in range(n_envs)]
        self._transitions: list[list] = [[] for _ in range(n_envs)]

    def generate_episodes(
        self, n_episodes: int,
    ) -> Generator[list[tuple], None, None]:
        """Generate n_episodes via batched play. Yields transition lists.

        Each transition: (state, action_enc, history, hist_len, mc_return)
        """
        for env in self.envs:
            env.reset()
        self._transitions = [[] for _ in range(self.n_envs)]

        episodes_completed = 0

        while episodes_completed < n_episodes:
            # Step forced moves and opponent moves without GPU
            self._step_trivial_moves()

            # Collect pending decisions from non-done envs needing RL action
            lead_pending, follow_pending = self._collect_decisions()

            # Batched GPU forward pass
            if lead_pending:
                self._batch_inference(lead_pending, self.q_lead)
            if follow_pending:
                self._batch_inference(follow_pending, self.q_follow)

            # Check for completed games
            for i in range(self.n_envs):
                if not self.envs[i].done:
                    continue

                rewards = self.envs[i].get_rewards()
                transitions = self._finalize_episode(i, rewards)
                yield transitions
                episodes_completed += 1

                if episodes_completed >= n_episodes:
                    return

                # Reset for next game
                self.envs[i].reset()
                self._transitions[i] = []

    def _step_trivial_moves(self) -> None:
        """Step forced moves (1 legal action) and opponent moves without GPU.

        Caches legal moves for non-trivial decisions in self._cached_legal
        so _collect_decisions() doesn't recompute them.
        """
        self._cached_legal: dict[int, list] = {}
        for i, env in enumerate(self.envs):
            if env.done:
                continue

            # Step repeatedly while move is trivial
            while not env.done:
                player = env.current_player

                # Opponent seats
                if self.opponent is not None and player in (1, 3):
                    env.step(self.opponent.act(env, player))
                    continue

                legal = env.legal_moves(player)
                if len(legal) == 1:
                    env.step(legal[0])
                    continue

                self._cached_legal[i] = legal
                break  # needs RL decision — handled by _batch_inference

    def _collect_decisions(self) -> tuple[list, list]:
        """Collect all pending decisions, split by lead/follow.

        Uses cached legal moves from _step_trivial_moves() to avoid redundant calls.
        """
        lead, follow = [], []
        for i, legal in self._cached_legal.items():
            env = self.envs[i]
            player = env.current_player
            is_leading = env.current_trick is None
            entry = (i, player, legal, is_leading)
            if is_leading:
                lead.append(entry)
            else:
                follow.append(entry)
        return lead, follow

    def _batch_inference(self, pending: list, q_net: QNetworkLSTM) -> None:
        """Batched forward: LSTM once per decision, MLP per action."""
        N = len(pending)
        states, hists, hlens, action_lists = [], [], [], []

        for (env_idx, player, legal, _) in pending:
            env = self.envs[env_idx]
            states.append(encode_state(env, player))
            h, hl = encode_history(env, player, env.level_rank)
            hists.append(h)
            hlens.append(hl)
            action_lists.append(
                [encode_action(m, env.hands[player], env.level_rank) for m in legal]
            )

        # Pad histories to [N, MAX_HISTORY, D_MOVE]
        hist_pad = torch.zeros(N, MAX_HISTORY, D_MOVE, dtype=torch.float32, device=self.device)
        for i, h in enumerate(hists):
            T = min(len(h), MAX_HISTORY)
            if T > 0:
                hist_pad[i, :T] = torch.tensor(h[:T], dtype=torch.float32)
        hlens_t = torch.tensor(hlens, dtype=torch.long)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)

        # LSTM once per decision
        with torch.no_grad():
            hist_emb = q_net.encode_history(hist_pad, hlens_t)  # [N, lstm_hidden]

        # Per-decision MLP + action selection
        for i, (env_idx, player, legal, is_leading) in enumerate(pending):
            B = len(legal)
            a_enc = torch.tensor(
                np.array(action_lists[i]), dtype=torch.float32, device=self.device
            )
            s_exp = states_t[i].unsqueeze(0).expand(B, -1)
            h_exp = hist_emb[i].unsqueeze(0).expand(B, -1)

            with torch.no_grad():
                q_vals = q_net.forward_from_embedding(s_exp, a_enc, h_exp)

            # ε-greedy
            if random.random() < self.epsilon:
                idx = random.randint(0, B - 1)
            else:
                idx = q_vals.argmax().item()

            # Record transition
            self._transitions[env_idx].append((
                states[i],
                np.array(action_lists[i][idx], dtype=np.float32),
                hists[i],
                hlens[i],
                player,
            ))

            self.envs[env_idx].step(legal[idx])

    def _finalize_episode(
        self, env_idx: int, rewards: dict[int, float],
    ) -> list[tuple]:
        """Assign MC returns and return finalized transitions."""
        partners = {0: 2, 1: 3, 2: 0, 3: 1}
        ts = self.team_spirit
        transitions = []
        for (state, action, history, hist_len, player) in self._transitions[env_idx]:
            G = (1 - ts) * rewards[player] + ts * rewards[partners[player]]
            transitions.append((state, action, history, hist_len, G))
        return transitions
