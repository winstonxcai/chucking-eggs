"""GuanZero self-play training: replay buffer + DMC training loop.

Self-play episode: GuanZeroNetwork controls all 4 seats, epsilon-greedy.
MC returns: team-level rewards from env.get_rewards() assigned to all transitions.
Training: MSE loss Q(s,a) vs terminal return G.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..cards import Rank
from ..game import GuanDanEnv
from .guanzero_encoding import (
    _count_valid_history_steps,
    combo_to_108,
    compute_behavior_flags,
    encode_base_state,
    encode_history,
    score_all_actions,
)

# Replay buffer dimensions
_D_NON_HISTORY = 1075
_D_ACTION = 108
_N_HIST_STEPS = 5
_D_HIST_STEP = 432


class ReplayBuffer:
    """Circular replay buffer for GuanZero self-play transitions."""

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self._idx = 0
        self.size = 0

        self.non_history = np.zeros((capacity, _D_NON_HISTORY), dtype=np.float32)
        self.history = np.zeros((capacity, _N_HIST_STEPS, _D_HIST_STEP), dtype=np.float32)
        self.hist_len = np.zeros(capacity, dtype=np.int64)
        self.action = np.zeros((capacity, _D_ACTION), dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        nh: np.ndarray,       # [1075]
        hist: np.ndarray,     # [5, 432]
        hl: int,
        action: np.ndarray,   # [108]
        mc_return: float,
    ) -> None:
        i = self._idx % self.capacity
        self.non_history[i] = nh
        self.history[i] = hist
        self.hist_len[i] = hl
        self.action[i] = action
        self.returns[i] = mc_return
        self._idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "non_history": torch.tensor(self.non_history[indices], device=device),
            "history": torch.tensor(self.history[indices], device=device),
            "hist_len": torch.tensor(self.hist_len[indices], device=device),
            "action": torch.tensor(self.action[indices], device=device),
            "return": torch.tensor(self.returns[indices], device=device),
        }

    def __len__(self) -> int:
        return self.size


def play_selfplay_episode(
    net,
    env: GuanDanEnv,
    epsilon: float,
    device: torch.device,
    level_rank: int,
) -> list[tuple[np.ndarray, np.ndarray, int, np.ndarray, float]]:
    """Play one full self-play game with net controlling all 4 seats.

    Returns:
        List of (non_history, history, hist_len, action_enc, mc_return) tuples.
    """
    env.reset()
    transitions: dict[int, list[tuple]] = {p: [] for p in range(4)}

    net.eval()
    while not env.done:
        player = env.current_player
        legal = env.legal_moves()

        if len(legal) == 1:
            env.step(legal[0])
            continue

        # Score all actions using LSTM-once optimization
        q_values = score_all_actions(net, env, player, legal, level_rank, device)

        # Epsilon-greedy
        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            idx = int(q_values.argmax().item())

        # Record transition for the chosen action
        base = encode_base_state(env, player, level_rank)
        behavior = compute_behavior_flags(env, player, legal[idx], legal)
        nh = np.concatenate([base, behavior])          # [1075]
        hist = encode_history(env, player)             # [5, 432]
        hl = _count_valid_history_steps(env, player)
        a_enc = combo_to_108(legal[idx])               # [108]

        transitions[player].append((nh, hist, hl, a_enc))
        env.step(legal[idx])

    # Assign team-level MC returns
    rewards = env.get_rewards()  # {0: ±r, 1: ∓r, 2: ±r, 3: ∓r}

    all_transitions = []
    for player, tlist in transitions.items():
        G = float(rewards[player])
        for nh, hist, hl, a_enc in tlist:
            all_transitions.append((nh, hist, hl, a_enc, G))

    return all_transitions


def train_dmc_step(
    net,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> float | None:
    """One MSE gradient step: Q(s,a) → terminal return G.

    Returns loss value, or None if buffer is too small.
    """
    if len(buffer) < batch_size:
        return None

    net.train()
    batch = buffer.sample(batch_size, device)

    # Run LSTM in batch, then MLP
    lstm_emb = net.batch_encode_history(batch["history"], batch["hist_len"])  # [B, 128]
    q_pred = net.forward_from_embedding(
        batch["non_history"], batch["action"], lstm_emb
    )  # [B]

    loss = F.mse_loss(q_pred, batch["return"])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def evaluate_vs(
    net,
    opponent,
    n_games: int,
    level_rank: int,
    device: torch.device,
) -> float:
    """Evaluate net (seats 0,2) vs opponent (seats 1,3). Returns win rate."""
    env = GuanDanEnv(level_rank=level_rank)
    wins = 0
    net.eval()
    for _ in range(n_games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p in (0, 2):
                legal = env.legal_moves()
                if len(legal) == 1:
                    env.step(legal[0])
                else:
                    q = score_all_actions(net, env, p, legal, level_rank, device)
                    env.step(legal[int(q.argmax().item())])
            else:
                env.step(opponent.act(env, p))
        if env.get_rewards()[0] > 0:
            wins += 1
    return wins / n_games


def train_selfplay(
    net,
    buffer: ReplayBuffer,
    device: torch.device,
    level_rank: int = Rank.TWO,
    total_episodes: int = 150_000,
    batch_size: int = 512,
    train_steps_per_ep: int = 4,
    eval_interval: int = 10_000,
    eval_games: int = 200,
    save_dir: Path | None = None,
    prod_ckpt_path: Path | None = None,
    patience: int = 3,
    min_delta: float = 0.01,
    verbose: bool = True,
) -> None:
    """Full self-play DMC training loop.

    Eval strategy:
    - vs Heuristic: used as the patience signal (reachable early in training)
    - vs Jidan: tracked for convergence but NOT used for early stopping
    - vs prod RL (optional): tracked alongside Jidan

    Early stops when wr_vs_heuristic fails to improve by ≥ min_delta for
    `patience` consecutive evals (default: 3 evals = 30K episodes).
    """
    from ..agents.heuristic_bot import HeuristicBot
    from ..agents.jidan_bot import JidanBot

    heuristic = HeuristicBot(level_rank)
    jidan = JidanBot()
    prod_agent = _load_prod_agent(prod_ckpt_path, device) if prod_ckpt_path else None

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_episodes, eta_min=3e-6
    )

    env = GuanDanEnv(level_rank=level_rank)
    best_patience_wr = 0.0
    no_improve_count = 0

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(total_episodes):
        # Epsilon: 0.15 → 0.03 over first 80% of training
        epsilon = max(0.03, 0.15 - 0.12 * ep / max(1, int(0.8 * total_episodes)))

        # Collect one self-play episode
        transitions = play_selfplay_episode(net, env, epsilon, device, level_rank)
        for nh, hist, hl, a_enc, G in transitions:
            buffer.push(nh, hist, hl, a_enc, G)

        # Train (scheduler only steps when optimizer actually ran)
        trained = False
        for _ in range(train_steps_per_ep):
            loss = train_dmc_step(net, optimizer, buffer, batch_size, device)
            if loss is not None:
                trained = True
        if trained:
            scheduler.step()

        # Evaluate
        if (ep + 1) % eval_interval == 0:
            wr_heuristic = evaluate_vs(net, heuristic, eval_games, level_rank, device)
            wr_jidan = evaluate_vs(net, jidan, eval_games, level_rank, device)
            tracking_wr = (wr_jidan + wr_heuristic) / 2.0

            if prod_agent is not None:
                wr_prod = evaluate_vs(net, prod_agent, eval_games, level_rank, device)
                tracking_wr = (wr_jidan + wr_heuristic + wr_prod) / 3.0
                if verbose:
                    print(
                        f"Episode {ep + 1:6d}: "
                        f"vs_heur={wr_heuristic:.3f}  vs_jidan={wr_jidan:.3f}  "
                        f"vs_prod={wr_prod:.3f}  tracking={tracking_wr:.3f}  buf={len(buffer)}"
                    )
            else:
                if verbose:
                    print(
                        f"Episode {ep + 1:6d}: "
                        f"vs_heur={wr_heuristic:.3f}  vs_jidan={wr_jidan:.3f}  "
                        f"tracking={tracking_wr:.3f}  buf={len(buffer)}"
                    )

            # Patience is based on heuristic WR only
            if wr_heuristic >= best_patience_wr + min_delta:
                best_patience_wr = wr_heuristic
                no_improve_count = 0
                if save_dir:
                    _save(net, save_dir / "guanzero_best.pt", level_rank)
            else:
                no_improve_count += 1
                if verbose:
                    print(f"  [Patience] heuristic WR no improvement {no_improve_count}/{patience} "
                          f"(best={best_patience_wr:.3f})")
                if no_improve_count >= patience:
                    if verbose:
                        print(f"[Selfplay] Early stop at episode {ep + 1}. "
                              f"Best heuristic WR={best_patience_wr:.3f}")
                    break

        if save_dir and (ep + 1) % (eval_interval * 5) == 0:
            _save(net, save_dir / f"guanzero_ep{ep + 1}.pt", level_rank)

    if save_dir:
        _save(net, save_dir / "guanzero_final.pt", level_rank)
        if verbose:
            print(f"[Selfplay] Training complete. Best heuristic WR={best_patience_wr:.3f}")


def _save(net, path: Path, level_rank: int) -> None:
    torch.save({"state_dict": net.state_dict(), "level_rank": level_rank}, path)


def _load_prod_agent(ckpt_path: Path, device: torch.device):
    """Load the prod RL checkpoint as an opponent agent."""
    from ..agents.rl_agent import RLAgentLSTM
    from .q_network import QNetworkLSTM, load_compat

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    q_lead = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    q_follow = QNetworkLSTM(lstm_hidden=256, hidden=1024).to(device)
    if "lead" in ckpt:
        q_lead.load_state_dict(ckpt["lead"])
        q_follow.load_state_dict(ckpt["follow"])
    else:
        load_compat(q_lead, ckpt.get("lead_state_dict", ckpt))
        load_compat(q_follow, ckpt.get("follow_state_dict", ckpt))
    q_lead.eval()
    q_follow.eval()
    return RLAgentLSTM(q_lead, q_follow, device=device)
