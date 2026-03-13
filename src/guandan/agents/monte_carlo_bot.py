"""MonteCarloBot — look-ahead agent using rollout simulation.

For each candidate move, simulates n_sims games using GreedyBot rollouts
and picks the move with the highest average reward. Uses strategic pruning
to keep candidates to ~6-10 moves, and multiprocessing for parallel rollouts.

Speed reference (8 cores, n_sims=50):
  ~10 candidates × 50 rollouts × 10ms/game ÷ 8 cores ≈ 0.6s per decision
  ~30 decisions/game → ~18s/game → 200 games ≈ 1 hour
"""

from __future__ import annotations

import copy
from multiprocessing import Pool

from ..cards import BOMB_TYPES, ComboType, Rank, level_order_key
from ..heuristic import _pass_combo
from .base import Agent
from .greedy_bot import GreedyBot

# Types using level-order key for sorting.
_LEVEL_ORDER_TYPES = frozenset({
    ComboType.SINGLE, ComboType.PAIR,
    ComboType.TRIPLE, ComboType.FULL_HOUSE,
})


# ─── Module-level worker (must be at top level for pickle / multiprocessing) ─

def _rollout_worker(env, player: int, move, level_rank: int) -> float:
    """Deep-copy env, play move, roll out with GreedyBot, return reward."""
    rollout_agent = GreedyBot(level_rank)
    sim = copy.deepcopy(env)
    sim.step(move)
    while not sim.done:
        p = sim.current_player
        sim.step(rollout_agent.act(sim, p))
    return sim.get_rewards()[player]


# ─── MonteCarloBot ──────────────────────────────────────────────────────────

class MonteCarloBot(Agent):
    """Monte Carlo simulation agent with strategic pruning and parallel rollouts.

    Parameters
    ----------
    n_sims:    Number of rollout simulations per candidate move.
    n_workers: Number of parallel processes (1 = sequential, no overhead).
    level_rank: Wild card rank for this game.
    """

    def __init__(
        self,
        n_sims: int = 50,
        n_workers: int = 8,
        level_rank: int = Rank.TWO,
    ):
        self.n_sims = n_sims
        self.n_workers = n_workers
        self.level_rank = level_rank

    def act(self, env, player: int):
        legal = env.legal_moves(player)

        # Trivial cases: nothing to choose between
        non_pass = [m for m in legal if m.type != ComboType.PASS]
        if len(non_pass) == 0:
            return _pass_combo(legal)
        if len(non_pass) == 1:
            return non_pass[0]

        # Prune to strategic candidates
        is_leading = env.current_trick is None
        candidates = (
            self._prune_lead(legal, env, player)
            if is_leading
            else self._prune_follow(legal, env, player)
        )

        # Single real candidate after pruning — skip simulation
        real = [m for m in candidates if m.type != ComboType.PASS]
        if len(real) == 0:
            return _pass_combo(legal)
        if len(real) == 1:
            return real[0]

        # Evaluate each candidate via rollout
        best_move = candidates[0]
        best_score = -float("inf")

        for move in candidates:
            score = self._evaluate_move(env, player, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # ─── Pruning ──────────────────────────────────────────────────────────

    def _prune_lead(self, legal: list, env, player: int) -> list:
        """Reduce leading moves to ~6-10 strategic candidates."""
        hand_size = len(env.hands[player])
        candidates = []

        # 1. Go-out plays (always consider)
        for m in legal:
            if m.type != ComboType.PASS and len(m.cards) == hand_size:
                candidates.append(m)

        # 2. Weakest (and strongest) of each ordinary combo type
        by_type: dict = {}
        for m in legal:
            if m.type == ComboType.PASS or m.type in BOMB_TYPES:
                continue
            by_type.setdefault(m.type, []).append(m)

        for ctype, moves in by_type.items():
            if ctype in _LEVEL_ORDER_TYPES:
                moves.sort(key=lambda m: level_order_key(m.key, self.level_rank))
            else:
                moves.sort(key=lambda m: m.key)
            candidates.append(moves[0])  # weakest
            if len(moves) >= 3:
                candidates.append(moves[-1])  # strongest (take-control intent)

        # 3. Weakest bomb (if any)
        bombs = [m for m in legal if m.type in BOMB_TYPES]
        if bombs:
            bombs.sort(key=lambda m: (m.type, m.key))
            candidates.append(bombs[0])

        return _deduplicate(candidates)

    def _prune_follow(self, legal: list, env, player: int) -> list:
        """Reduce following moves to ~4-7 strategic candidates."""
        hand_size = len(env.hands[player])
        trick = env.current_trick
        candidates = []

        # 1. Always consider PASS
        p = _pass_combo(legal)
        if p:
            candidates.append(p)

        # 2. Go-out plays
        for m in legal:
            if m.type != ComboType.PASS and len(m.cards) == hand_size:
                candidates.append(m)

        # 3. Same-type beats: cheapest, median, strongest
        same_type = [
            m for m in legal
            if m.type == trick.type and m.type != ComboType.PASS
        ]
        if same_type:
            if trick.type in _LEVEL_ORDER_TYPES:
                same_type.sort(key=lambda m: level_order_key(m.key, self.level_rank))
            else:
                same_type.sort(key=lambda m: m.key)
            candidates.append(same_type[0])  # cheapest
            if len(same_type) >= 5:
                candidates.append(same_type[len(same_type) // 2])  # median
            if len(same_type) >= 2:
                candidates.append(same_type[-1])  # strongest

        # 4. Bombs: weakest (and strongest if 3+)
        bombs = [m for m in legal if m.type in BOMB_TYPES]
        if bombs:
            bombs.sort(key=lambda m: (m.type, m.key))
            candidates.append(bombs[0])
            if len(bombs) >= 3:
                candidates.append(bombs[-1])

        return _deduplicate(candidates)

    # ─── Simulation ───────────────────────────────────────────────────────

    def _evaluate_move(self, env, player: int, move) -> float:
        """Simulate n_sims rollouts for `move` and return average reward."""
        args = [(env, player, move, self.level_rank)] * self.n_sims

        if self.n_workers <= 1:
            results = [_rollout_worker(*a) for a in args]
        else:
            with Pool(processes=self.n_workers) as pool:
                results = pool.starmap(_rollout_worker, args)

        return sum(results) / len(results)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _deduplicate(candidates: list) -> list:
    """Remove duplicate combos (same type + key + card set)."""
    seen: set = set()
    result = []
    for m in candidates:
        card_ids = frozenset((c.rank, c.suit, c.deck) for c in m.cards)
        key = (m.type, m.key, card_ids)
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result
