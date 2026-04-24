"""PartnerPIMCBot — Partner-Visible PIMC Search Agent (Direction A).

The agent is given full visibility of its partner's hand at act() time.
This removes ~27/81 hidden cards from the problem, making determinizations
much closer to reality than standard PIMC (which failed in Mar 2026 with
all 81 cards hidden — see LOGBOOK §11).

Architecture:
  1. Pre-filter: rank all legal moves with rank_sum heuristic.
     Keep only the top-K candidates to bound search cost.
  2. Determinize: sample N assignments of the 2 opponents' hidden cards.
     Opponent hand sizes are public knowledge (27 minus cards played).
  3. Rollout: for each determinization, score all K candidates in parallel.
     Each rollout uses GreedyBot for all 4 seats and stops after `depth_limit`
     steps; remaining hand advantage is used as the leaf value.
  4. Score: average team reward across rollouts per candidate.
  5. Return the candidate with the highest average score.

Speed optimisations vs original:
  - GreedyBot rollout (vs HeuristicBot): ~4× cheaper per step.
  - Depth-limited rollout + leaf value: ~3× fewer steps per rollout.
  - Multiprocessing: each determinization runs in a separate worker process,
    giving ~N_WORKERS× throughput on multi-core hardware.
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..cards import BOMB_TYPES, ComboType, make_deck
from ..combos import Combo
from ..game import GuanDanEnv
from .base import Agent
from .greedy_bot import GreedyBot

# Full 108-card deck built once at module load.
_FULL_DECK: frozenset = frozenset(make_deck())

# Number of parallel worker processes.
_N_WORKERS = min(os.cpu_count() or 4, 8)

# Path to ml/src so workers can import guandan after spawn.
_SRC_PATH = str(Path(__file__).resolve().parent.parent.parent)


# ── Env helpers ────────────────────────────────────────────────────────────────

def _clone_env(env: GuanDanEnv) -> GuanDanEnv:
    """Shallow-clone a GuanDanEnv for rollout use."""
    new = object.__new__(GuanDanEnv)
    new.level_rank = env.level_rank
    new.hands = [s.copy() for s in env.hands]
    new.played = [s.copy() for s in env.played]
    new.current_player = env.current_player
    new.current_trick = env.current_trick
    new.trick_winner = env.trick_winner
    new.consecutive_passes = env.consecutive_passes
    new.finish_order = env.finish_order.copy()
    new.is_out = env.is_out.copy()
    new.done = env.done
    new.move_history = []
    return new


def _determinize(
    env: GuanDanEnv,
    player: int,
    partner: int,
    rng: random.Random,
) -> GuanDanEnv:
    """Return a cloned env with opponent hands randomly re-dealt."""
    opp1 = (player + 1) % 4
    opp2 = (player + 3) % 4
    known: set = (
        env.hands[player]
        | env.played[player]
        | env.hands[partner]
        | env.played[partner]
        | env.played[opp1]
        | env.played[opp2]
    )
    hidden = list(_FULL_DECK - known)
    rng.shuffle(hidden)
    opp1_size = len(env.hands[opp1])
    det = _clone_env(env)
    det.hands[opp1] = set(hidden[:opp1_size])
    det.hands[opp2] = set(hidden[opp1_size : opp1_size + len(env.hands[opp2])])
    return det


# ── Leaf value ─────────────────────────────────────────────────────────────────

def _leaf_value(env: GuanDanEnv, player: int) -> float:
    """Hand-advantage heuristic when rollout is cut short.

    Returns a score in [-3, 3] matching the env.get_rewards() scale:
    positive = our team is ahead (fewer cards remaining than opponents).
    """
    partner = (player + 2) % 4
    opp1 = (player + 1) % 4
    opp2 = (player + 3) % 4
    our_cards = len(env.hands[player]) + len(env.hands[partner])
    opp_cards = len(env.hands[opp1]) + len(env.hands[opp2])
    # Max advantage ≈ 54 cards. Divide by 9 → [-6, 6], clamp to [-3, 3].
    return max(-3.0, min(3.0, (opp_cards - our_cards) / 9.0))


def _rollout_limited(
    env: GuanDanEnv,
    agents: list[Agent],
    depth_limit: int,
    player: int,
) -> float:
    """Run env for up to depth_limit steps, then score with leaf value."""
    steps = 0
    while not env.done and steps < depth_limit:
        p = env.current_player
        env.step(agents[p].act(env, p))
        steps += 1
    if env.done:
        return env.get_rewards()[player]
    return _leaf_value(env, player)


# ── Multiprocessing worker ─────────────────────────────────────────────────────

def _init_worker(src_path: str) -> None:
    """Initialise worker process: add ml/src to sys.path."""
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _pimc_worker(
    args: tuple,
) -> list[float]:
    """Score all candidates for one determinisation.

    Must be a module-level function so pickle can locate it across spawn.
    """
    det_env, candidates, player, depth_limit, level_rank = args
    # GreedyBot instances are cheap to create; build once per worker call.
    rollout_agents = [GreedyBot(level_rank) for _ in range(4)]
    scores = []
    for move in candidates:
        sim = _clone_env(det_env)
        sim.step(move)
        scores.append(_rollout_limited(sim, rollout_agents, depth_limit, player))
    return scores


# ── Agent ──────────────────────────────────────────────────────────────────────

class PartnerPIMCBot(Agent):
    """Perfect-partner-info PIMC search agent.

    Expects that env.hands[partner_seat] is the true partner hand at act()
    time (i.e., the caller has set up a full-visibility environment).

    Args:
        level_rank: current level rank (passed to sub-agents).
        n_det: number of determinizations per candidate evaluation.
        n_cands: maximum candidates to consider (top-K pre-filter).
        depth_limit: max rollout steps before leaf evaluation (0 = unlimited).
        n_workers: worker processes for parallelism (0 = single-process).
        seed: optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        level_rank: int = 2,
        n_det: int = 20,
        n_cands: int = 10,
        depth_limit: int = 30,
        n_workers: int = _N_WORKERS,
        seed: int | None = None,
    ):
        self.level_rank = level_rank
        self.n_det = n_det
        self.n_cands = n_cands
        self.depth_limit = depth_limit
        self.n_workers = n_workers
        self._rng = random.Random(seed)
        self._pool: ProcessPoolExecutor | None = None
        if n_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(_SRC_PATH,),
            )

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        partner = (player + 2) % 4
        candidates = env.legal_moves(player)

        if len(candidates) == 1:
            return candidates[0]

        candidates = self._prefilter(env, player, candidates)
        if len(candidates) == 1:
            return candidates[0]

        # Build determinizations
        dets = [
            _determinize(env, player, partner, self._rng)
            for _ in range(self.n_det)
        ]

        scores = [0.0] * len(candidates)

        if self._pool is not None:
            # Parallel: one task per determinization.
            worker_args = [
                (det, candidates, player, self.depth_limit, self.level_rank)
                for det in dets
            ]
            for det_scores in self._pool.map(_pimc_worker, worker_args):
                for i, s in enumerate(det_scores):
                    scores[i] += s
        else:
            # Single-process fallback.
            rollout_agents = [GreedyBot(self.level_rank) for _ in range(4)]
            for det in dets:
                for i, move in enumerate(candidates):
                    sim = _clone_env(det)
                    sim.step(move)
                    scores[i] += _rollout_limited(
                        sim, rollout_agents, self.depth_limit, player
                    )

        return candidates[max(range(len(candidates)), key=lambda i: scores[i])]

    def __del__(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)

    # ── Pre-filter ──────────────────────────────────────────────────────────────

    def _prefilter(
        self, env: GuanDanEnv, player: int, candidates: list[Combo]
    ) -> list[Combo]:
        """Rank candidates by cheap heuristic and return the top-K."""
        if len(candidates) <= self.n_cands:
            return candidates

        bombs = [c for c in candidates if c.type in BOMB_TYPES]
        passes = [c for c in candidates if c.type == ComboType.PASS]
        rest = [c for c in candidates if c.type not in BOMB_TYPES
                and c.type != ComboType.PASS]

        rest.sort(key=lambda combo: sum(c.rank for c in combo.cards))
        budget = max(1, self.n_cands - len(bombs) - len(passes))
        kept = passes + rest[:budget] + bombs

        seen: set = set()
        result = []
        for c in kept:
            key = (c.type, tuple(sorted((x.rank, x.suit, x.deck) for x in c.cards)))
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result
