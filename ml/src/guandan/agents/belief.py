"""BeliefModel — opponent-hand inference for partner-visible PIMC search.

Layers (per Direction D plan):

  Layer 1 — Card counting: per-rank counts of unaccounted-for cards
            (used by determinization for hypergeometric per-rank sampling).

  Layer 2 — Single-trick hard constraints derived from pass events:
            H1  passed on single of key R     → no single  > R (level order)
            H2  passed on pair of key R       → no pair    > R
            H3  passed on triple of key R     → no triple  > R
            (H5 no_bomb inference removed — players routinely sandbag bombs)

  (H4 — full-house/straight/etc same-type constraints — skipped in Phase 1a.
   Layer 3 soft signals — Phase 1b only.)

Constraints are accumulated by replaying `env.move_history`. Each opponent's
constraints are tightened over time (a later, harder pass overrides an earlier,
softer one). Constraints persist: a card excluded after a pass is excluded
forever (hands only shrink).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..cards import BOMB_TYPES, Card, ComboType, level_order_key
from ..combos import Combo
from ..game import GuanDanEnv


@dataclass
class OppConstraints:
    """Per-opponent hard constraints. Defaults = no constraint."""
    # H1-H3: Largest level-order-key the opp can hold for each single/pair/triple.
    # None means unconstrained.
    max_single_key: int | None = None
    max_pair_key: int | None = None
    max_triple_key: int | None = None
    no_bomb: bool = False  # True ⇒ opp holds no rank with count ≥ 4
    # H4: Largest natural-rank key for same-type combos (STRAIGHT, TUBE, PLATE).
    # FULL_HOUSE is keyed by level-order-key of the triple's rank.
    max_straight_key: int | None = None    # natural rank of highest card in straight
    max_tube_key: int | None = None        # natural rank of highest pair in tube (3 consec pairs)
    max_plate_key: int | None = None       # natural rank of highest triple in plate (2 consec triples)
    max_fullhouse_lo_key: int | None = None  # level-order-key of triple's rank in full house

    def tighten_max(self, attr: str, k: int) -> None:
        cur = getattr(self, attr)
        if cur is None or k < cur:
            setattr(self, attr, k)

    def violates(self, hand: set[Card], level_rank: int) -> bool:
        """Does this candidate hand violate any of the accumulated constraints?"""
        if not hand:
            return False

        # Build per-rank counts once (excludes jokers from natural-rank sequences)
        rank_counts: Counter[int] = Counter(c.rank for c in hand)

        # H1: passed on single of key K → no card with level_order_key > K
        if self.max_single_key is not None:
            for rank in rank_counts:
                if level_order_key(rank, level_rank) > self.max_single_key:
                    return True

        # H2: passed on pair of key K → no pair with level_order_key > K
        if self.max_pair_key is not None:
            for rank, cnt in rank_counts.items():
                if cnt >= 2 and level_order_key(rank, level_rank) > self.max_pair_key:
                    return True

        # H3: passed on triple of key K → no triple with level_order_key > K
        if self.max_triple_key is not None:
            for rank, cnt in rank_counts.items():
                if cnt >= 3 and level_order_key(rank, level_rank) > self.max_triple_key:
                    return True

        if self.no_bomb:
            for cnt in rank_counts.values():
                if cnt >= 4:
                    return True
            from ..cards import Rank
            bj = rank_counts.get(Rank.BLACK_JOKER, 0)
            rj = rank_counts.get(Rank.RED_JOKER, 0)
            if bj >= 2 and rj >= 2:
                return True

        # H4: same-type combo constraints (natural rank only; wilds not counted → safe under-constraint)
        # Jokers and level-rank wilds are excluded from consecutive-rank checks intentionally:
        # over-rejection from wild-assisted combos is harder to detect and rarer than base combos.
        natural_ranks = sorted(
            r for r, cnt in rank_counts.items()
            if r not in (level_rank,) and r < 15  # exclude wilds and jokers
            for _ in range(cnt)
        )

        if self.max_straight_key is not None and len(natural_ranks) >= 5:
            # Find any 5 consecutive distinct natural ranks with max > max_straight_key.
            distinct = sorted(set(natural_ranks))
            for i in range(len(distinct) - 4):
                seq = distinct[i:i + 5]
                if seq[-1] - seq[0] == 4 and seq[-1] > self.max_straight_key:
                    return True

        if self.max_tube_key is not None:
            # Tube = 3 consecutive ranks each with count ≥ 2; key = highest rank.
            pairs = sorted(r for r, cnt in rank_counts.items() if cnt >= 2 and r < 15 and r != level_rank)
            for i in range(len(pairs) - 2):
                if pairs[i + 1] == pairs[i] + 1 and pairs[i + 2] == pairs[i] + 2:
                    if pairs[i + 2] > self.max_tube_key:
                        return True

        if self.max_plate_key is not None:
            # Plate = 2 consecutive ranks each with count ≥ 3; key = highest rank.
            triples = sorted(r for r, cnt in rank_counts.items() if cnt >= 3 and r < 15 and r != level_rank)
            for i in range(len(triples) - 1):
                if triples[i + 1] == triples[i] + 1 and triples[i + 1] > self.max_plate_key:
                    return True

        if self.max_fullhouse_lo_key is not None:
            # Full house key = level-order-key of the triple's rank.
            for rank, cnt in rank_counts.items():
                if cnt >= 3 and level_order_key(rank, level_rank) > self.max_fullhouse_lo_key:
                    return True

        return False


class BeliefModel:
    """Mines `env.move_history` for hard constraints on opponent hands."""

    def constraints(self, env: GuanDanEnv) -> dict[int, OppConstraints]:
        """Per-seat constraints accumulated by replaying move history.

        Returns a dict {seat: OppConstraints} for all 4 seats. Caller picks
        out the opponent seats relative to the acting player.
        """
        result = {p: OppConstraints() for p in range(4)}
        level_rank = env.level_rank

        # Replay history, tracking trick state and applying constraints on each pass.
        current_trick: Combo | None = None
        trick_is_bomb: bool = False
        trick_key_lo: int = 0  # level-order-key of current trick

        for player, combo in env.move_history:
            if combo.type == ComboType.PASS:
                if current_trick is None:
                    continue  # pass with no trick on table — shouldn't happen but skip
                c = result[player]
                t = current_trick.type
                k = trick_key_lo

                if t == ComboType.SINGLE:
                    c.tighten_max("max_single_key", k)
                elif t == ComboType.PAIR:
                    c.tighten_max("max_pair_key", k)
                elif t == ComboType.TRIPLE:
                    c.tighten_max("max_triple_key", k)
                # H4: same-type combo constraints from passes.
                # STRAIGHT/TUBE/PLATE use natural key (not level-order) for comparison.
                elif t == ComboType.STRAIGHT:
                    c.tighten_max("max_straight_key", current_trick.key)
                elif t == ComboType.TUBE:
                    c.tighten_max("max_tube_key", current_trick.key)
                elif t == ComboType.PLATE:
                    c.tighten_max("max_plate_key", current_trick.key)
                elif t == ComboType.FULL_HOUSE:
                    # Full house key uses level-order-key (same as triple comparison).
                    c.tighten_max("max_fullhouse_lo_key", k)

                # H5 (no_bomb inference) removed: players routinely sandbag bombs
                # and save them for critical moments, even in greedy bots. Applying
                # this constraint over-rejects valid determinizations.
            else:
                # New trick or response within the trick.
                # In Guandan, a play is either the lead (current_trick is None)
                # or a beat (so a stronger combo). We just snapshot it as the
                # new current_trick so subsequent passes can use it.
                current_trick = combo
                trick_is_bomb = combo.type in BOMB_TYPES
                trick_key_lo = level_order_key(combo.key, level_rank)

            # Heuristic for trick reset: rather than reproduce env's pass-counting
            # logic, we let the next non-pass move overwrite current_trick. This
            # is correct: a new trick lead is always a non-pass play, and any
            # "intervening" passes refer to the most recent non-pass — which is
            # exactly what current_trick holds.

        return result

    def rank_distribution(
        self, env: GuanDanEnv, player: int
    ) -> dict[int, int]:
        """Per-rank: how many of the 8 copies are still in opposition hands.

        Returns {rank: count_unknown_in_opps} for ranks where count > 0.
        Used by `_determinize` for hypergeometric per-rank sampling.

        From the acting player's perspective: opposition = the 2 non-team seats.
        """
        partner = (player + 2) % 4
        # Count what's accounted for: own hand, partner hand, all played.
        accounted: Counter[int] = Counter()
        for c in env.hands[player]:
            accounted[c.rank] += 1
        for c in env.hands[partner]:
            accounted[c.rank] += 1
        for p in range(4):
            for c in env.played[p]:
                accounted[c.rank] += 1

        # Per-rank totals in the 108-card deck:
        #   ranks 2..A    → 8 each (4 suits × 2 decks)
        #   BLACK_JOKER   → 2
        #   RED_JOKER     → 2
        result: dict[int, int] = {}
        from ..cards import Rank
        for rank in range(2, 15):  # 2 through A (14)
            unknown = 8 - accounted.get(rank, 0)
            if unknown > 0:
                result[rank] = unknown
        for rank in (Rank.BLACK_JOKER, Rank.RED_JOKER):
            unknown = 2 - accounted.get(rank, 0)
            if unknown > 0:
                result[rank] = unknown
        return result
