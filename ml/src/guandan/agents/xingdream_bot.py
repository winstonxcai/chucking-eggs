"""XingDreamBot — heuristic agent ported from xingdream/guandan (mj branch).

Key differences from HeuristicBot/StrategicBot:
- Leads straights first, then multi-card combos, then small combos
- Leads singles early if holding Red Joker (flush out opponent cards)
- Overtakes teammate's low singles/pairs (<J) instead of always passing
- Dynamic bomb-spending threshold: more bombs or fewer cards → more willing to bomb
- Never breaks combos to follow
"""

from __future__ import annotations

from ..cards import BOMB_TYPES, ComboType, Rank, level_order_key
from .base import Agent


class XingDreamBot(Agent):
    """Heuristic bot based on xingdream/guandan strategy."""

    label = "Xingdream"
    description = "8th Place · 2020 NJUPT entry."
    source = "2020 NJUPT"
    award = "8th Place"

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

    def act(self, env, player: int):
        legal = env.legal_moves(player)
        if len(legal) == 1:
            return legal[0]

        if env.current_trick is None:
            return self._lead(env, player, legal)
        else:
            return self._follow(env, player, legal)

    # ── Leading ──────────────────────────────────────────

    def _lead(self, env, player, legal):
        by_type = self._group_by_type(legal)
        has_rj = any(c.rank == Rank.RED_JOKER for c in env.hands[player])

        # Priority order: Straight → Single(if RJ) → Tube → Plate
        #   → FullHouse(if pair<J) → Triple → Pair → Single → Bomb
        pick = self._lowest(by_type, ComboType.STRAIGHT)
        if pick:
            return pick

        # If we have Red Joker, lead a single to flush out opponent cards
        if has_rj:
            pick = self._lowest(by_type, ComboType.SINGLE)
            if pick:
                return pick

        pick = self._lowest(by_type, ComboType.TUBE)
        if pick:
            return pick

        pick = self._lowest(by_type, ComboType.PLATE)
        if pick:
            return pick

        # Full house — only if the pair component is low (<J)
        fh_moves = by_type.get(ComboType.FULL_HOUSE, [])
        if fh_moves:
            fh_moves.sort(key=lambda c: level_order_key(c.key, self.level_rank))
            best_fh = fh_moves[0]
            if level_order_key(best_fh.key, self.level_rank) < Rank.JACK:
                return best_fh

        pick = self._lowest(by_type, ComboType.TRIPLE)
        if pick:
            return pick

        pick = self._lowest(by_type, ComboType.PAIR)
        if pick:
            return pick

        pick = self._lowest(by_type, ComboType.SINGLE)
        if pick:
            return pick

        # Bombs as last resort
        bomb_moves = [m for m in legal if m.type in BOMB_TYPES]
        if bomb_moves:
            bomb_moves.sort(key=lambda c: (c.type, level_order_key(c.key, self.level_rank)))
            return bomb_moves[0]

        # Fallback (shouldn't reach here)
        return legal[0]

    # ── Following ────────────────────────────────────────

    def _follow(self, env, player, legal):
        trick = env.current_trick
        trick_winner = env.trick_winner

        # Determine relationship: teammate (distance=2) or opponent
        is_teammate = (abs(trick_winner - player) % 4) == 2

        # Separate pass, same-type beats, and bombs
        pass_move = next(m for m in legal if m.type == ComboType.PASS)
        same_type = [m for m in legal if m.type == trick.type and m.type not in BOMB_TYPES]
        same_type.sort(key=lambda c: level_order_key(c.key, self.level_rank))
        bombs = [m for m in legal if m.type in BOMB_TYPES]
        bombs.sort(key=lambda c: (c.type, level_order_key(c.key, self.level_rank)))

        if is_teammate:
            return self._follow_teammate(env, player, trick, same_type, pass_move)
        else:
            return self._follow_opponent(env, player, same_type, bombs, pass_move)

    def _follow_teammate(self, env, player, trick, same_type, pass_move):
        """Teammate is winning. Overtake only if their card is low (<J)."""
        trick_key = level_order_key(trick.key, self.level_rank)

        # Only consider overtaking singles and pairs
        if trick.type in (ComboType.SINGLE, ComboType.PAIR):
            if trick_key < Rank.JACK and trick.key != self.level_rank:
                if same_type:
                    return same_type[0]  # cheapest beat

        return pass_move

    def _follow_opponent(self, env, player, same_type, bombs, pass_move):
        """Opponent is winning. Match type or bomb based on threshold."""
        # Try same-type beat first
        if same_type:
            return same_type[0]  # cheapest beat

        # Bomb decision based on xingdream heuristic
        rest_me = len(env.hands[player])
        n_bombs = len(bombs)

        should_bomb = (
            n_bombs > 3
            or (n_bombs > 2 and rest_me < 20)
            or (n_bombs > 1 and rest_me < 15)
            or (n_bombs > 0 and rest_me < 10)
        )

        if should_bomb and bombs:
            return bombs[0]  # weakest bomb

        return pass_move

    # ── Helpers ──────────────────────────────────────────

    def _group_by_type(self, legal):
        """Group legal moves by ComboType, excluding PASS and bombs."""
        groups = {}
        for m in legal:
            if m.type == ComboType.PASS or m.type in BOMB_TYPES:
                continue
            groups.setdefault(m.type, []).append(m)
        return groups

    def _lowest(self, by_type, combo_type):
        """Return lowest-rank combo of given type, or None."""
        moves = by_type.get(combo_type)
        if not moves:
            return None
        moves.sort(key=lambda c: level_order_key(c.key, self.level_rank))
        return moves[0]
