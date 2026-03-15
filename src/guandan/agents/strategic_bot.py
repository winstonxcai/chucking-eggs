"""StrategicBot — strong amateur-level agent.

Adds hand planning, opponent awareness, and active partnership play
on top of the HeuristicBot foundation.
"""

from __future__ import annotations

from ..cards import BOMB_TYPES, ComboType, Rank, is_wild, level_order_key
from ..combos import generate_all_leads, generate_responses
from .heuristic_bot import HandPlan, _breaks_bomb, _find_combo, _pass_combo
from .base import Agent

# Types where level_order_key applies for comparison.
_LEVEL_ORDER_TYPES = frozenset({
    ComboType.SINGLE, ComboType.PAIR,
    ComboType.TRIPLE, ComboType.FULL_HOUSE,
})


def estimate_moves_to_empty(hand: set, level_rank: int) -> int:
    """Estimate the minimum number of plays to empty this hand.

    Lower = better hand. This is a greedy approximation, not optimal.
    Approach: count natural groups, adjust for wilds and jokers.
    """
    if len(hand) == 0:
        return 0

    wilds = [c for c in hand if is_wild(c, level_rank)]
    naturals = [c for c in hand if not is_wild(c, level_rank)]

    by_rank: dict[int, list] = {}
    for c in naturals:
        if c.rank not in (Rank.BLACK_JOKER, Rank.RED_JOKER):
            by_rank.setdefault(c.rank, []).append(c)

    # Count each rank group as one play
    remaining_groups = 0
    for rank, cards in by_rank.items():
        n = len(cards)
        if n >= 4:
            remaining_groups += 1  # bomb is one play
        elif n == 3:
            remaining_groups += 1  # triple (or full house)
        elif n == 2:
            remaining_groups += 1  # pair
        elif n == 1:
            remaining_groups += 1  # single — the worst

    # Jokers
    jokers = [c for c in hand if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)]
    if len(jokers) == 4:
        remaining_groups += 1  # joker bomb
    elif len(jokers) >= 2:
        remaining_groups += (len(jokers) + 1) // 2  # pair up jokers
    else:
        remaining_groups += len(jokers)  # each joker is a single

    # Wilds reduce move count (they fill gaps)
    remaining_groups = max(1, remaining_groups - len(wilds))

    return remaining_groups


class StrategicBot(Agent):
    """Strong amateur agent with hand planning and opponent awareness."""

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

    def act(self, env, player: int):
        legal = env.legal_moves(player)

        if env.current_trick is None:
            return self._strategic_lead(env, player, legal)
        else:
            return self._strategic_follow(env, player, legal)

    # ------------------------------------------------------------------
    # Leading
    # ------------------------------------------------------------------

    def _strategic_lead(self, env, player: int, legal: list):
        """Context-aware leading."""
        hand = env.hands[player]
        partner = (player + 2) % 4
        plan = HandPlan(hand, self.level_rank)

        # --- Endgame: go out if possible ---
        for combo in legal:
            if len(combo.cards) == len(hand):
                return combo

        # --- Partner is out → I need to finish fast ---
        if env.is_out[partner]:
            return self._aggressive_lead(legal, plan)

        # --- I'm close to winning (≤ 5 cards) → play aggressively ---
        if len(hand) <= 5:
            return self._aggressive_lead(legal, plan)

        # --- Partner is close to winning → lead combos that help them ---
        partner_cards = len(env.hands[partner])
        if partner_cards <= 5 and partner_cards > 0 and not env.is_out[partner]:
            return self._help_partner_lead(legal, plan)

        # --- Standard play: shed efficiently ---
        return self._efficient_lead(legal, plan)

    def _aggressive_lead(self, legal: list, plan: HandPlan):
        """Play strongest combos to finish fast."""
        non_pass = [m for m in legal if m.type != ComboType.PASS]
        # Prefer multi-card combos (shed more cards per play)
        non_pass.sort(key=lambda m: (-len(m.cards), m.type))
        return non_pass[0]

    def _help_partner_lead(self, legal: list, plan: HandPlan):
        """Lead combos that partner can likely follow or win."""
        # Lead small singles or pairs — partner with few cards
        # probably has some singles/pairs left
        if plan.singles:
            rank = plan.singles[0][0]
            combo = _find_combo(legal, ComboType.SINGLE, rank)
            if combo:
                return combo

        if plan.pairs:
            rank = plan.pairs[0][0]
            combo = _find_combo(legal, ComboType.PAIR, rank)
            if combo:
                return combo

        # Fallback to efficient lead
        return self._efficient_lead(legal, plan)

    def _efficient_lead(self, legal: list, plan: HandPlan):
        """Shed cards that reduce move count the most."""
        # Singles first (clears isolated cards)
        if plan.singles:
            rank = plan.singles[0][0]
            combo = _find_combo(legal, ComboType.SINGLE, rank)
            if combo:
                return combo

        # Straights/tubes/plates (shed 5-6 cards without breaking groups)
        for ctype in [ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE]:
            candidates = [c for c in legal if c.type == ctype]
            if candidates:
                return min(candidates, key=lambda c: c.key)

        # Pairs
        if plan.pairs:
            rank = plan.pairs[0][0]
            combo = _find_combo(legal, ComboType.PAIR, rank)
            if combo:
                return combo

        # Full houses (shed 5 cards)
        full_houses = [c for c in legal if c.type == ComboType.FULL_HOUSE]
        if full_houses:
            return min(full_houses,
                       key=lambda c: level_order_key(c.key, self.level_rank))

        # Triples
        if plan.triples:
            rank = plan.triples[0][0]
            combo = _find_combo(legal, ComboType.TRIPLE, rank)
            if combo:
                return combo

        # Last resort: anything non-bomb
        non_bomb = [m for m in legal
                    if m.type not in BOMB_TYPES and m.type != ComboType.PASS]
        if non_bomb:
            return min(non_bomb, key=lambda m: (
                len(m.cards),
                level_order_key(m.key, self.level_rank),
            ))

        # Only bombs left
        non_pass = [m for m in legal if m.type != ComboType.PASS]
        if non_pass:
            return min(non_pass, key=lambda m: (m.type, m.key))

        return legal[0]

    # ------------------------------------------------------------------
    # Following
    # ------------------------------------------------------------------

    def _strategic_follow(self, env, player: int, legal: list):
        """Context-aware following with opponent awareness."""
        hand = env.hands[player]
        partner = (player + 2) % 4
        plan = HandPlan(hand, self.level_rank)
        trick = env.current_trick
        winner = env.trick_winner

        partner_winning = (winner == partner)

        # --- Partner winning → pass (usually) ---
        if partner_winning:
            # EXCEPTION: if I can go out by playing, do it
            beats = [m for m in legal
                     if m.type != ComboType.PASS and len(m.cards) == len(hand)]
            if beats:
                return beats[0]  # go out!
            return _pass_combo(legal)

        # --- Opponent winning ---
        same_type = [m for m in legal
                     if m.type == trick.type and m.type != ComboType.PASS]
        bombs = [m for m in legal if m.type in BOMB_TYPES]

        # Try cheapest same-type beat (don't break bombs)
        if same_type:
            safe_beats = [m for m in same_type if not _breaks_bomb(m, plan)]
            if safe_beats:
                if trick.type in _LEVEL_ORDER_TYPES:
                    safe_beats.sort(
                        key=lambda m: level_order_key(m.key, self.level_rank))
                else:
                    safe_beats.sort(key=lambda m: m.key)
                return safe_beats[0]

        # --- Bomb decision ---
        if bombs:
            bombs.sort(key=lambda m: (m.type, m.key))

            opp1 = (player + 1) % 4
            opp2 = (player + 3) % 4
            opp_min_cards = min(
                len(env.hands[opp1]) if not env.is_out[opp1] else 99,
                len(env.hands[opp2]) if not env.is_out[opp2] else 99,
            )

            partner_close = (
                len(env.hands[partner]) <= 5
                and not env.is_out[partner]
            )

            should_bomb = (
                opp_min_cards <= 5             # opponent about to win
                or partner_close               # partner needs the lead
                or trick.type in BOMB_TYPES    # they bombed, bomb back
                or len(hand) <= 4              # I'm about to win
            )

            if should_bomb:
                return bombs[0]  # weakest bomb

        return _pass_combo(legal)
