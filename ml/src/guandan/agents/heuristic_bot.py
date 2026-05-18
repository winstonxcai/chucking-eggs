"""HeuristicBot — rule-based Guan Dan agent with hand decomposition.

Implements HandPlan decomposition, leading/following logic, and
a HeuristicBot class (Agent interface) plus standalone functions.
"""

from __future__ import annotations

from ..cards import BOMB_TYPES, ComboType, Rank, is_wild, level_order_key
from ..combos import Combo, generate_all_leads, generate_responses
from .base import Agent


class HandPlan:
    """Decompose a hand into natural groups for decision-making."""

    def __init__(self, hand: set, level_rank: int):
        self.hand = hand
        self.level_rank = level_rank
        self.wilds = [c for c in hand if is_wild(c, level_rank)]
        self.naturals = [c for c in hand if not is_wild(c, level_rank)]

        # Group naturals by rank
        self.by_rank: dict[int, list] = {}
        for c in self.naturals:
            self.by_rank.setdefault(c.rank, []).append(c)

        # Classify groups (sorted by level order — weakest first)
        self.singles = []
        self.pairs = []
        self.triples = []
        self.quads = []

        for rank, cards in sorted(
            self.by_rank.items(),
            key=lambda x: level_order_key(x[0], level_rank),
        ):
            n = len(cards)
            if n == 1:
                self.singles.append((rank, cards))
            elif n == 2:
                self.pairs.append((rank, cards))
            elif n == 3:
                self.triples.append((rank, cards))
            else:
                self.quads.append((rank, cards))

        # Jokers tracked separately
        self.jokers = [
            c for c in hand if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)
        ]
        self.quad_ranks = {r for r, _ in self.quads}
        self.n_bombs = len(self.quads) + (1 if len(self.jokers) == 4 else 0)


def _find_combo(combos: list[Combo], combo_type: ComboType, key_rank: int) -> Combo | None:
    """Find a combo matching type and key rank."""
    for c in combos:
        if c.type == combo_type and c.key == key_rank:
            return c
    return None


def _pass_combo(responses: list[Combo]) -> Combo:
    """Find the PASS action in responses."""
    for c in responses:
        if c.type == ComboType.PASS:
            return c
    return responses[-1]


def _breaks_bomb(combo: Combo, plan: HandPlan) -> bool:
    """Would playing this combo break up a quad (potential bomb)?"""
    for card in combo.cards:
        if card.rank in plan.quad_ranks:
            return True
    return False


def heuristic_lead(hand: set, level_rank: int) -> Combo:
    """Choose a combo to lead with (free lead)."""
    plan = HandPlan(hand, level_rank)
    all_leads = generate_all_leads(hand, level_rank)

    # Endgame: can we go out in one play?
    for combo in all_leads:
        if len(combo.cards) == len(hand):
            return combo

    # Shed weakest singles first
    if plan.singles:
        rank = plan.singles[0][0]
        found = _find_combo(all_leads, ComboType.SINGLE, rank)
        if found:
            return found

    # Shed weakest pairs
    if plan.pairs:
        rank = plan.pairs[0][0]
        found = _find_combo(all_leads, ComboType.PAIR, rank)
        if found:
            return found

    # Shed weakest triples — prefer full house to shed more cards
    if plan.triples:
        rank = plan.triples[0][0]
        fh = _find_combo(all_leads, ComboType.FULL_HOUSE, rank)
        if fh:
            return fh
        found = _find_combo(all_leads, ComboType.TRIPLE, rank)
        if found:
            return found

    # Multi-card combos (straights, tubes, plates) — play lowest-ranked
    for ctype in [ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE]:
        candidates = [c for c in all_leads if c.type == ctype]
        if candidates:
            return min(candidates, key=lambda c: c.key)

    # Bombs as last resort
    non_pass = [c for c in all_leads if c.type != ComboType.PASS]
    if non_pass:
        return min(non_pass, key=lambda c: (c.type, c.key))

    # Should never reach here (leader always has at least one play)
    return all_leads[0]


def heuristic_follow(
    hand: set,
    level_rank: int,
    trick: Combo,
    trick_winner: int,
    my_seat: int,
) -> Combo:
    """Choose a response to a trick in progress."""
    plan = HandPlan(hand, level_rank)
    responses = generate_responses(hand, level_rank, trick)

    partner = (my_seat + 2) % 4
    partner_is_winning = trick_winner == partner

    # If partner is winning → pass
    if partner_is_winning:
        return _pass_combo(responses)

    # Opponent is winning: find cheapest same-type beat
    same_type_beats = [
        c for c in responses if c.type == trick.type and c.type != ComboType.PASS
    ]

    if same_type_beats:
        # Sort by level order key for types that use it, else by natural key
        level_types = {
            ComboType.SINGLE, ComboType.PAIR,
            ComboType.TRIPLE, ComboType.FULL_HOUSE,
        }
        if trick.type in level_types:
            same_type_beats.sort(key=lambda c: level_order_key(c.key, level_rank))
        else:
            same_type_beats.sort(key=lambda c: c.key)

        cheapest = same_type_beats[0]

        # Don't break a bomb to follow
        if not _breaks_bomb(cheapest, plan):
            return cheapest

    # Bomb decision
    bombs = [c for c in responses if c.type in BOMB_TYPES]
    if bombs:
        bombs.sort(key=lambda c: (c.type, c.key))

        trick_is_strong = (
            trick.type in BOMB_TYPES
            or level_order_key(trick.key, level_rank) >= Rank.ACE
        )

        if trick_is_strong and len(bombs) >= 2:
            return bombs[0]  # use weakest bomb, save stronger ones

    return _pass_combo(responses)


class HeuristicBot(Agent):
    """Rule-based Guan Dan agent."""

    label = "Heuristic"
    description = "Intermediate rule-based play."
    source = "In-house"
    color = "#2ca02c"
    sample_tag = 4

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

    def act(self, env, player: int) -> Combo:
        """Given game state, return a Combo to play."""
        hand = env.hands[player]

        if env.current_trick is None:
            return heuristic_lead(hand, self.level_rank)
        else:
            return heuristic_follow(
                hand,
                self.level_rank,
                env.current_trick,
                env.trick_winner,
                player,
            )

    # Alias for tests that use choose_action
    choose_action = act


# Backward compatibility alias
HeuristicAgent = HeuristicBot
