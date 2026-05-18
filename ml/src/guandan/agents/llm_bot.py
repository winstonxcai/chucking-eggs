"""LLMBot — tool-augmented LLM agent.

The model is shown the rules of Guan Dan, the current observable game state,
and the full enumerated list of legal actions. It returns an index into that
list. Provider-agnostic via LiteLLM, so any of OpenAI / Anthropic / Gemini /
local models can be plugged in by changing the ``model`` name.

The rules prefix is sent as a separate system message and marked with
``cache_control`` so providers that support prompt caching (Anthropic) avoid
re-billing the ~1k-token prefix on every decision.

Install:  uv pip install -e ".[llm]"
Env:      OPENAI_API_KEY / ANTHROPIC_API_KEY / etc.

Example:
    from guandan.agents import make_agent
    bot = make_agent("llm", model="claude-haiku-4-5")
    bot.act(env, player)
"""

from __future__ import annotations

import os
import random
import re
from typing import TYPE_CHECKING

from ..cards import Card, ComboType, Rank, Suit
from ..combos import Combo
from .base import Agent

if TYPE_CHECKING:
    from ..game import GuanDanEnv


_SUIT_GLYPH = {
    Suit.SPADE: "♠",
    Suit.HEART: "♥",
    Suit.DIAMOND: "♦",
    Suit.CLUB: "♣",
}

_RANK_STR = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A",
}

_COMBO_NAME = {
    ComboType.PASS: "pass",
    ComboType.SINGLE: "single",
    ComboType.PAIR: "pair",
    ComboType.TRIPLE: "triple",
    ComboType.FULL_HOUSE: "full-house",
    ComboType.STRAIGHT: "straight",
    ComboType.TUBE: "tube",
    ComboType.PLATE: "plate",
    ComboType.BOMB_4: "4-bomb",
    ComboType.BOMB_5: "5-bomb",
    ComboType.STRAIGHT_FLUSH: "straight-flush",
    ComboType.BOMB_6: "6-bomb",
    ComboType.BOMB_7: "7-bomb",
    ComboType.BOMB_8: "8-bomb",
    ComboType.BOMB_9: "9-bomb",
    ComboType.BOMB_10: "10-bomb",
    ComboType.BOMB_JOKER: "joker-bomb",
}

RULES_PREFIX = """You are playing Guan Dan (掼蛋), a 4-player Chinese trick-taking partnership game.
Your goal: help your team go out (empty both partners' hands) before the opposing team.

GAME SETUP
- 108-card deck: two standard 52-card decks plus 4 jokers. 27 cards per player.
- 4 seats (0-3) arranged counterclockwise. Partners sit across: (0,2) and (1,3).
- Level card for this game is 2. Heart-2 is the wildcard (both copies of Heart-2 from
  the two decks are wild).

CARD RANKING (low → high)
3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2(level/wild) < Black Joker < Red Joker
The level card (2) ranks above ace; otherwise natural order.

ORDINARY COMBOS (must match type to beat, higher key rank wins within the type)
- single, pair, triple, full-house (triple + pair, 5 cards)
- straight (5 consecutive ranks, mixed suits)
- tube (3 consecutive pairs, 6 cards)
- plate (2 consecutive triples, 6 cards)
Straights, tubes, and plates use natural rank order (A high) but A-2-3-4-5 (A low) is
also a valid straight. The 2 stays in its natural position for sequence purposes.

BOMBS (beat any ordinary combo; bomb-vs-bomb compared by tier)
4-bomb  <  5-bomb  <  straight-flush  <  6-bomb  <  7-bomb  <  8-bomb  <  9-bomb  <  10-bomb  <  joker-bomb
Within the same tier, higher key rank wins. The joker-bomb (both Black + both Red Jokers)
is unique and unbeatable.

WILDCARDS (Heart-2)
- A wild can substitute for any non-joker rank inside any combo, including bombs.
- A wild cannot be a joker, so it cannot be part of the joker-bomb.
- A combo that uses a wild ("wild combo") loses ties to the same combo formed without
  wilds — e.g. a natural pair of Aces beats a wild pair of Aces of equal rank.
- Singles are always a literal single card (no wild substitution).

TRICK FLOW
- One player leads any legal combo. Play proceeds counterclockwise.
- On your turn you must either (a) play a strictly stronger combo of the SAME ordinary
  type, (b) play any bomb (which also beats stronger bombs of the same tier with higher
  rank), or (c) pass.
- Three consecutive passes end the trick. The last player who played leads the next trick.
- A player who has emptied their hand is out and is automatically skipped (counts as a pass).

WINNING
- The round ends when 3 players are out. The team with both partners finishing earliest
  wins. A 1st-and-2nd finish for one team is a bigger win than a 1st-and-3rd.

YOUR INFORMATION
You see your own hand, your partner's card count, both opponents' card counts, the recent
play log, and the enumerated list of LEGAL ACTIONS available to you. Choose the action you
believe maximizes your TEAM's expected probability of winning the round.

OUTPUT FORMAT
Reply with only the chosen action's integer index. Just the number — no explanation, no
prose, no punctuation. Example: 4

WORKED EXAMPLES

Example A — your partner is currently winning the trick.

  GAME STATE
  - Your hand (8 cards): 4♠ 6♦ 7♣ 9♥ J♠ J♥ Q♦ K♠
  - Partner cards: 5  |  Opponents: 9 and 11
  CURRENT TRICK
  current winning combo: K♦ K♣  (pair, key K)
  → currently winning: partner (seat 2)
  LEGAL ACTIONS
  [0] pass
  [1] J♠ J♥  (pair, key J — would be overruled, not legal)
  [2] 4♠ 4♥+2♥*  (4-bomb)
  Correct answer: 0
  Rationale: partner is winning; don't waste a bomb or cards to overcall your own team.
  Pass and conserve hand strength.

Example B — free lead with a short hand near the end of the round.

  GAME STATE
  - Your hand (4 cards): 5♦ 9♣ J♠ J♥
  - Partner cards: 3  |  Opponents: 7 and 8
  CURRENT TRICK
  free lead (no current trick to beat)
  LEGAL ACTIONS
  [0] 5♦
  [1] 9♣
  [2] J♠
  [3] J♥
  [4] J♠ J♥  (pair, key J)
  Correct answer: 4
  Rationale: leading the pair empties two cards at once and forces opponents to spend
  two cards to overcall, accelerating your team toward going out.
"""


def _card_str(card: Card, level_rank: int) -> str:
    """Compact card display, e.g. '7♠', 'K♥', '2♥*' (wild), 'BJ', 'RJ'."""
    if card.rank == Rank.BLACK_JOKER:
        return "BJ"
    if card.rank == Rank.RED_JOKER:
        return "RJ"
    rank_s = _RANK_STR[Rank(card.rank)]
    suit_s = _SUIT_GLYPH[Suit(card.suit)]
    wild_mark = "*" if (card.rank == level_rank and card.suit == Suit.HEART) else ""
    return f"{rank_s}{suit_s}{wild_mark}"


def _format_hand(hand: set[Card], level_rank: int) -> str:
    """Sort hand by rank then suit and render as a space-separated string."""
    cards = sorted(hand, key=lambda c: (c.rank, c.suit, c.deck))
    return " ".join(_card_str(c, level_rank) for c in cards)


def _format_combo(combo: Combo, level_rank: int) -> str:
    """Render a combo as 'cards (combo-name, key K[, wild])'."""
    if combo.type == ComboType.PASS:
        return "pass"
    cards_s = " ".join(_card_str(c, level_rank) for c in combo.cards)
    name = _COMBO_NAME[combo.type]
    key_s = _RANK_STR.get(Rank(combo.key), str(combo.key)) if combo.key in _RANK_STR else str(combo.key)
    wild_note = f", uses {combo.wild_count} wild" if combo.wild_count else ""
    return f"{cards_s}  ({name}, key {key_s}{wild_note})"


def _played_histogram(env: GuanDanEnv) -> str:
    """Per-rank count of cards already played this round, compressed."""
    counts: dict[int, int] = {}
    for seat_played in env.played:
        for c in seat_played:
            counts[c.rank] = counts.get(c.rank, 0) + 1
    ranks_in_order = list(range(Rank.THREE, Rank.ACE + 1)) + [Rank.TWO, Rank.BLACK_JOKER, Rank.RED_JOKER]
    parts = []
    for r in ranks_in_order:
        n = counts.get(r, 0)
        if n == 0:
            continue
        label = _RANK_STR.get(Rank(r)) if r in _RANK_STR else ("BJ" if r == Rank.BLACK_JOKER else "RJ")
        parts.append(f"{label}×{n}")
    return ", ".join(parts) if parts else "(none)"


def _format_recent_plays(env: GuanDanEnv, player: int, max_plays: int = 8) -> str:
    """Most recent move history, oldest first, as `seat S: <combo>` lines."""
    if not env.move_history:
        return "(none — you are leading the opening trick)"
    recent = env.move_history[-max_plays:]
    lines = []
    for seat, combo in recent:
        tag = " (partner)" if seat == ((player + 2) % 4) else (" (you)" if seat == player else "")
        lines.append(f"  seat {seat}{tag}: {_format_combo(combo, env.level_rank)}")
    return "\n".join(lines)


def _format_trick(env: GuanDanEnv, player: int) -> str:
    if env.current_trick is None:
        return "free lead (no current trick to beat)"
    winner = env.trick_winner
    winner_tag = "you" if winner == player else (
        "partner" if winner == ((player + 2) % 4) else f"opponent seat {winner}"
    )
    return (
        f"current winning combo: {_format_combo(env.current_trick, env.level_rank)}\n"
        f"  → currently winning the trick: {winner_tag} (seat {winner})\n"
        f"  → you must play a stronger combo of the same type, a bomb, or pass."
    )


def _build_user_message(env: GuanDanEnv, player: int, legal: list[Combo]) -> str:
    partner = ((player + 2) % 4)
    opp1 = (player + 1) % 4
    opp2 = (player + 3) % 4
    hand_s = _format_hand(env.hands[player], env.level_rank)

    action_lines = []
    for i, combo in enumerate(legal):
        action_lines.append(f"  [{i}] {_format_combo(combo, env.level_rank)}")
    actions_s = "\n".join(action_lines)

    return f"""GAME STATE
- Wild card: Heart-{_RANK_STR[Rank(env.level_rank)]} (* marks wild cards in your hand)
- Your seat: {player}  (partner is seat {partner})
- Your hand ({len(env.hands[player])} cards): {hand_s}
- Partner card count: {len(env.hands[partner])}
- Opponent seat {opp1} card count: {len(env.hands[opp1])}
- Opponent seat {opp2} card count: {len(env.hands[opp2])}

CURRENT TRICK
{_format_trick(env, player)}

RECENT PLAYS (oldest → newest)
{_format_recent_plays(env, player)}

CARDS ALREADY PLAYED THIS ROUND
{_played_histogram(env)}

LEGAL ACTIONS
{actions_s}

Reply with only the chosen action's integer index."""


_INDEX_RE = re.compile(r"-?\d+")


def _parse_index(text: str, n_actions: int) -> int | None:
    """Extract the first integer in text; return None if invalid / out of range."""
    if text is None:
        return None
    m = _INDEX_RE.search(text)
    if m is None:
        return None
    try:
        idx = int(m.group(0))
    except ValueError:
        return None
    if 0 <= idx < n_actions:
        return idx
    return None


class LLMBot(Agent):
    """Tool-augmented LLM agent. Selects from the enumerated legal action set.

    Args:
        model: LiteLLM model name (e.g. "gpt-4o-mini", "claude-haiku-4-5",
            "gemini/gemini-2.0-flash"). Required.
        level_rank: Round level rank — controls wild card identification. Default
            Rank.TWO to match the GuanDanBench fixed-level-2 setting.
        temperature: Sampling temperature. Default 0 for reproducibility.
        max_tokens: Generation cap. Default 10 (we expect just an index).
        fallback: How to choose when the LLM returns an unparseable / out-of-range
            answer. "random" (default) or "first".
        dedup: If True, collapse suit-permutation variants of strategically
            equivalent plays so the action list stays compact (opening leads
            can have 200+ raw legal moves). Default True.
        log_failures: If True, prints a one-line warning each time we fall back.
    """

    label = "LLM"
    description = "Tool-augmented LLM that picks from the enumerated legal action set."
    source = "In-house"
    color = "#6b6ecf"

    def __init__(
        self,
        model: str,
        level_rank: int = Rank.TWO,
        temperature: float = 0.0,
        max_tokens: int = 10,
        fallback: str = "random",
        dedup: bool = True,
        log_failures: bool = False,
    ):
        self.model = model
        self.level_rank = level_rank
        self.temperature = temperature
        self.max_tokens = max_tokens
        if fallback not in {"random", "first"}:
            raise ValueError(f"fallback must be 'random' or 'first', got {fallback!r}")
        self.fallback = fallback
        self.dedup = dedup
        self.log_failures = log_failures
        self._litellm = None  # lazy import
        self._dedup_fn = None  # lazy import

    def _get_litellm(self):
        if self._litellm is None:
            import litellm  # heavy import; defer to first .act()
            litellm.drop_params = True  # silently drop unsupported provider kwargs
            self._litellm = litellm
        return self._litellm

    def _get_dedup(self):
        if self._dedup_fn is None:
            from guandan.dart.utils.legal_utils import dedup_strategic
            self._dedup_fn = dedup_strategic
        return self._dedup_fn

    def act(self, env: GuanDanEnv, player: int) -> Combo:
        legal = env.legal_moves(player)
        if self.dedup:
            legal = self._get_dedup()(legal)
        if len(legal) == 1:
            return legal[0]

        user_msg = _build_user_message(env, player, legal)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": RULES_PREFIX,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self._get_litellm().completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response["choices"][0]["message"]["content"]
        except Exception as e:
            if self.log_failures:
                print(f"[LLMBot] API error ({type(e).__name__}: {e}); falling back.")
            return self._fallback_choice(legal)

        idx = _parse_index(text, len(legal))
        if idx is None:
            if self.log_failures:
                print(f"[LLMBot] could not parse index from {text!r}; falling back.")
            return self._fallback_choice(legal)
        return legal[idx]

    def _fallback_choice(self, legal: list[Combo]) -> Combo:
        if self.fallback == "first":
            return legal[0]
        return random.choice(legal)
