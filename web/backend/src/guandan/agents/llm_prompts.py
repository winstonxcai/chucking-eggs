"""LLM agent prompt templates and game state serialization for Guan Dan.

All functions use only imperfect (production-fair) information:
  ✅ env.hands[player]          — own hand
  ✅ len(env.hands[other])      — opponent/partner card counts (visible at table)
  ✅ env.played[other]          — publicly played cards
  ✅ env.move_history           — all moves are public
  ❌ env.hands[other] contents  — never accessed (would be cheating)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..cards import ComboType, Rank, Suit
from ..combos import Combo
from ._vendor.adapter import card_to_string, rank_to_string
from ._vendor.jidan.message_Reyn_CUR import get_point_val

if TYPE_CHECKING:
    from ..game import GuanDanEnv

# ── Display maps ──────────────────────────────────────────────────────────────

_RANK_DISPLAY: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
    Rank.BLACK_JOKER: "BJ", Rank.RED_JOKER: "RJ",
}

_SUIT_NAMES = {
    Suit.SPADE: "Spades", Suit.HEART: "Hearts",
    Suit.DIAMOND: "Diamonds", Suit.CLUB: "Clubs",
}

_COMBO_TYPE_DISPLAY: dict[ComboType, str] = {
    ComboType.PASS: "PASS",
    ComboType.SINGLE: "Single",
    ComboType.PAIR: "Pair",
    ComboType.TRIPLE: "Triple",
    ComboType.FULL_HOUSE: "Full House",
    ComboType.STRAIGHT: "Straight",
    ComboType.TUBE: "Tube",
    ComboType.PLATE: "Plate",
    ComboType.BOMB_4: "BOMB-quad",
    ComboType.BOMB_5: "BOMB-5oak",
    ComboType.STRAIGHT_FLUSH: "BOMB-SF",
    ComboType.BOMB_6: "BOMB-6oak",
    ComboType.BOMB_7: "BOMB-7oak",
    ComboType.BOMB_8: "BOMB-8oak",
    ComboType.BOMB_9: "BOMB-9oak",
    ComboType.BOMB_10: "BOMB-10oak",
    ComboType.BOMB_JOKER: "BOMB-joker",
}

_SUIT_LETTER = {Suit.SPADE: "S", Suit.HEART: "H", Suit.DIAMOND: "D", Suit.CLUB: "C"}


# ── Card helpers ──────────────────────────────────────────────────────────────

def _card_display(card, level_rank: int) -> str:
    """Single card → display string. Wild Heart marked with *."""
    r = _RANK_DISPLAY.get(card.rank, "?")
    if card.rank >= 16:
        return r
    s = _SUIT_LETTER[card.suit]
    if card.suit == Suit.HEART and card.rank == level_rank:
        return f"*{s}{r}"  # wild
    return f"{s}{r}"


def _combo_desc(combo: Combo, level_rank: int) -> str:
    if combo.type == ComboType.PASS:
        return "PASS"
    t = _COMBO_TYPE_DISPLAY.get(combo.type, "?")
    r = _RANK_DISPLAY.get(combo.key, "?")
    cards = " ".join(
        _card_display(c, level_rank)
        for c in sorted(combo.cards, key=lambda c: (c.rank, c.suit))
    )
    return f"{t} {r} [{cards}]"


# ── JidanBot scoring ──────────────────────────────────────────────────────────

def _jidan_score(combo: Combo, level_rank: int) -> float:
    """Sum of JidanBot point values. Higher = more valuable cards (bombs last)."""
    if combo.type == ComboType.PASS:
        return float("inf")
    lr = rank_to_string(level_rank)
    return sum(get_point_val(card_to_string(c), lr) for c in combo.cards)


# ── Cooperative flags ─────────────────────────────────────────────────────────

def compute_cooperative_flags(env: GuanDanEnv, player: int, legal: list[Combo]) -> dict:
    """Compute GuanZero-style cooperative flags using only fair (imperfect) info."""
    partner = (player + 2) % 4
    left_opp = (player + 1) % 4
    right_opp = (player + 3) % 4

    partner_remaining = len(env.hands[partner])
    left_remaining = len(env.hands[left_opp])
    right_remaining = len(env.hands[right_opp])
    min_opp_remaining = min(left_remaining, right_remaining)

    can_cooperate = (
        env.current_trick is not None
        and env.trick_winner == partner
        and any(c.type == ComboType.PASS for c in legal)
    )
    can_dwarf = any(
        len(c.cards) >= min_opp_remaining
        for c in legal if c.type != ComboType.PASS
    )
    # Only meaningful when partner is near finishing — avoid firing on every turn
    can_assist = (
        partner_remaining <= 10
        and any(
            len(c.cards) < partner_remaining
            for c in legal if c.type != ComboType.PASS
        )
    )

    return {
        "can_cooperate": can_cooperate,
        "can_dwarf": can_dwarf,
        "can_assist": can_assist,
        "partner": partner,
        "left_opp": left_opp,
        "right_opp": right_opp,
        "partner_remaining": partner_remaining,
        "left_remaining": left_remaining,
        "right_remaining": right_remaining,
        "min_opp_remaining": min_opp_remaining,
    }


# ── Move filtering ────────────────────────────────────────────────────────────

def filter_by_intent(
    legal: list[Combo],
    intent: str,
    flags: dict,
    level_rank: int,
    top_k: int = 8,
    scores: dict[int, float] | None = None,
) -> list[Combo]:
    """Filter and rank moves by cooperative intent.

    scores: optional {id(combo): q_value} from RL recommender.
    When provided, higher Q-value = better move (best K selected, sorted best-first).
    Falls back to static JidanBot scoring when scores is None or id not found.
    """
    pass_moves = [c for c in legal if c.type == ComboType.PASS]
    non_pass = [c for c in legal if c.type != ComboType.PASS]

    if intent == "cooperate":
        if pass_moves:
            return pass_moves  # stage 3 will be skipped
        intent = "normal"  # LLM said cooperate but PASS not legal — fall through

    if intent == "dwarf":
        min_opp = flags["min_opp_remaining"]
        filtered = [c for c in non_pass if len(c.cards) >= min_opp]
        if not filtered:
            filtered = non_pass  # fallback: no qualifying moves
    elif intent == "assist":
        partner_rem = flags["partner_remaining"]
        filtered = [c for c in non_pass if len(c.cards) < partner_rem]
        if not filtered:
            filtered = non_pass  # fallback
    else:
        filtered = non_pass

    def _score(combo: Combo) -> float:
        if scores is not None and id(combo) in scores:
            return -scores[id(combo)]  # negate: higher Q → sorts first (ascending sort)
        return _jidan_score(combo, level_rank)  # fallback: cheap plays first

    sorted_moves = sorted(filtered, key=_score)
    candidates = sorted_moves[:top_k]

    if pass_moves:
        candidates.append(pass_moves[0])

    return candidates


# ── Hand formatting ───────────────────────────────────────────────────────────

def _format_hand(hand: set, level_rank: int) -> str:
    by_suit: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    jokers: list = []

    for c in hand:
        if c.rank >= 16:
            jokers.append(c)
        else:
            by_suit[c.suit].append(c)

    lines = []
    for suit_id, name in [(0, "Spades"), (1, "Hearts"), (2, "Diamonds"), (3, "Clubs")]:
        cards = sorted(by_suit[suit_id], key=lambda c: c.rank)
        if cards:
            parts = []
            for c in cards:
                r = _RANK_DISPLAY[c.rank]
                if suit_id == Suit.HEART and c.rank == level_rank:
                    r = f"*{r}"  # wild marker
                parts.append(r)
            lines.append(f"  {name}: {' '.join(parts)}")

    if jokers:
        j_strs = ["BJ" if c.rank == Rank.BLACK_JOKER else "RJ" for c in jokers]
        lines.append(f"  Jokers: {' '.join(j_strs)}")

    return "\n".join(lines) if lines else "  (empty)"


def _format_played_cards(played: set, level_rank: int) -> str:
    if not played:
        return "(none yet)"
    strs = sorted(
        _RANK_DISPLAY.get(c.rank, "?") + _SUIT_LETTER.get(c.suit, "?")
        for c in played
        if c.rank < 16
    )
    joker_strs = ["BJ" if c.rank == Rank.BLACK_JOKER else "RJ"
                  for c in played if c.rank >= 16]
    all_strs = strs + joker_strs
    summary = " ".join(all_strs[:24])
    if len(all_strs) > 24:
        summary += "..."
    return summary


def _format_player_history(move_history: list, seat: int, n: int = 3) -> str:
    """Last n moves for a specific player seat. Returns space-joined compact string."""
    moves = [combo for p, combo in move_history if p == seat][-n:]
    if not moves:
        return "(none)"
    parts = []
    for combo in moves:
        if combo.type == ComboType.PASS:
            parts.append("PASS")
        else:
            t = _COMBO_TYPE_DISPLAY.get(combo.type, "?")
            r = _RANK_DISPLAY.get(combo.key, "?")
            parts.append(f"{t}{r}[{len(combo.cards)}c]")
    return " ".join(parts)


def _format_recent_moves(move_history: list, player: int, n: int = 8) -> str:
    recent = move_history[-n:]
    if not recent:
        return "(none)"
    partner = (player + 2) % 4
    parts = []
    for p, combo in recent:
        if p == player:
            label = "you"
        elif p == partner:
            label = "partner"
        else:
            label = "opp"
        if combo.type == ComboType.PASS:
            parts.append(f"P{p}({label}):PASS")
        else:
            t = _COMBO_TYPE_DISPLAY.get(combo.type, "?")
            r = _RANK_DISPLAY.get(combo.key, "?")
            n_cards = len(combo.cards)
            parts.append(f"P{p}({label}):{t}{r}[{n_cards}c]")
    return " | ".join(parts)


# ── Game state prompt ─────────────────────────────────────────────────────────

def format_game_state(
    env: GuanDanEnv,
    player: int,
    flags: dict,
    level_rank: int,
) -> str:
    """Format game state for LLM. Uses only production-fair (imperfect) info."""
    partner = flags["partner"]
    left_opp = flags["left_opp"]
    right_opp = flags["right_opp"]
    lr_str = _RANK_DISPLAY.get(level_rank, str(level_rank))

    hand = env.hands[player]
    team = "A (P0+P2)" if player % 2 == 0 else "B (P1+P3)"

    def danger(n: int) -> str:
        return " ⚠ DANGER" if n <= 5 else ""

    # Current trick
    if env.current_trick is None:
        trick_line = "FREE LEAD — you may play any valid combo"
    else:
        trick = env.current_trick
        w = env.trick_winner
        rel = "YOUR PARTNER" if w == partner else "OPPONENT"
        trick_line = f"{_combo_desc(trick, level_rank)} — currently won by P{w} ({rel})"

    # Finish info
    if env.finish_order:
        fin = " → ".join(f"P{p}" for p in env.finish_order)
        finish_line = f"Finished: {fin}"
    else:
        finish_line = "Nobody finished yet"

    # Cooperative options
    coop_parts = []
    if flags["can_cooperate"]:
        coop_parts.append(
            f"  [cooperate] PASS — partner P{partner} is winning this trick"
        )
    if flags["can_dwarf"]:
        coop_parts.append(
            f"  [dwarf]     Play ≥{flags['min_opp_remaining']} cards to force an opponent out"
        )
    if flags["can_assist"]:
        coop_parts.append(
            f"  [assist]    Play <{flags['partner_remaining']} cards to set partner P{partner} up"
        )
    if not coop_parts:
        coop_parts.append("  (none available this turn — choose: normal)")
    coop_section = "\n".join(coop_parts)

    return f"""=== YOUR TURN — Player {player} | Team {team} ===
LEVEL RANK: {lr_str}  (Heart {lr_str} marked with * is the wild card)

YOUR HAND ({len(hand)} cards):
{_format_hand(hand, level_rank)}

CARD COUNTS (visible at the table):
  Partner P{partner}: {flags['partner_remaining']} cards{danger(flags['partner_remaining'])}
  Left    P{left_opp}:  {flags['left_remaining']} cards{danger(flags['left_remaining'])}
  Right   P{right_opp}:  {flags['right_remaining']} cards{danger(flags['right_remaining'])}

PUBLICLY PLAYED CARDS:
  Partner P{partner}: {_format_played_cards(env.played[partner], level_rank)}
  Left    P{left_opp}:  {_format_played_cards(env.played[left_opp], level_rank)}
  Right   P{right_opp}:  {_format_played_cards(env.played[right_opp], level_rank)}

CURRENT TRICK:
  {trick_line}
{finish_line}

RECENT MOVES (last 8):
  {_format_recent_moves(env.move_history, player)}

COOPERATIVE OPTIONS:
{coop_section}"""


def format_tom_beliefs(
    env: GuanDanEnv,
    player: int,
    flags: dict,
    level_rank: int,
) -> str:
    """Format observable player profiles for ToM belief inference.

    Uses only public information: card counts, played cards, recent actions.
    Never accesses env.hands[other] contents.
    """
    roles = [
        (flags["partner"], "partner"),
        (flags["left_opp"], "left-opp"),
        (flags["right_opp"], "right-opp"),
    ]
    lines = ["PLAYER PROFILES (for belief reasoning):"]
    for seat, role in roles:
        if seat in env.finish_order:
            lines.append(f"P{seat}({role}): OUT")
            continue
        rem = len(env.hands[seat])
        played_cards = sorted(env.played[seat], key=lambda c: (c.rank, c.suit))[:16]
        played_str = " ".join(_card_display(c, level_rank) for c in played_cards)
        if len(env.played[seat]) > 16:
            played_str += f" +{len(env.played[seat]) - 16}"
        played_str = played_str or "(none)"
        last3 = _format_player_history(env.move_history, seat, n=3)
        lines.append(
            f"P{seat}({role}, rem={rem}): played=[{played_str}] last3=[{last3}]"
        )
    return "\n".join(lines)


def format_candidates(candidates: list[Combo], level_rank: int) -> str:
    """Format candidate moves for stage-3 move selection."""
    lines = ["\n\nSELECT A MOVE — output the integer index on the first line, nothing else before it:"]
    for i, combo in enumerate(candidates):
        lines.append(f"  [{i}] {_combo_desc(combo, level_rank)}")
    lines.append(
        f"\nFirst line: ONLY the integer index (0–{len(candidates)-1}). "
        "Second line (optional): one sentence explaining why."
    )
    return "\n".join(lines)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_intent(text: str) -> str:
    """Extract cooperative intent. Returns 'normal' on failure."""
    t = text.strip().lower()
    for intent in ("cooperate", "dwarf", "assist", "normal"):
        if intent in t:
            return intent
    return "normal"


def parse_move_index(text: str, n_candidates: int) -> int | None:
    """Extract valid move index from LLM response. Returns None on failure.

    Checks the first non-empty line first (expected format: number on first line),
    then scans remaining lines from last to first as fallback.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None
    # First line is the intended output format — check it first
    for line in [lines[0]] + list(reversed(lines[1:])):
        for m in reversed(re.findall(r"\b(\d+)\b", line)):
            idx = int(m)
            if 0 <= idx < n_candidates:
                return idx
    return None


# ── System prompt ─────────────────────────────────────────────────────────────

def get_system_prompt(level_rank: int) -> str:
    lr = _RANK_DISPLAY.get(level_rank, str(level_rank))
    return f"""You are an expert Guan Dan (掼蛋) card game player. Play strategically to win with your partner.

## GAME OVERVIEW
- 4 players, 2 teams: (P0,P2) vs (P1,P3). Your partner is always 2 seats away.
- 108-card double deck. Each player gets 27 cards. Play passes counterclockwise.
- Current level rank: {lr}. Heart {lr} (marked with *) is the wild card.
- Win condition: your TEAM (you + partner) both finish their hands before the other team.

## COMBO TYPES (weakest → strongest within same type)
Normal combos (must beat same type):
  Single, Pair, Triple, Full House (triple+pair), Straight (5 consec.), Tube (3 consec. pairs), Plate (2 consec. triples)
Bombs (beat any normal; higher bomb beats lower bomb):
  Quad → 5-of-a-kind → Straight Flush → 6 → 7 → 8 → 9 → 10-of-a-kind → Double Joker (highest)

## WILD CARD
Heart {lr} substitutes for any rank. Wild combos beat non-wild of the same type/rank.

## STRATEGY PRINCIPLES
1. Shed cheap cards early (2-7 singles/pairs) — they're hard to play later
2. Preserve bombs for emergencies: opponent near finishing, or to retake control
3. If your PARTNER is winning the trick → PASS and let them keep control
4. If an opponent has ≤5 cards left → consider bombing to stop them
5. When you or partner has ≤5 cards → prioritize emptying that hand

## COOPERATIVE ACTIONS (when available)
- cooperate: PASS to preserve partner's trick win
- dwarf: play a large combo forcing an opponent to finish (risky — check if worth it)
- assist: play a small combo to stay efficient and help partner
- normal: play the strategically best move

## OUTPUT FORMAT
When asked for intent: output exactly one word: cooperate | dwarf | assist | normal
When asked for beliefs: infer what each player likely holds. ≤100 words. End with DANGER: P<seat> or DANGER: NONE.
When asked for a move: first line is ONLY the integer index. Second line (optional): one sentence of reasoning."""
