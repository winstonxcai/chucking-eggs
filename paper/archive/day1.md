# Guan Dan DMC: 1-Day MVP Blueprint (Full Rules)

> **Goal**: Implement DMC end-to-end on **real Guan Dan** — correct rules, simplified training infrastructure
> **Budget**: 1 intense day (~10–12 hours). Training runs overnight on CPU.
> **Result**: Full Guan Dan engine + working DMC loop. Agent beats random within hours of training.
> **Non-negotiable**: Game rules are production-correct. No shortcuts on combo types, wilds, teams, or bombs.
> **Author**: Winston | Tsinghua SIGS — High Performance Geo-Computing Lab

---

## 0. Design Philosophy: Correct Engine, Minimal Infrastructure

The game engine is the foundation everything else builds on. A bug in movegen or wild card logic will silently corrupt every training run downstream. So we build the engine right, and keep everything around it dead simple.

| Component | Full Plan | This MVP | Rationale |
|-----------|-----------|----------|-----------|
| **Game rules** | **Full Guan Dan** | **Full Guan Dan** | Non-negotiable. No simplified variants. |
| **Combo types** | **All types** | **All types** | Singles, pairs, triples, straights, tubes, plates, full houses, 9 bomb tiers (quad→decuple + SF + 4-joker) |
| **Wild cards** | **♥ of level rank** | **♥ of level rank** | Correct wild substitution in all combos |
| **4 players, 2 teams** | **Yes** | **Yes** | Team dynamics are core to Guan Dan |
| **Level rank** | **Rotates** | **Fixed (start at 2)** | Semantics identical; simplifies one variable for debugging |
| **Scoring** | Rank-based (+3/+1/-1/-3) | Rank-based (+3/+1/-1/-3) | Same as full plan |
| Movegen | Cython (<1ms) | **Pure Python** | Correct first, fast later |
| Q-network | LSTM + ResNet (4M) | **MLP (100K)** | Same learning dynamics, faster iteration |
| History encoding | LSTM over 20 moves | **Flat: last trick only** | Defer LSTM to scale-up |
| Actors | 32 CPU (Modal) | **Single process** | Distributed is engineering, not algorithm |
| Replay buffer | 2M transitions | **100K transitions** | Enough for signal |
| Lead/follow split | 2 Q-networks | **1 Q-network** | Add split when engine is proven |
| Soft-start | Heuristic mixing | **Pure ε-greedy** | Add when baseline works |
| Oracle (PTIE) | Phase 2 | **Cut** | Phase 2 |
| Hand prediction | Phase 3 | **Cut** | Phase 3 |

---

## 1. Guan Dan Rules Reference (Source: pagat.com)

This section is the ground truth. Every line of game engine code must comply with these rules. Source: https://www.pagat.com/climbing/guan_dan.html

### 1.1 Deck, Players, and Deal

- **2 standard decks** + **4 jokers** = **108 cards total**
- Each standard deck: 13 ranks (2–A) × 4 suits (♠♥♦♣) = 52 cards
- Jokers: 2 Black Jokers (小王), 2 Red Jokers (大王)
- **Deal**: 27 cards to each of 4 players
- **Seating**: Players 0, 1, 2, 3. Teams: {0, 2} vs {1, 3}
- **Play direction**: **Counterclockwise** (right to left)

### 1.2 Card Ranking: Two Orderings

There are two distinct orderings. Getting this right is critical — most bugs come from mixing them up.

**Natural order** (high to low): RJ, BJ, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2, (A)
- Aces can be low, ranking below 2, for forming sequences only
- This ordering is used for: **straights, tubes, plates, straight flushes**

**Level order** (high to low): RJ, BJ, **[level cards]**, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2
- Level cards are promoted above aces, below black jokers
- Level cards' natural position is skipped (e.g., at level 7: ...8, 6, 5...)
- This ordering is used for: **singles, pairs, triples, full houses, ranking N-of-a-kind bombs**

**Example at level 2 (first hand)**:
- Natural order: RJ, BJ, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2, (A)
- Level order: RJ, BJ, **2**, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3

**Example at level 8**:
- Natural order: RJ, BJ, A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2, (A)
- Level order: RJ, BJ, **8**, A, K, Q, J, 10, 9, 7, 6, 5, 4, 3, 2

### 1.3 Level Rank and Wild Cards

- Each team has a **level** starting at 2, advancing through 3...K, A as they win hands
- The **level cards** are all cards matching the current level rank
- **Wild cards**: The two ♥ of the current level rank are wild (e.g., at level 2, both ♥2 are wild)
- **Wild substitution**: A wild can represent **any card except a joker**
- **Wild as single/pair**: When played as a single, a wild ranks as a level card (above A, below BJ). Two wilds together form a pair equal to a pair of level cards.
- **Wild in combos**: Can fill any position in any combination. Player must declare what each wild represents.
- **Two wilds can stand for different cards** in the same combination
- **Wilds in bombs**: Allowed in all bomb types **except** the four-joker bomb
- **For MVP**: Fix level at 2 for all hands (all mechanics are identical; just one fewer variable)

### 1.4 Ordinary Combinations (7 Types)

```
Type            Size    Description                                          Ranked by
──────────────────────────────────────────────────────────────────────────────────────────
SINGLE          1       One card                                             Level order
PAIR            2       Two cards of same rank                               Level order
                        (BJ+BJ or RJ+RJ ok; BJ+RJ is NOT a pair)
TRIPLE          3       Three cards of same rank                             Level order
                        (no triple of jokers possible → highest = 3 level cards)
FULL_HOUSE      5       Triple + Pair                                        Triple's rank
                        (ranked by triple in level order; pair rank irrelevant)      (level order)
STRAIGHT        5       5 consecutive cards in NATURAL order,                Top card
                        NOT all the same suit                                (natural order)
TUBE (连对)     6       3 consecutive PAIRS in NATURAL order                 Top pair
                        (no joker pairs; level cards in natural position)     (natural order)
PLATE (钢板)    6       2 consecutive TRIPLES in NATURAL order               Top triple
                        (level cards in natural position)                     (natural order)
```

**Critical rules for sequences (straights, tubes, plates)**:
- **Level cards take their NATURAL position**, not level position. At level 7, a 7 sits between 6 and 8 in a straight, NOT above A.
- **Aces can be high or low**: 10-J-Q-K-A (highest straight) and A-2-3-4-5 (lowest straight, ranked by the 5) are both valid
- **No wrapping**: K-A-2-3-4, Q-K-A-2-3, J-Q-K-A-2 are NOT valid
- **Jokers cannot appear** in straights, tubes, or plates
- **Tubes are exactly 3 pairs** (6 cards). Not more, not fewer.
- **Plates are exactly 2 triples** (6 cards). Not more, not fewer.
- **Straights are exactly 5 cards**. Not more, not fewer.
- **A straight must NOT be all one suit** — that would be a straight flush (a bomb)
- Lowest tube: A-A-2-2-3-3 (ranked by 3). Highest tube: Q-Q-K-K-A-A (ranked by A)
- Lowest plate: A-A-A-2-2-2 (ranked by 2). Highest plate: K-K-K-A-A-A (ranked by A)

### 1.5 Bombs (9 Types, Strictly Ordered Low→High)

```
Tier   Type              Size    Description
─────────────────────────────────────────────────────────────────────────
1      QUADRUPLE         4       4 cards of same rank
2      QUINTUPLE         5       5 cards of same rank
3      STRAIGHT FLUSH    5       5 consecutive same-suit cards (natural order)
4      SEXTUPLE          6       6 cards of same rank
5      SEPTUPLE          7       7 cards of same rank
6      OCTUPLE           8       8 cards of same rank
7      NONUPLE           9       9 cards of same rank (requires wilds)
8      DECUPLE           10      10 cards of same rank (requires wilds)
9      FOUR-JOKER        4       2 BJ + 2 RJ (highest bomb)
```

**Within each N-of-a-kind tier**: ranked by card rank in **level order** (level cards highest within their tier, since there aren't enough jokers to make joker-based N-of-a-kind)

**Straight flushes**: ranked by top card in **natural order**. Lowest: A♠-2♠-3♠-4♠-5♠. Highest: 10♠-J♠-Q♠-K♠-A♠. Level has no effect on straight flush ranking.

**Key hierarchy facts**:
- Any bomb beats any non-bomb combo
- Straight flush sits BETWEEN quintuple and sextuple
- The highest quintuple (5 level-cards) is beaten by the lowest straight flush (A-2-3-4-5 suited)
- Nonuple/decuple require wilds (only 2 wilds exist, so max 1 decuple or 2 nonuples per hand)
- Four-joker bomb beats everything

**Wild cards in bombs**: Wilds can be used in ALL bomb types except four-joker. Examples at level 2:
- 3 naturals of rank 9 + 1 wild = quadruple of 9s
- 4 naturals of rank 9 + 1 wild = quintuple of 9s
- 3 naturals + 2 wilds = quintuple
- 8 naturals of rank 9 (from double deck) + 2 wilds = decuple of 9s

### 1.6 Turn and Trick Mechanics

- **Leading**: The trick leader plays any ordinary combo or any bomb
- Play proceeds **counterclockwise**
- **Responding to ordinary combo**: Play a higher combo **of the same type**, or play any bomb, or pass
- **Responding to bomb**: Play a higher bomb, or pass
- **Passing does NOT lock you out**: A player who passes can still play on a later turn in the same trick
- **Trick end**: 3 consecutive passes after the last play → trick over, last player who played leads next trick
- **Players who are out**: Simply skipped on their turn. They always count as passing.

### 1.7 Going Out, Finish Order, and Game End

- A player **goes out** when they play their last card(s)
- **Play continues until both members of the winning team have gone out**
- The finish order (1st through 4th) determines the round outcome
- **Partner leads rule**: If a player whose turn it is to lead has no cards left, the lead passes to their **partner**

### 1.8 Scoring (Level Promotion)

The team of the player who finishes first wins the hand. The finish positions of both teammates determine promotion:

| Result | Promotion |
|--------|-----------|
| **1-2 win** (teammates finish 1st & 2nd) | Winners promoted **4 levels** |
| **1-3 win** (teammates finish 1st & 3rd) | Winners promoted **2 levels** |
| **1-4 win** (teammates finish 1st & 4th) | Winners promoted **1 level** |

**For the RL reward signal** (not the level system), we use position-based rewards:
```
1st out: +3.0,  2nd out: +1.0,  3rd out: -1.0,  4th out: -3.0
```
This is zero-sum (3+1-1-3=0) and captures the incentive to go out early.

### 1.9 Tribute (Skip for MVP)

From the second hand onward, the loser(s) of the previous hand pay tribute (their highest non-wild card) to the winner(s), who return an unwanted card. This affects who leads first. **Skip for MVP** — just pick first leader randomly. Add in scale-up when implementing multi-hand sessions.

### 1.10 First Hand

- First hand is always at level 2
- First player chosen randomly (or by designated start card in traditional rules)

### 1.11 Rule Subtleties to Watch For

1. **Wild as itself**: A wild IS a ♥[level_rank]. It can always be played "as itself" without using wild power. E.g., at level 2, ♥2 can be played as a normal 2 in a straight containing a 2.
2. **Pair of wilds**: Two wilds together = pair of level cards. They can also be used as two different cards in a combo (e.g., two wilds in a straight each substituting for different missing ranks).
3. **No joker pairs across colors**: BJ+BJ is a pair. RJ+RJ is a pair. BJ+RJ is NOT a pair.
4. **Level cards in natural position for sequences**: This is the most common implementation error. At level 7: 5-6-7-8-9 is a valid straight. 5-6-8-9-10 is NOT (7 is missing, not promoted out).
5. **Straight vs straight flush**: 5 consecutive cards NOT all same suit = ordinary straight. All same suit = straight flush BOMB. The movegen must check suit uniformity.

---

## 2. Schedule (10–12 Hours, Honest Estimate)

### Block 1 (Hours 1–4): Game Engine

This is the bulk of the work and the most important part. No shortcuts.

**Hour 1: `cards.py` — Card model + constants**

```python
# cards.py

from enum import IntEnum
from typing import NamedTuple

class Suit(IntEnum):
    SPADE = 0; HEART = 1; DIAMOND = 2; CLUB = 3

class Rank(IntEnum):
    # Natural order values. 2 is lowest normal rank, A=14 is highest.
    # In sequences, Ace can also be LOW (=1), handled in movegen.
    TWO = 2; THREE = 3; FOUR = 4; FIVE = 5; SIX = 6; SEVEN = 7
    EIGHT = 8; NINE = 9; TEN = 10; JACK = 11; QUEEN = 12
    KING = 13; ACE = 14
    BLACK_JOKER = 16; RED_JOKER = 17

class Card(NamedTuple):
    rank: int
    suit: int   # 0-3 for normal, 0/1 for jokers
    deck: int   # 0 or 1 (which copy from the double deck)

class ComboType(IntEnum):
    PASS = 0
    # ── Ordinary (7 types) ──
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FULL_HOUSE = 4      # triple + pair (5 cards)
    STRAIGHT = 5        # exactly 5 consecutive, natural order, NOT all same suit
    TUBE = 6            # exactly 3 consecutive pairs (6 cards), natural order
    PLATE = 7           # exactly 2 consecutive triples (6 cards), natural order
    # ── Bombs (9 tiers, ordered low→high by enum value) ──
    BOMB_4 = 8          # quadruple
    BOMB_5 = 9          # quintuple
    STRAIGHT_FLUSH = 10 # 5 consecutive same suit (BETWEEN 5 and 6-of-a-kind!)
    BOMB_6 = 11         # sextuple
    BOMB_7 = 12         # septuple
    BOMB_8 = 13         # octuple
    BOMB_9 = 14         # nonuple (requires wilds)
    BOMB_10 = 15        # decuple (requires wilds)
    BOMB_JOKER = 16     # 2BJ + 2RJ — highest bomb

BOMB_TYPES = {ComboType.BOMB_4, ComboType.BOMB_5, ComboType.STRAIGHT_FLUSH,
              ComboType.BOMB_6, ComboType.BOMB_7, ComboType.BOMB_8,
              ComboType.BOMB_9, ComboType.BOMB_10, ComboType.BOMB_JOKER}

# Bomb tier is just the ComboType int value (8..16). Higher = stronger.

def level_order_key(rank, level_rank):
    """
    Comparison key for LEVEL ORDER (used for singles, pairs, triples,
    full houses, N-of-a-kind bombs).
    Level cards rank above A(14), below BJ(16).
    """
    if rank == level_rank:
        return 15  # above A(14), below BJ(16)
    return rank

def make_deck():
    """Create 108-card double deck."""
    cards = []
    for deck_id in range(2):
        for rank in range(2, 15):  # 2 through A(=14)
            for suit in range(4):
                cards.append(Card(rank, suit, deck_id))
        cards.append(Card(Rank.BLACK_JOKER, 0, deck_id))
        cards.append(Card(Rank.RED_JOKER, 0, deck_id))
    return cards

def is_wild(card, level_rank):
    """Is this card a wild (♥ of level rank)?"""
    return card.rank == level_rank and card.suit == Suit.HEART
```

**Hours 2–3: `combos.py` — Combo generation and comparison**

This is the hardest file. Every combo type, wild substitution, and the beats() function.

```python
# combos.py — combo classification, generation, and comparison

class Combo:
    def __init__(self, combo_type, key_rank, cards, length=0, wild_count=0,
                 key_is_level_order=False, level_rank=None):
        self.type = combo_type
        self.key = key_rank          # primary rank for comparison
        self.cards = tuple(cards)    # actual cards played
        self.length = length         # for straights only (always 5 for now)
        self.wild_count = wild_count
    
    def beats(self, other, level_rank):
        """Can this combo beat 'other'? (other = current trick on table)"""
        if other is None:
            return True  # free lead
        
        my_bomb = self.type in BOMB_TYPES
        their_bomb = other.type in BOMB_TYPES
        
        # Any bomb beats any non-bomb
        if my_bomb and not their_bomb:
            return True
        if not my_bomb and their_bomb:
            return False
        
        # Both bombs: compare by tier (ComboType int value), then rank
        if my_bomb and their_bomb:
            if self.type != other.type:
                # Different bomb tier → higher enum value wins
                return self.type > other.type
            # Same bomb tier
            if self.type == ComboType.STRAIGHT_FLUSH:
                # Ranked by top card in NATURAL order (no level promotion)
                return self.key > other.key
            if self.type == ComboType.BOMB_JOKER:
                return False  # can't beat another four-joker (there's only one)
            # N-of-a-kind: ranked in LEVEL ORDER
            return level_order_key(self.key, level_rank) > level_order_key(other.key, level_rank)
        
        # Neither is bomb: must match type
        if self.type != other.type:
            return False
        
        # Ordinary combos: comparison depends on type
        if self.type == ComboType.STRAIGHT:
            # Ranked by top card in NATURAL order
            return self.key > other.key
        
        if self.type in (ComboType.TUBE, ComboType.PLATE):
            # Ranked by top pair/triple in NATURAL order
            return self.key > other.key
        
        # Singles, pairs, triples, full houses: LEVEL ORDER
        return level_order_key(self.key, level_rank) > level_order_key(other.key, level_rank)

# ─── MOVEGEN ───────────────────────────────────────────

def generate_all_leads(hand, level_rank):
    """Generate every legal combo from hand (free lead)."""
    combos = []
    wilds = [c for c in hand if is_wild(c, level_rank)]
    naturals = [c for c in hand if not is_wild(c, level_rank)]
    
    by_rank = {}  # rank → list of Card
    for c in naturals:
        by_rank.setdefault(c.rank, []).append(c)
    
    # --- Singles ---
    # Every distinct card can be a single
    for rank, cards in by_rank.items():
        for c in _unique_cards(cards):
            combos.append(_make(ComboType.SINGLE, rank, [c]))
    # Wilds as singles (rank as level card)
    for w in wilds:
        combos.append(_make(ComboType.SINGLE, level_rank, [w]))
    
    # --- Pairs ---
    # Natural pairs (including BJ+BJ and RJ+RJ, but NOT BJ+RJ)
    for rank, cards in by_rank.items():
        if len(cards) >= 2:
            combos.append(_make(ComboType.PAIR, rank, cards[:2]))
    # Wild-augmented pairs (wild + any non-joker card)
    _add_wild_pairs(combos, by_rank, wilds, level_rank)
    # Pair of wilds (= pair of level cards)
    if len(wilds) >= 2:
        combos.append(_make(ComboType.PAIR, level_rank, wilds[:2], wild_count=2))
    
    # --- Triples ---
    for rank, cards in by_rank.items():
        if len(cards) >= 3:
            combos.append(_make(ComboType.TRIPLE, rank, cards[:3]))
    _add_wild_triples(combos, by_rank, wilds, level_rank)
    
    # --- Full houses (triple + pair) ---
    _add_full_houses(combos, by_rank, wilds, level_rank)
    
    # --- Straights (exactly 5 consecutive, natural order, NOT all same suit) ---
    _add_straights(combos, by_rank, wilds, level_rank)
    
    # --- Tubes (exactly 3 consecutive pairs, natural order) ---
    _add_tubes(combos, by_rank, wilds, level_rank)
    
    # --- Plates (exactly 2 consecutive triples, natural order) ---
    _add_plates(combos, by_rank, wilds, level_rank)
    
    # --- N-of-a-kind bombs (4 through 10) ---
    _add_nofakind_bombs(combos, by_rank, wilds, level_rank)
    
    # --- Straight flushes (5 consecutive, same suit, natural order) ---
    _add_straight_flushes(combos, hand, wilds, level_rank)
    
    # --- Four-joker bomb ---
    jokers = [c for c in hand if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)]
    if len(jokers) == 4:
        combos.append(_make(ComboType.BOMB_JOKER, 99, jokers))
    
    return _deduplicate(combos)

def generate_responses(hand, level_rank, trick):
    """Generate all combos that beat the current trick, plus PASS."""
    leads = generate_all_leads(hand, level_rank)
    responses = [c for c in leads if c.beats(trick, level_rank)]
    responses.append(Combo(ComboType.PASS, 0, []))
    return responses

# ─── HELPER GENERATORS ────────────────────────────────

def _make(ctype, key, cards, length=0, wild_count=0):
    return Combo(ctype, key, cards, length, wild_count)

def _unique_cards(cards):
    """Deduplicate identical cards (same rank+suit but different deck)."""
    seen = set()
    result = []
    for c in cards:
        k = (c.rank, c.suit)
        if k not in seen:
            seen.add(k)
            result.append(c)
    return result

def _add_wild_pairs(combos, by_rank, wilds, level_rank):
    """Wild + any non-joker natural = pair of that rank."""
    if not wilds:
        return
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue  # wilds can't become jokers
        if len(cards) >= 1:
            combos.append(_make(ComboType.PAIR, rank, [cards[0], wilds[0]], wild_count=1))

def _add_wild_triples(combos, by_rank, wilds, level_rank):
    """Wild(s) + naturals to form triples."""
    n_wild = len(wilds)
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue
        n = len(cards)
        if n == 2 and n_wild >= 1:
            combos.append(_make(ComboType.TRIPLE, rank, cards[:2] + wilds[:1], wild_count=1))
        if n == 1 and n_wild >= 2:
            combos.append(_make(ComboType.TRIPLE, rank, cards[:1] + wilds[:2], wild_count=2))

def _add_full_houses(combos, by_rank, wilds, level_rank):
    """Triple + Pair. Enumerate all triple/pair combos including wild-assisted."""
    # TODO: implement — enumerate all (triple, pair) combos where
    # triple and pair use different ranks and total wilds used ≤ len(wilds)
    pass

def _sequence_ranks():
    """Valid ranks for sequences in natural order. Ace can be high or low.
    Returns list of rank values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    where 1 = Ace-low, 2-13 = normal, 14 = Ace-high.
    """
    return list(range(1, 15))  # 1(A-low), 2, 3, 4, ..., 13(K), 14(A-high)

def _add_straights(combos, by_rank, wilds, level_rank):
    """Exactly 5 consecutive cards, natural order. NOT all same suit.
    Ace-low: A(=1)-2-3-4-5. Ace-high: 10-J-Q-K-A(=14).
    No wrapping (K-A-2-3-4 invalid). Jokers never in straights.
    Level cards in NATURAL position.
    """
    # TODO: implement
    # For each starting rank in [1, 2, ..., 10] (5-card window):
    #   Check if hand has cards for each of the 5 consecutive ranks
    #   (using wilds to fill gaps, max wilds used ≤ available wilds)
    #   Ensure NOT all same suit (check after selecting cards)
    #   Rank 1 = Ace-low, Rank 14 = Ace-high
    #   Key = top rank of straight (natural order)
    pass

def _add_tubes(combos, by_rank, wilds, level_rank):
    """Exactly 3 consecutive pairs, natural order.
    Ace-low: AA-22-33. Ace-high: QQ-KK-AA.
    No joker pairs. Level cards in natural position.
    """
    # TODO: implement
    # For each starting rank: need 2 cards of each of 3 consecutive ranks
    # Wilds can fill gaps
    pass

def _add_plates(combos, by_rank, wilds, level_rank):
    """Exactly 2 consecutive triples, natural order.
    Ace-low: AAA-222. Ace-high: KKK-AAA.
    Level cards in natural position.
    """
    # TODO: implement
    pass

def _add_nofakind_bombs(combos, by_rank, wilds, level_rank):
    """N-of-a-kind bombs (4 through 10). Wilds extend natural groups."""
    n_wild = len(wilds)
    for rank, cards in by_rank.items():
        if rank >= Rank.BLACK_JOKER:
            continue  # no joker bombs via N-of-a-kind
        n = len(cards)  # max 8 from double deck
        # Generate all possible sizes from 4 to min(n + n_wild, 10)
        for size in range(4, min(n + n_wild, 10) + 1):
            needed_wilds = max(0, size - n)
            if needed_wilds > n_wild:
                continue
            # Map size to bomb type
            bomb_type = {4: ComboType.BOMB_4, 5: ComboType.BOMB_5,
                         6: ComboType.BOMB_6, 7: ComboType.BOMB_7,
                         8: ComboType.BOMB_8, 9: ComboType.BOMB_9,
                         10: ComboType.BOMB_10}.get(size)
            if bomb_type:
                used_cards = cards[:min(n, size)] + wilds[:needed_wilds]
                combos.append(_make(bomb_type, rank, used_cards, wild_count=needed_wilds))

def _add_straight_flushes(combos, hand, wilds, level_rank):
    """5 consecutive same-suit cards, natural order.
    Ace high or low. Wilds can fill gaps (declared as needed suit).
    Jokers cannot be used. Level cards in natural position.
    """
    # TODO: implement
    # For each suit, for each starting rank:
    #   Find natural cards of that suit at each of 5 consecutive ranks
    #   Fill gaps with wilds (max wilds ≤ available)
    #   Key = top rank (natural order)
    pass

def _deduplicate(combos):
    """Remove duplicate combos (same type + key + card set)."""
    seen = set()
    result = []
    for c in combos:
        # Use frozenset of card identities for dedup
        card_ids = frozenset((card.rank, card.suit, card.deck) for card in c.cards)
        key = (c.type, c.key, c.length, card_ids)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
```

**Implementation notes on movegen**:
- Wild cards make movegen combinatorially harder. The key insight: you only have **0, 1, or 2 wilds**, so you can enumerate wild placements exhaustively.
- For straights/tubes/plates with wilds: iterate over all possible start ranks and lengths, count how many gaps exist, check if wilds can fill them.
- **Deduplication**: Multiple wild placements might produce the "same" combo (same type + key). Deduplicate by (combo_type, key_rank, length, frozenset of card IDs).
- **Test heavily**: The movegen is the #1 source of bugs in every Guan Dan RL implementation.

**Hour 3–4: `game.py` — Full game loop with 4 players**

```python
# game.py — 4-player Guan Dan game environment

class GuanDanEnv:
    def __init__(self, level_rank=Rank.TWO):
        self.level_rank = level_rank
        self.reset()
    
    def reset(self):
        deck = make_deck()
        random.shuffle(deck)
        self.hands = [
            set(deck[0:27]),  set(deck[27:54]),
            set(deck[54:81]), set(deck[81:108])
        ]
        self.played = [set() for _ in range(4)]  # cards each player has played
        self.current_player = 0         # who acts next
        self.current_trick = None       # Combo or None (free lead)
        self.trick_winner = None        # who played the current winning combo
        self.consecutive_passes = 0     # consecutive passes since last play
        self.finish_order = []          # [first_out, second_out, ...]
        self.is_out = [False] * 4       # who has gone out
        self.done = False
    
    def _next_player(self, from_player):
        """Next player COUNTERCLOCKWISE (to the right). Skips players who are out."""
        p = (from_player - 1) % 4  # counterclockwise = subtract 1
        for _ in range(4):
            if not self.is_out[p]:
                return p
            p = (p - 1) % 4
        return None  # all out
    
    def _partner(self, player):
        """Return partner seat. 0↔2, 1↔3."""
        return (player + 2) % 4
    
    def _active_count(self):
        """How many players are still in."""
        return sum(1 for x in self.is_out if not x)
    
    def _winning_team_done(self):
        """Check if both members of the winning team (team of 1st finisher) are out."""
        if len(self.finish_order) < 1:
            return False
        first = self.finish_order[0]
        partner = self._partner(first)
        return self.is_out[first] and self.is_out[partner]
    
    def step(self, combo):
        """
        Execute a play. combo is a Combo object from legal_moves().
        Returns (current_player, done).
        """
        player = self.current_player
        
        if combo.type == ComboType.PASS:
            self.consecutive_passes += 1
            
            # Trick ends when 3 consecutive passes after the last play
            if self.consecutive_passes >= 3:
                self._end_trick()
                self.current_player = self._resolve_leader()
                return self.current_player, self.done
            
            self.current_player = self._next_player(player)
            # Skip players who are out (they auto-pass and count toward consecutive_passes)
            # Actually: "A player with no card left in hand passes every opportunity to play"
            # The _next_player already skips out players, but we need to count their
            # auto-passes. For simplicity, we skip them in _next_player and only
            # count explicit passes. This means consecutive_passes counts only
            # among active players. The rule "3 consecutive passes" means 3 passes
            # from other players after the last play. Since out players are skipped,
            # this works correctly when active_count adjusts:
            # - 4 active: need 3 passes to end trick
            # - 3 active: need 2 passes (only 2 others besides trick_winner)
            # - 2 active: need 1 pass
            # Wait — re-reading: "three consecutive players pass" and out players
            # always pass. So if player A plays, then B passes, C is out (auto-pass),
            # D passes → that's 3 consecutive passes → trick ends.
            # For correct implementation: count out players' auto-passes too.
            # Simplest approach: after each non-pass play, count how many players
            # (including out players) have passed since then.
            return self.current_player, self.done
        
        # --- Non-pass play ---
        for card in combo.cards:
            self.hands[player].discard(card)
            self.played[player].add(card)
        
        self.current_trick = combo
        self.trick_winner = player
        self.consecutive_passes = 0
        
        # Check if player goes out
        if len(self.hands[player]) == 0:
            self.finish_order.append(player)
            self.is_out[player] = True
            
            # Game ends when both members of the winning team are out
            if self._winning_team_done():
                # Add remaining players to finish order
                for p in range(4):
                    if p not in self.finish_order:
                        self.finish_order.append(p)
                self.done = True
                return self.current_player, True
        
        # Advance to next active player (counterclockwise, skip out players)
        next_p = self._next_player(player)
        
        # If we've come back to the trick winner, trick is over
        if next_p == self.trick_winner:
            self._end_trick()
            self.current_player = self._resolve_leader()
        else:
            self.current_player = next_p
        
        return self.current_player, self.done
    
    def _end_trick(self):
        """Reset trick state for new trick."""
        self.current_trick = None
        self.consecutive_passes = 0
    
    def _resolve_leader(self):
        """
        Who leads the next trick?
        - Normally: the trick winner
        - If trick winner has gone out: their PARTNER leads
          (pagat: "If the player whose turn it is to lead has no cards left, 
           the lead passes to that player's partner")
        - If partner also out: next active player counterclockwise
        """
        leader = self.trick_winner
        if not self.is_out[leader]:
            return leader
        partner = self._partner(leader)
        if not self.is_out[partner]:
            return partner
        # Both out — shouldn't happen if game end is checked properly
        return self._next_player(leader)
    
    def legal_moves(self, player=None):
        if player is None:
            player = self.current_player
        if self.current_trick is None:
            return generate_all_leads(self.hands[player], self.level_rank)
        else:
            return generate_responses(self.hands[player], self.level_rank,
                                      self.current_trick)
    
    def get_rewards(self):
        """Rank-based zero-sum rewards from finish order."""
        position_rewards = [3.0, 1.0, -1.0, -3.0]
        rewards = {}
        for pos, player in enumerate(self.finish_order):
            rewards[player] = position_rewards[pos]
        return rewards
    
    def is_leading(self, player=None):
        return self.current_trick is None
```

**Hour 4 checkpoint — TEST THE ENGINE**:

```python
# test_game.py — run this BEFORE writing any training code

def test_random_games(n=10_000):
    env = GuanDanEnv()
    crashes = 0
    for i in range(n):
        env.reset()
        steps = 0
        while not env.done and steps < 500:
            legal = env.legal_moves()
            assert len(legal) > 0, f"No legal moves for player {env.current_player}"
            move = random.choice(legal)
            env.step(move)
            steps += 1
        
        if not env.done:
            crashes += 1
            continue
        
        assert len(env.finish_order) == 4, f"Expected 4 finishers, got {len(env.finish_order)}"
        assert set(env.finish_order) == {0, 1, 2, 3}
        # Verify winning team: first finisher's team both done
        first = env.finish_order[0]
        partner = (first + 2) % 4
        assert env.is_out[first] and env.is_out[partner], "Winning team not both out"
    
    print(f"Ran {n} random games: {crashes} crashes, "
          f"{n - crashes} completed successfully")

def test_team_balance(n=50_000):
    """Random play should give ~50% winrate to each team."""
    team_02_wins = 0
    env = GuanDanEnv()
    for _ in range(n):
        env.reset()
        while not env.done:
            env.step(random.choice(env.legal_moves()))
        r = env.get_rewards()
        if r[0] + r[2] > 0:
            team_02_wins += 1
    print(f"Team {{0,2}} winrate: {team_02_wins/n:.1%} (expect ~50%)")

def test_counterclockwise():
    """Verify play direction is counterclockwise."""
    env = GuanDanEnv()
    env.reset()
    env.current_player = 0
    # After player 0 plays, next should be player 3 (counterclockwise)
    legal = env.legal_moves()
    env.step(legal[0])  # player 0 plays something
    if not env.done:
        # Next player should be 3 (0-1 = -1 mod 4 = 3)
        assert env.current_player == 3 or env.is_out[3], \
            f"Expected player 3 next (CCW), got {env.current_player}"
    print("Counterclockwise play direction: OK")

def test_all_combo_types_appear(n=50_000):
    """Verify every combo type gets generated at least once across many games."""
    seen_types = set()
    env = GuanDanEnv()
    for _ in range(n):
        env.reset()
        while not env.done:
            legal = env.legal_moves()
            for m in legal:
                seen_types.add(m.type)
            env.step(random.choice(legal))
    
    expected = {ComboType.SINGLE, ComboType.PAIR, ComboType.TRIPLE,
                ComboType.FULL_HOUSE, ComboType.STRAIGHT, ComboType.TUBE,
                ComboType.PLATE, ComboType.BOMB_4, ComboType.BOMB_5,
                ComboType.STRAIGHT_FLUSH}
    # Rare types (BOMB_6+, BOMB_JOKER) may not appear in 50K random games
    missing_critical = expected - seen_types
    print(f"Combo types seen: {sorted(seen_types)}")
    if missing_critical:
        print(f"WARNING — missing critical types: {missing_critical}")
    else:
        print("All common combo types generated")

def test_bomb_hierarchy():
    """Verify bomb ordering: 4 < 5 < SF < 6 < 7 < 8 < 9 < 10 < JokerBomb"""
    level_rank = Rank.TWO
    # Create mock combos
    b4 = Combo(ComboType.BOMB_4, Rank.FIVE, [], wild_count=0)
    b5 = Combo(ComboType.BOMB_5, Rank.THREE, [], wild_count=0)  # lowest quintuple
    sf = Combo(ComboType.STRAIGHT_FLUSH, Rank.FIVE, [], wild_count=0)  # A-2-3-4-5
    b6 = Combo(ComboType.BOMB_6, Rank.THREE, [], wild_count=0)
    bj = Combo(ComboType.BOMB_JOKER, 99, [], wild_count=0)
    
    assert b5.beats(b4, level_rank), "Quintuple should beat quadruple"
    assert sf.beats(b5, level_rank), "Straight flush should beat quintuple"
    assert b6.beats(sf, level_rank), "Sextuple should beat straight flush"
    assert bj.beats(b6, level_rank), "Joker bomb should beat sextuple"
    assert not b4.beats(b5, level_rank), "Quadruple should NOT beat quintuple"
    print("Bomb hierarchy: OK")

def test_level_order_vs_natural():
    """Verify level cards rank correctly in different contexts."""
    level_rank = 7  # level 7
    
    # In LEVEL ORDER: 7 ranks above A
    assert level_order_key(7, 7) > level_order_key(Rank.ACE, 7), \
        "Level 7 should rank above Ace in level order"
    
    # In NATURAL ORDER (for straights): 7 sits between 6 and 8
    # This means 5-6-7-8-9 is a valid straight, and 7 doesn't jump above A
    assert Rank.SIX < Rank.SEVEN < Rank.EIGHT, \
        "In natural order, 7 should be between 6 and 8"
    print("Level order vs natural order: OK")

def test_ace_low_straight():
    """Verify A-2-3-4-5 is a valid straight (ace low)."""
    # This should be generated by movegen when hand contains A, 2, 3, 4, 5
    # The straight's key should be 5 (ranked by top card = 5 in natural order)
    # 10-J-Q-K-A should also work with key = A(14)
    # K-A-2-3-4 should NOT be valid
    print("Ace-low straight test: implement after movegen is complete")

def test_straight_vs_straight_flush():
    """5 consecutive same-suit = straight flush BOMB, not ordinary straight."""
    # Movegen must check suit uniformity when generating straights
    # and route all-same-suit combos to STRAIGHT_FLUSH instead
    print("Straight vs SF test: implement after movegen is complete")
```

**Do not proceed until all tests pass.** A broken engine wastes every subsequent hour.

---

### Block 2 (Hours 5–7): Encoding + Q-Network + DMC Loop

Once the engine is verified, this is the same DMC core from the previous plan — just wired to the real game.

**Hour 5: `encoding.py`**

```python
# encoding.py — full Guan Dan state/action to tensors

import numpy as np

# Card matrix: 15 ranks × 4 suits
# 15 ranks: 2,3,4,5,6,7,8,9,10,J,Q,K,A,BJ,RJ
# 4 columns: one per suit, values are counts (0/1/2 from double deck)

NUM_RANKS = 15   # 2..A(14), BJ, RJ
NUM_COLS = 4     # suits, with count values

def _rank_index(rank):
    """Map rank to matrix row index (0-14)."""
    if rank <= 14:
        return rank - 2  # 2→0, 3→1, ..., A(14)→12
    if rank == 16:       # BLACK_JOKER
        return 13
    if rank == 17:       # RED_JOKER
        return 14
    raise ValueError(f"Unknown rank {rank}")

def cards_to_matrix(cards):
    """Cards → [15, 4] count matrix (suit-level counts, 0/1/2)."""
    mat = np.zeros((NUM_RANKS, NUM_COLS), dtype=np.float32)
    for c in cards:
        ri = _rank_index(c.rank)
        si = c.suit if c.rank <= 14 else 0
        mat[ri, si] = min(mat[ri, si] + 1, 2)
    return mat

NUM_COMBO_TYPES = 17  # ComboType enum has values 0..16

def encode_state(env, player):
    """
    State features for player (relative perspective).
    """
    opp_L = (player - 1) % 4  # left = previous in CCW order
    partner = (player + 2) % 4
    opp_R = (player + 1) % 4  # right = next in CCW order

    hand       = cards_to_matrix(env.hands[player]).flatten()        # 60
    played_me  = cards_to_matrix(env.played[player]).flatten()       # 60
    played_par = cards_to_matrix(env.played[partner]).flatten()      # 60
    played_opL = cards_to_matrix(env.played[opp_L]).flatten()        # 60
    played_opR = cards_to_matrix(env.played[opp_R]).flatten()        # 60
    
    # Cards unaccounted for
    all_played = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[player] | all_played
    unknown = set(make_deck()) - known
    remaining = cards_to_matrix(unknown).flatten()                   # 60
    
    # Card counts (normalized, relative order: me, right, partner, left)
    counts = np.array([len(env.hands[(player + i) % 4]) / 27.0
                       for i in range(4)], dtype=np.float32)         # 4
    
    # Out flags (relative order)
    out_flags = np.array([float(env.is_out[(player + i) % 4])
                          for i in range(4)], dtype=np.float32)      # 4
    
    # Level rank (one-hot over 13 normal ranks: 2..A)
    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0                               # 13
    
    # Wild cards in hand
    wilds_in_hand = sum(1 for c in env.hands[player]
                        if is_wild(c, env.level_rank))
    wild_flags = np.array([float(wilds_in_hand >= 1),
                           float(wilds_in_hand >= 2)], dtype=np.float32)  # 2
    
    # Current trick info
    is_leader = np.array([float(env.current_trick is None)], dtype=np.float32)  # 1
    trick_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)      # 17
    trick_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)             # 15
    trick_is_bomb = np.array([0.0], dtype=np.float32)                # 1
    if env.current_trick is not None:
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_is_bomb[0] = float(env.current_trick.type in BOMB_TYPES)
    
    return np.concatenate([
        hand, played_me, played_par, played_opL, played_opR,  # 300
        remaining, counts, out_flags,                          # 68
        level_oh, wild_flags, is_leader,                       # 16
        trick_type_oh, trick_key_oh, trick_is_bomb             # 33
    ])
    # Total: ~417 dims

def encode_action(combo, hand=None):
    """Action features."""
    cards_played = cards_to_matrix(combo.cards).flatten()              # 60
    combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)       # 17
    combo_type_oh[combo.type] = 1.0
    combo_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)              # 15
    if 2 <= combo.key <= 17:
        combo_key_oh[_rank_index(combo.key)] = 1.0
    num_cards = np.array([len(combo.cards) / 10.0], dtype=np.float32) # 1
    is_bomb = np.array([float(combo.type in BOMB_TYPES)], dtype=np.float32) # 1
    wild_oh = np.zeros(3, dtype=np.float32)                           # 3
    wild_oh[min(combo.wild_count, 2)] = 1.0
    
    return np.concatenate([
        cards_played, combo_type_oh, combo_key_oh,
        num_cards, is_bomb, wild_oh
    ])
    # Total: ~97 dims
```

**Hour 6: `q_network.py` + `replay.py`** — Same MLP and buffer as before, just adjust dims:

```python
# q_network.py
class QNetwork(nn.Module):
    def __init__(self, d_state=417, d_action=97, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state + d_action, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)
```

**Hour 7: `train.py`** — Identical DMC loop, 4 players instead of 2:

```python
def play_episode(env, q_net, epsilon):
    env.reset()
    transitions = {p: [] for p in range(4)}
    
    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        state_enc = encode_state(env, player)
        action_encs = np.array([encode_action(m) for m in legal])
        
        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                s = torch.tensor(state_enc).unsqueeze(0).expand(len(legal), -1)
                a = torch.tensor(action_encs)
                idx = q_net(s, a).argmax().item()
        
        transitions[player].append((state_enc, action_encs[idx]))
        env.step(legal[idx])
    
    # Terminal rewards → same return for all of that player's transitions
    rewards = env.get_rewards()
    all_trans = []
    for player, tlist in transitions.items():
        G = rewards[player]
        for (s, a) in tlist:
            all_trans.append((s, a, G))
    return all_trans
```

### Block 3 (Hours 8–10): Train + Debug + Evaluate

- Run training: 20K–50K episodes on CPU
- Expect ~1–3 episodes/second (Python movegen is slow with wilds — that's OK)
- **Overnight run**: kick off 50K episodes before bed, check in morning
- Evaluate: Q-agent (all 4 seats) vs random (all 4 seats), measure team winrate

**What you should see**:
```
Ep  1000 | ε=0.28 | buf=35000 | WR vs random: 52%   ← noise
Ep  5000 | ε=0.24 | buf=100K  | WR vs random: 58%   ← signal
Ep 20000 | ε=0.14 | buf=100K  | WR vs random: 68%   ← learning
Ep 50000 | ε=0.05 | buf=100K  | WR vs random: 75%+  ← target
```

**If winrate is stuck at 50%**: The most likely cause is a movegen bug, especially in wild card handling. Run `test_all_combo_types_appear()` and manually verify 20 random games by printing every move.

### Hours 10–12: Stretch Goals (Pick One)

1. **Add `remaining_after_play`** to action encoding (Section 3.2 of full plan)
2. **Add heuristic agent** and eval against it
3. **Profile movegen** and identify the hot path for future Cython port
4. **Add lead/follow network split** (2 Q-networks)

---

## 3. Project Structure

```
chucking-eggs/
├── cards.py          # Card, Rank, Suit, ComboType, level_order_key, wild detection (~80 lines)
├── combos.py         # Combo class, beats(), ALL movegen functions (~400-500 lines)
│                     #   Hardest file: 9 bomb tiers, wild substitution, ace-low sequences,
│                     #   straight vs straight-flush routing, deduplication
├── game.py           # GuanDanEnv: 4-player CCW game loop (~200 lines)
├── encoding.py       # state/action → tensors (~100 lines)
├── q_network.py      # MLP Q(s,a) (~20 lines)
├── replay.py         # Circular buffer (~30 lines)
├── train.py          # DMC loop: play + learn + eval (~100 lines)
├── heuristic.py      # Rule-based baseline (~40 lines)
└── test_game.py      # Engine correctness tests (~120 lines)
```

**Total**: ~1090–1190 lines. The bulk (~500 lines) is in `combos.py` because movegen with wilds, ace-low sequences, 9 bomb tiers, and straight/straight-flush disambiguation is genuinely complex.

---

## 4. Honest Time Estimate

| Block | Hours | What | Risk |
|-------|-------|------|------|
| Game engine (cards + combos + game loop) | 5–6 | Correct Guan Dan with all combos, wilds, 9 bomb tiers, ace-low sequences, CCW play | **High**: wild card movegen + ace-low + straight/SF routing. Budget extra time. |
| Engine tests | 1–1.5 | Random games, combo coverage, team balance, bomb hierarchy, direction | Gate: don't proceed until passing |
| Encoding + Q-net + DMC loop | 2 | Wire DMC to real engine | Low: same algorithm, different dims |
| Training + debug | 2–3 | Run, observe, fix | Medium: slow episodes on CPU |
| **Total** | **10–12.5** | | |

The game engine is 50%+ of the work. The corrected rules add complexity vs. the previous version: 9 bomb tiers (not 5), ace-low sequences, exact sizes for tubes/plates, straight vs. straight-flush disambiguation, and counterclockwise play. This is the right investment — this engine carries forward unchanged.

---

## 5. Scaling Path (After MVP Day)

Once the MVP is working, add upgrades **one at a time**, measuring the impact of each:

```
Day 1 (today):  Full rules + MLP + single process + ε-greedy
                 └→ WR vs random: ~70-75%

Day 2:           + remaining_after_play encoding
                 + lead/follow network split  
                 + heuristic eval opponent
                 └→ WR vs heuristic: ~55-60%

Day 3:           + LSTM history encoder (replace MLP portion)
                 + larger buffer (500K) + larger batch (1024)
                 └→ WR vs heuristic: ~65%

Day 4:           + Cython movegen (10× speedup → 10× more episodes)
                 + soft-start (heuristic mixing)
                 └→ WR vs heuristic: ~75%

Day 5-7:         + Modal deployment (32 actors + GPU)
                 + ResNet blocks
                 + 2M buffer
                 → Full plan from original blueprint

Week 2-3:        + Oracle guiding (PTIE)
                 + Hand prediction
                 → Human-competitive play
```

Each step is a single PR-sized change with a measurable before/after.