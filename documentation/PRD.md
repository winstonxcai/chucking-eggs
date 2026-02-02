# Chucking Eggs - Product Requirements Document (PRD)

**Version:** 1.0  
**Date:** February 2026  
**Status:** Final - Ready for Engineering

---

## 1. EXECUTIVE SUMMARY

**Product Name:** Chucking Eggs (掼蛋)  
**Platform:** iOS (React Native + Expo)  
**Target Users:** Guan Dan players (assumes rules knowledge)  
**Core Value Prop:** Fast, friction-free multiplayer Guan Dan with friends  
**MVP Scope:** 4-player games, no AI, no social features beyond room codes

---

## 2. PRODUCT VISION & PHILOSOPHY

### Vision
A minimal, fast Guan Dan card game for iOS where players with existing rules knowledge can immediately jump into social multiplayer games with friends - no tutorials, no distractions, pure gameplay.

### Core Philosophy
- **Assume Competence:** Players know the rules; no onboarding needed
- **Speed First:** Get in a game in <2 minutes
- **Multiplayer Only:** No single-player training wheels (AI)
- **Friends-Based:** Room codes for direct play, no matchmaking
- **Zero Communication:** Players are together or on video call already

---

## 3. GAMEPLAY REQUIREMENTS

### 3.1 Game Structure
- **Mode:** Multiplayer only (4 players, 2 fixed partnerships)
- **Players:** 4 humans (no AI opponents)
- **Team Assignment:** Player-chosen at lobby (North/South or East/West)
- **Session Duration:** ~5-15 minutes per hand; multiple hands per session
- **Cards:** 2 standard decks + 4 jokers (108 total)
- **Hand Size:** 27 cards per player

### 3.2 Core Rules (Reference)

#### Card Rankings (per level)
- Red Joker > Black Joker > Level Card > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3 > 2

#### Playable Combinations (7 types)
1. **Single** - One card
2. **Pair** - Two same rank
3. **Triple** - Three same rank
4. **Full House** - Triple + Pair (5 cards)
5. **Straight** - Exactly 5 consecutive cards (any suits)
6. **Tube** - Exactly 6 cards (3 consecutive pairs)
7. **Plate** - Exactly 6 cards (2 consecutive triples)

**Note:** See Section 3.4 for complete rule specifications and edge cases.

#### Bombs (9 types, lowest to highest)
1. Quadruple (4 of a kind)
2. Quintuple (5 of a kind)
3. Straight Flush (exactly 5 consecutive, same suit)
4. Sextuple through Decuple (6-10 of a kind)
5. Four-Joker Bomb (2 red + 2 black) - HIGHEST

**Note:** See Section 3.4.3 for complete bomb specifications including wild card usage and level card limitations.

#### Wild Cards
- Hearts of current level rank only
- Can substitute for any card EXCEPT jokers
- Validator automatically recognizes

#### Scoring/Promotion
- **1-4 win:** +1 level (4 gives card to 1)
- **1-3 win:** +2 levels (4 gives card to 1)
- **1-2 win (双上):** +3 levels (4 gives to 1, 3 gives to 2)
- First hand: Always level 2

#### Tribute Card Distribution (Who Gives to Whom)
- **1-4 win:** 4th place player gives card to 1st place player
- **1-3 win:** 4th place player gives card to 1st place player
- **1-2 win:** 4th place player gives to 1st place, 3rd place player gives to 2nd place

#### Tribute Leader Selection (Who Goes First After Tribute)
- **Leader is determined by who gave the highest card**
- If 4 gave ♠K and 3 gave ♠Q → Player 4 goes first
- If both gave same card (e.g., both gave ♠K) → Randomly choose one to go first

#### Tribute System (2nd hand onwards)
- Losers give highest non-wild card to winners
- Winners return unwanted cards in exchange
- Exception: If players hold both red jokers, tribute cancelled
- Leader selection depends on tribute status (see 3.3)

### 3.3 Leader Selection (Tribute-Based) ✅

**First Hand:**
- Random player chosen by server

**Subsequent Hands (After Tribute):**
- **Leader determined by who gave the highest tribute card**
- If 4 gave ♠K and 3 gave ♠Q → Player 4 leads (highest card given)
- If both gave identical cards → Randomly select one to lead
- This replaces the old "winner leads" rule

**For 1-2 Win Scenario:**
- Player 4 gives card to Player 1, Player 3 gives card to Player 2
- Compare: if 4's card > 3's card → Player 4 leads
- If equal → Random selection

### 3.4 Complete Game Rules Reference (Detailed)

This section provides comprehensive rule specifications for implementation. All edge cases and ambiguities resolved.

#### 3.4.1 Card Rankings & Special Cards

**Level Cards:**
- ALL 8 cards of current level rank are "level cards"
- When level is 7: all eight 7s (♠7 ♠7 ♥7 ♥7 ♦7 ♦7 ♣7 ♣7 from both decks)
- Exception: ♥7 are ALSO wild cards (dual role)
- Level cards rank at 98 (numeric ranking)
- Ranking: Red Joker (100) > Black Joker (99) > Level Card (98) > A (14) > K (13)...
- Single level card BEATS single Ace
- Level cards are the 3rd strongest cards in the game

**Wild Cards (Hearts of Current Level):**
- Hearts of current level rank have dual role: wild card AND level card
- Context determines usage:
  - In triple 7s when level is 7: ♥7 acts as level card
  - In triple Kings with ♥7: ♥7 acts as wild-as-King
- Player doesn't explicitly declare - game engine infers from combination context
- Wild cards can substitute for any card EXCEPT jokers
- Can be used in ALL combo types (singles, pairs, triples, straights, tubes, plates, bombs)
- Wild cards CANNOT be given as tribute (even if highest card)

**Level Cards in Combinations:**
- In straights: Use natural rank position (level 2 → 2 is rank 2 in straight)
- In pairs/triples/bombs: Use level card power (rank 98)
- Context-dependent usage based on combination type

#### 3.4.2 Combination Rules (Exact Specifications)

**1. Single**
- One card of any rank
- Beats: Higher rank single only
- Comparison: By card rank

**2. Pair**
- Two cards of same rank
- Beats: Higher rank pair only
- Comparison: By rank of the pair
- Wild cards allowed to complete pair

**3. Triple**
- Three cards of same rank
- Beats: Higher rank triple only
- Comparison: By rank of the triple
- Wild cards allowed to complete triple

**4. Full House**
- Format: Triple + Pair (5 cards total)
- Example: 8-8-8-K-K
- Beats: Higher rank full house only
- Comparison: By TRIPLE rank only (pair rank irrelevant)
  - 8-8-8-K-K beats 7-7-7-A-A (compare triples: 8 > 7)
- Wild cards allowed in both triple and pair portions

**5. Straight**
- Length: EXACTLY 5 cards (not 5+, not variable)
- Consecutive ranks, any suits
- Beats: Higher rank straight only (by highest card)
- Comparison: By highest card in straight
  - 10-J-Q-K-A (highest straight possible) beats A-2-3-4-5
- Ace flexibility: Can be HIGH (10-J-Q-K-A) OR LOW (A-2-3-4-5)
- NO wrapping: K-A-2-3-4 is INVALID
- Wild cards allowed
- Level cards use natural rank position when in straights

**6. Tube**
- Definition: EXACTLY 6 cards (3 consecutive pairs)
- Example: 3-3, 4-4, 5-5
- FIXED LENGTH (cannot be 4 pairs or 5 pairs)
- Beats: Higher rank tube only (by highest pair)
- Comparison: By highest pair in tube
- Wild cards allowed

**7. Plate**
- Definition: EXACTLY 6 cards (2 consecutive triples)
- Example: 5-5-5, 6-6-6
- FIXED LENGTH (cannot be 3 triples or more)
- Beats: Higher rank plate only (by highest triple)
- Comparison: By highest triple in plate
- Wild cards allowed

#### 3.4.3 Bomb Rules (Complete Specifications)

**Bomb Hierarchy (Lowest to Highest):**
1. Quadruple (4 of a kind)
2. Quintuple (5 of a kind)
3. Straight Flush (5 consecutive, same suit)
4. Sextuple (6 of a kind)
5. Septuple (7 of a kind)
6. Octuple (8 of a kind)
7. Nonuple (9 of a kind)
8. Decuple (10 of a kind)
9. Four-Joker Bomb (2 red + 2 black) - HIGHEST

**Bomb Comparison Rules:**
- Same type: Higher card rank wins (Quad 9s > Quad 5s)
- Different type: Bomb tier wins (Quintuple > Quadruple always)
- Bombs beat ALL non-bombs regardless of type or strength

**Bombs with Wild Cards:**
- Wild cards CAN complete bombs
- ♠9 ♥9 ♦9 ♥7(wild) = Valid quadruple bomb
- Same power as natural quadruple (no penalty)
- Wild cards universally applicable to all bomb types

**Straight Flush Bombs:**
- Length: EXACTLY 5 cards (not variable)
- 5 consecutive ranks, all same suit
- Wild cards CAN cross suits (♥7-wild can become ♠7 for straight flush)
- Rank #3 in bomb hierarchy (above quintuple, below sextuple)
- Comparison: By highest card if same length

**Level Card Bombs (Important Limitation):**
- Level cards CAN form bombs
- If level is 7, four 7s = valid quadruple bomb
- **Maximum with level cards: 8-bomb (octuple)**
  - Reason: 6 natural level cards (non-heart suits) + 2 wild ♥ level cards = 8 total
  - Wild ♥ level cards counted once (not double-counted as both wild and level)
- **For nonuple (9) and decuple (10): MUST use NON-level cards**
  - Example at level 2: All Aces (8 cards) + both ♥2 wilds = 10-bomb (decuple Aces)
  - Cannot make 9-10 bomb with level cards themselves

#### 3.4.4 Beating & Comparison Rules (Core Mechanic)

**Fundamental Rule: MUST match combination type to beat**
- Pair can ONLY beat pair (with higher rank)
- Triple can ONLY beat triple (with higher rank)
- Straight can ONLY beat straight (with higher highest card)
- Full house can ONLY beat full house (with higher triple)
- Tube can ONLY beat tube (with higher highest pair)
- Plate can ONLY beat plate (with higher highest triple)

**EXCEPTION: Bombs beat ALL non-bombs**
- ANY bomb beats ANY non-bomb regardless of type or strength
- Bomb vs bomb: Compare by bomb tier first, then card rank if same tier

**Same Rank Ties:**
- If two players play identical rank combos (e.g., both triple 8s)
- LAST play always wins
- No suit hierarchy needed (not implemented in MVP)

#### 3.4.5 Finish Order & Round Scoring

**Finish Order Determination:**
- Based on ORDER cards were played, NOT trick winner
- P1 plays last cards → P1 finishes first (even if beaten by P2's bomb)
- P2 beats with bomb using last cards → P2 finishes second
- Plays CAN be beaten after someone finishes (doesn't affect finish order)
- Order recorded by timestamp when last card leaves hand

**Level Progression Scoring:**
- First place team ALWAYS wins the round
- Scoring based on finish pattern:
  - **1-2 finish** (same team 1st & 2nd): +3 levels
  - **1-3 finish** (same team 1st & 3rd): +2 levels
  - **1-4 finish** (same team 1st & 4th): +1 level
- No other patterns possible (round ends when one team gets 1st & 2nd)

**Round End Conditions:**
- Round ends IMMEDIATELY when ONE TEAM finishes 1st & 2nd
- If NS team gets positions 1 & 2 → round ends, game freezes
- Remaining EW players don't continue playing
- 3rd/4th order determined later via tribute card comparison (see Tribute System)

**Teams Start at Same Level:**
- Both partnerships always start at level 2
- Levels can diverge throughout game
- No catch-up mechanics (intended)
- Game ends when one team reaches and WINS at Ace level

#### 3.4.6 Tribute System (Complete Rules)

**Tribute Exchange Rules:**
- Losers give HIGHEST NON-WILD card
- Wild ♥ level cards can NEVER be given as tribute (even if highest card in hand)
- Winners can return ANY cards (including tribute cards themselves)
- No restrictions on which cards winners return

**Tribute Distribution:**
- **1-4 finish:** 4th gives to 1st (one exchange)
- **1-3 finish:** 4th gives to 1st (one exchange)
- **1-2 finish:** 4th gives to 1st, 3rd gives to 2nd (two exchanges)

**Tribute Cancellation:**
- If ONE loser has both red jokers (🔴🔴) → ALL tributes cancelled
- Applies to entire tribute phase (both exchanges in 1-2 win)
- Joker power hierarchy: Red jokers (100) > Level cards (98)
- If cancelled: Skip directly to leader selection

**3rd/4th Place Determination (1-2 Finish Only):**
- For 1-2 finish: Round ends before 3rd/4th finish naturally
- 3rd/4th determined BY TRIBUTE CARDS (strategic paradox):
  - Higher tribute card given = 4th place (worse finish)
  - Lower tribute card given = 3rd place (better finish)
  - If tied: Random selection
- **Strategic dilemma:** Must give highest card (rule) but that makes finish worse
- **Rule is intentional:** Creates interesting endgame decisions

**Tribute Seating (Physical Clockwise):**
- "Clockwise" means physical seat positions
- Seats: P1(North), P2(East), P3(South), P4(West)
- Clockwise order: N → E → S → W → N
- Used for determining turn order and game flow

**Leader After Tribute:**
- Leader = player who gave HIGHEST tribute card
- Compare tribute cards using normal card ranking
- If tribute cancelled: Previous round winner leads (or random if Round 1)
- If tied tribute cards: Random selection between tied players

**Tribute Timing:**
- Tribute happens BETWEEN rounds, as first phase of next round
- Flow: Round N ends → Show results → Tribute exchange → Determine leader → Deal Round N+1

#### 3.4.7 Game Flow & Edge Cases

**Leader When Trick Winner Finishes:**
- Winner finishes last cards → next player CAN play on top of winner's hand
- If BOTH opponents pass → winner's teammate leads next trick
- Leadership passes within team if winner is out
- Clockwise rotation continues until someone leads

**Timer Timeout on Leader:**
- When leading player times out: Auto-play LOWEST SINGLE CARD
- Leader cannot "pass" when leading (no current trick to pass on)
- Ensures game always progresses
- Applies 30-second timer rule

**Pass Count Reset:**
- ANY play resets consecutive pass count to 0
- Example: P2 passes (1) → P3 passes (2) → P4 plays bomb → count resets to 0
- 3 consecutive passes ends trick
- Pass count is NOT cumulative across plays

**Team Persistence:**
- Teams are PERMANENT for entire game session (all rounds to Ace)
- Cannot switch teams between rounds
- Partnerships are locked at lobby
- If player disconnects, game cancels (no team substitution)

**Level Affects Card Ranking:**
- Current level determines which cards are "level cards" (rank 98)
- When level is 7: All 7s become powerful (rank 98, beat Aces)
- When level is 2: All 2s become powerful (rank 98, beat Aces)
- When level is A: All Aces remain at rank 14 (NOT level cards, since there's no higher rank)
- Level cards are context-dependent: powerful in pairs/triples, natural rank in straights

**Wild Card Substitution Mechanics:**
- Wild card value INFERRED from combination context
- ♠K ♥K ♥7(wild) = Automatically triple Kings
- No explicit declaration needed by player
- Game engine determines substitution from surrounding cards
- If ambiguous (shouldn't happen in valid combos), validation fails

**Pass Mechanics:**
- Passing means "skip THIS trick only"
- NOT "skip rest of round"
- Can play on NEXT trick after passing
- Pass is always available except when leading
- One-tap confirmation: "Are you sure?"

---

## 4. FEATURE REQUIREMENTS

### 4.1 Multiplayer & Rooms

#### Room Creation
- Generate 6-character alphanumeric room code (e.g., "F8E6FC")
- Display code prominently with copy button
- Host waits for 3 other players

#### Room Joining
- Text input field for room code
- Validation: Code exists? Room full? Game in progress?
- Clear error messages

#### Lobby
- Show 4 player slots (name + team selection)
- Team selection: Radio buttons (North/South vs East/West)
- Prevent multiple players on same team
- Start button enabled only when all 4 ready

#### Team Selection
- Players manually choose North/South or East/West
- Cannot select same team as other players
- All teams must be assigned before game starts

### 4.2 Disconnect & Reconnection ✅

- **Disconnect Detection:** Player AFK after 30 seconds
- **Reconnect Window:** 30 seconds to rejoin
- **If Reconnects:** Game resumes from exact state
- **If Timeout:** **Game cancelled immediately**
  - All players returned to lobby
  - Notification: "[Player X] disconnected - game cancelled"
  - Session not saved

### 4.3 Game Actions

#### Turn Timer
- **Duration:** 30 seconds per turn (poker now style UI)
- **Time Additions:** +5 seconds each (unlimited additions)
- **Penalty:** None in MVP (simplifies logic and reduces new player frustration)

#### Valid Plays & Bombs
- **Bombs can only be played on your turn** (simplifies multiplayer logic)
  - Bombs are treated as special combinations you can play when responding
  - Like all plays, must wait for your turn in trick rotation
  - Bombs beat all non-bomb combos and lower-ranked bombs
- Auto-detect valid combinations from selected cards
- Show "Invalid combination" message if illegal
- "Play" button disabled until valid combo selected
- "Pass" button always available (with one-tap confirmation)

#### Game Flow
1. Dealing animation (3 seconds)
2. Player 1 leads with any valid combo
3. Players 2-4 respond in order
4. Continue until 3 consecutive passes
5. Winner of trick leads next
6. Repeat until 2+ players have no cards left
7. Calculate finish order (1st, 2nd, 3rd, 4th)
8. Show round results
9. Process tribute (if applicable)
10. Select leader for next round
11. Repeat

### 4.4 What's NOT in MVP

| Feature | Reason |
|---------|--------|
| AI Opponents | Multiplayer-only focus |
| Chat | Players are together/on video |
| Emojis/Reactions | Not needed for MVP |
| Game History | Minimal retention focus |
| Leaderboards | Friends-only play |
| Achievements | Out of scope |
| Dark Mode | Nice-to-have only |
| Landscape Mode | Portrait only |
| Accessibility Features | Post-launch |

---

## 5. SUCCESS CRITERIA

### Technical Correctness
- ✅ All 7 combinations validate correctly
- ✅ All 9 bombs rank correctly
- ✅ Wild cards work in any valid combo
- ✅ Tribute system enforces rules
- ✅ Level progression calculates correctly
- ✅ Leader selection matches rules

### Multiplayer Reliability
- ✅ 4 phones sync in real-time (<500ms latency)
- ✅ Disconnect handled cleanly (30-sec window)
- ✅ Network errors don't crash game
- ✅ 99.9% uptime target

### User Experience
- ✅ Room creation/joining takes <30 seconds
- ✅ Game start after all 4 ready: <10 seconds
- ✅ Turn time (select + play): <30 seconds
- ✅ Card selection UX feels natural
- ✅ No "stuck" states (always a valid action)

---

## 6. TESTING STRATEGY

### Game Logic Testing
- Unit tests for all combinations
- Unit tests for bomb ranking
- Unit tests for wild card validation
- Unit tests for tribute logic
- Unit tests for level progression
- Edge case: simultaneous hand completion

### Multiplayer Testing
- Room creation/joining (success & error cases)
- Real-time sync between 4 phones
- Disconnect/reconnect within 30 seconds
- Network latency handling
- Team assignment validation

### User Testing (Beta Phase)
- Recruit 20-30 existing Guan Dan players
- Test real multiplayer games
- Collect feedback on card selection UX, timer settings, clarity
- Validate game flow matches rules expectations

---

## 7. TIMELINE & ROADMAP

### MVP (v1.0) - 4 weeks
- Phase 1-3: Game engine & core logic
- Phase 4: UI components
- Phase 5: Multiplayer integration
- Phase 6: Polish & testing
- Phase 7: TestFlight beta

### Post-MVP (v1.1+)
- Sound effects & haptics
- Dark mode
- Game history/stats
- Landscape orientation
- Android version
- Advanced features (chat, leaderboards, etc.)

---

## 8. OPEN QUESTIONS RESOLVED ✅

| Question | Decision |
|----------|----------|
| Multiplayer or AI? | Multiplayer only |
| Onboarding tutorial? | None (assumes rules knowledge) |
| Card sorting? | Auto-sorted by rank (default) |
| Straight detector? | 4 suit buttons with cycling |
| Time additions? | +5 seconds, unlimited |
| Time penalties? | None (MVP) |
| Card grouping? | Client-side only, persist until hand ends |
| Previous tricks display? | None (not shown) |
| Disconnect handling? | Game cancels after 30s timeout |
| Leader selection? | Based on highest tribute card given |
| Authentication? | Anonymous with display name only |
| Bomb timing? | Only on your turn (not anytime) |
| Wild card display? | Gold border on card |
