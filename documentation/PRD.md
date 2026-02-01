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
4. **Full House** - Triple + Pair
5. **Straight** - 5+ consecutive cards (any suits)
6. **Tube** - 3+ consecutive pairs
7. **Plate** - 2+ consecutive triples

#### Bombs (9 types, lowest to highest)
1. Quadruple (4 of a kind)
2. Quintuple (5 of a kind)
3. Straight Flush (5+ consecutive, same suit)
4. Sextuple through Decuple (6-10 of a kind)
5. Four-Joker Bomb (2 red + 2 black) - HIGHEST

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
