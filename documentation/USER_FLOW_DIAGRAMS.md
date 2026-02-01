# Chucking Eggs - User Flow Diagrams

All flows use ASCII diagrams. For detailed visual mockups, these can be converted to Figma/design tools.

---

## 1. MAIN USER FLOWS

### 1.1 First-Time User Flow (Home → Create Room → Game)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CHUCKING EGGS HOME                          │
│                                                                     │
│                    [🥚] Chucking Eggs [🥚]                          │
│                         掼 蛋                                         │
│                                                                     │
│                     ┌──────────────────┐                           │
│                     │  Create New Room │                           │
│                     └────────┬─────────┘                           │
│                              │                                      │
│                     ┌────────▼─────────┐                           │
│                     │  Join Room Code  │                           │
│                     └──────────────────┘                           │
│                                                                     │
│                     ┌──────────────────┐                           │
│                     │    Settings      │                           │
│                     └──────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [Create New Room]
                              │
                ┌─────────────▼──────────────┐
                │   ROOM CREATION SCREEN     │
                │                            │
                │  Room Code: F8E6FC         │
                │  [Copy to Clipboard]       │
                │                            │
                │  Waiting for players...    │
                │                            │
                │  🦊 You (Player 1)         │
                │  ⏳ Waiting for Player 2   │
                │  ⏳ Waiting for Player 3   │
                │  ⏳ Waiting for Player 4   │
                │                            │
                └────────────┬───────────────┘
                             │
                    [3 players join]
                             │
         ┌───────────────────▼────────────────────┐
         │      LOBBY SCREEN - TEAM SELECTION    │
         │                                       │
         │  ┌─ NORTH/SOUTH ─┐ ┌─ EAST/WEST ─┐  │
         │  │                │ │              │  │
         │  │ 🦊 You (✓ NS)  │ │ 🐼 Player 2 │  │
         │  │                │ │ (✓ EW)      │  │
         │  │ ⏳ Player 3     │ │              │  │
         │  │ (ns/ew?)       │ │ ⏳ Player 4  │  │
         │  │                │ │ (ns/ew?)    │  │
         │  └────────────────┘ └──────────────┘  │
         │                                       │
         │  [Start Game] (disabled until 4/4)   │
         │                                       │
         └───────────────────┬────────────────────┘
                             │
                    [All teams assigned]
                             │
         ┌───────────────────▼────────────────────┐
         │    CARDS DEALING + ANIMATION          │
         │                                       │
         │         Dealing 27 cards...           │
         │                                       │
         │              ▌▌▌ ▌▌▌ ▌▌▌             │
         │                                       │
         │    (3 second animation)               │
         │                                       │
         └───────────────────┬────────────────────┘
                             │
         ┌───────────────────▼────────────────────┐
         │          GAME BOARD - YOUR TURN       │
         │                                       │
         │         ┌─ GAME STATE ─┐             │
         │         │ Level: 7 ♥️   │             │
         │         │ Random leader │             │
         │         │ selected      │             │
         │         └───────────────┘             │
         │                                       │
         │         [GAME BOARD LAYOUT]           │
         │                                       │
         │         Your hand: 27 cards           │
         │         [Play] [Pass] [+5s]           │
         │         ⏱️ 30s                         │
         │                                       │
         └───────────────────┬────────────────────┘
                             │
                   [Game proceeds]
                             │
```

---

### 1.2 Existing User Flow (Home → Join Room → Game)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CHUCKING EGGS HOME                          │
│                                                                     │
│                    [🥚] Chucking Eggs [🥚]                          │
│                                                                     │
│                     ┌──────────────────┐                           │
│                     │  Create New Room │                           │
│                     └──────────────────┘                           │
│                                                                     │
│                     ┌──────────────────┐                           │
│                     │  Join Room Code  │◄────── [User taps]       │
│                     └──────────────────┘                           │
│                                                                     │
│                     ┌──────────────────┐                           │
│                     │    Settings      │                           │
│                     └──────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                      [Join Room Code]
                              │
                ┌─────────────▼──────────────┐
                │    JOIN ROOM SCREEN        │
                │                            │
                │  Enter Room Code:          │
                │  ┌──────────────────────┐  │
                │  │ F8E6FC               │  │
                │  └──────────────────────┘  │
                │                            │
                │      [Join Room]           │
                │                            │
                └────────────┬───────────────┘
                             │
                   [Validate code]
                             │
         ┌─────────No─────────┤─────────Yes────────┐
         │                    │                    │
    ┌────▼────────┐      [Valid]        ┌──────────▼──────────┐
    │Error Screen │                     │  LOBBY SCREEN      │
    │             │                     │                    │
    │ "Room not   │                     │ Waiting room has:  │
    │  found"     │                     │ 🦊 Player 1 (NS)   │
    │             │                     │ 🐼 Player 2 (EW)   │
    │ [Back]      │                     │ 🦁 Player 3 (EW)   │
    │             │                     │ ⏳ Waiting for     │
    └─────────────┘                     │    Player 4...     │
                                        │                    │
                                        │ Team Selection:    │
                                        │ ◉ North/South      │
                                        │ ○ East/West        │
                                        │                    │
                                        │ [Ready]            │
                                        │                    │
                                        └──────────┬─────────┘
                                                   │
                                        [Select team & Ready]
                                                   │
                                        ┌──────────▼──────────┐
                                        │ Waiting for all     │
                                        │ players to ready    │
                                        │ (3/4 ready)         │
                                        │                    │
                                        │ [Cancel]            │
                                        └──────────┬──────────┘
                                                   │
                                        [All 4 ready]
                                                   │
                                        ┌──────────▼──────────┐
                                        │  GAME STARTS        │
                                        │  [Same as above]    │
                                        │                    │
                                        └────────────────────┘
```

---

## 2. CORE GAMEPLAY FLOWS

### 2.1 Turn Flow (Single Player's Perspective)

⚠️ **IMPORTANT RULE - Bombs on Your Turn:**
Bombs can ONLY be played on your turn, like any other combination. This simplifies multiplayer synchronization and prevents race conditions. Bombs are powerful plays that beat most combos, but must wait for your turn in the trick rotation.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      YOUR TURN - GAME BOARD                         │
│                                                                     │
│  Level: 7 ♥️  |  Previous: You won (2-card play)                   │
│                                                                     │
│  Player 2 (8 cards)          Player 3 (12 cards)                   │
│                                                                     │
│  Player 4 (15 cards)    [CURRENT TRICK]          [PREV TRICKS]     │
│                         Player 1 led with        1. [7♠][7♦][7♣]  │
│                         [7♠][7♦][7♣]            2. [8♠][8♠][8♠]  │
│                                                                     │
│                         [YOUR PLAY]                                │
│                         Must beat 3-card combo                     │
│                         (7-high)                                   │
│                                                                     │
│  YOUR HAND (27 CARDS - SORTED BY RANK):                           │
│                                                                     │
│  [Straight Finder] [Group by Suit]                                │
│                                                                     │
│  ♠A  ♠K  ♠Q  ♠J  ♠10  ♠9  ♠8  ♠7  ♠6  ...  2♦  2♣              │
│                                                                     │
│  [Selected: ♠8 ♦8 ♣8]   (Triple - beats 7s)                       │
│                                                                     │
│  [Clear] [Play (3 cards)] [Pass]  [+5s]                          │
│                                                                     │
│  ⏱️ 15 seconds remaining  [Time additions used: 2]                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [User selects cards]
                              │
         ┌───────────────────┬────────────────────┐
         │                   │                    │
    [Valid combo]      [Invalid combo]      [Pass selected]
         │                   │                    │
    ┌────▼────┐         ┌────▼────┐       ┌──────▼──────┐
    │ Play     │         │ Error   │       │Confirm Pass │
    │ (3 cards)│         │Message: │       │             │
    │ Button   │         │ "Need 3 │       │ "Are you    │
    │ enabled  │         │  of     │       │  sure?"     │
    │          │         │  same   │       │             │
    │[Play]    │         │ rank"   │       │ [Yes] [No]  │
    │          │         │         │       │             │
    └────┬─────┘         └─────────┘       └──────┬──────┘
         │                                        │
      [User clicks]                            [Yes]
         │                                        │
    ┌────▼──────────────────────────────────────▼────┐
    │        PLAY SUBMITTED TO SERVER                │
    │                                                │
    │ Animation: Your 3 cards fly to center pile    │
    │ Server validates + broadcasts to other       │
    │ players                                       │
    │                                                │
    └────────────────────┬──────────────────────────┘
                         │
            [Next player's turn starts]
                         │
         ┌───────────────▼────────────────┐
         │   WAIT FOR OTHER PLAYERS       │
         │                                │
         │   Player 2's turn (30s timer)  │
         │   [Pass] [Plays ♦8 ♣8 ♠8]     │
         │                                │
         │   Result: Beats your triple!   │
         │                                │
         │   Current Trick (Player 2 has):│
         │   [♦8][♣8][♠8]                │
         │                                │
         └───────────────┬────────────────┘
                         │
            [Continue around table]
                         │
              [Your turn comes again]
                         │
    ┌────────────────────▼──────────────────────┐
    │    NEXT ACTION: Pass or Play Higher?      │
    │                                           │
    │    Current trick: [♦8][♣8][♠8]           │
    │    (Player 2 playing 3 of a kind - 8s)   │
    │                                           │
    │    Your options:                         │
    │    - Play 4-of-a-kind (quad bomb)        │
    │    - Play straight flush                 │
    │    - Play higher 3-of-a-kind             │
    │    - Pass                                │
    │                                           │
    │    [Straight Finder] highlighted         │
    │    ♠9 ♦9 ♣9 highlighted (beats 8s)      │
    │                                           │
    │    [Clear] [Play (3)] [Pass] [+5s]       │
    │    ⏱️ 50s                                 │
    │                                           │
    └────────────────────┬──────────────────────┘
                         │
                    [User action]
                         │
```

---

### 2.2 Straight Flush Detector Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   YOUR HAND (27 CARDS)                              │
│                                                                     │
│  ♠A ♠K ♠Q ♠J ♠10 ♠9 ♠8 ♠7 ♠6 ♠5 ♠4 ... 2♦ 2♣                   │
│                                                                     │
│  [Straight Finder] ◄── User taps this button                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [Button clicked]
                              │
         ┌────────────────────▼──────────────────┐
         │   DETECTOR ANALYZES HAND              │
         │                                       │
         │   Checking for straights...           │
         │   - A-K-Q-J-10 ✓ (Straight)          │
         │   - K-Q-J-10-9 ✓ (Straight)          │
         │   - Q-J-10-9-8 ✓ (Straight)          │
         │   - J-10-9-8-7 ✓ (Straight)          │
         │   - 10-9-8-7-6 ✓ (Straight)          │
         │   - 9-8-7-6-5 ✓ (Straight)           │
         │                                       │
         │   Checking for straight flushes...   │
         │   - ♠A ♠K ♠Q ♠J ♠10 ✓ (Flush!)       │
         │   - ♠K ♠Q ♠J ♠10 ♠9 ✓ (Flush!)       │
         │   (etc.)                             │
         │                                       │
         │   Best option: Straight Flush        │
         │   (beats all 7-of-a-kind)            │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  HIGHLIGHT BEST STRAIGHT FLUSH       │
         │                                       │
         │  ♠A ♠K ♠Q ♠J ♠10 highlighted         │
         │  (shown in gold/bright color)        │
         │                                       │
         │  Message: "Straight Flush ♠ found!" │
         │           "Will beat 6-of-a-kind"    │
         │                                       │
         │  [Play] button now ENABLED           │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────Yes─────────┤─────────No──────┐
         │                    │                 │
    [User clicks         [User wants]      [User wants]
    Play]                different          another
         │                 straight          option
         │                   │                 │
    ┌────▼─────┐        ┌────▼────┐    ┌──────▼──────┐
    │Auto-play  │        │Tap to   │    │Tap to clear │
    │5 cards    │        │select   │    │selection &  │
    │(♠A-10)    │        │another  │    │try again    │
    │           │        │flush    │    │             │
    │Submit to  │        │         │    │[Clear]      │
    │server     │        │[Select] │    │             │
    │           │        │         │    │             │
    │[Playing]  │        └─────────┘    └─────────────┘
    │           │
    └─────────────┘
```

---

### 2.3 Card Grouping Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   YOUR HAND (27 CARDS)                              │
│                   Default: Sorted by Rank                           │
│                                                                     │
│  ♠A ♠K ♠Q ♠J ♠10 ♠9 ♠8 ♠7 ♠6 ♠5 ♠4 ♠3 ♠2                        │
│  ♦A ♦K ♦Q ♦J ♦10 ♦9 ♦8 ♦7 ♦6 ♦5 ♦4 ♦3 ♦2                        │
│  ♣A ♣K (more...)                                                   │
│                                                                     │
│  [Straight Finder] [Group by Suit] ◄── User taps                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [Group by Suit clicked]
                              │
         ┌────────────────────▼──────────────────┐
         │   HAND REORGANIZED BY SUIT            │
         │                                       │
         │  ♠ SPADES (9 cards)                   │
         │  ♠A ♠K ♠Q ♠J ♠10 ♠9 ♠8 ♠7 ♠6       │
         │  │                                    │
         │  ♥ HEARTS (8 cards) [WILD: ♥7]       │
         │  ♥A ♥K ♥Q ♥J ♥10 ♥8 ♥7(wild) ♥6   │
         │  │                                    │
         │  ♦ DIAMONDS (5 cards)                 │
         │  ♦A ♦K ♦Q ♦J ♦10                     │
         │  │                                    │
         │  ♣ CLUBS (5 cards)                    │
         │  ♣A ♣K ♣Q ♣J ♣10                     │
         │                                       │
         │  [Back to Default] [Group by Rank]   │
         │                                       │
         └───────────────────────────────────────┘
                              │
         ┌────────────────────┴──────────────────┐
         │                                       │
    [User can now]                          [Or back to
    easily see all              Default]
    cards of same suit                    │
    for finding flushes                    │
         │                             ┌───▼──────┐
         │                             │Reset view│
         │                             │to rank   │
         │                             │sort      │
         │                             └──────────┘
         │
    [Group by Rank clicked]
         │
    ┌────▼──────────────────────────────────┐
    │   HAND REORGANIZED BY RANK            │
    │                                       │
    │  A (4 cards)                          │
    │  ♠A ♥A ♦A ♣A                         │
    │                                       │
    │  K (4 cards)                          │
    │  ♠K ♥K ♦K ♣K                         │
    │                                       │
    │  (... more ranks ...)                 │
    │                                       │
    │  7 (5 cards) [WILD: ♥7]              │
    │  ♠7 ♥7(wild) ♦7 ♣7 [duplicate 7]    │
    │                                       │
    │  [Back to Default] [Group by Suit]   │
    │                                       │
    └───────────────────────────────────────┘
         │
    [User can now easily]
    identify pairs/triples
    for forming combos
```

---

### 2.4 Timer & Time Addition Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       YOUR TURN                                     │
│                                                                     │
│  Timer Display:                                                     │
│                                                                     │
│        ┌─────────────────────────┐                                 │
│        │                         │                                 │
│        │      ⏱️ 30 seconds      │                                 │
│        │                         │                                 │
│        │   [======░░░░░░░]  60%  │                                 │
│        │                         │                                 │
│        └─────────────────────────┘                                 │
│                                                                     │
│  You're thinking... selecting cards...                             │
│                                                                     │
│  [Play] [Pass] [+5s]                                               │
│                                                                     │
│  Time additions used: 0 (unlimited, no penalties in MVP)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                   [Time ticks down]
                              │
         ┌────────────────────▼──────────────────┐
         │        45 SECONDS REMAINING           │
         │                                       │
         │   [========░░░░░░░░]  75%             │
         │                                       │
         │   (Timer is still green)              │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
                   [Time continues]
                              │
         ┌────────────────────▼──────────────────┐
         │        10 SECONDS REMAINING           │
         │                                       │
         │   [=================░░]  17%          │
         │                                       │
         │   (Timer turns YELLOW)                │
         │   Tick-tick-tick audio plays          │
         │                                       │
         │   You still haven't decided!          │
         │   [Tap +5s to add time]               │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────┴──────────────────┐
         │                                       │
    [User taps          [User plays        [Timeout]
     +5s button]         in time]               │
         │                   │                 │
    ┌────▼─────┐        ┌────▼─────┐    ┌──────▼──────┐
    │CONFIRMED │        │Sent to   │    │Auto-PASS    │
    │Time added│        │Server    │    │(Server      │
    │          │        │          │    │ discards    │
    │ 30s timer│        │(Normal   │    │ selection)  │
    │ +5s      │        │flow)     │    │             │
    │ = 35s    │        │          │    │No penalty   │
    │          │        │          │    │in MVP       │
    │Times used│        │          │    │             │
    │ = 1      │        │          │    │Next turn    │
    │(tracking │        │          │    │starts       │
    │ only)    │        │          │    │             │
    │          │        │          │    │             │
    └──────────┘        └──────────┘    └─────────────┘
         │                   │                 │
    [Continue          [Game continues]  [Penalized]
     playing]               │                  │
                            │
                     [Next player's turn]
```

---

### 2.5 Tribute System Flow (2nd Hand Onwards)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ROUND 1 COMPLETE                                 │
│                                                                     │
│  Winner: Player 1 + Player 3 (1-2 win)                             │
│  Losers: Player 2 + Player 4                                       │
│                                                                     │
│  Level Jump: 7 → 10 (+3 levels for 1-2 win)                        │
│                                                                     │
│  Tribute Required (Round 2 setup):                                 │
│  - Player 2 gives highest non-wild to winners                      │
│  - Player 4 gives highest non-wild to winners                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [Round 1 ends animation]
                              │
         ┌────────────────────▼──────────────────┐
         │    TRIBUTE SCREEN (Player 2's view)   │
         │                                       │
         │  You lost round 1!                    │
         │                                       │
         │  Your hand value:                     │
         │  - Highest non-wild: ♠K               │
         │                                       │
         │  💰 Give ♠K to the winners?           │
         │                                       │
         │  [Confirm Tribute]                    │
         │                                       │
         │  ⚠️  Hold 🔴🔴? → Tribute cancelled   │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  TRIBUTE EXCHANGE BEGINS              │
         │                                       │
         │  Player 2 → ♠K (given)                │
         │  Player 4 → ♥A (given)                │
         │                                       │
         │  Winners review cards...              │
         │  - Want to keep ♠K? (high value)      │
         │  - Want to keep ♥A? (if wild?)       │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  WINNERS' RETURN (Player 1's view)    │
         │                                       │
         │  You received:                        │
         │  - ♠K from Player 2                   │
         │  - ♥A from Player 4                   │
         │                                       │
         │  Which cards to return?               │
         │  (Must equal # cards received: 2)     │
         │                                       │
         │  [Select 2 unwanted cards]            │
         │  ♦3 ♣2 selected (2 cards)             │
         │                                       │
         │  [Confirm Exchange]                   │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  TRIBUTE COMPLETE                     │
         │                                       │
         │  Exchange Summary:                    │
         │  - Player 1 received: ♠K, ♥A          │
         │  - Player 1 gave back: ♦3, ♣2        │
         │  - Player 2 received: ♦3              │
         │  - Player 4 received: ♣2              │
         │                                       │
         │  ✅ Ready for Round 2                 │
         │                                       │
         │  System calculating who leads...      │
         │                                       │
         │  Cards given in tribute:              │
         │  - Player 2 gave: ♠K (Rank: 13)      │
         │  - Player 4 gave: ♥A (Rank: 14)      │
         │                                       │
         │  Leader: Player 4                     │
         │  (gave highest card: ♥A)              │
         │                                       │
         │  Level: 10                            │
         │                                       │
         │  [Start Round 2]                      │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  LEADER SELECTION CONFIRMED           │
         │                                       │
         │  Player 4 will go first in Round 2    │
         │  (based on highest tribute card: ♥A) │
         │                                       │
         │  [3 second countdown]                 │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  CARDS DEALING ANIMATION              │
         │  (Shuffle & redeal 27 cards)          │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
                      [Round 2 begins]
                      (Same as before)
```

### 2.5b Tribute Refusal Flow (Red Jokers)

```
┌─────────────────────────────────────────────────────────────────────┐
│              BEFORE TRIBUTE - PLAYER 4's VIEW                       │
│                                                                     │
│  Round 1 ended. You lost.                                          │
│  Your highest non-wild card: ♥K                                    │
│                                                                     │
│  Wait... I have 🔴🔴 (Red Jokers)!                                 │
│  Tribute can be CANCELLED!                                         │
│                                                                     │
│  [Show Red Jokers] ◄── Button appears                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    [Click "Show Red Jokers"]
                              │
         ┌────────────────────▼──────────────────┐
         │    TRIBUTE CANCELLED!                 │
         │                                       │
         │    🔴 🔴 You have both red jokers!    │
         │                                       │
         │    Tribute is CANCELLED for this     │
         │    hand.                              │
         │                                       │
         │    (Your team keeps your cards)      │
         │    (Other team keeps theirs)         │
         │                                       │
         │    [Continue to Round 2]              │
         │    Level: 10 (same as before)        │
         │    No card exchange                   │
         │                                       │
         │    First player: Player 2             │
         │    (Previous winner - tribute not    │
         │     given, so leader rule doesn't    │
         │     apply)                           │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
                      [Round 2 begins]
                    (with full original hands)
```

### 2.5c Leader Selection - Equal Cards Scenario

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIBUTE COMPLETE                                 │
│                                                                     │
│  Exchange Summary:                                                  │
│  - Player 1 received: ♠K, ♠K                                        │
│  - Player 1 gave back: ♦3, ♣2                                       │
│  - Player 2 received: ♦3                                            │
│  - Player 4 received: ♣2                                            │
│                                                                     │
│  ✅ Ready for Round 2                                               │
│                                                                     │
│  System calculating who leads...                                    │
│                                                                     │
│  Cards given in tribute:                                            │
│  - Player 2 gave: ♠K (Rank: 13)                                     │
│  - Player 4 gave: ♠K (Rank: 13)                                     │
│                                                                     │
│  ⚠️  EQUAL CARDS GIVEN!                                              │
│  Randomly selecting leader...                                       │
│                                                                     │
│  [Spinning wheel animation - 2 seconds]                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  LEADER SELECTION CONFIRMED           │
         │                                       │
         │  Player 2 will go first in Round 2    │
         │  (randomly selected - both gave ♠K)  │
         │                                       │
         │  Level: 10                            │
         │                                       │
         │  [3 second countdown]                 │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │  CARDS DEALING ANIMATION              │
         │  (Shuffle & redeal 27 cards)          │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
                      [Round 2 begins]
                      (Same as before)
```

---

## 3. MULTIPLAYER COORDINATION FLOWS

### 3.1 Real-Time Sync Flow (What Happens Behind Scenes)

```
┌──────────────────┐        ┌──────────────────┐
│ PLAYER 1's PHONE │        │ PLAYER 2's PHONE │
│ (Plays ♠8 ♦8)   │        │ (Waiting)        │
│                  │        │                  │
│ [Tap Play]       │        │                  │
│ Selection sent   │        │                  │
│ to Supabase      │        │                  │
│                  │        │                  │
│ ────────────────►│        │                  │
│                  │        │                  │
│                  │        │ Supabase         │
│                  │        │ Realtime UPDATE  │
│                  │        │ (latency: <500ms)│
│                  │        │                  │
│                  │        │◄─────────────────│
│                  │        │                  │
│ Animation starts │        │ [Update received]│
│ (500ms)          │        │ Animation shows  │
│                  │        │ P1's play        │
│                  │        │                  │
│ Timer: 50s       │        │ Timer: 45s       │
│ Next turn: P2    │        │ [YOUR TURN NOW]  │
│                  │        │                  │
│                  │        │ Options appear:  │
│                  │        │ - Play higher    │
│                  │        │ - Pass           │
│                  │        │                  │
└──────────────────┘        └──────────────────┘
        │                           │
        └─────── All phones ────────┘
         synced in real-time
         via Supabase Realtime
```

### 3.2 Disconnect & Reconnect Flow

```
┌──────────────────────────────────────────────────────────────────┐
│            PLAYER 3 LOSES CONNECTION                             │
│            (WiFi drops / Network error)                          │
│                                                                 │
│ Player 3's phone shows:                                         │
│ ⚠️  "Connection lost"                                           │
│ "Attempting to reconnect..."                                    │
│                                                                 │
│ Other players see:                                              │
│ ⏳ Player 3 is AFK (timeout: 30 seconds)                         │
│                                                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                   [30-second window opens]
                              │
         ┌────────────────────┴────────────────┐
         │                                     │
    [P3 reconnects          [P3 doesn't      [Server
     within 30s]            rejoin in time]  action]
         │                       │             │
    ┌────▼────┐           ┌──────▼───┐  ┌─────▼──────┐
    │Supabase │           │Game      │  │Game ends   │
    │restores │           │aborted   │  │ or P3 is   │
    │session  │           │          │  │ replaced   │
    │         │           │ Team     │  │ by bot?    │
    │P3 can   │           │ penaliz- │  │            │
    │resume   │           │ ed -0.5  │  │ (tbd)      │
    │game     │           │ level    │  │            │
    │from last│           │          │  │            │
    │state    │           │P3 can    │  │            │
    │         │           │rejoin    │  │            │
    │Network  │           │next game │  │            │
    │stable   │           │          │  │            │
    │         │           │          │  │            │
    │Game     │           └──────────┘  └────────────┘
    │resumes  │
    │         │
    └─────────┘
```

---

## 4. END-OF-GAME FLOWS

### 4.1 Round Complete & Results

```
┌──────────────────────────────────────────────────────────────────┐
│             ROUND COMPLETE - RESULTS SCREEN                      │
│                                                                  │
│  🎉 NORTH/SOUTH WINS! 🎉                                        │
│                                                                  │
│  Finish Order:                                                   │
│  1. Player 1 (N) - 0 cards left ✅                              │
│  2. Player 3 (S) - 0 cards left ✅                              │
│  3. Player 2 (E) - 3 cards left                                 │
│  4. Player 4 (W) - 2 cards left                                 │
│                                                                  │
│  Outcome: 1-2 WIN (双上)                                         │
│  Promotion: 7 → 10 (+3 levels)                                  │
│                                                                  │
│  Team Scores:                                                    │
│  🏆 N/S: Level 10                                               │
│  📉 E/W: Remain at Level 7                                      │
│                                                                  │
│  --- TRIBUTE REQUIRED (Next Round) ---                          │
│  Losers (E/W) must give highest cards to winners (N/S)          │
│                                                                  │
│  [Continue to Next Round]  [End Session]                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴──────────────────┐
         │                                       │
    [Continue]                               [End Session]
         │                                       │
    [Show Tribute                          ┌─────▼────────┐
     Screen]                               │Session ends  │
         │                                 │              │
    [Round 2]                              │Would you like│
                                           │to play again?│
                                           │              │
                                           │[New Game]    │
                                           │[Home]        │
                                           │              │
                                           └──────────────┘
```

### 4.2 Session End Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    SESSION SUMMARY                               │
│                                                                  │
│  🎮 Chucking Eggs - 3 Rounds Played                             │
│                                                                  │
│  Final Standings:                                                │
│  ┌─ NORTH/SOUTH ─┐        ┌─ EAST/WEST ─┐                     │
│  │               │        │             │                      │
│  │ Final: Level A│        │ Final: Lv 9 │                      │
│  │ Rounds Won: 2 │        │ Rounds Won: 1│                      │
│  │               │        │             │                      │
│  │ 🦊 Player 1   │        │ 🐼 Player 2 │                      │
│  │ 🦁 Player 3   │        │ 🐨 Player 4 │                      │
│  │               │        │             │                      │
│  └───────────────┘        └─────────────┘                      │
│                                                                  │
│  Most Impressive Plays:                                         │
│  - ♠10-J-Q-K-A (Straight Flush by P1)                          │
│  - 4-Joker Bomb played (Round 1)                               │
│                                                                  │
│  [Share Results] [New Game] [Home]                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┴──────────────────┐
         │                                       │
    [New Game]                               [Home]
         │                                       │
    ┌────▼────┐                            ┌────▼────┐
    │[Create   │                            │[Return  │
    │ or Join] │                            │ to      │
    │          │                            │ Home]   │
    │Game 2    │                            │         │
    │starts    │                            │Session  │
    │          │                            │saved    │
    └──────────┘                            └─────────┘
```

---

## 5. ERROR & EDGE CASE FLOWS

### 5.1 Invalid Play Attempt

```
┌──────────────────────────────────────────────────────────────┐
│            INVALID COMBO SELECTED                            │
│                                                              │
│  User selects: [♠8][♦8][♣7]                                │
│  (2 eights + 1 seven - NOT valid)                           │
│                                                              │
│  [Play] button remains DISABLED                             │
│                                                              │
│  Error message appears:                                      │
│  ⚠️  "Invalid combination"                                   │
│      "Cannot mix ranks - select same rank or valid combo"   │
│                                                              │
│  [Clear] button appears                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────▼──────────────────┐
         │        USER CORRECTS MISTAKE          │
         │                                       │
         │  Taps [Clear]                         │
         │  All cards deselected                 │
         │                                       │
         │  Selects: [♠8][♦8][♣8]               │
         │  (Valid triple of 8s)                 │
         │                                       │
         │  Error message disappears             │
         │  [Play] button ENABLED                │
         │                                       │
         │  [Play (3 cards)]                     │
         │                                       │
         └────────────────────┬──────────────────┘
                              │
                      [Play submitted]
```

### 5.2 Cannot Beat Current Play

```
┌──────────────────────────────────────────────────────────────┐
│        CURRENT TRICK - YOU CANNOT BEAT                       │
│                                                              │
│  Current card on table: 4-Joker Bomb                         │
│  (Highest possible)                                          │
│                                                              │
│  Your options:                                               │
│  ✅ Pass (skip your turn)                                    │
│  ❌ Cannot play anything (bombs only beat bombs)             │
│                                                              │
│  Message: "4-Joker Bomb cannot be beaten - must pass"       │
│                                                              │
│  [Pass] button HIGHLIGHTED                                  │
│  [Play] button DISABLED                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                         [User passes]
                              │
         ┌────────────────────▼──────────────────┐
         │      TURN PASSES TO NEXT PLAYER       │
         │      (Round ends if 3 pass)           │
         │                                       │
         └────────────────────────────────────────┘
```

---

## 6. SETTINGS & ACCOUNT FLOWS

### 6.1 Home Screen → Settings

```
┌─────────────────────────────────────────────────────────────┐
│              CHUCKING EGGS HOME                             │
│                                                             │
│         [🥚] Chucking Eggs [🥚]                            │
│              掼 蛋                                           │
│                                                             │
│          ┌──────────────────┐                              │
│          │ Create New Room  │                              │
│          └──────────────────┘                              │
│          ┌──────────────────┐                              │
│          │ Join Room Code   │                              │
│          └──────────────────┘                              │
│          ┌──────────────────┐                              │
│          │   ⚙️ Settings     │◄─── User taps               │
│          └──────────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    [Tap Settings]
                              │
         ┌────────────────────▼──────────────────┐
         │         SETTINGS SCREEN               │
         │                                       │
         │  Account (Anonymous)                 │
         │  ┌──────────────────────────────────┐ │
         │  │ Display Name: Alex               │ │
         │  │ [Change Name] [Change Avatar]    │ │
         │  └──────────────────────────────────┘ │
         │                                       │
         │  Display Name                        │
         │  ┌──────────────────────────────────┐ │
         │  │ Alex                     [Edit]  │ │
         │  └──────────────────────────────────┘ │
         │                                       │
         │  Gameplay (MVP: Basic only)          │
         │  □ Sound Effects (v1.1)              │
         │  □ Haptic Feedback (v1.1)            │
         │  □ Dark Mode (v1.1)                  │
         │                                       │
         │  Timer Settings (MVP)                │
         │  Default Time: [30 seconds]          │
         │                                       │
         │  About                               │
         │  Version: 1.0                        │
         │  [Privacy Policy]                    │
         │  [Terms of Service]                  │
         │                                       │
         │  [Back]                              │
         │                                       │
         └───────────────────────────────────────┘
```

---

## 7. SUMMARY - KEY DECISION POINTS IN FLOWS

| Flow | Decision Point | Status |
|------|---|---|
| Room Creation | Room code format (6-char) | ✅ FIXED |
| Team Selection | Auto vs Player-chosen | ✅ Player-chosen |
| Card Selection | Single vs Multi-select | ✅ Multi-select |
| Straight Detector | Auto-select best or show options | ✅ Suit buttons, manual cycle |
| Timer | Duration | ✅ 30 seconds |
| Penalties | Time addition penalties? | ✅ None (MVP) |
| Disconnect | Game aborts or continues? | ✅ Game cancels after 30s |
| Reconnect Window | Duration | ✅ 30 seconds |
| Leader Selection | How determined? | ✅ Highest tribute card given |
| Bomb Timing | Anytime or on turn? | ✅ Only on your turn |
| Authentication | Method? | ✅ Anonymous with display name |
| Wild Card Display | How shown? | ✅ Gold border |

---

**These flows should be implemented as clickable prototypes in Figma/Sketch for validation before Phase 4 (UI) engineering begins.**

