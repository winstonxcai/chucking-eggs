# Chucking Eggs - UI/UX Documentation

**Version:** 1.0  
**Date:** February 2026  
**Audience:** Design & Frontend Engineering

---

## 1. DESIGN SYSTEM

### Color Palette
```
Primary Background:  #FFF9E6 (warm cream)
Card Area:           #FFFFFF (white)
Text Primary:        #1A1A1A (dark gray)
Text Secondary:      #666666 (medium gray)
Accent (Highlight):  #FF6B6B (coral red)
Accent (Success):    #51CF66 (green)
Error:               #FF4757 (red)
Neutral:             #E0E0E0 (light gray)
```

### Typography
```
Font Family:         SF Pro (iOS native system font)
                     - No custom fonts needed for MVP
                     - Zero bundle size overhead
                     - Automatic iOS version compatibility
Headlines:           24-28pt, Bold, #1A1A1A
Subheading:          16-18pt, Semibold, #1A1A1A
Body Text:           14-16pt, Regular, #333333
Labels:              12-14pt, Medium, #666666
Buttons:             16pt, Semibold, #FFFFFF
```

### Spacing
```
Margin:              16pt (standard)
Padding:             16pt (containers)
Card Gap:            8pt (overlapping cards)
Button Gap:          12pt (between buttons)
Safe Area:           16pt from edges
```

### Interactive Elements
```
Button Height:       48pt (min tap target)
Border Radius:       12pt (rounded corners)
Shadow:              0 2px 8px rgba(0,0,0,0.1)
Animation Duration:  300-500ms
Card Overlap:        20px (hand fan layout)
```

---

## 2. SCREEN LAYOUTS & WIREFRAMES

### 2.1 Home Screen

```
┌─────────────────────────────────────────┐
│                  Safe Area              │
│                                         │
│                                         │
│         [🥚] Chucking Eggs [🥚]        │
│              掼 蛋                       │
│                                         │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │   Create New Room               │  │
│  │   (Tap to start hosting)         │  │
│  └─────────────────────────────────┘  │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │   Join Room Code                │  │
│  │   (Tap to enter code)            │  │
│  └─────────────────────────────────┘  │
│                                         │
│                                         │
│  ┌──────────────┐                      │
│  │      ⚙️      │                      │
│  │  Settings    │                      │
│  └──────────────┘                      │
│                                         │
└─────────────────────────────────────────┘
```

**Interactive Elements:**
- "Create New Room" button → Navigate to room creation
- "Join Room Code" button → Navigate to room joining
- ⚙️ Settings icon (bottom) → Navigate to settings

**States:**
- Initial load: Buttons enabled
- Offline: Show warning badge, disable "Join"
- Loading: Show spinner on selected button

---

### 2.2 Create Room Screen

```
┌─────────────────────────────────────────┐
│                  Safe Area              │
│                                         │
│  < Back                                 │
│                                         │
│  Creating Your Room...                  │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │  Share this code with friends:  │  │
│  │                                 │  │
│  │      F8E6FC                     │  │
│  │                                 │  │
│  │          [Copy]                 │  │
│  └─────────────────────────────────┘  │
│                                         │
│  Waiting for players...                 │
│                                         │
│  🦊 You (Player 1)                      │
│  ⏳ Waiting for Player 2...             │
│  ⏳ Waiting for Player 3...             │
│  ⏳ Waiting for Player 4...             │
│                                         │
│  [Cancel]                               │
│                                         │
└─────────────────────────────────────────┘
```

**Interactive Elements:**
- Back button → Return to home
- [Copy] button → Copy code to clipboard
- [Cancel] button → Destroy room, return to home

**Real-Time Updates:**
- Player list updates as others join
- Show "✓ Player 2 joined" notification
- Once 4 joined → Auto-navigate to team selection

---

### 2.3 Join Room Screen

```
┌─────────────────────────────────────────┐
│                  Safe Area              │
│                                         │
│  < Back                                 │
│                                         │
│  Enter Room Code                        │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │ Enter code (e.g., F8E6FC)       │  │
│  │                                 │  │
│  │ ┌─────────────────────────────┐ │  │
│  │ │ [_________________]          │ │  │
│  │ │                              │ │  │
│  │ └─────────────────────────────┘ │  │
│  │                                 │  │
│  │     [Join Room]                 │  │
│  │     (disabled until input is 6 chars)  │  │
│  └─────────────────────────────────┘  │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

**Interactive Elements:**
- Text input field: Accepts 6 alphanumeric characters
- [Join Room] button: Disabled until valid code entered
- Back button: Return to home

**Validation:**
- Valid code (6 chars): Button enabled
- Invalid code: Show "Invalid code format"
- Room not found: Show "Room not found"
- Room full (4/4): Show "Room is full"
- Game in progress: Show "Game in progress, cannot join"

---

### 2.4 Lobby Screen (Team Selection)

```
┌─────────────────────────────────────────┐
│                  Safe Area              │
│                                         │
│  < Back     Room: F8E6FC  [Copy]        │
│                                         │
│  ┌────────────────────────────────┐   │
│  │  Select Your Team              │   │
│  └────────────────────────────────┘   │
│                                         │
│  ┌─ NORTH/SOUTH ┐    ┌─ EAST/WEST ─┐ │
│  │               │    │             │ │
│  │  🦊 You       │    │ 🐼 Player 2 │ │
│  │  ◉ Select    │    │ ◉ Select    │ │
│  │               │    │             │ │
│  │  ⏳ Player 3   │    │ ⏳ Player 4  │ │
│  │  ○ Select    │    │ ○ Select    │ │
│  │               │    │             │ │
│  └───────────────┘    └─────────────┘ │
│                                         │
│  [Cancel]            [Start Game]      │
│  (leaves room)       (disabled)         │
│                                         │
└─────────────────────────────────────────┘
```

**Interactive Elements:**
- Radio buttons: Select North/South or East/West for your seat
- [Cancel]: Leave room, return to home
- [Start Game]: Enabled only when all 4 seats have teams selected

**Interaction Rules:**
- Cannot select same team as another player (radio button disabled)
- Once a team is selected by one player, that option is unavailable to others
- Visual feedback: Selected radio button = filled circle
- [Start Game] disabled until all 4 players have selections

**Real-Time Updates:**
- See other players select teams in real-time
- Show "Player X selected East/West" notification

---

### 2.5 Game Board - Main Layout

```
┌─────────────────────────────────────────┐
│                  Safe Area              │
│                                         │
│ Lv 7 ♥️  │  Your Team: 2 lvls ahead    │
├─────────────────────────────────────────┤
│                                         │
│           🦊 Player 3 (12 cards)        │
│               East/West                 │
│                                         │
│     Player 2              Player 4      │
│     North/South           East/West     │
│     (8 cards)             (15 cards)    │
│                                         │
│          ┌─────────────────┐            │
│          │  [TRICK PILE]   │            │
│          │  3 cards shown  │            │
│          │  (animation)    │            │
│          └─────────────────┘            │
│          (Your turn indicator)          │
│          ⏱️ 45s  [Playing]              │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│     [YOUR HAND - 27 CARDS]              │
│                                         │
│     ♠A  ♠K  ♠Q  ♠J  ♠10 ♠9 ... 2♦ 2♣ │
│     (Fan layout with overlap)           │
│     [Selected: ♠8 ♦8 ♣8]               │
│     (elevated, highlighted)             │
│                                         │
│ [Straight Flush Finder]  [♠] [♥] [♦] [♣]   │
│                                         │
│ [Clear]  [Play (3)]  [Pass]  [+5s]    │
│                                         │
└─────────────────────────────────────────┘
```

**Starting Hand Layout (default sort by rank):**
- Cards sorted by rank left-to-right (highest → lowest: A, K, Q … 2).
- Same rank stacked vertically in a column (not grouped; e.g. three 8s = three separate cards in one column).
- Grouped cards (melds) appear on the left; each group is also vertically stacked. Ungrouped cards follow in rank order.

```
┌─────────────────────────────────────────┐
│     [YOUR HAND - 27 CARDS] (starting)   │
│                                         │
│  Grouped (left)     Ungrouped by rank   │
│  (each vertical)    (high → low)        │
│  ┌────┐  ┌────┐    ♠A   ♠K   ♠Q   ♠J   │
│  │ ♠8 │  │ ♠7 │     │    │    │    │    │
│  │ ♦8 │  │ ♥7 │    ♥A   ♥K   ♥Q   ♥J   │
│  │ ♣8 │  │ ♦7 │     │    │    │    │    │
│  └────┘  └────┘    ♦A   ♦K   ...  ...  │
│  (meld)  (meld)     │    │              │
│                    ♣A   ...             │
│                    │                    │
│  (groups appended  (vertical stacks per │
│   to left; each    rank; same rank =   │
│   group stacked    one column)          │
│   vertically)                            │
│                                         │
│ [Straight Finder]  [♠] [♥] [♦] [♣]   │
│ [Clear]  [Play]  [Pass]  [+5s]        │
└─────────────────────────────────────────┘
```

**Key Information Always Visible:**
- Current level with suit indicator
- Player positions with card counts
- Only display card count of opponents if less than or equal to 10 cards
  - **Strategic gameplay rule:** When opponent has >10 cards, show card backs without count
  - When ≤10 cards, display exact number (e.g., "7 cards")
  - Forces players to track cards mentally during early game
- Your hand (sorted by rank)
- Current trick in center
- Timer (poker now style; 30 second timer)
- Action buttons

---

### 2.6 Hand Display (Card Stack) - Detailed View

```
Card Layout (27 cards):
- VERTICAL STACK (not horizontal fan)
- Each card 20-30px offset below previous
- Cards displayed top-to-bottom in columns
- Selected cards pop to front with highlight

Interaction:
- Tap card to toggle selection
- Tap again to deselect
- Visual feedback: pop to front + highlight color

Visual Format:
┌────┐
│ ♠A │ ← Top
│ ♠K │
│ ♠Q │
│ ♠J │
│ ♠10│
│ ... │ ← Bottom
└────┘

Card Sorting Options:
┌────────────────────────────────┐
│ [Straight Flush Finder] (dynamic)│
│ [Order by Rank]   (toggle)     │
└────────────────────────────────┘

- Note that Order by rank should only consider the cards that are not in a group

Straight Finder Button Details:
- Shows 4 suit buttons: [♠] [♥] [♦] [♣]
- If straight flush exists in suit, button is highlighted
- Tap button → highlight cards for that straight flush
- Tap again → cycle to next highest straight flush in that suit
- Auto-groups cards by suit (vertical stacks per suit)
```

---

### 2.7 Straight Detector - Interaction Rules

```
SCENARIO 1: One straight flush per suit
┌─────────────────────────────────┐
│ Your hand: ♠A ♠K ♠Q ♠J ♠10    │
│           (plus 22 other cards) │
│                                 │
│ [♠] highlighted (has straight) │
│ [♥] grayed out (no straight)   │
│ [♦] grayed out (no straight)   │
│ [♣] grayed out (no straight)   │
│                                 │
│ User taps [♠]:                  │
│ → ♠A ♠K ♠Q ♠J ♠10 selected     │
│ → Message: "Straight Flush ♠"  │
│ → [Play] button enabled         │
│                                 │
│ User taps [♠] again:           │
│ → No other straights → nothing  │
│ → Stay on same selection        │
└─────────────────────────────────┘

SCENARIO 2: Multiple straight flushes in one suit
┌─────────────────────────────────┐
│ Your hand: ♠A♠K♠Q♠J♠10         │
│           ♠K♠Q♠J♠10♠9          │
│           (plus 19 other cards) │
│                                 │
│ [♠] highlighted                 │
│                                 │
│ User taps [♠]:                  │
│ → Select: ♠A♠K♠Q♠J♠10 (highest)│
│ → Message: "Straight Flush ♠"  │
│                                 │
│ User taps [♠] again:            │
│ → Select: ♠K♠Q♠J♠10♠9 (next)   │
│ → Message: "Straight Flush ♠"  │
│                                 │
│ User taps [♠] again:            │
│ → No more straights → reset     │
│ → Clear selection               │
└─────────────────────────────────┘
```

**Interaction Design:**
- Straight detector shows highlighted buttons for available suits
- Tapping cycles through all possible straights in that suit (highest first)
- Auto-selects cards on display
- User can still manually select different cards (overrides detector)
- When suit is selected, cards are grouped by that suit visually

---

### 2.7.1 Wild Card Visual Treatment

**Wild Cards:** Hearts of the current level rank only (e.g., ♥7 when level is 7)

**Visual Styling:**
```css
.wild-card {
  border: 2px solid #FFD700; /* Gold border */
  box-shadow: 0 0 8px rgba(255, 215, 0, 0.5); /* Subtle gold glow */
}
```

**Display Rules:**
- Gold (#FFD700) border replaces standard card border
- Subtle glow effect for clarity
- No text badge or label (keep card clean)
- Card suit/rank still clearly visible
- Animation: gentle pulse on selection (optional)

**Example:**
```
Normal card:     Wild card:
┌────────┐      ┌────────┐
│ ♥7     │      │ ♥7     │ ← Gold border
│        │      │        │
│     ♥7 │      │     ♥7 │
└────────┘      └────────┘
  (normal)        (wild)
```

---

### 2.8 Timer Display (Poker Now Style)

```
Circular Timer Layout:
┌──────────────────┐
│   ⏱️ 30 seconds   │
│   [====░░░░░░]   │  (progress ring)
│                  │
│    100% - Green  │
│    75% - Green   │
│    50% - Yellow  │
│    25% - Yellow  │
│    10% - Red     │
│    0%  - Red     │
│                  │
│   (Tick-tick-   │
│    tick sound)   │
│   when < 10s     │
└──────────────────┘

Layout on Game Board:
┌─────────────────────────────────┐
│                                 │
│          [TRICK]                │
│                                 │
│     ⏱️ 45s  [Currently Playing]  │
│                                 │
│     [progress ring visual]      │
│                                 │
└─────────────────────────────────┘

Actions Available During Timer:
- [Play] - if valid combo selected
- [Pass] - always available
- [+5s] - add 5 seconds (unlimited)
```

---

### 2.9 Action Buttons

```
Layout at Bottom of Screen:
┌─────────────────────────────────┐
│ [Clear]  [Play (3)]  [Pass] [+5s]│
└─────────────────────────────────┘

Button States:

[Clear] Button:
- Enabled when: ≥1 card selected
- Disabled when: No cards selected
- Tap: Deselect all cards

[Play] Button:
- Enabled when: Valid combo selected
- Disabled when: No cards OR invalid combo
- Shows: "Play (N cards)" where N = selected
- Tap: Submit play to server
- Visual feedback: Button lightens on press

[Pass] Button:
- Enabled always
- Tap: Show confirmation "Are you sure?"
- One-tap confirm if [Pass] tapped again
- Or tap [Play] to cancel pass

[+5s] Button:
- Enabled always
- Tap: Add 5 seconds to timer
- Updates timer immediately
- No penalty (MVP)
```

---

### 2.10 Error & Validation Messages

```
Invalid Combination:
┌──────────────────────────────────┐
│ ⚠️  Invalid combination          │
│                                  │
│ (Message dismisses after 3s)     │
└──────────────────────────────────┘

Cannot Beat Pass Automatically:

Not Your Turn:
┌──────────────────────────────────┐
│ Waiting for Player 2...          │
│ (All action buttons disabled)    │
└──────────────────────────────────┘
```

---

### 2.11 Round Results Screen

```
┌─────────────────────────────────┐
│                                 │
│    🎉 NORTH/SOUTH WINS! 🎉      │
│                                 │
│  Finish Order:                  │
│  1. Player 1 (N) ✅              │
│  2. Player 3 (S) ✅              │
│  3. Player 2 (E)                │
│  4. Player 4 (W)                │
│                                 │
│  Outcome: 1-2 WIN               │
│  Promotion: 7 → 10 (+3 levels)  │
│                                 │
│  ┌─────────────────────────┐   │
│  │ N/S: Level 10 🏆         │   │
│  │ E/W: Level 7             │   │
│  └─────────────────────────┘   │
│                                 │
│  [Continue] [End Session]       │
│                                 │
└─────────────────────────────────┘
```

**Interactive Elements:**
- [Continue]: Show tribute screen → next round
- [End Session]: Show session summary, return to home

---

### 2.12 Tribute Screen

```
(From Losers' Perspective)
┌─────────────────────────────────┐
│ You lost round 1!               │
│                                 │
│ Your highest non-wild card:     │
│ ♠K                              │
│                                 │
│ Tribute required:               │
│ Give ♠K to winners              │
│                                 │
│ [Confirm Tribute]               │
└─────────────────────────────────┘

- In 1-4 or 1-3, if 4 has both red jokers, it must be announced to the table
- In 1-2, if 3 or 4 individually has both red jokers, it must be announced to the table. if each 3 and 4 have one, it must be announced to the table

These two should be done automatically and no tribute screen is needed

(From Winners' Perspective)
┌─────────────────────────────────┐
│ You received:                   │
│ - ♠K (from Player 2)            │
│ - ♥A (from Player 4)            │
│                                 │
│ Which to return? (select 2)     │
│                                 │
│ ┌─────────────┐ ┌─────────────┐│
│ │ ♦3 [✓]      │ │ ♣2 [✓]      ││
│ │ ♠7 [ ]      │ │ ♥5 [ ]      ││
│ │ ♠4 [ ]      │ │ ♠9 [ ]      ││
│ │ (etc.)      │ │ (etc.)      ││
│ └─────────────┘ └─────────────┘│
│                                 │
│ [Confirm Exchange]              │
│                                 │
└─────────────────────────────────┘
```

---

## 3. INTERACTION RULES & PATTERNS

### 3.1 Card Selection
```
Tap Interaction:
- Single tap card → adds to selection (visual elevation)
- Tap again → removes from selection
- Multi-select → build combinations

Visual Feedback:
- Selected cards: Elevated 12pt, highlight border
- Invalid combo: Red border, error message
- Valid combo: Green border, [Play] enabled

Drag Interaction (Future Enhancement):
- Drag to reorder within selection
- Drag outside to deselect (v1.1)
```

### 3.2 Grouping Persistence
```
Rule: Grouping preference persists during entire hand (client-side only)

Storage Location:
- Stored in component state (React useState)
- NOT synced to server/Supabase
- NOT persisted across app restarts
- Each player can have different grouping (independent)

Visual Format: VERTICAL STACK (not horizontal overlap)
- Cards displayed as a vertical stack
- First card on top, each card slightly offset below
- When grouped: Each group forms its own vertical stack
- Click card to select (pops to front/highlighted)

Reset Triggers:
- Hand finishes (all cards played)
- Round ends
- New game starts
- App backgrounds/restarts
```

### 3.3 Turn Timer Rules
```
Start: 30 seconds when your turn begins

Actions:
- Play: Submits immediately (no timer drain)
- Pass: Confirms, timer stops, next player's turn
- +5s: Adds 5 seconds (unlimited, no penalties in MVP)

Timer Expiration:
- <10s: Red ring + tick sound
- 0s: Auto-pass, show "[Player X] passed (timeout)"
```

### 3.4 Hand Sorting Defaults
```
Default Sort: By Rank
- A: 4 cards (♠A ♥A ♦A ♣A)
- K: 4 cards (♠K ♥K ♦K ♣K)
- ...
- 2: 4 cards (♠2 ♥2 ♦2 ♣2)
- Jokers: 4 cards (🔴 🔴 ⚫ ⚫)

Display Order: Highest rank first

When User Changes Sort:
- Tap [Straight Finder] → highlight that suit
- Sort persists until round ends
```

### 3.5 Invalid Play Prevention
```
Client-Side:
- Only valid combos enable [Play] button
- Visual feedback: Error message for invalid attempts
- No submission until valid

Server-Side (Double-Check):
- Validate combo again
- Check player's hand matches
- Enforce turn order
- Prevent race conditions

User Experience:
- Never get to "your play was rejected" screen
- All validation happens before they can play
```

### 3.6 Pass Confirmation
```
First tap [Pass]:
- Show: "Are you sure?"
- [Yes, Pass] [Cancel]

If [Yes, Pass]:
- Submit pass to server
- Turn moves to next player

If [Cancel]:
- Return to card selection
- Timer continues
```

---

## 4. ANIMATIONS & TRANSITIONS

### 4.1 Dealing Animation
```
Timing: 3 seconds total
- Cards appear one-by-one (fade in)
- Brief rotate animation
- Fall into hand (bounce)
- Shuffle/sort (reorganize)

Performance: 60 FPS, GPU-accelerated
```

### 4.2 Playing Cards
```
Timing: 500ms
- Selected cards fly to center
- Scale down slightly
- Stack in pile
- Next player's timer starts

Sound: Card flip noise (optional)
```

### 4.3 Round Result Slide
```
Timing: 1 second
- Slide up from bottom
- Confetti animation (winner only)
- Show results

Sound: Victory/defeat jingle (optional)
```

### 4.4 Player Transitions
```
Timing: 300ms
- Fade out current player's timer
- Fade in next player's timer
- Highlight next player's seat

Sound: Turn notification (optional)
```

---

## 5. ACCESSIBILITY CONSIDERATIONS

### Color & Contrast
- All text: WCAG AA contrast minimum (4.5:1)
- Cards: Distinct by suit (not color-only)
- Error messages: Red + icon + text

### Touch Targets
- Minimum 44pt height for all buttons
- Minimum 48pt spacing between touch targets
- Card overlap: 20px (but minimum 16pt visible area)

### Text Size
- Default 16pt body text
- Scaling tested: 100%-200% zoom

### Future (Post-MVP)
- Dynamic font scaling
- Dark mode support
- VoiceOver support (iOS accessibility)

---

## 6. RESPONSIVE DESIGN

### Breakpoints
```
iPhone SE (375px):  Minimum width
iPhone 14 (390px):  Standard
iPhone 14 Pro (430px): Maximum tested
```

### Adaptations
- Safe area insets for notch
- Portrait only (MVP)
- Bottom-heavy layout (hand at bottom)
- Landscape support: v1.1

### Card Fan Overflow
```
27 cards × ~40px width = 1080px total
Screen width: 375px

Solution: Overlap cards by 20px
→ 27 cards fit in 375px
→ Minimum visible area per card: 16px
```

---

## 7. DARK MODE (Post-MVP)

**Not in MVP but planned for v1.1**

```
Dark Mode Colors:
Background:  #1A1A1A (dark gray)
Cards:       #2A2A2A (lighter gray)
Text:        #F0F0F0 (light gray)
Accent:      #FF8888 (lighter coral)
```

---

*UI/UX Documentation v1.0 - February 2026*
