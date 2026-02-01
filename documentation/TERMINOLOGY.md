# Chucking Eggs - Terminology Guide

**Version:** 1.0
**Date:** February 2026
**Purpose:** Ensure consistent terminology across all documentation and code

---

## Game Structure Terms

| Term | Definition | Usage Example |
|------|------------|---------------|
| **Game** | Entire session from start to Ace level | "We played 2 games last night" |
| **Round** | One complete dealing/playing cycle | "Round 3 starts at level 5" |
| **Hand** | 27 cards dealt to one player | "My hand has 3 bombs!" |
| **Trick** | One cycle of 4 plays around table | "I won the trick with my triple" |
| **Turn** | One player's opportunity to play | "It's your turn - 30 seconds left" |

---

## Multiplayer Terms

| Term | Definition | Usage Example |
|------|------------|---------------|
| **Room** | Multiplayer game lobby/session | "Join my room with code F8E6FC" |
| **Room Code** | 6-character alphanumeric ID | "Share the room code with friends" |
| **Room ID** | (Same as Room Code) | **Use "Room Code" in user-facing text** |
| **Session** | (Same as Game) | **Use "Game" for consistency** |

---

## Technical Terms

| Term | Definition | Usage Example |
|------|------------|---------------|
| **Level** | Current rank being played (2-A) | "We're at level 7 now" |
| **Wild Card** | Hearts of current level | "♥7 is wild this round" |
| **Tribute** | Card exchange after round | "Give your highest in tribute" |
| **Combination** | Valid group of cards to play | "Select a valid combination" |
| **Bomb** | Special powerful combination | "Play a bomb to beat that!" |

---

## Consistency Rules

### Always Use:
- ✅ **"Room Code"** (not "Room ID") in UI text
- ✅ **"Round"** (not "Hand") for game cycles
- ✅ **"Hand"** for player's cards
- ✅ **"Turn"** (not "Move") for player actions
- ✅ **"Trick"** for one cycle of plays
- ✅ **"Level"** (not "Rank") for progression

### Never Use:
- ❌ "Room ID" in user-facing documentation
- ❌ "Hand" to mean a round of play
- ❌ "Session" (use "Game" instead)
- ❌ "Move" (use "Turn" instead)

---

## Code Examples

### Correct:
```typescript
interface GameState {
  currentLevel: Rank;        // ✅ "level"
  currentRound: number;      // ✅ "round"
  currentPlayerIndex: number; // ✅ player's turn
}

interface Player {
  hand: Card[];              // ✅ player's cards
}

interface Trick {
  cards: Card[];             // ✅ one cycle of plays
}
```

### Incorrect:
```typescript
interface GameState {
  currentRank: Rank;         // ❌ use "level"
  currentHand: number;       // ❌ use "round"
  currentMove: number;       // ❌ use "turn"
}
```

---

*Terminology Guide v1.0 - February 2026*
