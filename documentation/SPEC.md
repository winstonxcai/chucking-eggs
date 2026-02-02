# Chucking Eggs - Technical Documentation

**Version:** 1.0  
**Date:** February 2026  
**Audience:** Engineering Team

---

## 1. TECHNOLOGY STACK

### Core Stack
- **Framework:** React Native + Expo (SDK 54)
- **Language:** TypeScript
- **State Management:** Zustand (lightweight, performant)
- **Navigation:** Expo Router (file-based routing)
- **UI Animations:** React Native Reanimated 2 + Gesture Handler
- **Backend:** Supabase (PostgreSQL + Realtime)
- **Deployment:** EAS (Expo Application Services)
- **Testing:** Jest + React Native Testing Library

### Why These Choices?
- **Expo:** Zero local build complexity, instant updates via OTA
- **EAS:** Cloud builds, direct TestFlight distribution
- **Zustand:** Minimal boilerplate, works with React Native
- **Supabase:** Open-source Firebase alternative, real-time multiplayer built-in
- **Reanimated 2:** 60 FPS card animations on mobile

---

## 2. PROJECT STRUCTURE

```
chucking-eggs/
├── app/                              # Expo Router pages
│   ├── (tabs)/
│   │   ├── index.tsx                 # Home screen
│   │   └── settings.tsx              # Settings
│   ├── game/
│   │   ├── [roomId].tsx              # Game board
│   │   └── local.tsx                 # Local test mode
│   ├── lobby/
│   │   ├── create.tsx                # Create room
│   │   └── join.tsx                  # Join room
│   └── _layout.tsx                   # Root layout
│
├── src/
│   ├── components/
│   │   ├── cards/
│   │   │   ├── Card.tsx              # Single card
│   │   │   ├── CardFan.tsx           # Hand display (27 cards)
│   │   │   ├── CardStack.tsx         # Trick pile
│   │   │   └── CardAnimations.tsx    # Animations (deal, play, etc)
│   │   │
│   │   ├── game/
│   │   │   ├── GameBoard.tsx         # Main game layout
│   │   │   ├── PlayerPosition.tsx    # Player seats (4 corners)
│   │   │   ├── PlayArea.tsx          # Center trick display
│   │   │   ├── ActionButtons.tsx     # Play/Pass/+Time
│   │   │   ├── TimerDisplay.tsx      # Poker now timer
│   │   │   └── LevelDisplay.tsx      # Level badge
│   │   │
│   │   ├── lobby/
│   │   │   ├── RoomCard.tsx
│   │   │   ├── PlayerSlot.tsx
│   │   │   └── TeamSelector.tsx
│   │   │
│   │   └── ui/
│   │       ├── Avatar.tsx            # Emoji avatars
│   │       ├── Button.tsx
│   │       ├── Modal.tsx
│   │       └── Toast.tsx
│   │
│   ├── game/
│   │   ├── engine/
│   │   │   ├── GameEngine.ts         # Main game state machine
│   │   │   ├── CardDeck.ts           # Deck creation & shuffle
│   │   │   ├── CardRanking.ts        # Card comparison logic
│   │   │   ├── CombinationValidator.ts
│   │   │   ├── CombinationDetector.ts
│   │   │   ├── BombDetector.ts
│   │   │   └── TributeManager.ts
│   │   │
│   │   └── types/
│   │       ├── Card.ts
│   │       ├── Combination.ts
│   │       ├── Player.ts
│   │       ├── GameState.ts
│   │       └── Trick.ts
│   │
│   ├── store/
│   │   ├── gameStore.ts              # Game state (Zustand)
│   │   ├── userStore.ts              # Auth & profile
│   │   └── settingsStore.ts          # User preferences
│   │
│   ├── services/
│   │   ├── supabase.ts               # Supabase client
│   │   ├── multiplayer.ts            # Realtime room sync
│   │   └── audio.ts                  # Sound effects (future)
│   │
│   ├── hooks/
│   │   ├── useGame.ts                # Game state logic
│   │   ├── useCardSelection.ts       # Card multi-select
│   │   ├── useAnimations.ts          # Animation timing
│   │   └── useTimer.ts               # Turn timer
│   │
│   ├── utils/
│   │   ├── cardUtils.ts              # Card helpers
│   │   └── gameUtils.ts              # Game helpers
│   │
│   └── constants/
│       ├── cards.ts                  # Card definitions
│       ├── theme.ts                  # Colors, fonts
│       └── avatars.ts                # Emoji avatars (9 animals)
│
├── assets/
│   ├── cards/                        # SVG card images
│   ├── sounds/                       # (future)
│   └── fonts/
│
├── app.json                          # Expo config
├── eas.json                          # EAS build config
├── tsconfig.json
├── package.json
└── .env.local                        # Supabase URL, anon key
```

**Avatar Constants (src/constants/avatars.ts):**
```typescript
export const AVATAR_EMOJIS = [
  '🦊', // 0 - Fox
  '🐼', // 1 - Panda
  '🦁', // 2 - Lion
  '🐨', // 3 - Koala
  '🐯', // 4 - Tiger
  '🐸', // 5 - Frog
  '🐷', // 6 - Pig
  '🐵', // 7 - Monkey
  '🐔', // 8 - Chicken
] as const;

export type AvatarIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

export const getAvatarEmoji = (index: AvatarIndex): string => {
  return AVATAR_EMOJIS[index];
};
```

---

## 3. CORE TYPE DEFINITIONS

### Card Type
```typescript
type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades';
type Rank = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K' | 'A';
type JokerType = 'red' | 'black';

interface Card {
  id: string;                // Unique: "hearts-7-0" (deck 0 = first deck)
  suit?: Suit;               // undefined for jokers
  rank?: Rank;               // undefined for jokers
  jokerType?: JokerType;     // 'red' | 'black'
  deckIndex: 0 | 1;          // Which of 2 decks
}
```

### Combination Type
```typescript
type CombinationType = 
  | 'single' | 'pair' | 'triple' | 'fullHouse' 
  | 'straight' | 'tube' | 'plate' | 'bomb';

type BombType =
  | 'quadruple' | 'quintuple' | 'straightFlush'
  | 'sextuple' | 'septuple' | 'octuple' | 'nonuple' | 'decuple'
  | 'fourJoker';

interface Combination {
  type: CombinationType;
  cards: Card[];
  rank: number;              // For comparison (higher = stronger)
  bombType?: BombType;       // Only if type === 'bomb'
  bombRank?: number;         // Ranking among bombs (0-8, 8=highest)
}
```

### Player & Game State
```typescript
interface Player {
  id: string;                // Supabase user ID
  name: string;
  avatarIndex: number;       // 0-8 (which emoji)
  hand: Card[];              // 27 cards (sorted)
  seat: 0 | 1 | 2 | 3;      // N=0, E=1, S=2, W=3
  team: 'NS' | 'EW';         // Partnership
  cardsRemaining: number;    // Quick lookup
  hasPassedThisTrick: boolean;
  finishOrder: number | null; // 1=first, 2=second, null=unfinished
}

interface GameState {
  id: string;                // Room ID = Game ID
  phase: 'lobby' | 'dealing' | 'tribute' | 'playing' | 'roundEnd';
  
  currentLevel: Rank;        // 2-A
  currentRound: number;      // 1, 2, 3, ...
  
  players: Player[];         // 4 players
  currentPlayerIndex: number; // 0-3
  currentTrick: Combination | null;
  trickCards: Card[];        // Cards played this trick (visual)
  playedBy: number[];        // Player indices who played to trick
  passCount: number;         // 3 = trick over
  
  tributes: Tribute[];       // (optional) Pending tributes
  
  finishOrder: number[];     // [1, 3, 0, 2] = order of players
  
  // Timer
  timeRemaining: number;     // Seconds left
  timerId?: NodeJS.Timeout;
  
  // Metadata
  createdAt: number;
  updatedAt: number;
  hostId: string;
}

interface Trick {
  cards: Card[];
  playedBy: number[];        // Which players played
  winner: number;            // Player index
  winningTeam: 'NS' | 'EW';
}
```

---

## 4. GAME ENGINE ARCHITECTURE

### 4.1 CardDeck
```typescript
class CardDeck {
  static create(): Card[] {
    // Creates 108 cards (2 decks + 4 jokers)
    // All cards are unique by ID
  }
  
  static shuffle(cards: Card[]): Card[] {
    // Fisher-Yates shuffle
  }
  
  static deal(cards: Card[]): Card[][] {
    // Returns 4 hands of 27 cards each
    // Hands are pre-sorted
  }
}
```

### 4.2 CardRanking
```typescript
class CardRanking {
  constructor(private currentLevel: Rank) {}
  
  getEffectiveRank(card: Card): number {
    // Red joker: 100
    // Black joker: 99
    // Level card: 98
    // A: 14, K: 13, ..., 2: 2
  }
  
  isWild(card: Card): boolean {
    // True if heart of current level
  }
  
  compare(a: Card, b: Card): number {
    // Returns -1, 0, or 1
  }
}
```

### 4.3 CombinationValidator
```typescript
class CombinationValidator {
  constructor(private ranking: CardRanking) {}
  
  validate(cards: Card[]): Combination | null {
    // Detects what combo type (if any) these cards form
    // Returns null if not a valid combination
  }
  
  canBeat(current: Combination, attempt: Combination): boolean {
    // True if attempt beats current
    // Bombs beat everything except higher bombs
  }
}
```

### 4.4 CombinationDetector
```typescript
class CombinationDetector {
  constructor(private ranking: CardRanking) {}
  
  findAllCombinations(hand: Card[]): Combination[] {
    // Returns all valid combos in hand
    // Used when leading a trick
  }
  
  findBeatingCombinations(
    hand: Card[], 
    currentTrick: Combination
  ): Combination[] {
    // Returns only combos that beat current
    // Or empty array if none can beat
  }
  
  detectStraightFlushes(hand: Card[]): Map<Suit, Combination[]> {
    // Returns all straight flushes grouped by suit
    // Highest first in each suit
  }
}
```

### 4.5 BombDetector
```typescript
class BombDetector {
  constructor(private ranking: CardRanking) {}
  
  detectAllBombs(hand: Card[]): Combination[] {
    // Returns all bombs in hand, ranked 0-8
    // [quadruple, quintuple, ..., fourJoker]
  }
}
```

### 4.6 GameEngine (State Machine)
```typescript
class GameEngine {
  constructor(state: GameState) {}
  
  // Main flow
  startGame(): void {}
  deal(): void {}
  processPlay(playerIdx: number, combo: Combination): Result {}
  processPass(playerIdx: number): Result {}

  // Checks
  getValidPlays(playerIdx: number): Combination[] {}
  canBeat(currentTrick: Combination): boolean {}
  
  // Transitions
  finishTrick(): void {}
  finishRound(): void {}
  processTribute(): void {} // Tracks which card each player gives
  selectNextLeader(): number {} // NEW: Based on highest tribute card given
  
  // Utilities
  getState(): GameState {}
  serialize(): string {} // For Supabase
}
```

### 4.7 TributeManager
```typescript
class TributeManager {
  private tributeCards: Map<number, Card> = new Map();

  getTributeRequirements(finishOrder: number[]): Tribute[] {
    // Determines which players must exchange which cards
    // 1-4 win: 4 gives to 1
    // 1-3 win: 4 gives to 1
    // 1-2 win: 4 gives to 1, 3 gives to 2
  }

  canCancelTribute(player: Player): boolean {
    // True if player has both red jokers
  }

  recordTributeCard(playerId: number, card: Card): void {
    // Store which card this player gave in tribute
    this.tributeCards.set(playerId, card);
  }

  selectLeaderByTribute(ranking: CardRanking): number {
    // Returns player index of highest tribute card giver
    // Uses CardRanking to compare card values
    // If equal: random selection using Math.random()

    let highestCard: Card | null = null;
    let leaderId: number | null = null;
    const tiedPlayers: number[] = [];

    this.tributeCards.forEach((card, playerId) => {
      const comparison = highestCard
        ? ranking.compare(card, highestCard)
        : 1;

      if (comparison > 0) {
        // This card is higher
        highestCard = card;
        leaderId = playerId;
        tiedPlayers.length = 0;
        tiedPlayers.push(playerId);
      } else if (comparison === 0) {
        // Tied - need random selection
        tiedPlayers.push(playerId);
      }
    });

    if (tiedPlayers.length > 1) {
      // Random selection among tied players
      return tiedPlayers[Math.floor(Math.random() * tiedPlayers.length)];
    }

    return leaderId ?? 0; // Fallback to player 0 if no tributes
  }

  clearTributeCards(): void {
    // Reset for next round
    this.tributeCards.clear();
  }

  processExchange(from: number, to: number, card: Card): void {
    // Actually move card between players
  }
}
```

---

## 5. SUPABASE SCHEMA

### Tables

#### `users`
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  display_name TEXT NOT NULL,
  avatar_index INT DEFAULT 0 CHECK (avatar_index >= 0 AND avatar_index <= 8),
  device_id TEXT UNIQUE,
  created_at TIMESTAMP DEFAULT NOW(),
  last_active TIMESTAMP DEFAULT NOW()
);

-- No email field for MVP (anonymous authentication)
-- Users created on first app launch
-- device_id optional for device recognition
```

#### `rooms`
```sql
CREATE TABLE rooms (
  id TEXT PRIMARY KEY,  -- 6-char code
  host_id UUID REFERENCES users(id),
  status TEXT DEFAULT 'waiting',  -- waiting, playing, finished
  game_state JSONB,     -- Serialized GameState
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_rooms_status ON rooms(status);
```

#### `room_players`
```sql
CREATE TABLE room_players (
  room_id TEXT REFERENCES rooms(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id),
  seat INT,             -- 0-3 (can be null before assignment)
  team TEXT,            -- 'NS' or 'EW'
  PRIMARY KEY (room_id, user_id)
);
```

#### `game_actions` (for audit/replay)
```sql
CREATE TABLE game_actions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  room_id TEXT REFERENCES rooms(id),
  player_id UUID REFERENCES users(id),
  action_type TEXT,     -- 'play', 'pass', '+time'
  action_data JSONB,    -- Cards, etc
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Realtime Subscriptions
```typescript
// Subscribe to room changes
supabase
  .channel(`room:${roomId}`)
  .on('postgres_changes', 
    { event: 'UPDATE', schema: 'public', table: 'rooms' },
    (payload) => {
      // Broadcast game state updates to all players
    }
  )
  .subscribe();
```

---

## 6. STATE MANAGEMENT (Zustand)

### gameStore
```typescript
const useGameStore = create((set) => ({
  gameState: null as GameState | null,
  
  initGame: (state: GameState) => set({ gameState: state }),
  updateState: (partial: Partial<GameState>) => 
    set((state) => ({
      gameState: { ...state.gameState, ...partial }
    })),
  
  playCard: (playerIdx: number, combo: Combination) => {
    // Update game state with play
  },
  
  passCard: (playerIdx: number) => {
    // Update game state with pass
  },
}));
```

### userStore
```typescript
const useUserStore = create((set) => ({
  user: null as User | null,
  isLoading: false, // No async auth needed for anonymous

  // Create anonymous user on first launch
  initAnonymousUser: async (displayName: string, avatarIndex: number) => {
    const { data, error } = await supabase
      .from('users')
      .insert({ display_name: displayName, avatar_index: avatarIndex })
      .select()
      .single();

    if (data) {
      set({ user: data });
      // Store user ID in AsyncStorage for persistence
      await AsyncStorage.setItem('userId', data.id);
    }
  },

  // Load existing user from device
  loadUser: async () => {
    const userId = await AsyncStorage.getItem('userId');
    if (userId) {
      const { data } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .single();

      if (data) set({ user: data });
    }
  },

  updateDisplayName: async (newName: string) => {
    // Update display name in database
  },

  updateAvatar: async (newIndex: number) => {
    // Update avatar index in database
  },
}));
```

---

## 7. MULTIPLAYER SYNC STRATEGY

### Event Flow
```
Player 1 clicks [Play]
  ↓
Client validates combo locally
  ↓
Client sends to Supabase: { action: 'play', combo }
  ↓
Server processes (double-check validation)
  ↓
Server updates GameState in room
  ↓
Supabase broadcasts to all 4 players
  ↓
Clients receive update, animate cards
  ↓
Turn moves to Player 2
```

### Latency Budget
- Player action → Server: <300ms
- Server validation: <50ms
- Broadcast to other 3: <100ms
- **Total:** <500ms (acceptable for turn-based game)

---

## 8. DEPLOYMENT (EAS)

### eas.json
```json
{
  "build": {
    "preview": {
      "ios": {
        "simulator": true
      }
    },
    "production": {
      "ios": {
        "enterpriseProvisioning": "universal"
      }
    }
  },
  "submit": {
    "production": {
      "ios": {
        "ascAppId": "APP_ID_HERE"
      }
    }
  }
}
```

### Build Commands
```bash
# Local development
npx expo start

# Build preview (simulator)
eas build --platform ios --profile preview

# Build production
eas build --platform ios

# Submit to TestFlight
eas submit --platform ios
```

---

## 9. PERFORMANCE TARGETS

### Acceptable Latencies
- Card selection: instant (<16ms per frame)
- Valid play detection: <100ms
- Network sync: <500ms
- Animation (card deal/play): 300-500ms (intentional)

### Memory Usage
- Game state: <5MB
- 27-card hand: <100KB
- Full game session: <10MB

### Battery Impact
- Realtime multiplayer: ~5% per 15-min game
- Animations: GPU-accelerated (minimal CPU)

---

## 10. ERROR HANDLING

### Network Errors
- **Connection lost:** Show "Reconnecting..." toast message
- **Reconnection window:** 30 seconds maximum
- **If reconnects within 30s:** Game state restored, player resumes from exact state
- **If timeout (30s expired):**
  - Game immediately cancelled
  - All 4 players receive notification: "[Player X] disconnected - game cancelled"
  - Room status set to 'cancelled' in database
  - All players returned to home screen
  - Game state NOT saved (no partial game recovery)
- **Server unreachable:** Offline mode disabled (multiplayer only app)

**Implementation Details:**
- Supabase Realtime automatically detects disconnects via heartbeat
- Client-side: Set timeout timer when disconnect detected
- Server-side: Update room status to 'cancelled' after 30s
- Broadcast cancellation event to all connected players
- Clean up room from active games table

### Game Logic Errors
- Invalid play: Show error message, don't submit
- Duplicate plays: Server deduplicates
- Race conditions: Server enforces strict turn order

### Server Rejection of Client Plays

**Scenario:** Client-validated play is rejected by server due to state desync

**Causes:**
- Network latency causes client to have stale game state
- Client thinks it's their turn, server knows it's not
- Client validation bug (rare edge case)
- Player attempts to play out of turn

**Handling:**
1. Server returns error: `{ error: "INVALID_PLAY", reason: "Not your turn", gameState: {...} }`
2. Client receives rejection
3. Display to user: "Connection issue - refreshing game state"
4. Force client to re-sync from server's authoritative state
5. Update all UI components with fresh state
6. Resume gameplay from corrected state

**Implementation:**
```typescript
onServerRejection(error: ServerError) {
  // Show user-friendly message
  showToast("Connection issue - refreshing game state");

  // Force re-sync from server
  const freshState = await fetchGameState();
  gameStore.setState(freshState);

  // Log for debugging (but don't show to user)
  console.error("Server rejection:", error);
}
```

**User Experience:**
- Never show technical error messages ("Invalid play rejected")
- Always treat as network sync issue to avoid frustration
- Game continues seamlessly after refresh

### Crash Recovery
- Game state auto-saved after every turn
- If app crashes: Rejoin room to resume
- If room invalid: Return to home

---

## 11. TESTING STRATEGY

### Unit Tests
- CardDeck shuffle randomness
- CardRanking comparisons
- CombinationValidator all 7 types + 9 bombs
- BombDetector ranking
- TributeManager logic

### Integration Tests
- Full game flow (13 rounds)
- Disconnect/reconnect
- Team assignment validation
- Leader selection logic

### E2E Tests (via EAS)
- 4 phones playing simultaneously
- Real network conditions
- TestFlight beta feedback

---

## 12. GETTING STARTED

```bash
# Install dependencies
npm install -g eas-cli
npx create-expo-app chucking-eggs -t expo-template-blank-typescript
cd chucking-eggs

# Install packages
npx expo install expo-router react-native-reanimated
npm install zustand @supabase/supabase-js

# Set up EAS
eas login
eas build:configure --platform ios

# Development
npx expo start

# Build
eas build --platform ios --profile preview
```

---

*Technical Documentation v1.0 - February 2026*
