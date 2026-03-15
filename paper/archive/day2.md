# Guan Dan DMC: Day 2 — Heuristic Opponent + Encoding Upgrade

> **Status**: DMC agent hits 82-96% WR vs random at 4K episodes on GPU. Engine is correct.
> **Day 2 Goal**: Build a rule-based heuristic opponent that exploits the DMC agent's weaknesses,
> add the `remaining_after_play` encoding from the full plan, and establish proper evaluation protocol.
> **Expected outcome**: DMC agent initially loses to heuristic (~40-45% WR), then trains up to ~55-60%.

---

## 0. Why a Heuristic Matters

Random play is a terrible benchmark. A random agent:
- Leads bombs on the first trick (wasting them)
- Passes when it could easily beat a low single
- Never helps its partner
- Has no concept of hand structure

Beating random at 90% just means your agent learned "play cards > don't play cards." The DanZero paper evaluated against **8 rule-based bots from the first Chinese AI for GuanDan Competition**. Even the weakest competition bots crushed random, and DanZero still only hit ~80-90% against most of them after 30 days on 160 CPUs. A decent heuristic will immediately expose whether the DMC agent has learned any real strategy.

More importantly, the heuristic serves three purposes going forward:
1. **Evaluation**: A stable, deterministic opponent gives reproducible winrate measurements
2. **Soft-start mixing** (Day 4): Blend heuristic actions with ε-greedy to accelerate early training
3. **Curriculum**: Train against heuristic first, then self-play, to avoid early-training drift

---

## 1. Heuristic Agent Design

The heuristic has two modes: **leading** (free lead, you choose what to play) and **following** (someone played, you must beat it or pass). Each mode has a decision procedure.

### 1.1 Core Data Structure: Hand Decomposition

Before deciding what to play, the heuristic decomposes its hand into **natural groups**. This is the key abstraction — it answers "what combos does my hand naturally contain?"

```python
# heuristic.py

class HandPlan:
    """Decompose a hand into natural groups for decision-making."""
    
    def __init__(self, hand, level_rank):
        self.hand = hand
        self.level_rank = level_rank
        self.wilds = [c for c in hand if is_wild(c, level_rank)]
        self.naturals = [c for c in hand if not is_wild(c, level_rank)]
        
        # Group naturals by rank
        self.by_rank = {}
        for c in self.naturals:
            self.by_rank.setdefault(c.rank, []).append(c)
        
        # Classify groups
        self.singles = []    # ranks with exactly 1 card
        self.pairs = []      # ranks with exactly 2 cards
        self.triples = []    # ranks with exactly 3 cards
        self.quads = []      # ranks with 4+ cards (potential bombs)
        
        for rank, cards in sorted(self.by_rank.items(),
                                   key=lambda x: level_order_key(x[0], level_rank)):
            n = len(cards)
            if n == 1:
                self.singles.append((rank, cards))
            elif n == 2:
                self.pairs.append((rank, cards))
            elif n == 3:
                self.triples.append((rank, cards))
            else:
                self.quads.append((rank, cards))
        
        # Jokers (tracked separately)
        self.jokers = [c for c in hand
                       if c.rank in (Rank.BLACK_JOKER, Rank.RED_JOKER)]
        self.n_bombs = len(self.quads) + (1 if len(self.jokers) == 4 else 0)
    
    def weakest_single_rank(self):
        """Lowest-ranked single card (level order). None if no singles."""
        return self.singles[0][0] if self.singles else None
    
    def weakest_pair_rank(self):
        return self.pairs[0][0] if self.pairs else None
    
    def weakest_triple_rank(self):
        return self.triples[0][0] if self.triples else None
```

### 1.2 Leading Logic

When the heuristic has a free lead, it follows this priority:

```
LEADING PRIORITY (top to bottom):
  
  1. If I have ≤ 3 cards left and can win in 1-2 plays → play to go out
  2. If partner is out and I need to finish → play aggressively (strongest leads)
  3. Play weakest SINGLE (shed low garbage)
  4. If no singles: play weakest PAIR
  5. If no pairs: play weakest TRIPLE  
  6. If no triples: play weakest FULL HOUSE (if any triple+pair exists)
  7. If forced: play weakest STRAIGHT / TUBE / PLATE (don't break these voluntarily)
  8. Last resort: play a BOMB (shouldn't happen unless hand is all bombs)
  
  NEVER lead with wilds as singles unless nothing else is possible.
  NEVER voluntarily break a bomb to lead singles/pairs.
```

```python
def heuristic_lead(hand, level_rank):
    """Choose a combo to lead with (free lead)."""
    plan = HandPlan(hand, level_rank)
    all_leads = generate_all_leads(hand, level_rank)
    
    # --- Endgame: can we go out in one play? ---
    for combo in all_leads:
        if len(combo.cards) == len(hand):
            return combo  # play everything, go out immediately
    
    # --- Shed weakest singles first ---
    if plan.singles:
        rank = plan.singles[0][0]  # weakest single by level order
        cards = plan.singles[0][1]
        return _find_combo(all_leads, ComboType.SINGLE, rank)
    
    # --- Shed weakest pairs ---
    if plan.pairs:
        rank = plan.pairs[0][0]
        return _find_combo(all_leads, ComboType.PAIR, rank)
    
    # --- Shed weakest triples ---
    if plan.triples:
        rank = plan.triples[0][0]
        # Prefer full house (triple + pair) to shed more cards
        fh = _find_combo(all_leads, ComboType.FULL_HOUSE, rank)
        if fh:
            return fh
        return _find_combo(all_leads, ComboType.TRIPLE, rank)
    
    # --- Multi-card combos (straights, tubes, plates) ---
    # Play the lowest-ranked one available
    for ctype in [ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE]:
        candidates = [c for c in all_leads if c.type == ctype]
        if candidates:
            return min(candidates, key=lambda c: c.key)
    
    # --- Bombs as last resort ---
    non_pass = [c for c in all_leads if c.type != ComboType.PASS]
    if non_pass:
        # Play weakest bomb
        return min(non_pass, key=lambda c: (c.type, c.key))
    
    # Should never reach here (leader always has at least one play)
    return all_leads[0]
```

### 1.3 Following Logic

When responding to a trick, the heuristic considers:
- **Who is currently winning the trick?** Partner or opponent?
- **How expensive is the cheapest beat?**
- **Should I bomb?**

```
FOLLOWING PRIORITY:
  
  1. If PARTNER is winning the trick → PASS (let partner keep control)
  2. If opponent is winning:
     a. Find the CHEAPEST beat of the same type (lowest rank that beats)
     b. If cheapest beat requires breaking a bomb → PASS instead
     c. If cheapest beat is a singleton from a quad → PASS (don't break bomb)
     d. Otherwise → play the cheapest beat
  3. BOMB decision (only if we passed on same-type beat):
     a. If opponent played something very strong (A or higher single, 
        or any combo with key ≥ A) AND I have 2+ bombs → bomb with weakest
     b. If opponent has ≤ 5 cards left (about to win) → bomb with weakest
     c. Otherwise → PASS
  4. Default → PASS
```

```python
def heuristic_follow(hand, level_rank, trick, trick_winner, my_seat):
    """Choose a response to a trick in progress."""
    plan = HandPlan(hand, level_rank)
    responses = generate_responses(hand, level_rank, trick)
    
    partner = (my_seat + 2) % 4
    partner_is_winning = (trick_winner == partner)
    
    # --- If partner is winning → pass ---
    if partner_is_winning:
        return _pass_combo(responses)
    
    # --- Opponent is winning: find cheapest same-type beat ---
    same_type_beats = [c for c in responses
                       if c.type == trick.type and c.type != ComboType.PASS]
    
    if same_type_beats:
        # Sort by key (cheapest first)
        same_type_beats.sort(key=lambda c: level_order_key(c.key, level_rank)
                              if c.type in (ComboType.SINGLE, ComboType.PAIR,
                                           ComboType.TRIPLE, ComboType.FULL_HOUSE)
                              else c.key)
        cheapest = same_type_beats[0]
        
        # Don't break a bomb to follow
        if not _breaks_bomb(cheapest, plan):
            return cheapest
    
    # --- Bomb decision ---
    bombs = [c for c in responses if c.type in BOMB_TYPES]
    if bombs:
        bombs.sort(key=lambda c: (c.type, c.key))  # weakest bomb first
        
        # Bomb if: opponent is close to winning, or trick is very strong
        opponent_close = False  # would need env info — approximate with trick strength
        trick_is_strong = (trick.type in BOMB_TYPES or
                          level_order_key(trick.key, level_rank) >= Rank.ACE)
        
        if trick_is_strong and len(bombs) >= 2:
            return bombs[0]  # use weakest bomb, save stronger ones
        
        # Conservative: don't bomb weak tricks
    
    return _pass_combo(responses)


def _breaks_bomb(combo, plan):
    """Would playing this combo break up a quad (potential bomb)?"""
    # Check if any card in combo comes from a rank where we have 4+
    for card in combo.cards:
        if card.rank in [r for r, _ in plan.quads]:
            return True
    return False

def _pass_combo(responses):
    """Find the PASS action in responses."""
    for c in responses:
        if c.type == ComboType.PASS:
            return c
    return responses[-1]  # fallback

def _find_combo(combos, combo_type, key_rank):
    """Find a combo matching type and key rank."""
    for c in combos:
        if c.type == combo_type and c.key == key_rank:
            return c
    return None
```

### 1.4 Top-Level Agent Interface

```python
class HeuristicAgent:
    """Rule-based Guan Dan agent."""
    
    def __init__(self, level_rank=Rank.TWO):
        self.level_rank = level_rank
    
    def choose_action(self, env, player):
        """Given game state, return a Combo to play."""
        hand = env.hands[player]
        
        if env.current_trick is None:
            # Free lead
            return heuristic_lead(hand, self.level_rank)
        else:
            # Following
            return heuristic_follow(
                hand, self.level_rank,
                env.current_trick, env.trick_winner, player
            )
```

### 1.5 What This Heuristic Does Well (and Poorly)

**Does well** (compared to random):
- Sheds weak cards first instead of randomly dumping strong ones
- Doesn't waste bombs on low tricks
- Lets partner win tricks instead of needlessly overplaying
- Has basic endgame awareness (play everything if you can go out)

**Does poorly** (what the RL agent should exploit):
- No card counting (doesn't track what's been played to infer opponents' hands)
- No sequence planning (doesn't think "if I lead this, I can follow with that")
- Rigid priority ordering (always singles→pairs→triples, never adapts)
- No bluffing or deception
- Doesn't consider partner's hand size or likely holdings
- Bomb timing is crude (only bombs "strong" tricks, misses tactical opportunities)
- Doesn't help partner go out — only passive cooperation (passing when partner wins)

This gives the RL agent a clear skill ladder to climb. Beating this heuristic requires genuine strategic understanding.

---

## 2. `remaining_after_play` Encoding Upgrade

This is the highest-value encoding upgrade from the full DanZero plan. The idea: when evaluating an action, also encode **what your hand looks like after playing it**. This helps the network reason about future playability.

From the full plan (§3.2):
> The `remaining_after_play` matrix is critical. NetEase's commercial system found that encoding what your hand looks like after a play helps the network reason about hand structure and future playability.

### Implementation

```python
# In encoding.py — modify encode_action()

def encode_action(combo, hand, level_rank):
    """
    Action features WITH remaining_after_play.
    hand = current hand BEFORE playing this combo.
    """
    cards_played = cards_to_matrix(combo.cards).flatten()              # 60
    
    # NEW: remaining hand after this play
    remaining = set(hand) - set(combo.cards)
    remaining_matrix = cards_to_matrix(remaining).flatten()            # 60  ← NEW
    
    combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)       # 17
    combo_type_oh[combo.type] = 1.0
    combo_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)              # 15
    if 2 <= combo.key <= 17:
        combo_key_oh[_rank_index(combo.key)] = 1.0
    num_cards = np.array([len(combo.cards) / 10.0], dtype=np.float32) # 1
    is_bomb = np.array([float(combo.type in BOMB_TYPES)], dtype=np.float32) # 1
    wild_oh = np.zeros(3, dtype=np.float32)                           # 3
    wild_oh[min(combo.wild_count, 2)] = 1.0
    
    # NEW: summary stats about remaining hand
    n_remaining = len(remaining)
    n_singles_after = 0
    n_bombs_after = 0
    by_rank = {}
    for c in remaining:
        by_rank.setdefault(c.rank, []).append(c)
    for rank, cards in by_rank.items():
        if len(cards) == 1:
            n_singles_after += 1
        if len(cards) >= 4:
            n_bombs_after += 1
    
    hand_stats = np.array([
        n_remaining / 27.0,         # cards left (normalized)
        n_singles_after / 10.0,     # isolated singles (bad — hard to shed)
        n_bombs_after / 3.0,        # bombs remaining (good — control)
    ], dtype=np.float32)                                               # 3  ← NEW
    
    return np.concatenate([
        cards_played, remaining_matrix, combo_type_oh, combo_key_oh,
        num_cards, is_bomb, wild_oh, hand_stats
    ])
    # Total: 60 + 60 + 17 + 15 + 1 + 1 + 3 + 3 = 160 dims (was 97)
```

**Why this helps**: Without `remaining_after_play`, the network has to separately compute "hand matrix minus action cards" from the state and action encodings — a subtraction that MLPs struggle with. By precomputing it, we make the network's job easier. The `hand_stats` features (singles count, bomb count) give the network direct access to hand-quality signals it would otherwise have to learn from scratch.

**Update Q-network input dim**: `d_action` goes from 97 → 160.

```python
class QNetwork(nn.Module):
    def __init__(self, d_state=417, d_action=160, hidden=256):
        ...
```

**Important**: You must retrain from scratch after this change (the action encoding dimension changed). Don't try to load old checkpoints.

---

## 3. Evaluation Protocol

Proper evaluation is critical now that we have two opponents (random and heuristic).

### 3.1 Evaluation Function

```python
def evaluate(q_net, opponent='random', n_games=500, level_rank=Rank.TWO):
    """
    Evaluate DMC agent vs opponent.
    DMC agent plays as team {0, 2}. Opponent plays as team {1, 3}.
    Returns: winrate, avg_reward, finish_position_distribution.
    """
    heuristic = HeuristicAgent(level_rank) if opponent == 'heuristic' else None
    env = GuanDanEnv(level_rank)
    
    wins = 0
    total_reward = 0.0
    finish_positions = {0: [0,0,0,0], 2: [0,0,0,0]}  # track where our players finish
    
    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            
            if player in (0, 2):
                # DMC agent
                legal = env.legal_moves()
                state_enc = encode_state(env, player)
                action_encs = [encode_action(m, env.hands[player], level_rank)
                               for m in legal]
                with torch.no_grad():
                    s = torch.tensor(state_enc).unsqueeze(0).expand(len(legal), -1)
                    a = torch.tensor(np.array(action_encs))
                    idx = q_net(s, a).argmax().item()
                env.step(legal[idx])
            else:
                # Opponent
                if heuristic:
                    action = heuristic.choose_action(env, player)
                else:
                    action = random.choice(env.legal_moves())
                env.step(action)
        
        rewards = env.get_rewards()
        team_reward = rewards[0] + rewards[2]
        if team_reward > 0:
            wins += 1
        total_reward += team_reward
        
        # Track finish positions
        for pos, p in enumerate(env.finish_order):
            if p in (0, 2):
                finish_positions[p][pos] += 1
    
    winrate = wins / n_games
    avg_reward = total_reward / n_games
    
    return {
        'winrate': winrate,
        'avg_reward': avg_reward,
        'finish_positions': finish_positions,
        'n_games': n_games
    }
```

### 3.2 What to Log

During training, evaluate every `eval_interval` episodes against BOTH opponents:

```python
# In train.py main loop

if episode % eval_interval == 0:
    # Eval vs random (sanity check — should stay high)
    r = evaluate(q_net, opponent='random', n_games=200)
    print(f"  vs Random:    WR={r['winrate']:.1%}  AvgR={r['avg_reward']:+.2f}")
    
    # Eval vs heuristic (the real benchmark)
    h = evaluate(q_net, opponent='heuristic', n_games=200)
    print(f"  vs Heuristic: WR={h['winrate']:.1%}  AvgR={h['avg_reward']:+.2f}")
    
    # Finish position breakdown (tells you HOW you're winning/losing)
    # e.g., {0: [80, 40, 50, 30], 2: [70, 60, 40, 30]} 
    # means player 0 finished 1st in 80 games, 2nd in 40, etc.
```

### 3.3 Expected Progression

```
Training vs self-play, evaluating vs both:

Ep    0   | vs Random: 50% | vs Heuristic: 25-30%    ← heuristic crushes untrained agent
Ep  2000  | vs Random: 85% | vs Heuristic: 35-40%    ← learning to play, but heuristic still dominant
Ep  5000  | vs Random: 90% | vs Heuristic: 42-48%    ← closing the gap
Ep 10000  | vs Random: 92% | vs Heuristic: 50-55%    ← crossing even with remaining_after_play
Ep 20000  | vs Random: 93% | vs Heuristic: 55-60%    ← target for Day 2
```

If WR vs heuristic plateaus below 45%, the agent is likely stuck in local optima from pure self-play. This is where **soft-start mixing** (Day 4) would help — blend heuristic actions into self-play so the agent sees better trajectories early on.

---

## 4. Training Configuration Changes

### 4.1 Self-Play Remains the Training Regime

The agent still trains via self-play (all 4 seats use the Q-network with ε-greedy). The heuristic is used **only for evaluation**, not for training. This is deliberate — training against a fixed heuristic would overfit to exploiting its specific weaknesses rather than learning general strategy.

### 4.2 Hyperparameter Adjustments for Day 2

```python
# train.py config

EPISODES = 30_000           # GPU should handle this in a few hours
BATCH_SIZE = 512            # bump from 256 — more stable gradients
BUFFER_SIZE = 200_000       # bump from 100K — more diverse experience
LR = 1e-4                   # keep same
EPSILON_START = 0.25        # slightly lower start (agent has some skill now)
EPSILON_END = 0.02          # lower floor for more exploitation
EPSILON_DECAY_EPISODES = 20_000
EVAL_INTERVAL = 500
EVAL_GAMES = 200
HIDDEN_DIM = 256            # keep same
```

### 4.3 Fresh Start vs Fine-Tune

Because the action encoding dimension changed (97 → 160), you **must train from scratch**. This is actually fine — with GPU training and the encoding upgrade, you should surpass your Day 1 results within the first 2-3K episodes.

---

## 5. Day 2 Implementation Checklist

### Morning (~3 hours)

- [ ] **Implement `HandPlan` class** — hand decomposition into groups
- [ ] **Implement `heuristic_lead()`** — the leading priority chain
- [ ] **Implement `heuristic_follow()`** — follow/pass/bomb logic
- [ ] **Test heuristic vs random** — run 1000 games, verify heuristic wins 70-80%
- [ ] **Test heuristic vs heuristic** — run 1000 games, verify ~50% for each team

### Midday (~2 hours)

- [ ] **Add `remaining_after_play`** to `encode_action()` — the 60-dim matrix + 3-dim stats
- [ ] **Update `QNetwork` input dims** — d_action = 160
- [ ] **Update `play_episode()`** — pass `hand` to `encode_action()`
- [ ] **Update `evaluate()`** — same change
- [ ] **Verify shapes** — run one episode, print tensor shapes, confirm no crashes

### Afternoon (~3 hours)

- [ ] **Train from scratch with new encoding** — kick off 30K episodes on GPU
- [ ] **Monitor dual eval** — both vs random and vs heuristic every 500 episodes
- [ ] **Log everything** — loss, Q-value stats, both winrates, finish position distribution

### Evening

- [ ] **Analyze results** — where does the agent finish? 1st? 4th? Does it help its partner?
- [ ] **Identify next bottleneck** — is it the encoding? Network capacity? Training volume?

---

## 6. Heuristic Validation: Expected Behavior

Before using the heuristic as an eval opponent, verify it plays sensibly:

```python
def validate_heuristic(n=20):
    """Print games to verify heuristic makes reasonable decisions."""
    env = GuanDanEnv()
    agent = HeuristicAgent()
    
    for game_idx in range(n):
        env.reset()
        moves = []
        while not env.done:
            player = env.current_player
            action = agent.choose_action(env, player)
            moves.append((player, action.type.name, action.key, len(action.cards)))
            env.step(action)
        
        if game_idx < 3:  # print first 3 games
            print(f"\n=== Game {game_idx} ===")
            print(f"Finish order: {env.finish_order}")
            print(f"Rewards: {env.get_rewards()}")
            print(f"Total moves: {len(moves)}")
            # Print last 10 moves
            for p, tname, key, ncards in moves[-10:]:
                print(f"  P{p}: {tname} key={key} ({ncards} cards)")
```

**What to check**:
- Games terminate (no infinite loops)
- Heuristic doesn't lead with bombs early
- Heuristic passes when partner is winning
- Average game length is ~80-120 moves (shorter than random — heuristic plays more efficiently)
- Heuristic wins 70-80% vs random

---

## 7. Diagnostic: Finish Position Analysis

The most informative metric isn't just winrate — it's **where each player finishes**. This tells you about partnership cooperation:

```python
def analyze_finish_positions(results):
    """
    results['finish_positions'] = {0: [n_1st, n_2nd, n_3rd, n_4th],
                                    2: [n_1st, n_2nd, n_3rd, n_4th]}
    """
    for player in [0, 2]:
        counts = results['finish_positions'][player]
        total = sum(counts)
        pcts = [c/total*100 for c in counts]
        print(f"Player {player}: 1st={pcts[0]:.0f}% 2nd={pcts[1]:.0f}% "
              f"3rd={pcts[2]:.0f}% 4th={pcts[3]:.0f}%")
    
    # Partnership analysis
    p0 = results['finish_positions'][0]
    p2 = results['finish_positions'][2]
    total = sum(p0)
    
    # How often do teammates finish 1-2 (best outcome)?
    # This requires game-level tracking, but we can approximate:
    # If both players have high 1st+2nd rates, cooperation is good
    team_top2 = (p0[0]+p0[1]+p2[0]+p2[1]) / (2*total)
    print(f"Team top-2 rate: {team_top2:.0%} (higher = better cooperation)")
```

**What good cooperation looks like**:
- When Player 0 finishes 1st, Player 2 finishes 2nd or 3rd (not 4th)
- Neither player dominates — both contribute roughly equally
- Against heuristic, you want to see 1-2 wins (4 level jumps), not just 1-4 wins

---

## 8. Day 3 Preview: What Comes Next

After Day 2, the bottleneck shifts from "does the agent play at all" to "can it count cards and remember history." The MLP sees only the current state — it has no memory of what happened earlier in the trick sequence.

Day 3 adds:
- **LSTM history encoder**: Feed the last 15-20 moves through an LSTM, concat with state encoding. This lets the network learn card-counting and opponent modeling implicitly.
- **Larger buffer (500K)**: More experience diversity as the agent gets stronger.
- **Larger batch (1024)**: Smoother gradients for the now-larger network.
- **Expected improvement**: WR vs heuristic jumps from ~55-60% to ~65%.