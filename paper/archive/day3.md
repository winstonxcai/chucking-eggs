# Guan Dan DMC: Day 3 — LSTM History + Lead/Follow Split

> **Status**: 59% WR vs heuristic with MLP + remaining_after_play. ~92% vs random.
> **Day 3 Goal**: Add move history via LSTM so the agent can implicitly count cards and
> read opponent patterns. Split into lead/follow Q-networks. Scale up buffer and batch.
> **Expected outcome**: WR vs heuristic climbs from 59% → ~65-68%.

---

## 0. Why the MLP Plateaued

The current MLP sees a snapshot of the world: your hand, what's been played total, the current trick, card counts. But it has **zero memory of the sequence of play**. It can't answer questions like:

- "Player 3 passed on a low pair — do they have bombs?" (A pass on something easy to beat signals strength elsewhere)
- "Player 1 has been leading low singles — they're probably setting up to dump a straight" (Behavioral patterns over multiple tricks)
- "The last 4 plays were all passes — everyone is weak in this combo type" (Trick-level dynamics)
- "3 Kings have appeared so far — the 4th is probably in player 2's hand" (Fine-grained card counting from the play sequence, beyond the aggregate `played_by_X` matrices)

The `played_by_X` card matrices in the state encoding give aggregate counts, but they lose **ordering information**. Knowing that player 1 played K♠ *after* seeing player 3 play Q♠ is different from knowing they both played those cards at some point. The LSTM recovers this temporal signal.

The DanZero paper uses a 513-dim state + 54-dim action with an MLP, but relies on massive scale (160 CPUs, 30 days) to compensate for the lack of history. The full blueprint (§2) adds LSTM over the last 20 moves. DouZero proved this gives 2-5% winrate improvement at the same training budget.

---

## 1. Move History Encoding

Each past move becomes a fixed-size vector. We store the last **T=15 moves** (covers ~4 tricks, enough for pattern detection without excessive padding).

### 1.1 Per-Move Event Vector

```python
# encoding.py — add this

def encode_move_event(actor_relative, combo, level_rank):
    """
    Encode a single move (by any player) as a fixed-size vector.
    
    actor_relative: int 0-3 (0=me, 1=right, 2=partner, 3=left in CCW)
    combo: Combo object (the play made), or None for PASS
    
    Returns: numpy array of ~83 dims
    """
    # Who played (one-hot, relative to me)
    actor_oh = np.zeros(4, dtype=np.float32)                     # 4
    actor_oh[actor_relative] = 1.0
    
    # What they played (card matrix, flattened)
    if combo is not None and combo.type != ComboType.PASS:
        card_mat = cards_to_matrix(combo.cards).flatten()         # 60
        is_pass = np.array([0.0], dtype=np.float32)               # 1
        combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)  # 17
        combo_type_oh[combo.type] = 1.0
        is_bomb = np.array([float(combo.type in BOMB_TYPES)],
                           dtype=np.float32)                      # 1
    else:
        card_mat = np.zeros(60, dtype=np.float32)                 # 60
        is_pass = np.array([1.0], dtype=np.float32)               # 1
        combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)  # 17
        is_bomb = np.array([0.0], dtype=np.float32)               # 1
    
    return np.concatenate([actor_oh, card_mat, is_pass, combo_type_oh, is_bomb])
    # Total: 4 + 60 + 1 + 17 + 1 = 83 dims per move event

D_MOVE = 83  # constant for network architecture
```

### 1.2 Collecting History During Play

The game environment needs to record the move sequence. Add a history buffer to `GuanDanEnv`:

```python
# game.py — modifications

class GuanDanEnv:
    def reset(self):
        # ... existing reset code ...
        self.move_history = []  # list of (player, combo) tuples
    
    def step(self, combo):
        player = self.current_player
        # Record this move BEFORE executing it
        self.move_history.append((player, combo))
        # ... rest of existing step logic ...
```

### 1.3 Building the History Tensor for a Player

```python
# encoding.py

MAX_HISTORY = 15  # last 15 moves

def encode_history(env, player, level_rank):
    """
    Encode the last MAX_HISTORY moves from this player's perspective.
    
    Returns:
        history: np.array of shape [T, D_MOVE] where T <= MAX_HISTORY
        length:  int, actual number of moves encoded (for packing)
    """
    # Get relative player mapping (me=0, right(CCW)=1, partner=2, left=3)
    def relative(p):
        return (player - p) % 4  # CCW-relative
    
    recent = env.move_history[-MAX_HISTORY:]
    
    if len(recent) == 0:
        # No history yet — return a single zero vector
        return np.zeros((1, D_MOVE), dtype=np.float32), 1
    
    events = []
    for (actor, combo) in recent:
        events.append(encode_move_event(relative(actor), combo, level_rank))
    
    history = np.stack(events, axis=0)  # [T, D_MOVE]
    return history, len(events)
```

---

## 2. New Q-Network: LSTM + MLP

Replace the plain MLP with an LSTM that processes history, then concatenates the LSTM output with the state and action encodings before feeding into the value head.

### 2.1 Architecture

```
                    ┌──────────────┐
  move_history ────►│  LSTM(128)   │──── history_emb [B, 128]
  [B, T, 83]       └──────────────┘
                           │
           ┌───────────────┼───────────────┐
           │ concat(state, action, history) │
           │     [B, 417 + 160 + 128]      │
           │          = [B, 705]            │
           └───────────────┬───────────────┘
                           │
                    ┌──────▼──────┐
                    │ Linear(512) │
                    │    ReLU     │
                    ├─────────────┤
                    │ Linear(512) │
                    │    ReLU     │
                    ├─────────────┤
                    │ Linear(512) │
                    │    ReLU     │
                    ├─────────────┤
                    │  Linear(1)  │──── Q(s,a) scalar
                    └─────────────┘

Parameters: ~750K (up from ~100K)
```

### 2.2 Implementation

```python
# q_network.py — replace QNetwork class

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetworkLSTM(nn.Module):
    """
    Q(s, a, h) network with LSTM history encoder.
    
    Inputs:
        state:       [B, d_state]        (417 dims)
        action:      [B, d_action]       (160 dims)
        history:     [B, T, d_move]      (T up to 15, each 83 dims)
        history_len: [B]                 (actual lengths for packing)
    
    Output:
        q_value:     [B]                 (scalar Q-value per sample)
    """
    
    def __init__(self, d_state=417, d_action=160, d_move=83,
                 lstm_hidden=128, hidden=512, n_layers=3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=d_move,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        input_dim = d_state + d_action + lstm_hidden
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize LSTM forget gate bias to 1 (helps learning)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                # Set forget gate bias to 1.0
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, state, action, history, history_len):
        """
        Forward pass. history_len is needed for packing variable-length sequences.
        """
        B = state.size(0)
        
        # Clamp lengths to at least 1 (avoid empty sequences)
        history_len = history_len.clamp(min=1).cpu()
        
        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            history, history_len, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n.squeeze(0)  # [B, lstm_hidden]
        
        # Concatenate all features
        x = torch.cat([state, action, h], dim=-1)  # [B, 705]
        return self.mlp(x).squeeze(-1)              # [B]
```

### 2.3 Handling Variable-Length History in the Replay Buffer

The replay buffer now stores history tensors of different lengths. Two approaches:

**Option A (simple, recommended for Day 3)**: Pad all histories to MAX_HISTORY and store the length.

```python
# replay.py — modifications

class ReplayBuffer:
    def __init__(self, capacity, d_state, d_action, d_move, max_history):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        
        self.states = np.zeros((capacity, d_state), dtype=np.float32)
        self.actions = np.zeros((capacity, d_action), dtype=np.float32)
        self.histories = np.zeros((capacity, max_history, d_move), dtype=np.float32)
        self.hist_lens = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, history, hist_len, G):
        i = self.idx % self.capacity
        self.states[i] = state
        self.actions[i] = action
        # Pad history to max_history
        T = min(hist_len, self.histories.shape[1])
        self.histories[i, :T] = history[:T]
        self.histories[i, T:] = 0.0  # zero-pad
        self.hist_lens[i] = T
        self.returns[i] = G
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'state': torch.tensor(self.states[indices]),
            'action': torch.tensor(self.actions[indices]),
            'history': torch.tensor(self.histories[indices]),
            'hist_len': torch.tensor(self.hist_lens[indices]),
            'return': torch.tensor(self.returns[indices]),
        }
```

Memory estimate: 500K × (417 + 160 + 15×83 + 1 + 1) floats × 4 bytes ≈ **3.6 GB**. Fits comfortably in RAM.

---

## 3. Lead/Follow Network Split

The full plan (§1.2) uses two separate Q-networks: one for leading (free lead) and one for following (responding to a trick). The strategic considerations are fundamentally different:

- **Leading**: "What combo type should I lead? Should I shed weak cards or attack?"
- **Following**: "Can I beat this cheaply? Should I bomb? Should I let partner win?"

Sharing one network forces it to handle both with the same weights. Splitting gives each network a simpler learning task.

### 3.1 Implementation

```python
# train.py — modifications

# Create two networks
q_lead = QNetworkLSTM(d_state=417, d_action=160, d_move=83).cuda()
q_follow = QNetworkLSTM(d_state=417, d_action=160, d_move=83).cuda()

opt_lead = torch.optim.Adam(q_lead.parameters(), lr=1e-4)
opt_follow = torch.optim.Adam(q_follow.parameters(), lr=1e-4)

# Two replay buffers (or one with a tag — two buffers is simpler)
buf_lead = ReplayBuffer(capacity=250_000, ...)
buf_follow = ReplayBuffer(capacity=250_000, ...)
```

### 3.2 Routing During Play

```python
def play_episode(env, q_lead, q_follow, epsilon, level_rank):
    env.reset()
    transitions = {p: [] for p in range(4)}
    
    while not env.done:
        player = env.current_player
        legal = env.legal_moves()
        is_leading = env.current_trick is None
        
        state_enc = encode_state(env, player)
        action_encs = [encode_action(m, env.hands[player], level_rank)
                       for m in legal]
        history, hist_len = encode_history(env, player, level_rank)
        
        # Route to correct network
        q_net = q_lead if is_leading else q_follow
        
        if random.random() < epsilon:
            idx = random.randint(0, len(legal) - 1)
        else:
            with torch.no_grad():
                B = len(legal)
                s = torch.tensor(state_enc).unsqueeze(0).expand(B, -1).cuda()
                a = torch.tensor(np.array(action_encs)).cuda()
                h = torch.tensor(history).unsqueeze(0).expand(B, -1, -1).cuda()
                hl = torch.tensor([hist_len]).expand(B).cuda()
                q_vals = q_net(s, a, h, hl)
                idx = q_vals.argmax().item()
        
        transitions[player].append({
            'state': state_enc,
            'action': action_encs[idx],
            'history': history,
            'hist_len': hist_len,
            'is_leading': is_leading,
        })
        
        env.step(legal[idx])
    
    # Assign returns and push to appropriate buffer
    rewards = env.get_rewards()
    for player, tlist in transitions.items():
        G = rewards[player]
        for t in tlist:
            buf = buf_lead if t['is_leading'] else buf_follow
            buf.push(t['state'], t['action'], t['history'], t['hist_len'], G)
    
    return rewards
```

### 3.3 Training Both Networks

```python
def train_step(q_lead, q_follow, buf_lead, buf_follow, opt_lead, opt_follow,
               batch_size=1024):
    """One training step for both networks."""
    losses = {}
    
    for name, q_net, buf, opt in [
        ('lead', q_lead, buf_lead, opt_lead),
        ('follow', q_follow, buf_follow, opt_follow),
    ]:
        if buf.size < batch_size:
            continue
        
        batch = buf.sample(batch_size)
        s = batch['state'].cuda()
        a = batch['action'].cuda()
        h = batch['history'].cuda()
        hl = batch['hist_len'].cuda()
        G = batch['return'].cuda()
        
        q_pred = q_net(s, a, h, hl)
        loss = F.mse_loss(q_pred, G)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
        opt.step()
        
        losses[name] = loss.item()
    
    return losses
```

---

## 4. Updated Evaluation

```python
def evaluate(q_lead, q_follow, opponent='heuristic', n_games=300, level_rank=Rank.TWO):
    """Evaluate DMC agent (team 0,2) vs opponent (team 1,3)."""
    heuristic = HeuristicAgent(level_rank) if opponent == 'heuristic' else None
    env = GuanDanEnv(level_rank)
    
    results = {'wins': 0, 'total_reward': 0.0, 'n_games': n_games,
               'finish_1_2': 0, 'finish_1_3': 0, 'finish_1_4': 0}
    
    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            
            if player in (0, 2):
                # DMC agent
                legal = env.legal_moves()
                is_leading = env.current_trick is None
                q_net = q_lead if is_leading else q_follow
                
                state_enc = encode_state(env, player)
                action_encs = [encode_action(m, env.hands[player], level_rank)
                               for m in legal]
                history, hist_len = encode_history(env, player, level_rank)
                
                with torch.no_grad():
                    B = len(legal)
                    s = torch.tensor(state_enc).unsqueeze(0).expand(B, -1).cuda()
                    a = torch.tensor(np.array(action_encs)).cuda()
                    h = torch.tensor(history).unsqueeze(0).expand(B, -1, -1).cuda()
                    hl = torch.tensor([hist_len]).expand(B).cuda()
                    idx = q_net(s, a, h, hl).argmax().item()
                
                env.step(legal[idx])
            else:
                if heuristic:
                    env.step(heuristic.choose_action(env, player))
                else:
                    env.step(random.choice(env.legal_moves()))
        
        rewards = env.get_rewards()
        team_r = rewards[0] + rewards[2]
        if team_r > 0:
            results['wins'] += 1
        results['total_reward'] += team_r
        
        # Track win type
        if env.finish_order[0] in (0, 2) and env.finish_order[1] in (0, 2):
            results['finish_1_2'] += 1
        elif env.finish_order[0] in (0, 2) and env.finish_order[2] in (0, 2):
            results['finish_1_3'] += 1
        elif env.finish_order[0] in (0, 2):
            results['finish_1_4'] += 1
    
    wr = results['wins'] / n_games
    avg_r = results['total_reward'] / n_games
    
    return results, wr, avg_r
```

### 4.1 What to Log

```python
if episode % EVAL_INTERVAL == 0:
    _, wr_rand, ar_rand = evaluate(q_lead, q_follow, 'random', 200)
    res_h, wr_h, ar_h = evaluate(q_lead, q_follow, 'heuristic', 300)
    
    loss_str = f"L_lead={losses.get('lead', 0):.4f} L_follow={losses.get('follow', 0):.4f}"
    print(f"Ep {episode:6d} | ε={epsilon:.3f} | {loss_str} | "
          f"vs Rand: {wr_rand:.1%} | vs Heur: {wr_h:.1%} "
          f"(1-2: {res_h['finish_1_2']}, 1-3: {res_h['finish_1_3']}, "
          f"1-4: {res_h['finish_1_4']})")
```

The **1-2 / 1-3 / 1-4 breakdown** is the most important new diagnostic. A strong agent doesn't just finish first — it helps its partner finish second. Watch for:
- Early training: mostly 1-4 wins (agent goes out fast, partner is abandoned)
- Mid training: more 1-3 wins (starting to help partner)
- Late training: some 1-2 wins (real cooperation — agent leads combos partner can follow)

---

## 5. Hyperparameters

```python
# Day 3 config

EPISODES = 30_000            # GPU training, should take 4-8 hours depending on speed
BATCH_SIZE = 1024            # up from 512 — LSTM benefits from larger batches
BUFFER_SIZE_LEAD = 250_000   # split buffers
BUFFER_SIZE_FOLLOW = 250_000
LR = 1e-4                   # keep stable
EPSILON_START = 0.20         # agent has base skill, less exploration needed
EPSILON_END = 0.02
EPSILON_DECAY_EPISODES = 20_000
EVAL_INTERVAL = 500
EVAL_GAMES_HEURISTIC = 300
EVAL_GAMES_RANDOM = 200

# Network
LSTM_HIDDEN = 128
MLP_HIDDEN = 512
MLP_LAYERS = 3
MAX_HISTORY = 15
D_MOVE = 83

# Training steps per episode (do multiple gradient steps per episode collected)
TRAIN_STEPS_PER_EPISODE = 4  # more gradient steps, since episodes are expensive
```

The key change: **TRAIN_STEPS_PER_EPISODE = 4**. Each episode generates ~30 transitions per player (~120 total), but with a 500K buffer, we can do multiple gradient steps per episode without overfitting. This extracts more learning from each game.

---

## 6. Implementation Checklist

### Morning (~3 hours)

- [ ] **Add `move_history` to `GuanDanEnv`** — append (player, combo) on each step, clear on reset
- [ ] **Implement `encode_move_event()`** — 83-dim vector per move
- [ ] **Implement `encode_history()`** — returns [T, 83] array + length
- [ ] **Test history encoding** — play 5 random games, print history tensors, verify shapes and content
- [ ] **Implement `QNetworkLSTM`** — LSTM(128) + 3-layer MLP(512)
- [ ] **Test forward pass** — random tensors through network, verify output shape [B]

### Midday (~2 hours)

- [ ] **Update `ReplayBuffer`** for history storage — padded [MAX_HISTORY, D_MOVE] + length
- [ ] **Create two buffers** — buf_lead, buf_follow
- [ ] **Update `play_episode()`** — collect history, route to lead/follow networks, push to correct buffer
- [ ] **Update `train_step()`** — train both networks from respective buffers
- [ ] **Smoke test** — run 100 episodes, verify no crashes, check buffer sizes growing

### Afternoon (~3-4 hours)

- [ ] **Train from scratch** — 30K episodes on GPU with all Day 3 changes
- [ ] **Monitor** — loss for both networks, WR vs both opponents, 1-2/1-3/1-4 breakdown
- [ ] **Compare to Day 2 baseline** — save Day 2 final checkpoint for comparison

### Evening

- [ ] **Analyze lead vs follow performance** — which network improves more? (Usually follow improves faster because history is more informative when responding)
- [ ] **Check history utilization** — are Q-values more spread after LSTM? (Q-value std across legal actions should increase vs Day 2, meaning the network differentiates moves better)

---

## 7. Expected Progression

```
Ep    0   | vs Heur: ~30%  | 1-2: 0   1-3: 0   1-4: lots  ← untrained
Ep  2000  | vs Heur: ~45%  | 1-2: few 1-3: some 1-4: most ← recovering Day 2 level fast
Ep  5000  | vs Heur: ~55%  | 1-2: 5%  1-3: 20% 1-4: 25%   ← matching Day 2 peak
Ep 10000  | vs Heur: ~60%  | 1-2: 8%  1-3: 22% 1-4: 25%   ← surpassing Day 2
Ep 20000  | vs Heur: ~65%  | 1-2: 12% 1-3: 25% 1-4: 23%   ← LSTM signal kicking in
Ep 30000  | vs Heur: ~68%  | 1-2: 15% 1-3: 25% 1-4: 22%   ← target
```

The jump from Day 2 (59%) to Day 3 (65-68%) comes from two sources:
1. **LSTM enables card counting** — the network learns "4 of the 8 Kings have appeared, so my opponent probably has the rest" from the move sequence
2. **Lead/follow split** — each network specializes, learning its role faster

If you don't see improvement past 60%, the likely bottleneck is **training volume** — the LSTM needs more episodes to learn temporal patterns than the MLP needed to learn snapshot patterns. Consider running overnight to 50K+ episodes.

---

## 8. Debugging the LSTM

### 8.1 Is the LSTM Learning Anything?

Quick diagnostic: compare Q-value distributions with and without history.

```python
def check_lstm_impact(q_net, env, player, level_rank):
    """Compare Q-values with real history vs zeroed history."""
    legal = env.legal_moves()
    state = encode_state(env, player)
    actions = [encode_action(m, env.hands[player], level_rank) for m in legal]
    history, hist_len = encode_history(env, player, level_rank)
    
    B = len(legal)
    s = torch.tensor(state).unsqueeze(0).expand(B, -1).cuda()
    a = torch.tensor(np.array(actions)).cuda()
    
    # Real history
    h_real = torch.tensor(history).unsqueeze(0).expand(B, -1, -1).cuda()
    hl_real = torch.tensor([hist_len]).expand(B).cuda()
    q_real = q_net(s, a, h_real, hl_real).cpu().numpy()
    
    # Zeroed history (as if no moves happened)
    h_zero = torch.zeros_like(h_real)
    hl_zero = torch.ones(B, dtype=torch.long).cuda()
    q_zero = q_net(s, a, h_zero, hl_zero).cpu().numpy()
    
    # If LSTM is learning, these should diverge
    diff = np.abs(q_real - q_zero).mean()
    print(f"Mean |Q_real - Q_zero|: {diff:.4f}")
    print(f"Q_real range: [{q_real.min():.3f}, {q_real.max():.3f}]")
    print(f"Q_zero range: [{q_zero.min():.3f}, {q_zero.max():.3f}]")
    # Early training: diff ≈ 0 (LSTM not yet useful)
    # After 5K+ episodes: diff > 0.1 (LSTM contributing to decisions)
    # After 20K+ episodes: diff > 0.3 (LSTM is critical)
```

### 8.2 Gradient Flow Through LSTM

Check that the LSTM receives gradients and its hidden state norm grows during training:

```python
def check_lstm_gradients(q_net):
    """Call after loss.backward(), before optimizer.step()."""
    for name, param in q_net.lstm.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  LSTM {name}: grad_norm={grad_norm:.6f}")
    # If all grad norms are 0 or tiny (< 1e-8), the LSTM is disconnected
```

### 8.3 Common LSTM Pitfalls

1. **History always length 1**: If you forgot to append to `move_history` or clear it wrong, the LSTM gets trivial input. Print history lengths — they should grow throughout the game (1 → 15).

2. **History not relative to current player**: Each player should see themselves as actor 0. If you forget the relative mapping, the LSTM can't learn player-specific patterns.

3. **Packing errors**: `pack_padded_sequence` with `enforce_sorted=False` is essential because history lengths in a batch are random. If you see CUDA errors from the LSTM, this is usually why.

4. **Forgetting to zero-pad in buffer**: If the buffer stores garbage in the padded region, the LSTM gets corrupted input for short sequences.

---

## 9. Day 4 Preview

After Day 3, the remaining bottlenecks are **training speed** (Python movegen is slow) and **exploration quality** (pure self-play can loop in local optima). Day 4 targets both:

- **Cython movegen**: Port `generate_all_leads()` and `generate_responses()` to Cython for ~10× speedup. This directly translates to 10× more episodes per hour.
- **Soft-start mixing**: Blend heuristic actions into early self-play (α decays from 0.5→0 over 10K episodes). This bootstraps better trajectories before the agent is skilled enough to generate them itself.
- **Expected**: WR vs heuristic climbs to ~72-75%.