# Building a Deep Monte Carlo RL Agent for Guan Dan

## 1. Introduction

Guan Dan (掼蛋, literally "egg-tossing") is a four-player, partnership trick-taking card game popular in eastern China. It is played with a 108-card double deck (two standard 52-card decks plus four jokers) and features 17 distinct combo types, 9 bomb tiers, wild cards, and a level-rank mechanic that shifts the power hierarchy each round. The combination of imperfect information (each player sees only their own 27 cards), a massive action space (hundreds of legal moves per turn), and team dynamics makes Guan Dan a challenging domain for reinforcement learning.

This project builds an end-to-end pipeline from a full game engine through a hierarchy of handcrafted agents to a Deep Monte Carlo (DMC) RL agent with an LSTM history encoder. The goal is to train a neural agent that surpasses the heuristic baselines through self-play and curriculum-scheduled opponent training.

## 2. Game Engine

The environment (`GuanDanEnv`) implements the full Guan Dan ruleset:

- **Players**: 4 seats in fixed partnerships (0+2 vs 1+3), counterclockwise play order.
- **Deck**: 108 cards --- two copies of the standard 52-card deck plus 2 Black Jokers and 2 Red Jokers. Each player is dealt 27 cards.
- **Wild cards**: The heart-suited card of the current level rank acts as a wild, substitutable into most combo types.
- **Level rank**: One of the 13 standard ranks (2 through Ace). Level-rank cards are promoted above Ace in the ordering hierarchy for singles, pairs, triples, full houses, and N-of-a-kind bombs.

### 2.1 Combo Types

| Category | Types | Details |
|----------|-------|---------|
| Ordinary (7) | Single, Pair, Triple, Full House, Straight, Tube, Plate | Full House = triple+pair (5 cards); Straight = 5 consecutive (natural order, not all same suit); Tube = 3 consecutive pairs (6 cards); Plate = 2 consecutive triples (6 cards) |
| Bombs (9 tiers) | BOMB_4 through BOMB_10, STRAIGHT_FLUSH, BOMB_JOKER | Ordered: 4-of-a-kind < 5-of-a-kind < Straight Flush < 6-of-a-kind < ... < 10-of-a-kind < 4-Joker. Any bomb beats any non-bomb. |

Combo generation (`combos.py`) handles wild-card substitution, ace-low/ace-high sequences, straight-flush routing, and deduplication. Move generation produces all legal leads (free play) or all legal responses that beat the current trick, plus PASS.

### 2.2 Trick Resolution and Game End

A trick ends when 3 consecutive passes occur (including auto-passes from eliminated players) or when play returns to the trick winner. When a player empties their hand, they are eliminated; the game ends when both members of the leading team finish. Finish order determines rewards: positions 1st through 4th receive `[+3, +1, -1, -3]` (zero-sum across teams).

## 3. Agent Hierarchy

Six agents of increasing strength, all implementing a common `Agent.act(env, player) -> Combo` interface:

| Agent | Strategy | Approx. Strength |
|-------|----------|-------------------|
| **RandomBot** | Uniform random over legal moves | Baseline floor |
| **GreedyBot** | Always plays the cheapest legal beat; saves bombs; prefers smallest combos when leading | Weak but consistent |
| **HeuristicBot** | HandPlan decomposition (singles/pairs/triples/quads), shed weakest groups first, partner-pass rule, selective bombing | Moderate |
| **StrategicBot** | Adds opponent awareness, partner-help leading, aggressive endgame, context-dependent bomb decisions (opponent card count, partner proximity) | Strong amateur |
| **MonteCarloBot** | For each candidate move (pruned to 6-10 via strategic heuristics), runs N rollout simulations with GreedyBot, picks highest average reward. Supports multiprocessing. | Strongest handcrafted |
| **RLAgentLSTM** | Neural Q-network with LSTM history encoder; separate lead/follow networks | Learning target |

### 3.1 HandPlan Decomposition

The `HandPlan` class (used by HeuristicBot and StrategicBot) decomposes a hand into natural groups by rank count:

- **Singles**: isolated cards (1 copy of a rank)
- **Pairs**: 2 copies
- **Triples**: 3 copies
- **Quads**: 4+ copies (potential bombs)
- **Wilds and Jokers**: tracked separately

This decomposition drives leading priority (shed singles first, then pairs, then triples/full-houses, then multi-card sequences) and following decisions (avoid breaking potential bombs).

## 4. State and Action Encoding

### 4.1 State Encoding (417 dimensions)

The state is encoded from the perspective of the acting player:

| Feature | Dims | Description |
|---------|------|-------------|
| Hand matrix | 60 | `[15 ranks x 4 suits]` count matrix, flattened |
| Played cards (self) | 60 | Cards this player has already played |
| Played cards (partner) | 60 | Partner's played cards |
| Played cards (left opp) | 60 | Left opponent's played cards |
| Played cards (right opp) | 60 | Right opponent's played cards |
| Unknown cards | 60 | Cards not in hand and not seen played |
| Hand sizes | 4 | Normalized card counts `[me, right, partner, left]` / 27 |
| Out flags | 4 | Binary: is each player eliminated |
| Level rank | 13 | One-hot over ranks 2-A |
| Wild flags | 2 | `[has >= 1 wild, has >= 2 wilds]` |
| Is leader | 1 | 1.0 if free lead, 0.0 if following |
| Trick type | 17 | One-hot over 17 ComboType values |
| Trick key rank | 15 | One-hot over 15 rank indices |
| Trick is bomb | 1 | Binary |
| **Total** | **417** | |

All spatial features use a `[15, 4]` card matrix (15 rank slots covering 2-A plus Black/Red Joker, 4 suit columns). Counts are capped at 2 (from the double deck).

### 4.2 Action Encoding (160 dimensions)

Each candidate action is encoded independently, enabling variable-size action sets:

| Feature | Dims | Description |
|---------|------|-------------|
| Cards played | 60 | Card matrix of the combo being played |
| Remaining hand | 60 | Card matrix of hand after this play |
| Combo type | 17 | One-hot |
| Combo key rank | 15 | One-hot |
| Num cards | 1 | `len(cards) / 10` |
| Is bomb | 1 | Binary |
| Wild count | 3 | One-hot: `[0, 1, 2+]` wilds used |
| Hand stats after | 3 | `[remaining_size/27, n_singles/10, n_bombs/3]` |
| **Total** | **160** | |

The "remaining hand" and "hand stats after" features give the network forward-looking information about the consequence of each action, without needing to learn it from scratch.

### 4.3 History Encoding (83 dimensions per move)

Each move in the game history is encoded as an 83-dimensional event vector:

| Feature | Dims | Description |
|---------|------|-------------|
| Actor (relative) | 4 | One-hot: `[me, right, partner, left]` |
| Card matrix | 60 | Cards played (zeros for PASS) |
| Is pass | 1 | Binary |
| Combo type | 17 | One-hot |
| Is bomb | 1 | Binary |
| **Total per move** | **83** | |

The last `MAX_HISTORY = 15` moves are encoded and fed to the LSTM. The sequence is returned as a `[T, 83]` tensor with the actual sequence length for packed/padded processing.

## 5. Network Architecture

### 5.1 QNetworkLSTM

The production architecture uses two separate `QNetworkLSTM` instances --- one for leading decisions, one for following decisions. Each network computes `Q(state, action, history) -> scalar`:

```
History [B, T, 83] --> LSTM (hidden=128, 1 layer) --> h_n [B, 128]
                                                         |
State [B, 417] ----+                                     |
                   |---> concat [B, 705] --> MLP --> Q [B]
Action [B, 160] ---+                                     |
                                                         |
h_n [B, 128] --------------------------------------------|
```

**LSTM encoder**: Single-layer LSTM with `input_size=83`, `hidden_size=128`. Forget gate bias initialized to 1.0 for stable long-range gradients. On MPS (Apple Silicon), runs on full padded sequences since `pack_padded_sequence` is unsupported; on CUDA/CPU, uses proper packing.

**MLP head**: 3-layer MLP with hidden dimension 512:
```
Linear(705, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 1)
```

Input dimension: `417 (state) + 160 (action) + 128 (LSTM hidden) = 705`.

Parameter count: approximately 800K per network, 1.6M total for the lead/follow pair.

### 5.2 Legacy QNetwork (MLP-only)

A simpler `QNetwork` with 3-layer MLP on concatenated `[state, action]` (577-dim input, hidden=256) serves as the Day 1-2 baseline. No history encoding.

## 6. Training

### 6.1 Deep Monte Carlo (DMC)

Training follows the DMC paradigm from DouZero: play full episodes, then assign the terminal reward to every transition in that episode as the target Q-value. This avoids bootstrapping and temporal-difference bias at the cost of higher variance.

**Episode collection**: The RL agent plays seats {0, 2}; the opponent plays seats {1, 3}. At each decision point, the agent encodes the state, all legal actions, and the move history, then selects via epsilon-greedy (exploit the Q-network or explore uniformly). After the game ends, position-based rewards `[+3, +1, -1, -3]` are assigned to all transitions for each player.

**Gradient update**: MSE loss between predicted Q-values and episode returns, with gradient clipping at `max_norm=1.0`. Both lead and follow networks train on a single shared replay buffer (avoids lead-transition starvation). Multiple gradient steps per episode (`train_steps=4` by default).

### 6.2 Replay Buffer

A pre-allocated circular buffer with capacity 500K transitions. Each entry stores:
- State vector `[417]`
- Action vector `[160]`
- History tensor `[15, 83]`
- History length (int)
- Episode return (float)

Uniform random sampling; no prioritization.

### 6.3 Curriculum Opponent Scheduling

Training uses a 3-stage curriculum:

| Stage | Opponent | Promotion Gate | Consecutive Evals Required |
|-------|----------|---------------|---------------------------|
| 0 | RandomBot | Win rate >= 65% | 2 |
| 1 | GreedyBot | Win rate >= 60% | 2 |
| 2 | HeuristicBot | (terminal) | -- |

On promotion, the best win rate and patience counter reset, allowing the agent to adapt to the stronger opponent without premature early stopping.

### 6.4 Exploration Schedule

Epsilon decays linearly from 0.30 to 0.05 over 85% of total episodes. The slower decay (compared to typical DQN schedules) accounts for the high variance of DMC returns and the large action space.

### 6.5 Hyperparameters

| Parameter | Default |
|-----------|---------|
| Episodes | 30,000 |
| Batch size | 1,024 |
| Learning rate | 1e-4 (Adam) |
| Buffer capacity | 500,000 |
| LSTM hidden | 128 |
| MLP hidden | 512 |
| MLP layers | 3 |
| Train steps per episode | 4 |
| Eval interval | 2,000 episodes |
| Eval games | 300 |
| Patience (early stop) | 20 evals |
| Gradient clip | 1.0 |

A `--quick` mode caps training at 8,000 episodes with 1,000-episode eval intervals for rapid iteration (~4-5 hours on M1 Pro).

## 7. Evaluation

### 7.1 Mid-Training Eval

Every `eval_interval` episodes, the agent (greedy, epsilon=0) plays `eval_games` against the current curriculum opponent. Metrics tracked:

- **Win rate**: fraction of games where team {0,2} has positive total reward.
- **Average reward**: mean of `reward[0] + reward[2]` across games.
- **Finish breakdown**: counts of 1-2 (both teammates finish first), 1-3, and 1-4 finishes.

### 7.2 Ladder Eval

Post-training evaluation against all opponent tiers (random, greedy, heuristic, strategic) to measure absolute strength.

## 8. Current Results

### 8.1 Curriculum Training Run (test_curriculum2)

The most complete run used curriculum scheduling with 1,000-episode eval intervals:

| Episode | Opponent | Win Rate | Avg Reward | Stage |
|---------|----------|----------|------------|-------|
| 1,000 | Random | 56% | +1.76 | 0 |
| 2,000 | Random | 61% | +1.80 | 0 |
| 3,000 | Random | 67% | +2.04 | 0 |
| 4,000 | Random | 75% | +2.54 | 0 |
| 5,000 | Greedy | 44% | +0.80 | 1 (promoted) |
| 6,000 | Greedy | 48% | +1.00 | 1 |
| 7,000 | Greedy | 50% | +1.24 | 1 |

The agent reached 75% win rate vs Random by episode 4,000 and was promoted to the Greedy stage, where it reached 50% by episode 7,000 with epsilon approaching its floor.

### 8.2 Direct vs Heuristic (no curriculum)

An earlier run trained directly against HeuristicBot without curriculum:

| Episode | Win Rate vs Heuristic | Avg Reward |
|---------|-----------------------|------------|
| 2,000 | 20% | -0.97 |
| 4,000 | 18% | -0.99 |
| 6,000 | 18% | -1.06 |

This stalled at ~18-20%, confirming that curriculum scheduling is necessary --- the agent cannot learn meaningful signal when the opponent is too strong relative to the randomly-initialized network.

### 8.3 Baseline Reference

For context, a uniformly random agent wins approximately 50% against another random agent (by symmetry). The HeuristicBot baseline against random wins roughly 80-85%. The gap between the RL agent's current 50% vs Greedy and the target of beating Heuristic represents the primary challenge for the next training phase.

## 9. Summary and Next Steps

The pipeline provides a complete framework: a faithful Guan Dan engine with all 17 combo types and 9 bomb tiers, a 6-level agent hierarchy for benchmarking, a 417+160+83-dim encoding scheme with LSTM history, and a DMC training loop with curriculum scheduling. Early results confirm that curriculum training produces meaningful learning (75% vs Random, 50% vs Greedy at 7K episodes), while direct training against strong opponents fails. The next milestone is sustained training through the Greedy stage and into the Heuristic stage, targeting >55% win rate vs HeuristicBot as evidence that the LSTM architecture can leverage game history for strategic play.
