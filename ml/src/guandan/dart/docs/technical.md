# Dart Technical Q&A

---

## What is the current blessed training path?

**DART is the production path.** New training configs should use
`model_type: Dart`, the role-aware encoder, and batched local actor inference.
We tested centralized GPU inference-server variants, including cross-actor
lane batching, and local actor-side batching was faster for this model because
it amortizes Q-forward work without IPC.

Validated production configs:

- Local Apple Silicon: `ml/src/guandan/dart/configs/dart_mps.yaml`
- Modal L4: `ml/src/guandan/dart/configs/dart_l4.yaml`

---

## How does distributed training work?

Three process types, two communication channels.

**Main process** spawns the learner first, blocks until `weight_dir/latest.txt` appears, then spawns N actors. After that it just polls `metrics_learner.jsonl` every 2s to drive a tqdm progress bar, and sets `stop_event` when done.

**Learner process** owns the authoritative Q-nets and replay buffer. Tight loop: (1) drain the queue into the buffer, (2) run one gradient step when every seat has `buffer_min_size` samples, (3) publish weights / checkpoint / log on periodic ticks.

**Actor processes** each hold a local eval-mode copy of the Q-nets. Loop: roll one or more episode lanes with `play_episodes_batched`, accumulate samples, push a pre-stacked batch to the queue every `actor_push_batch_size` steps. Sync weights from disk every `sync_interval_updates` episodes (non-blocking — skips if already current).

Communication — four channels, three directions:

**1. Actors → Learner: bounded `mp.Queue`**

Created in the main process with the `spawn` context ([train.py:152-153](../train.py#L152-L153)):

```python
ctx          = mp.get_context("spawn")
sample_queue = ctx.Queue(maxsize=cfg.sample_queue_maxsize)   # default 64 batches
```

Actor side — accumulate `actor_push_batch_size` samples, pre-stack into 9 contiguous numpy arrays, then put ([worker.py:99-111](../worker.py#L99-L111)):

```python
stacked = {k: np.stack([d[k] for d in buf_dicts], axis=0) for k in ENCODE_CHANNEL_KEYS}
msg = {"actor_id": actor_id, "version": local_version,
       "stacked": stacked,
       "players": np.asarray(buf_players, dtype=np.int8),
       "returns": np.asarray(buf_returns, dtype=np.float32)}
try:
    sample_queue.put(msg, timeout=5)
except Exception:
    pass   # queue full — drop and continue
```

Learner side — drain up to `max_drain_batches_per_loop` per iteration ([learner.py:327-336](../learner.py#L327-L336)):

```python
while drained < cfg.max_drain_batches_per_loop:
    try:
        msg = sample_queue.get_nowait()
        buffer.push_stacked(msg["stacked"], msg["players"], msg["returns"])
        fresh_samples_total += len(msg["players"])
        drained += 1
    except Exception:
        break
```

Pre-stacking matters: pickling one `(512, 108)` float32 array is ~6× faster to unpickle than 512 individual dicts of small arrays (measured: 3.81 ms → 0.63 ms per push).

**2. Learner → Actors: filesystem poll**

Learner writes weights every `publish_interval_updates` steps. Both `os.replace` calls are POSIX renames — atomic on macOS and Linux ([learner.py:175-197](../learner.py#L175-L197)):

```python
torch.save({"version": version, "state_dicts": {...}},
           weight_dir / f"weights_{version}.tmp")
os.replace(tmp, weight_dir / f"weights_{version}.pt")   # atomic

ver_tmp.write_text(str(version))
os.replace(ver_tmp, weight_dir / "latest.txt")           # atomic
# delete stale weight files
```

Actor polls every `sync_interval_updates` episodes. Non-blocking — silently skips if the file is transiently unreadable during an atomic replacement ([worker.py:23-37](../worker.py#L23-L37)):

```python
snapshot = load_latest_weights(weight_dir)   # reads latest.txt then weights_{ver}.pt
if snapshot is None or snapshot.version <= local_version:
    return local_version
for p in range(4):
    q_nets[p].load_state_dict(snapshot.state_dicts[p])
    q_nets[p].eval()
return snapshot.version
```

**3. Main → all processes: `mp.Event`**

A single `stop_event` is created in the main process ([train.py:154](../train.py#L154)), set on clean finish, KeyboardInterrupt, or crash ([train.py:264](../train.py#L264)):

```python
stop_event.set()
```

Both `actor_loop` ([worker.py:117](../worker.py#L117)) and `learner_loop` ([learner.py:324](../learner.py#L324)) gate their main loop on it:

```python
while not stop_event.is_set():
    ...
```

**4. Learner → Main: metrics file (progress polling)**

The main process has no direct handle to the learner's state. It reads progress by tailing `metrics_learner.jsonl` every 2 seconds ([train.py:240-244](../train.py#L240-L244)):

```python
while True:
    time.sleep(_WATCHER_POLL_INTERVAL_S)          # 2s
    count = _read_update_count(run_dir)            # reads last line of metrics_learner.jsonl
    delta = min(count - last_count, target_updates - last_count)
    bar.update(delta * cfg.batch_size)
```

This decouples the progress display from the learner process entirely — no shared memory, no IPC, just a log file.

Processes are spawned with `mp.get_context("spawn")` — required on macOS to avoid PyTorch deadlocks with the default `fork`.

---

## How is the training target created?

`targets` in `Learner.update()` is a `(B,)` float32 tensor of MC returns, one per sample in the batch.

Chain from game end to tensor:
1. `env.get_rewards()` → `{player: ±3/±2/±1}` depending on team finish order
2. `normalize_terminal_rewards()` divides by 3 → values in `[-1, 1]`
3. `compute_mc_returns()` assigns each step on player p's trajectory `G_t = γ^(n−1−j) · R_p`
4. Stored as `TrainSample.mc_return` in the per-seat `ReplayBuffer`
5. `collate()` stacks `[s.mc_return for s in samples]` into the targets tensor

With `gamma=1.0` (paper default), every step a player takes in a game gets the same target — their normalized terminal reward. No credit assignment, no bootstrapping.

---

## What is the MC return formula mathematically?

Let player p take turns at global indices `T_p = {t_0 < t_1 < ... < t_{n-1}}` with normalized terminal reward `R_p`. The return assigned to p's j-th turn:

```
G_{t_j} = γ^(n−1−j) · R_p
```

The exponent counts **p's own remaining turns** after turn j, not global timesteps. With `γ=1.0` this collapses to `G_{t_j} = R_p` for all j — every decision gets the same flat signal.

With `γ<1`, earlier decisions are discounted more. For example, p's first decision with n=27 turns and γ=0.99 gets `0.99^26 · R_p ≈ 0.77 · R_p`. This would bias the network toward weighting late-game plays more heavily.

---

## What is the `encoded` field in `TrainSample`?

A Python dict with named numpy arrays — one per channel. Current DART runs use the role-aware schema; the older base encoder has a smaller 9-key schema. The dict stays structured until `collate()` stacks it into tensors for the forward pass.

| Key | Shape | Content |
|---|---|---|
| `own_hand` | (108,) | multi-hot over the actor's current hand |
| `partner_hand` | (108,) | partner hand when `is_partner_visible=true`; zeros otherwise |
| `others_hand` | (108,) | union of the two opponents' current hands; oracle information in the current checkpoint |
| `player_blocks` | (4, 256) | public per-role features: played cards, last non-pass action, remaining count, bomb tiers, trick position |
| `global_features` | (13,) | current level rank, one-hot |
| `behavior` | (9,) | cooperation/dwarfing/assisting flags |
| `history_actions` | (20, 108) | recent moves as multi-hot card vectors |
| `history_roles` | (20, 4) | relative actor role for each history row |
| `history_is_pass` | (20, 1) | pass marker for each history row |
| `candidate_action` | (108,) | the action being scored, multi-hot |
| `trick_head_id` | scalar | routing head: leading / first responder / across / last responder |

108 dims per card channel = one bit per physical card in the double deck (52×2 + 4 jokers). The dict structure exists so individual channels can be swapped for ablations without touching the rest of the pipeline.

In the Q-network, the history channels go through the LSTM; `player_blocks`, global hand/features, and the candidate action go through separate MLP branches before the routed Q head.

---

## Why is epsilon needed?

Without epsilon, every actor always plays the greedy action. Two problems:

1. **Unvisited states never get Q-value updates.** If the network always plays the bomb, it never generates samples from states where the bomb wasn't played, so those Q-values stay at random initialization forever.

2. **Self-play creates a feedback loop.** All four seats run the same greedy policy against each other. If they converge to a narrow strategy early, the buffer fills with samples from that one corner of the game tree and the network overfits to beating that specific strategy.

Epsilon injects random decisions that force the policy to encounter and recover from a wider variety of game states.

---

## How is epsilon usually selected?

No universal rule — it's empirically driven. Common patterns:

**Start/end values.** Literature default is start=1.0, end=0.01–0.05. Dart uses start=0.1 (conservative) because self-play + replay buffer already provides diversity without needing fully random early play.

**Decay horizon.** Set to roughly when the buffer first fills and losses stabilize — before that point Q-values are noise and epsilon barely matters; after that you want it low. DART uses `epsilon_decay_updates`; the actor evaluates the schedule against the latest learner update count, so exploration tracks optimization progress rather than local episode count.

**Per-actor variation (distributed).** DouZero gives each actor a different fixed epsilon rather than a shared decaying one. DART does not currently use per-actor epsilon bands; actors share the update-based epsilon schedule and diversify through independent seeds, lane interleaving, mixed opponents, and replay.

**Guan Dan branching factor.** With 100+ legal moves, even ε=0.01 generates meaningful randomness — picking uniformly from 100 actions is a lot of variance. Could go lower than 0.01 without losing exploration coverage.

---

## How are episodes and updates related?

They are **completely decoupled** in the distributed setup.

- An **episode** is one full game → produces ~100 `TrainSample`s (one per decision across all four seats) → pushed into the queue.
- An **update** is one gradient step → samples 512 items per seat from the replay buffer → computes `MSE(Q(s,a), G_t)` → steps Adam. Nothing is consumed from the buffer; samples stay for future updates.

The connection is the replay buffer. Episodes fill it; updates sample from it. The learner doesn't wait for a specific number of episodes — it updates whenever every seat has `buffer_min_size` samples, regardless of how many episodes have run.

In the single-process `train()` there is an explicit coupling (`learn_every_episodes: 4` = 4 episodes then 1 update). The distributed version removes this entirely because actor and learner run at their own speeds.

Practical consequence: if actors are slow, the queue drains and the learner idles. If actors are fast, the queue fills and actors start dropping batches (the `queue.put(..., timeout=5)` silently discards). The `n_actors=1` optimum on M1 exists because one actor already saturates the MPS learner — more actors starve the learner of CPU time.

---

## Are the 512 samples per seat drawn randomly or are they the most recent?

Uniformly random without replacement — `random.sample(range(n), k)` picks k distinct indices from the entire buffer with no recency bias. The 512 samples for seat 0 could come from any 512 of the up to 50,000 stored samples, old or new.

---

## Why random sampling — why not the most recent?

Three reasons:

1. **Break temporal correlation.** Consecutive steps within an episode are highly correlated — same game, same opponent, same hand evolving one card at a time. Sequential batches would cause the network to overfit to the current game state and destabilize.

2. **Break policy correlation.** All samples in a single episode were generated by the same Q-net version. Training only on recent samples means chasing a moving target — the policy that generated the data shifts every few updates. Random sampling across the buffer mixes many policy versions, smoothing the gradient signal.

3. **Seat balance.** Each per-seat buffer is sampled independently, so every position always contributes exactly 512 samples per update regardless of how many turns that seat took in recent episodes.

This works well here because targets are MC returns (actual game outcomes), not Bellman bootstraps. A sample from 5000 updates ago still has a valid label — the game result didn't change retroactively.

---

## What is the optimal buffer size?

A tradeoff between two failure modes:

- **Too small:** samples are dominated by the current policy — high correlation, network oscillates.
- **Too large:** too many samples from old, weak policies whose state distribution the current policy no longer visits. Less harmful with MC returns than with bootstrapping, but still slows convergence.

Practical heuristic: `capacity ≈ 10–100× batch_size per seat`. The current 50k vs 512 is ~100×, which is on the generous side.

The real constraint is memory. One encoded sample is ~13 KB (the `history` channel alone is 20×108×4 bytes = 8.6 KB). At 50k per seat × 4 seats: `200,000 × 13 KB ≈ 2.6 GB`. Cutting to 20k per seat saves ~1 GB with minimal training impact.

`buffer_min_size` (5000) is a separate knob — controls when training *starts*, not steady-state diversity.

---

## Are the learner and actor loops run in parallel?

Yes — each is a separate OS process (`multiprocessing.Process`) with its own Python interpreter, memory space, and GIL. They run simultaneously on different CPU cores, not just concurrently in a threading sense.

On M1 (`n_actors=1`): the actor runs on a CPU core doing game simulation + encoding + Q-net inference, while the learner runs on MPS doing forward + backward + Adam simultaneously. The queue is the handoff point between them.

`mp.get_context("spawn")` is required on macOS — the default `fork` copies PyTorch's internal CUDA/MPS state into child processes causing deadlocks. `spawn` starts each child fresh, which is why `actor_loop` and `learner_loop` do their imports lazily at the top of the function.

---

## What happens when the queue gets full? Are "queue" and "buffer" interchangeable?

**When the queue gets full:** the actor tries `sample_queue.put(msg, timeout=5)`. If the learner hasn't drained space within 5 seconds, the entire batch of 512 samples is silently dropped and the actor moves on. No retry, no backpressure. The actor's job is to generate fresh experience, not guarantee delivery.

**Queue and ReplayBuffer are not the same:**

| | Queue | ReplayBuffer |
|---|---|---|
| Type | `multiprocessing.Queue` | Python list per seat |
| Lives in | shared between processes | learner process only |
| Capacity | 64 batches | 50,000 samples per seat |
| Drop policy | actor drops on full | circular overwrite on full |
| Purpose | IPC transport | random-access training store |

The queue is a short transit pipe for crossing the process boundary. The replay buffer is the actual training dataset inside the learner.

---

## If the buffer stays full is that a bad thing?

Depends on which one.

**Queue full (bad):** learner is the bottleneck — actors are generating faster than the learner can consume. Samples are being dropped. Watch `queue_depth` in `metrics_learner.jsonl`; consistently at `sample_queue_maxsize=64` means a problem.

**ReplayBuffer full (good, expected):** designed to stay full at steady state. Once it hits `capacity_per_player=50000` it circularly overwrites the oldest samples. A full replay buffer means maximum diversity in batches and stale samples being evicted. `buffer_total` hitting 200,000 (4 × 50k) is just normal healthy operation.

---

## How are learner loss buckets named?

Loss-bucket metrics in `metrics_learner.jsonl` are schema-backed rather than
integer-cell encoded. The schema lives in
`runtime/learners/loss_bucket_schema.py` and defines every axis label used by
`runtime/learners/loss_buckets.py`.

Grid metrics use:

```text
<metric>_<axis_a_label>__<axis_b_label>_<stat>
```

For example, `phase_role_opening__leading_loss` is the MSE for samples where
the acting player is in the opening phase and is leading a new trick.
Marginal metrics use:

```text
<metric>_<axis_label>_<stat>
```

For example, `q_gap_pivotal_n` counts sampled decisions where the greedy
Q-value gap was below 0.05. Each emitted cell has `_n`, `_frac`, and `_loss`;
`_loss` is `null` when the cell has too few samples for a stable estimate.

The top-level prefixes (`phase_role_`, `q_gap_`, etc.) are still stable so the
learner can split scalar losses from diagnostic bucket payloads without knowing
the full axis cardinality.

---

## What is the queue used for exactly?

Its only job is transferring sample batches from actor processes to the learner's ReplayBuffer. The actor and learner live in separate OS processes with separate memory — the actor cannot call `buffer.push()` directly. The queue is the only shared channel.

```
Actor process                      Learner process
─────────────                      ───────────────
play_episodes_batched()
accumulate per-lane samples
stack into numpy arrays
sample_queue.put(msg)    ──►    sample_queue.get_nowait()
                                  → buffer.push_stacked()
                                       → lives in ReplayBuffer
```

Batches are pre-stacked before pushing (9 contiguous numpy arrays rather than 512 individual sample dicts) because pickling one large array is ~6× faster to unpickle across the process boundary (measured: 3.81ms → 0.63ms per push).

---

## How would the queue ever actually get full?

The learner drains in a tight loop, so in normal operation the queue stays nearly empty. It fills only when the **learner is slower than the actors combined**.

The drain cap is `max_drain_batches_per_loop=32` per learner iteration — between drains the learner must do a gradient step. If the gradient step is slow (cold MPS, `torch.compile` warmup, large batch on a tired GPU), actors push faster than the learner drains.

Concrete example on M1 with `n_actors=1`:
- Actor: ~1 episode/sec → fills a 512-sample push batch every ~5 episodes → ~1 push every 5 seconds
- Learner: ~8 updates/sec, drains up to 32 batches per loop → easily keeps up

With `n_actors=4` on the same M1:
- 4 actors pushing simultaneously → ~4 pushes every 5 seconds
- Learner now competes with 4 actor processes for CPU → slower per update
- Queue backs up → eventually hits `maxsize=64` → drops begin

This is why `n_actors=1` is the M1 optimum — adding actors starves the learner of CPU and causes queue saturation.

---

## Is `mc_return` the same as the terminal reward?

In this config (`gamma=1.0`), yes — but the names mean different things in general:

- **Terminal reward** `R_p`: what `env.get_rewards()` returns, normalized by 3. Fixed per game.
- **MC return** `G_t = γ^(n−1−j) · R_p`: the discounted version. Differs per timestep.

With `γ=1.0`, `γ^(anything) = 1`, so `G_t = R_p` for every step on player p's trajectory. The code uses `mc_return` as the field name because it's the general concept — at `γ<1` they would diverge.

The training loop uses `mc_return` as the regression target: `MSE(Q(s,a), G_t)`. The network learns to predict the discounted game outcome from any (state, action) pair.

---

## How large can a legal-action set get?

Empirically (5000 random deals, full 27-card hand at game start):

| | Count |
|---|---|
| Raw max (before dedup) | **1,404** |
| After `dedup_strategic` | **400** |
| Median after dedup | 78 |
| p90 | 229 |
| p99 | 339 |

Dart no longer truncates legal actions for policy reasons. Actors score the full
post-dedup legal set.

---

## How does the weight update mechanism work?

Filesystem-mediated. Learner writes weights to disk; actors poll for new versions.

**Learner publishes — `publish_weights()`** (every `publish_interval_updates` steps):

```python
torch.save({...}, weight_dir / f"weights_{ver}.tmp")
os.replace(tmp, weight_dir / f"weights_{ver}.pt")   # atomic
ver_tmp.write_text(str(version))
os.replace(ver_tmp, weight_dir / "latest.txt")       # atomic
# delete prior weight files
```

Both `os.replace` calls are POSIX renames — atomic on macOS and Linux. An actor reading mid-publish sees either the old complete file or the new complete file, never a partial write.

**Actor syncs — `maybe_sync_weights()`** (every `sync_interval_updates`):

```python
snapshot = load_latest_weights(weight_dir)
if snapshot is None or snapshot.version <= local_version:
    return local_version    # no-op if not newer
for p in range(4):
    q_nets[p].load_state_dict(snapshot.state_dicts[p])
```

Reads `latest.txt`, compares versions, loads only if strictly newer. Non-blocking — if the file isn't readable it silently returns.

**Staleness gap.** Actors are typically 1-2 published versions behind the learner (~100-200 gradient steps). This is fine: DMC is off-policy and the replay buffer already mixes samples from many policy versions. Blocking actors on every weight update would serialize the parallel pipeline.
