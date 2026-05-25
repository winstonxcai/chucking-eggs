# Tradeoffs and Design Decisions

This doc records the non-obvious choices behind DART and the web app, in ADR
format. Each entry: **Context → Options → Decision → Consequences**. The goal
is to make the reasoning legible to reviewers (and future-me) so the same
debates do not get re-litigated from scratch.

Entries are stable once written. When a decision is revisited, append a
`### Update (YYYY-MM-DD)` block rather than rewriting history.

---

## 1. Filesystem weight publishing

**Context.** The learner produces new Q-net weights ~every few seconds; 32 actor
processes each need to refresh their local policy copies asynchronously.

**Options.** (a) Shared memory tensors with a version counter, (b) NCCL or
`torch.distributed` broadcast, (c) filesystem with atomic rename.

**Decision.** Filesystem. The learner writes `weights_{version}.tmp`, calls
`os.replace()` to `weights_{version}.pt`, then atomically rewrites
`latest.txt`. Actors poll `latest.txt` and load when their lag threshold is
crossed. See [`runtime/weights.py`](../ml/src/guandan/dart/runtime/weights.py)
and [`runtime/actor/runtime.py`](../ml/src/guandan/dart/runtime/actor/runtime.py).

**Consequences.**
- ✅ Trivial to reason about, works across `multiprocessing` and across Modal's
  container boundaries with a shared volume mount.
- ✅ Crash-safe: a half-written `.tmp` is never read because the rename is atomic.
- ❌ Each refresh costs a `torch.load()` deserialization per actor, which is
  not free on the actor's CPU budget.
- ❌ On Modal the shared volume is network-attached, so refresh latency is
  higher than it would be on a single host. This has not been a measured
  bottleneck at the current refresh cadence.

---

## 2. Local actor-side batched inference vs centralized inference server

**Context.** Actor-side Q-forwards dominate actor CPU. A natural alternative is
to ship encoded states to a single GPU inference server and batch across
actors.

**Options.** (a) Each actor runs its own local Q-net on CPU and batches across
its own lanes, (b) one centralized GPU inference server that batches across
all 32 actors.

**Decision.** Local actor-side batching with 128 lanes per actor.

**Consequences.**
- ✅ No IPC on the inference path — encoded state stays in-process.
- ✅ Linearly scales with actor count without a separate service to operate.
- ✅ Confirmed empirically: `32 × 128` reaches ~19.7k–22.4k accepted samples/sec
  on L4/A10G vs ~4.6k–4.9k for the actor-limited `32 × 1` shape (4.3–4.6×).
- ❌ Each actor pays the cost of loading a full Q-net into its own process and
  refreshing it. Memory per actor is non-trivial.
- ❌ Forecloses CUDA actor inference without redesign. If we ever need GPU
  actors, the centralized variant becomes more attractive.
- ⚠️ The centralized variant *was* discussed but not implemented; this is a
  designed-out option, not an A/B result.

---

## 3. Four trick-position heads vs single head with role as input

**Context.** Decision quality depends heavily on a player's role in the
current trick: leading is offensive, last responder is heavily constrained.

**Options.** (a) Four `nn.Linear` Q-heads keyed by `trick_head_id`, (b) one
head fed a one-hot role feature, (c) four full networks per absolute seat.

**Decision.** Four trick-position heads sharing one trunk. See
[`model/q_network.py:276-336`](../ml/src/guandan/dart/model/q_network.py).
The router is deterministic — `trick_head_id` selects the head, not a learned
gate.

**Consequences.**
- ✅ Shared trunk amortizes capacity across the most-frequent state features
  while letting each head specialize on tactical role.
- ✅ Far cheaper than per-seat networks (4× parameters → 1× trunk + 4 small
  linear heads).
- ❌ **No ablation has been run** comparing this against the single-head +
  one-hot variant. The architectural argument is theoretically sound but the
  measured contribution is unknown. This is the most likely "easy win to drop"
  if a reviewer pushes hard.
- ❌ A bug in role assignment poisons the entire policy, not one head.

---

## 4. Partner-visible state as a research assumption

**Context.** Guan Dan is a partnership game where many strong moves only make
sense if the model knows what cards the partner is holding.

**Options.** (a) Strict hidden information — model infers partner's hand from
play, (b) partner-visible — partner's hand is in the state encoding, (c)
full perfect-information — also expose opponent hands.

**Decision.** Partner-visible (`is_partner_visible=True` in
[`config.py:42`](../ml/src/guandan/dart/config.py)). The released checkpoint
*also* includes an `others_hand` opponent channel (see [§10](#10-othershand-channel-in-the-released-checkpoint)).

**Consequences.**
- ✅ Lets the network spend capacity on *cooperation*, not teammate inference.
  This is the actual research question.
- ✅ Reproduces cleanly: partner visibility is a well-defined assumption other
  researchers can match or remove.
- ❌ The result is not a human-deployable hidden-information policy. README
  and MODEL_CARD label it as such; reviewers should still expect this caveat
  to come up first.
- ❌ Comparison against other published Guan Dan agents is harder because
  most do not assume partner visibility.

---

## 5. Pure Monte Carlo targets vs bootstrapped Q-learning

**Context.** Guan Dan episodes are short (one hand, ~30 moves per seat) and
team-level rewards are sparse. The DouZero/GuanZero lineage uses MC returns.

**Options.** (a) MSE against MC returns, (b) DQN-style bootstrapped targets
with a target network, (c) actor-critic with advantage estimates.

**Decision.** MSE against MC returns with `gamma=1.0`. See
[`learners/dart.py:103`](../ml/src/guandan/dart/runtime/learners/dart.py).

**Consequences.**
- ✅ No target network, no bootstrap bias, no off-policyness from stale
  targets. Replay samples remain valid across policy updates because the
  target is the realized terminal reward, not a re-estimated value.
- ✅ Matches the DouZero/DanZero/GuanZero family, so results are directly
  comparable.
- ❌ Sample-inefficient relative to bootstrapped methods — needs substantial
  actor throughput to keep up. This is why the system invests so much in
  actor-side lane batching.
- ❌ Slow to credit-assign through long sequences of weak signal.

---

## 6. Uniform replay sampling vs prioritized

**Context.** The role-aware replay buffer holds ~400k samples. Some samples
(e.g., bombs, tribute decisions) are tactically more informative than others.

**Options.** (a) Uniform sampling within per-trick-head buckets, (b)
prioritized experience replay (PER) weighted by TD error or other importance
score, (c) curriculum-driven sampling.

**Decision.** Uniform per-bucket via
[`buffer.py:462-487`](../ml/src/guandan/dart/data/buffer.py). An optional
`replay_mix` config lets the user weight bucket selection, but within-bucket
sampling is uniform.

**Consequences.**
- ✅ MC targets are unbiased estimates of realized outcomes, so old samples
  retain their statistical validity — prioritization's main upside (correcting
  for outdated values) does not apply cleanly here.
- ✅ Simple, no IS correction needed.
- ❌ Likely leaves performance on the table for rare-but-decisive events. PER
  is worth revisiting if Yaoji-specific weakness traces to a small number of
  high-leverage decision types.

---

## 7. Single-seed training run

**Context.** The 1.25M-update L4 release run costs roughly $215–$225 in Modal
GPU time, plus eval sweeps.

**Options.** (a) Single seed with paired-deck eval CIs, (b) 3 seeds with
shared eval setup, (c) full 5-seed sweep.

**Decision.** Single seed.

**Consequences.**
- ✅ Affordable. Eval variance is reported via binomial standard errors over
  5000 games, which is the largest visible source of uncertainty in the
  *eval* phase.
- ❌ No run-to-run variance estimate. The published Glicko-2 number could
  shift across reruns and we cannot quantify by how much.
- ❌ A reviewer pushing on this point is correct; the right response is to
  show the dollar cost and the eval-game CIs, not to defend single-seed as
  scientifically equivalent to a sweep.
- 🔜 The cheapest mitigation is a 3-seed sweep at a smaller scale (say
  300k–500k updates) to bound run-to-run variance, then state the headline
  number is from a single full-scale extension of one of those seeds.

---

## 8. Rust + Python dual move generator with pure-Python fallback

**Context.** Legal-move generation is the hottest path in episode rollout.
Rust is dramatically faster but adds an install step and a non-trivial
toolchain dependency.

**Options.** (a) Rust-only via `pyo3` + `maturin`, (b) Python-only, (c) Rust
preferred with a pure-Python fallback.

**Decision.** Rust preferred, Python fallback. The
[`guandan_rs/__init__.py`](../ml/src/guandan_rs/__init__.py) module tries
`from _guandan_rs import ...` and sets `HAS_NATIVE=True`; on `ImportError` it
falls back to a pure-Python implementation and sets `HAS_NATIVE=False`.

**Consequences.**
- ✅ Repo works out of the box (tests + local play) without the user building
  the Rust extension.
- ✅ Training and large-scale eval benefit from the native generator when
  installed.
- ❌ Two implementations to maintain. **No differential testing** currently
  guarantees the two paths return identical legal-move sets. A bug in either
  could pass CI silently as long as the relied-on path is correct.
- ❌ `uv sync` wipes the editable `_guandan_rs` install, which has bitten the
  install flow repeatedly. README and DEVELOPMENT.md document the
  `maturin develop` re-step.

---

## 9. Web backend topology: FastAPI + Mongo + Redis + WebSocket + in-process bot inference

**Context.** The web app needs persistent player/game data, real-time game
state to multiple browser tabs per room, and bot inference for solo/AFK seats.

**Options.** (a) Monolithic FastAPI with in-process model, (b) split bot
inference into a separate service, (c) move room state to a managed
real-time backend.

**Decision.** Single FastAPI service. Mongo for player/game persistence,
Redis (with in-memory fallback) as a generic KV cache (see
[`redis_client.py`](../web/backend/app/redis_client.py)), and bot agents
loaded once at startup in
[`ai_service.py`](../web/backend/app/ai_service.py) with synchronous
inference in a thread pool.

**Consequences.**
- ✅ One process to deploy on Fly.io. No service mesh, no extra cost.
- ✅ The web app shares the engine code with the ML side, so version drift
  between training and serving is impossible.
- ❌ In-process bot inference means model latency directly impacts request
  latency, and a slow `DartBot` move blocks a worker thread.
- ❌ Horizontal scaling assumes Redis carries any cross-instance state. Some
  invariants ("one signed player, one active game") rely on a single backend
  instance today; multi-instance correctness is **not yet validated**.
- 🔜 If concurrent load grows, the first split is likely bot inference into
  a separate service so model GPU/CPU budget can be scaled independently of
  HTTP throughput.

---

## 10. `others_hand` channel as a precomputed deduction

**Context.** The role-aware encoder includes an `others_hand` channel: a
108-dim multihot vector representing the *union* of the two opponents' hands.
See [`role_encoder.py:218`](../ml/src/guandan/dart/model/encoding/role_encoder.py):
`others_hand = env.hand_multihot[next_opp_seat] | env.hand_multihot[prev_opp_seat]`.

The natural reading is "this leaks opponent cards" — but it does not.
`others_hand` is the *complement* of information the player already has. A
normal player knows their own hand, so by deck-subtraction they know exactly
which cards are not in their own hand. Partner-visible adds the partner's
hand, so the same subtraction now yields the set of cards collectively held
by the two opponents. `others_hand` is precisely that subtraction result.
The channel does not reveal *which* opponent holds a given card; that
per-opponent split would be oracle information, and the encoder explicitly
ORs them.

**Options.** (a) Include the precomputed union, (b) omit it and let the
model learn the deduction from `own_hand`, `partner_hand`, and the play
history channels, (c) include the per-opponent split (true oracle).

**Decision.** Include the precomputed union. The encoder does not provide
per-opponent information.

**Consequences.**
- ✅ Removes a card-counting bookkeeping subtask from the LSTM/encoder
  budget. The shallow LSTM does not need to maintain a running tally of
  played cards just to know what is still in circulation.
- ✅ Not oracle information — a careful human or rule-bot has the same
  signal.
- ❌ Slightly muddies the "partner-visible only" framing in the README and
  model card. The current language calls this "oracle-style benchmarking,"
  which overstates what the channel actually contains. README §"Why
  Partner-Visible" and §Limitations should be revised to clarify that
  `others_hand` is a derived union, not per-opponent leakage.
- ❌ An ablation removing `others_hand` would still be a worthwhile
  diagnostic: it would measure how much the network was leaning on the
  precomputed deduction vs learning to derive it. But it is not a "fix
  for oracle leakage" — there is no oracle leakage to fix.

---

## 11. Mode-agnostic multiplayer abstraction

**Context.** The product supports solo (1 human + 3 bots), duo (2 humans + 2
bots), and quad (4 humans) games. A naive implementation duplicates room
logic per mode.

**Options.** (a) Separate `SoloRoom` / `DuoRoom` / `QuadRoom` classes, (b)
one `GameRoom` parameterized by a "which seats are human" mask, (c) generic
`Room` with a strategy object per mode.

**Decision.** Single `GameRoom` parameterized by
`_human_seats_for_mode(mode)` returning a seat set: `{0}` for solo, `{0,2}`
for duo, `{0,1,2,3}` for quad. The underlying `GuanDanEnv` is identical
across modes. See
[`game_room.py:43-62`](../web/backend/app/game_room.py).

**Consequences.**
- ✅ Adding a new mode is a one-line change to the seat-set mapping.
- ✅ Bot logic, AFK takeover, disconnect handling, forfeit, and Elo all
  share a single code path.
- ❌ Mode-specific UX rules (lobby behavior, ready-up logic) leak into the
  shared room and are guarded by `if mode == ...` checks. Worth watching
  for further accumulation.

---

## Decisions worth revisiting

A short watchlist of entries above where the current answer is the *cheap*
answer, not the *defended* answer:

- §3: no ablation against the single-head variant.
- §6: PER could plausibly help Yaoji-specific weakness.
- §7: at minimum a 3-seed mid-scale sweep to bound run-to-run variance.
- §8: no differential test between Rust and Python move generators.
- §10: README/model-card language calling this "oracle-style" overstates the
  channel; revise to "partner-visible with precomputed deck-complement
  feature." An ablation without `others_hand` remains a useful diagnostic
  but is not a leak fix.
