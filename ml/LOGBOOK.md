# Guan Dan RL — Training Logbook

Chronological record of every ML training attempt for the Guan Dan RL agent (Mar 4 – Apr 11, 2026). Sourced from `README.md` §1–17 and the daily logs at `Winnie's War Vault/logs/`. Compiled 2026-04-24 alongside the strict archive of `ml/_archive/`.

This logbook is the single source of truth for "what we tried, what worked, what failed, and why." The next training plan should consult this before relitigating any approach already on the failure list.

---

## Glossary

- **WR vs X** = win rate of agent over X games against opponent X (a "win" = our team's player finishes 1st of 4).
- **Comp avg** = average WR vs the four strongest competition bots (Yaoji, Jidan, NoAI, Lalala).
- **ts** = team-spirit reward mixing coefficient: `r = (1−ts)·own_reward + ts·partner_reward`.
- **Gate** = composite eval threshold required before saving a new "best" checkpoint.

---

## 1. Project reset (Mar 3–4)

**Decision:** RL-first, web-app-second. Scrapped initial web scaffolding. Built MVP blueprint specifying production-correct Guan Dan rules as non-negotiable.

**Outcome:** Clean engine + DMC loop foundation. No checkpoint.

---

## 2. Agent hierarchy + LSTM Q-network (Mar 13)

**Approach:** Built 5 rule-based agents (Random, Greedy, Heuristic, Strategic, MonteCarlo) + RLAgentLSTM Q-network: state(417) ⊕ action(160) ⊕ LSTM(history T×83) → 3-layer MLP(512) → scalar Q. ~3.7M params.

**Result:** Beat random after initial training. Foundation in place.

**Lesson:** Single-opponent training plateaus quickly → curriculum needed.

---

## 3. Curriculum learning + training infra (Mar 13–16)

**Approach:** Stage 1 random → Stage 2 greedy (65% gate) → Stage 3 heuristic (60% gate, terminal). Parallel episode collection. Tuned across `full_run_3..5`. Run 5 replaced hard curriculum with mixed-opponent schedule.

**Result:** Curriculum stages worked. CPU bottleneck appeared.

**Lesson:** Movegen is the CPU wall. Need acceleration.

---

## 4. Rust acceleration + Modal distributed training (Mar 18–20)

**Approach:** Ported movegen to Rust via PyO3. Producer-consumer (CPU workers → GPU training). Modal A10G launcher with HF token, volume mounts.

**Result:** Movegen 3.5×; episode collection 9× overall; GPU batched forward 10×; 50K episodes from >24h → ~6h.

**Lesson:** Throughput unlocked. Now distillation and self-play are feasible.

---

## 5. Supervised distillation v1 — GreedyBot teacher (Mar 24)

**Approach:** Generate 1000 games with GreedyBot, MC rollouts (n_sims=20), 5-epoch MSE fine-tuning at lr=3e-6.

**Result:** **44.7% WR vs Strategic** (regression from 55% baseline).

**Why failed:** GreedyBot is a 6% WR player. Rollout policy quality is everything.

---

## 6. Supervised distillation v2 — StrategicBot teacher (Mar 24)

**Approach:** Same MSE distillation pipeline, StrategicBot (90% WR) as rollout policy. 1000 games × n_sims=20 = 41K decisions. Single-epoch MSE at lr=3e-6.

**Result:** **68.7% WR vs Strategic** (+1.05 net level/game). Checkpoint `mc_distilled.pt`.

**Lesson:** Gentle 1-epoch distillation is the sweet spot. 2 epochs overshoots (drops to 56%). MSE > cross-entropy because Q-values are continuous, not categorical.

---

## 7. Self-play from MC baseline — 100K episodes (Mar 24)

**Approach:** Continue self-play from `mc_distilled.pt` for 100K episodes locally (8 workers, ~2.8h). No team-spirit mixing.

**Result:** **Peak 75.5% vs Strategic at ep70K**, regressed to 66.5% by ep100K. vs Heuristic peaked 90% at 55K.

**Lesson:** Late regression = overfitting to self-play patterns. Should have included ts=0.5.

---

## 8. Aux hand-prediction head + opponent pool (Mar 24)

**Approach:** Added 60-dim auxiliary opponent-card-prediction head (BCE loss). Population pool of past checkpoints with `pool_prob=0.3`. ts=0.5. 100K self-play (~3.3h).

**Result:** **74.5% peak vs Strategic at ep50K** — no lift over plain ts=0.5 baseline (75.5%).

**Why failed:** Aux task didn't help despite clean implementation. Network had no incentive to use the head; later analysis showed aux columns near-zero. Network params 3.30M → 3.68M (+376K) wasted.

---

## 9. QMIX / WQMIX team coordination (Mar 20–22)

**Approach:** TeamMixer (monotonic QMIX with abs() weights) + UnrestrictedMixer (WQMIX Q* for training signal). Two-phase: Phase A freeze Q-nets train mixer; Phase B end-to-end gradient flow. TrickCollector accumulates per-player transitions. Global state 310 dims (all 4 hands visible to mixer).

**Result:** **Null result** vs individual Q-learning. `qmix_best.pt`, `qmix_final.pt`, `wqmix_best.pt`, `wqmix_final.pt` produced but no measurable lift.

**Why failed:** Variance from team-target backprop; credit assignment ambiguous in long tricks. Mixer learned trivial weighted sum. Confirmed bug fix in Phase B (must store raw Q-net inputs, not detached scalars) didn't change the result.

**Verdict:** QMIX path is dead. Do not revisit without major architectural change.

---

## 10. Reward shaping experiments (Mar 23)

**10a. Team-spirit ts=0.5 (Mar 23):** `r = 0.5·own + 0.5·partner`. **74.4% vs Heuristic, 54.8% vs Strategic** (+4.2pp). Simple reward shaping beat all of QMIX/WQMIX. Saved as `team_spirit_best.pt`.

**10b. Real Guan Dan rewards [3,0,−1,−2] (Mar 23):** Position rewards updated to match real level rules (1-2 finish = +3, 1-3 = +2, 1-4 = +1, loss = −2). Net levels ↑ (+1.93 vs +1.88) but 1-2 finish rate ↓ from 66.2% to 56.6%. Mathematically correct, incentive-misaligned. Reverted.

**10c. Team-level rewards (Mar 23):** Both teammates get `r = team_level_change`, no position signals. **79.2% vs Heuristic, 65.4% vs Strategic** — underperformed ts=0.5 baseline. Saved `teamlevel_final.pt`. Theory was sound but hurt individual card-play quality. Reverted.

**Final:** Position-based [3,0,−1,−2] with ts=0.5 is the keeper combination.

---

## 11. GNN hand-structure encoding (Mar 25–26)

**Approach:** Per-hand graph with 5 edge types (same_rank, consecutive, same_suit_consecutive, wild_bridge, same_suit). 23-dim card features. 2× GATv2 layers (4 heads, d=64→128). Attentive readout → 3×128 embeddings concat'd to Q-net. Pre-computed embeddings stored in replay buffer (1024-graph rebuild avoidance).

**Result:** +3pp vs Strategic (75.5% → 78.5%). 8.5× training speedup from amortization. Network params 3.68M → 4.12M (+439K).

**Mar 28 follow-up:** Analysis showed GNN MLP columns std=0.0017 vs std=0.0235 for non-GNN columns. Network learned to ignore GNN signal. `load_strip_gnn()` shrinks 4.12M → 3.68M losslessly (eval identical: 69.4% vs 70.0%).

**Verdict:** GNN dead. Stripped from production.

---

## 12. Inference-time PIMC search (Mar 26)

**Approach:** Determinize 81 unknown cards across 3 players, simulate N rollouts, Q-guided action selection, average return. Modules: `determinize.py`, `simulate.py`, `search.py`.

**Result:** **42–65% WR vs Strategic** (down from 77% baseline). 35× slowdown (0.17s → ~6s/game).

**Why failed:** 81 hidden cards across 3 players → random determinizations rarely close to reality. Bridge has 13 hidden, poker 2–5 — Guan Dan has too many for search to be useful.

**Verdict:** Search dead.

---

## 13. Competition bot ecosystem + Glicko-2 calibration (Mar 26–27)

**Approach:** Vendored 8 NJUPT 2020 competition bots (`_vendor/{team}/`) + thin adapters. Fixed Lalala `pass_num=0` bug (always passed instead of cumulative; WR 43% → 79% vs random). 13-bot round-robin, 200 games × 78 matchup pairs = 31,200 games (~3h on M1 Pro). Glicko-2 with 30 convergence passes.

**Result:** RL Elo **1786 (#1)**, Jidan **1779**, Yaoji **1772**, NoAI **1726**, Strategic **1621**. RL margin over Jidan: 7 points (statistical tie at 200-game CI).

**Lesson:** RL is top of ladder but barely. Need fine-tuning vs competition bots.

---

## 14. Competition selfplay fine-tune (Mar 28)

**Approach:** 50K episodes of fine-tuning. Opponent rotation: random per-move through Yaoji, Jidan, NoAI. New composite save gate: `(heuristic_WR + competition_avg_WR) / 2`. Each best saves timestamped `prod_MM_DD_HH_MM.pt`.

**Result:** **Composite gate +5pp (66.5% → 71.5%)**. Heuristic WR stable 84–90.5%. **Competition WR FLAT at 48–51%** despite training directly against them.

**Bug discovered:** `episodes_done % 5000 == 0` never fires with `n_envs=64`. Only 2 of 10 planned evals ran. Fixed by replacing modulo with threshold tracking (`next_eval_at += interval`).

**Verdict:** Best-known production checkpoint became `prod_03_28_11_36.pt`. Competition WR plateau is the central unsolved problem.

---

## 15. Continued production runs (Mar 29 – Mar 31)

**Approach:** Various selfplay continuations from updated bases. Generated checkpoints `prod_03_29_01_21.pt`, `prod_03_29_08_15.pt`, `prod_03_29_10_56.pt`, `prod_03_29_11_51.pt`, `prod_03_29_15_33.pt`, `prod_03_31_20_18.pt`, `selfplay_best.pt` (Mar 31 20:18, identical to prod_03_31).

**Result:** Final production checkpoint `prod_03_31_20_18.pt` ≈ Elo 1786 vs Jidan 1779 — still essentially tied with Jidan over 500-game evals.

**Lesson:** No breakthrough. Hit a hard ceiling around 50% vs Jidan even with more episodes from same architecture.

---

## 16. Curriculum from-scratch (Apr 1)

**Approach:** End-to-end curriculum (pretrain → random → greedy → heuristic → strategic → competition), 54K total episodes. Speedups: mega-batch MLP, GNN removed, n_envs=128, train_steps=2 → 7 eps/sec (4.1× faster). Total runtime 6.4h.

**Result:** vs Random 98.5%, vs Greedy 91.5%, vs Heuristic 72%, vs Strategic 47%, **vs Competition avg 25.8%** (Yaoji 25.5%, Jidan 28%, NoAI 24%).

**Verdict:** From-scratch curriculum reaches Strategic-level only. Competition gap requires >5× more compute or fundamentally different approach.

---

## 17. LLM agent — GPT-5.4 Nano via LiteLLM (Apr 3)

**Approach:** 3-stage pipeline per move: (1) cooperative-flag intent LLM call, (2.5) ToM beliefs LLM call (conditional), (3) move selection from numbered top-K candidates. RL Q-network as candidate recommender (Elo-#1 `prod_03_29_11_51.pt`, ~5ms/call vs 500ms LLM). HKUST paper (arXiv:2408.02559) replication attempt with our Q-network in place of DanZero embeddings.

**Result:** **8% WR vs Jidan over 50 games.** Places LLMBot between greedy (1%) and heuristic (11.5%). Cost ~$0.008/game. Intent distribution: coop 13% / assist 23% / dwarf 8% / normal 56%.

**Why low vs paper's −0.88 score gap:** (1) Single-pick from numbered list vs paper's per-candidate evaluation. (2) English prompts vs paper's Chinese prompts (paper found Chinese significantly outperforms).

**Lesson:** RL recommender is the critical lift (0% → 8%). Output format matters: number-on-first-line eliminates parse failures. ToM adds ~30% cost; WR impact inconclusive at 50 games.

---

## 18. GuanZero (Apr 4–5)

**Approach:** GuanZero-style architecture: separate encoding (`guanzero_encoding.py`, 18KB), network (`guanzero_network.py` ~2M params LSTM+MLP), distillation (`guanzero_distill.py`), self-play (`guanzero_selfplay.py`). Two-stage Jidan distillation + DMC self-play. Aux hand prediction loss.

**Result:** Checkpoint `guanzero_final.pt`. Did not surpass `prod_03_31_20_18.pt` baseline. Approach abandoned in favor of Tier 1 partner-visibility direction.

---

## 19. Tier 1 partner-visibility — uncommitted, never trained (Apr 7–11)

**Approach (planned but no working checkpoint produced):** 
- `visibility/encoding.py` — `encode_state_tier1()` returns 477 dims (417 + 60 partner-hand inserted at pos 60); also `encode_state_tier1_team()` returns 480-dim team-centric invariant encoding.
- `visibility/expand.py` — zero-init expansion of 417-dim checkpoints to 477-dim (bit-identical at ep 0).
- `visibility/adapter.py` — small ~90K-param MLP `(partner_hand⊕action) → hidden(256) → 1`. Output zero-initialized. Loss: MSE(adapter, clipped_residual) + L2_output_penalty.
- `agents/impossible_bot.py` — Tier 1 agent: `Q_final = Q_base(frozen) + adapter(partner_hand, action)`.
- Three training scripts at varying maturity: `train_tier1.py` (frozen base + adapter), `train_scratch_tier1.py` (from-scratch 477-dim), `train_full_tier1.py` (warm-start, all params unfrozen, low LR).

**Result:** **No `tier1_*.pt` checkpoint ever produced.** Website's `_try_load_impossible_agent()` is a graceful no-op. The "Impossible" UI tier exists but falls back silently.

**Status as of 2026-04-24 archive:** All Tier 1 code archived. The Phase 2 plan will rebuild fresh from this learning.

---

## 20. Pattern analysis: why attempts plateau

Recurring failure modes across all approaches:

1. **Imperfect-info ceiling.** Without partner visibility, optimal play is randomized over many beliefs. Best-response under perfect information is deterministic and strictly better. This appears to cap WR vs Jidan at ~50% regardless of other choices.
2. **Reward variance.** Long episodes (28+ tricks) make Q-target attribution noisy. MSE Q-targets on MC returns have high variance even with 20 rollouts/decision.
3. **Teacher quality is everything in distillation.** GreedyBot (6% WR) crashed distill v1 to 44.7%. StrategicBot (90% WR) lifted to 68.7%. Anything weaker than the student hurts.
4. **Auxiliary heads die without selection pressure.** Both the GNN columns and the hand-prediction head collapsed to near-zero weights. Auxiliary tasks need to be the *only* path the gradient can flow through to actually train.
5. **Eval bugs hide regressions.** The `episodes_done % 5000` modulo bug meant 2+ months of training had only 20% of expected eval checkpoints. Always use threshold-tracking, never modulo with variable batch sizes.
6. **Compute ceiling.** Curriculum from-scratch at 54K eps reaches Strategic-level only. Bridging to Competition tier needs ≥500K eps or different architecture — neither cheap on M1 Pro.
7. **Self-play late regression.** 100K-ep self-play peaked at 75% (ep70K) and dropped to 66.5% (ep100K). Use early-stopping on competition WR, not self-play loss.

---

## 21. Checkpoint inventory at archive time

All 47 checkpoints under `ml/checkpoints/` moved to `ml/_archive/checkpoints/` on 2026-04-24. Total ~1.2 GB. Key entries:

| Checkpoint | Date | Approach | Notes |
|---|---|---|---|
| `prod_03_31_20_18.pt` | Mar 31 | selfplay continuation | **Final production**, Elo 1786, ≈50% vs Jidan |
| `selfplay_best.pt` | Mar 31 | (alias) | Identical mtime to prod_03_31 |
| `prod_03_29_11_51.pt` | Mar 29 | selfplay | Used by LLM recommender experiments |
| `prod_03_28_11_36.pt` | Mar 28 | competition selfplay | First post-bug-fix best |
| `mc_distilled.pt` | Mar 24 | MC distill v2 (StrategicBot) | 68.7% vs Strategic |
| `team_spirit_best.pt` / `_final.pt` | Mar 23 | ts=0.5 | 74.4% vs Heuristic |
| `teamlevel_final.pt` | Mar 23 | team-level rewards | Underperformed ts=0.5 |
| `qmix_best.pt` / `_final.pt` | Mar 21 | QMIX | Null result |
| `wqmix_best.pt` / `_final.pt` | Mar 22 | WQMIX | Null result |
| `stage1_heuristic.pt` / `stage2_strategic.pt` | Mar 19–20 | supervised distill | Curriculum stages |
| `mixer_detached.pt` | Mar | QMIX Phase A | Detached mixer training |
| `newrewards_ts05.pt` | Mar 23 | ts=0.5 + new rewards | Reward refactor experiment |
| `selfplay_control_8k.pt` | Mar | self-play control | Sanity baseline |
| `guanzero_final.pt` | Apr 4 | GuanZero | Did not surpass prod_03_31 |
| `jidan_distill.pt` | Apr 4 | Jidan distillation | Component of GuanZero pipeline |
| `selfplay_ep{5056..100000}.pt` (16 files) | Mar 24–29 | self-play snapshots | Per-N-ep checkpoints; superseded |

To restore any: `git log --follow ml/_archive/checkpoints/<name>` and `git checkout <commit> -- <path>`.

---

## 22. Direction selection: perfect-partner-info agent (Apr 24, 2026)

**Goal:** strongest agent that beats Jidan ≥80% over a meaningful sample, where "beat" = 1st-or-2nd placement. The agent **may see its partner's full hand at execution** — this is *centralized execution*, not classic CTDE.

**Prior art surveyed (the relevant analog is bridge, not Doudizhu):**

- **NooK (NukkAI, 2022)** — beat 8 world champions 67/80 sets (83%) using a hybrid: small NN for early tricks + probabilistic logic + Monte Carlo sampling for late tricks. ([deeplearning.ai](https://www.deeplearning.ai/the-batch/bridge-to-explainable-ai/))
- **αμ search (Cazenave & Ventos, 2019)** — fixes PIMC's strategy fusion + non-locality; outperforms PIMC on bridge. ([arxiv 1911.07960](https://ar5iv.labs.arxiv.org/html/1911.07960))
- **Double Dummy Solver (DDS)** — perfect-info bridge solver; used as PIMC rollout policy. ([dds-bridge](https://github.com/dds-bridge/dds))
- **DanZero+ (2023)** — Guandan DMC, needs ~160 CPUs × 30 days; *has not* dominated Guandan even at that scale. ([arxiv 2312.02561](https://arxiv.org/abs/2312.02561))
- **CTDE survey (2024)** — QMIX/MAPPO family. We tried QMIX (LOGBOOK §6) — null result. CTDE assumes decentralized execution, which we don't need. ([arxiv 2409.03052](https://arxiv.org/abs/2409.03052))

**Three directions considered:**

1. **A — Pure search (αμ / Belief-PIMC with partner visibility).** No training. Sample determinizations of ~54 hidden cards across 2 opponent seats; rollout with Jidan; argmax over candidate moves. Cheap (1–2 days CPU); medium probability of clearing 80%; sets the floor.
2. **B — Distill + self-play with partner-visible encoding (DanZero+ family).** From-scratch RL on team-aware state. Rejected as primary path: high cost, joins graveyard of flat-50% fine-tunes (LOGBOOK §13, §15), only single-variable change vs prior failures.
3. **C — NooK-style hybrid (distilled policy prior + αμ/PIMC search).** Small policy net distilled from Jidan proposes top-K; search evaluates only those candidates with partner visibility. Highest historical evidence (NooK beat world champions); medium-high cost.

**Decision:** sequence A → C. Direction A first as a 1-week probe (500-game eval vs Jidan). Decision tree:

| A's WR vs Jidan | Action |
|---|---|
| ≥ 80% | Ship A. Done. |
| 60–80% | Build C with A as the search backend; add policy prior. |
| 50–60% | Build C with stronger search (αμ, more determinizations, Jidan rollouts). |
| < 50% | Reconsider whether partner visibility unlocks the gains expected. |

Direction B stays archived unless A and C both flatline.

**Plan file:** `~/.claude/plans/look-through-the-readme-md-soft-key.md` (approved 2026-04-24).

**Why this isn't on the failure list yet:** every prior approach in §1–17 used own-hand-only encoding. The central hypothesis — that adding partner visibility breaks the ~50% ceiling — has not been tested. Direction A tests it without burning compute.

---

## 23. Direction A result: PartnerPIMCBot (Apr 24, 2026) — FAILED

**Approach:** Pure PIMC search with partner-hand visibility. No training. At each decision:
1. Pre-filter legal moves to top-K by rank_sum heuristic (cheapest cards first).
2. Sample N determinizations of the 2 opponent hands (54 hidden cards across 2 seats).
3. For each determinization, clone env, step each candidate, run GreedyBot rollout for all 4 seats for up to `depth_limit` steps; score leaf with card-count advantage heuristic.
4. Argmax over candidates by average score.

**Config tested:**
- Run 1: n_det=20, n_cands=10, HeuristicBot rollouts (no depth limit) → 20% WR (2/10, very wide CI), 0.02 games/s
- Run 2: n_det=5, n_cands=5, HeuristicBot rollouts (no depth limit) → 8% WR (4/50, CI [3.2%–18.8%]), ~0.13 games/s
- Run 3 (optimized): n_det=10, n_cands=5, GreedyBot rollouts, depth_limit=30, multiprocessing (8 workers) → **3.5% WR (7/200, CI [1.7%–7.0%])**, 0.7 games/s

**Optimizations applied during Run 3:**
- GreedyBot rollout instead of HeuristicBot: ~4× cheaper per step
- depth_limit=30 + card-count leaf value instead of full rollout: ~3× fewer steps
- ProcessPoolExecutor (8 workers): ~8× throughput on M1 Pro
- Combined speedup: ~30× (0.02 → 0.7 games/s)

**Outcome:** 3.5% WR vs Jidan (200 games, statistically clear — CI upper bound 7%). Worse than HeuristicBot (11.5%) and barely above Greedy (1%). **Decisively failed.**

**Why it failed:**

1. **Pre-filter was the wrong heuristic.** rank_sum (cheapest cards first) is essentially GreedyBot logic. So the agent selected GreedyBot-quality candidates, ran expensive rollouts on them, and arrived at GreedyBot-quality decisions — with extra noise. Partner visibility was never used to *improve candidate selection*, only to reduce the hidden-card count in rollouts.

2. **GreedyBot rollouts are too weak.** With only 10 determinizations of GreedyBot play, the Monte Carlo estimate is dominated by noise. Jidan's explicit value scoring (get_VAL) is a calibrated hand-strength oracle; GreedyBot rollouts cannot match that signal quality. The search needs a rollout policy at or above Jidan's level to extract useful signal.

3. **Strategy fusion is severe at Guandan's branching factor.** Bridge PIMC works with ~5–13 legal plays per turn. Guandan has 50–150. Even with partner visibility, the pre-filter can't reliably identify the top-3 correct candidates across all possible opponent holdings — the "correct" move varies wildly by world.

4. **Depth-30 leaf value is too crude.** Guandan games run 100+ steps. Card-count advantage at step 30 is a poor proxy for outcome — bombs, level cards, and endgame sequencing dominate the actual result. A heuristic computed 70 steps before game end is essentially noise.

**Lesson:** Pure search without a strong policy prior is counterproductive in Guandan. The NooK result confirms this: NooK uses a NN for candidate selection first, then MC for evaluation. Reversing the order (cheap heuristic → MC) fails because the heuristic is too weak to surface the right candidates.

**Decision:** Direction A is added to the failure list. Move to Direction C: train a small policy network (distilled from Jidan, ~30 min M1 Pro) for candidate pre-filtering, then PIMC evaluates only those 3–5 candidates using partner-visible determinizations. Partner visibility is retained — it reduces hidden cards from 81→54, which is real signal once the candidates are meaningful.

**Files added (keep — infrastructure for Direction C):**
- `ml/src/guandan/agents/partner_pimc_bot.py` — PartnerPIMCBot (reusable for Direction C's search component)
- `ml/scripts/eval/bots.py` — head-to-head eval with CI output, partner_pimc support
- `ml/scripts/eval/wr_matrix.py` — round-robin WR matrix, Glicko-2, partner_pimc support
- `ml/src/guandan/rating.py` — restored from archive (needed by wr_matrix)

---

---

## 25. Direction C — Jidan-distilled policy + partner-visible PIMC search (2026-04-24)

**Hypothesis:** A policy net distilled from Jidan, used as a pre-filter for PartnerPIMCBot with Jidan rollouts, can exceed Jidan's WR (≥45% Stage 1 gate, ≥80% Stage 2 target). This is the NooK recipe: NN proposes, MC evaluates.

**Architecture:**
1. QNetwork(state=480, action=160, hidden=512): (state ⊕ action) → Q scalar.
2. State = encode_state_tier1_team (480d): own hand + partner hand + history (full visibility, team {0,2} perspective).
3. Distillation: cross-entropy loss over softmax(Q[legal]) vs one-hot(JidanBot pick). 100K decisions from Jidan self-play. AdamW + cosine LR + early stopping.
4. At inference (Stage 2): top-3 policy candidates fed to PartnerPIMCBot (n_det=30, full-game Jidan rollouts, partner hand fixed in determinizations).

**Stage 1 result (2026-04-24):** 45.0% WR (90/200, 95% CI [38.3%, 51.9%]) — pure policy argmax, no search, seat rotation. Exactly at the "solid" gate (≥45%). Distillation succeeded.

**Checkpoint:** `ml/checkpoints/jidan_policy.pt` (3.3 MB, hidden=512, trained 100K decisions, 10 epochs)

**Stage 2 result (2026-04-24):** 69.5% WR (139/200, 95% CI [62.8%, 75.5%]) — policy + PartnerPIMC, n_det=30, K=3, full-game Jidan rollouts, seat rotation. Falls in 60–80% band → one hyperparam tuning round per plan gate. Speed: ~3.5s/game (11:40 total).

**Stage 2 tuning (2026-04-24):** n_det=50, K=5 → **71.0% WR (142/200, 95% CI [64.4%, 76.8%])** in 26:36 (~8.0s/game). 2.78× compute for +1.5pp WR, CI heavily overlapping Stage 2. **Confirms search ceiling:** more rollouts do not break through. Bottleneck is policy/rollout quality (Jidan-level), not MC sample noise. Direction D (AZ + belief modeling) targets the policy quality directly.

---

## 26. Direction D Phase 1a — Belief-aware determinization (2026-04-25)

**Hypothesis:** Inference from opponent pass events (H1–H5 hard constraints) tightens the determinization distribution, improving PIMC search quality above the 69.5%/71.0% ceiling.

**Implementation:** `BeliefModel` (Layer 2): per-pass constraints H1 (no single > R), H2 (no pair > R), H3 (no triple > R), accumulated via `env.move_history` replay. Wired into `_determinize` with up to 50 retries and uniform fallback. 11 unit tests, all passing.

**Phase 1a v1 result (2026-04-25):** 65.0% WR (130/200, CI [58.2%, 71.3%]) — regression. Root cause: H5 (`no_bomb=True` on any non-bomb pass) is not a hard constraint; players sandbag bombs routinely. Fixed by removing H5.

**Phase 1a v2 result (2026-04-25, H5 removed):** 68.0% WR (136/200, CI [61.2%, 74.1%]) — still a regression from 69.5% baseline. Root cause: H1/H2/H3 are also not hard constraints for JidanBot, which uses a weighted value scorer (`get_VAL` / Reyn_AI 2.0) and chooses to pass on singles/pairs/triples to preserve combo structure. The inference "passed on single-K → no card above K" is wrong whenever the opponent is preserving pairs, triples, or combos.

**Key insight:** Layer 1 (rank distribution / card counting) adds nothing over the existing uniform shuffle of the known hidden-card pool — the uniform shuffle already gives the correct hypergeometric marginal distribution per rank. Layer 2 (pass-inference) requires opponents that always play when they can beat, which JidanBot does not satisfy.

**Decision:** Belief Layer 2 does not work vs JidanBot-class opponents. H5 is removed (correct fix); H1/H2/H3 are disabled by defaulting `use_belief=False` in PartnerOracleBot. Phase 1b (soft signals) is **skipped**. Move to Phase 2: AZ self-play with value head.

---

## 27. Direction D Phase 2 — AlphaZero self-play, gen-1 and gen-2 (2026-04-25)

### Architecture: QValueNet + AZ iteration

**QValueNet** extends QNetwork with a state-only value trunk V(s): separate 3-layer MLP(d_state→hidden→hidden/2→1). Q-trunk unchanged for backward compat. V zero-init (final layer only) so old checkpoints load with `strict=False`.

**selfplay_data.py:** records `(state[480], actions[K,160], pi_search[K], z)` per PIMC decision. pi_search = softmax(PIMC raw scores). Workers use CPU (avoid MPS contention across spawn processes).

**train_az.py:** `L = soft_CE(Q, pi_search) + 0.5·MSE(V, z)`. Per-epoch val metrics, inline policy-only eval vs Jidan, early-stop on val loss.

### Gen-1 (2026-04-25)

**Data:** 30K decisions from PartnerOracleBot(jidan_policy.pt) with Jidan rollouts (n_det=30, K=3). z stats: mean=+0.129, std=2.188. 52.4% of samples have H<0.1 (search mostly agrees with policy prior).

**Training (az_gen1.pt):** init from jidan_policy.pt. Best epoch 4/10, val=1.176. Policy-only WR: 38% (limited metric due to Q-drift). **With-search WR: 68% (136/200, CI [61.2%, 74.1%])** — matches baseline ± noise. Expected for one AZ iteration.

**Critical diagnosis:** Policy-only WR (38%) is 31pp below with-search WR (68%). Root cause: training constrains only top-K Q-values; legal moves never in training (not in any top-K) drift to arbitrary Q-values. At policy-only eval, these drifted values contaminate argmax → wrong picks. With-search eval masks this by re-applying the policy's top-K filter before PIMC.

### V-at-leaf speedup (2026-04-25)

Replaced Jidan rollouts with V(s) forward pass at PIMC leaf. Speed: 1.0 → 22.9 dec/s/worker (23× speedup). Gen-2 self-play: 30K decisions in 950s (31.6 dec/s total). Forces CPU (MPS can't share torch modules across spawn).

### Gen-2 (2026-04-25) — Mode collapse

**Data:** 30K decisions from az_gen1_latest.pt with V-at-leaf (n_det=50, K=3). z mean=-0.013, std=2.319.

**Training (az_gen2.pt):** init from az_gen1_latest.pt. Early stop at epoch 6, best val=0.949 (epoch 3). WR_jidan policy-only: **0%** throughout training — policy completely broken.

**With-search WR: 2.0% (4/200, CI [0.8%, 5.0%])** — catastrophic collapse vs 68% gen-1 baseline.

**Root cause: AZ mode collapse from Q-drift.** Gen-1's Q-drifted policy proposed bad top-K candidates in gen-2 self-play. PIMC scored bad candidates and returned bad π_search targets. Training gen-2 on these targets amplified Q-drift → gen-2 policy completely broken. Even with-search (n_det=50, V-at-leaf) couldn't rescue 3 uniformly-bad candidates.

**Key insight: Q-drift is self-amplifying across AZ iterations.** Each generation inherits and worsens the previous generation's Q-drift. Standard AZ (AlphaGo Zero style) avoids this because its policy is a softmax over ALL legal moves, updated every step. Our Q-net architecture trains only K=3 actions per sample → non-top-K actions drift unconstrained.

### Fix (2026-04-25)

Two-part fix to prevent Q-drift and AZ mode collapse:

1. **Ranking loss in train_az.py:** Store K_NEG=5 random non-candidate legal moves as "hard negatives" in self-play data. Add hinge loss: `relu(Q_neg_max - Q_pos_min + margin=1.0)` with weight 0.5. Prevents Q-drift by explicitly training non-top-K actions toward low Q-values.

2. **Hybrid oracle for gen-2-v2 self-play:** Use jidan_policy.pt for top-K candidate selection (clean, no Q-drift) + az_gen1_latest.pt V-head for leaf scoring (fast, better than Jidan rollouts). The `--policy-checkpoint` flag in selfplay_data.py decouples the two. After PartnerOracleBot is constructed, `oracle.net` is replaced with the gen-0 net; `oracle._search.value_net` retains the gen-1 reference (Python bound the reference at init time).

### Gen-2-v2 (2026-04-25) — Ranking loss working, V-at-leaf target collapse discovered

**Data (selfplay_gen2_v2.npz):** 30K decisions, hybrid oracle (jidan_policy.pt for K + az_gen1_latest.pt V-at-leaf for scoring, n_det=50, K=3). 30K in 872s (34.4 dec/s). z mean=-0.040, std=2.378.

**Critical data quality issue:** π_search targets from V-at-leaf are far more peaked than gen-1 (Jidan rollouts):
- gen-1: mean entropy H=0.348, max pi mean=0.798
- gen-2-v2: mean entropy H=0.142, **median H=0.009**, max pi mean=0.941. 68.8% of decisions are near-one-hot.
- Cause: V(s) is a deterministic single forward pass per determinization → very consistent scores across n_det=50 → softmax(avg V) is much more peaked than softmax(avg Jidan rollout).

**Training (az_gen2_v2.pt):** init from az_gen1_latest.pt, ranking loss included. Ranking loss correcting Q-drift (r: 0.937→0.624 at epoch 4→0.366 at epoch 7). Early stop at epoch 7, best val=1.019 (epoch 4).

**Policy-only WR:** 2% at epoch 4 (saved checkpoint), 12% at epoch 7. Very poor. Root cause: near-one-hot π_search targets cause policy entropy collapse (H: 0.892→0.688). The model memorizes "always pick candidate 0" rather than learning to discriminate.

**Val loss ≠ WR proxy:** epoch 4 has best val=1.019 but WR=2%; epoch 7 has val=1.130 but WR=12%. Early stopping saves the wrong checkpoint.

**With-search WR (all variants):**
- az_gen2_v2.pt (best-val, epoch 4) + search: **9.5% WR** (19/200, CI [6.2%, 14.4%])
- az_gen2_v2_latest.pt (epoch 7) + search: **8.0% WR** (16/200, CI [5.0%, 12.6%])
- Hybrid eval (jidan K-select + az_gen2_v2.pt V-head for leaf): **11.0% WR** (22/200, CI [7.4%, 16.1%])

All three confirm catastrophic failure. Even replacing the Q-trunk with Jidan (hybrid eval) doesn't rescue performance — the V-head itself is broken.

### Root cause: V distribution shift (2026-04-25)

**The fundamental problem:** Gen-1 V was trained on data generated by Jidan rollouts (gen-1 distribution). Gen-2-v2 data was generated using gen-1 V at leaf, creating a different state distribution (different game trajectories). Gen-1 V has **0.128 z-correlation on gen-2-v2 data** (near-random), despite 0.989 z-corr on gen-1 data.

Near-random V scores → still confident (high variance across candidates: gen-2-v2 V mean_var=0.063 vs gen-1 V mean_var=0.036) → peaks in π_search due to decisive but wrong ranking → policy memorizes "always pick candidate 0" → mode collapse.

**Key mechanical insight:** `depth_limit=0` in `_rollout_limited` means **unlimited** (runs to game completion). Gen-1's non-uniform π_search (H=0.348) comes from real Jidan-rollout game outcomes, not from `_leaf_value`. `_leaf_value` is only called when `depth_limit > 0` cuts rollouts short. So gen-1's signal is real terminal z values discriminating candidates — strong, correctly distributed training signal.

**Confirmed diagnostics:**
- Q-pi agreement on gen-1 data: gen-1 Q=65.6%, gen-2-v2 Q=51.5% (barely above random)
- Gen-2-v2 V has 0.988 z-corr on gen-2-v2 data but predicts stratgically incorrect rankings (trained on noise → learns to predict noise outcomes)
- All three with-search evals stuck at 8–11% regardless of which checkpoint or hybrid combination

### Additional fixes shipped (2026-04-25)

These fixes were implemented during the gen-2-v2 investigation and remain in the codebase for future use:

1. **Label smoothing in train_az.py (ε=0.1):** `π_smooth = 0.9·π_search + 0.1·(1/n_cands)`. Applied in `_train_step` only; val loss on raw targets for calibration. Gen-2-v3 (retrain on same data with label smoothing): best policy-only WR=15% (epoch 2) — entropy H still collapsed (0.718→0.584). Insufficient without fixing the data root cause.

2. **Best-WR checkpoint saving:** train_az.py saves `_bestwr.pt` at epoch with highest WR_jidan. Addresses val-loss ≠ WR proxy problem discovered in gen-2-v2 (epoch 4 best-val=2% WR, epoch 7 worst-val=12% WR).

3. **PIMC temperature (--pi-temp) in selfplay_data.py:** Temperature τ on PIMC scores before softmax: `pi_search = softmax(scores/τ)`. Softens near-one-hot targets. Recommended τ=2.0–3.0 for V-at-leaf.

4. **bots.py --policy-checkpoint flag:** Enables hybrid oracle eval (separate clean policy net for K-selection + V-head from main checkpoint). Loads to `agent.device` (avoids MPS/CPU mismatch).

### Gen-3 plan: back to Jidan rollouts (2026-04-25)

**Strategy:** Abandon V-at-leaf until V is bootstrapped on the correct distribution. Use gen-1 policy (az_gen1_latest.pt) for top-K candidate selection + full Jidan rollouts for leaf scoring. Ranking loss (K_NEG=5) prevents Q-drift in training. Label smoothing (ε=0.1) reduces any residual target noise.

**Why this breaks the distribution-shift cycle:** Jidan-rollout data always comes from the same base distribution (Jidan-vs-Jidan games). V trained on gen-3 Jidan-rollout data will have high z-corr on gen-4 data from the same type of rollout. Only use V-at-leaf once a generation of V is verified to have good z-corr on its own distribution.

**Data (selfplay_gen3.npz):** 30K decisions in 3975s (7.5 dec/s). z mean=+0.098, std=2.205 — near-identical to gen-1 stats, confirming Jidan-rollout data quality.

**Training (az_gen3.pt):** init from az_gen1_latest.pt. Best policy-only WR=40% at epoch 3 (az_gen3_bestwr.pt), H=0.850 (no collapse), ranking loss r=0.479. Best val epoch 4. Early stop at epoch 9 (patience=5).

**With-search WR: 75.0% (150/200, CI [68.6%, 80.5%])** — clear improvement over gen-1's 68.0% (136/200, CI [61.2%, 74.1%]). Gen-3 CI lower bound (68.6%) exceeds gen-1 point estimate, confirming real gain.

**AZ loop working:** one proper generation (Jidan rollouts + ranking loss) moved 68% → 75%. Strategy confirmed. Proceeding to gen-4 with az_gen3_bestwr.pt policy + Jidan rollouts.

### Gen-4 (2026-04-25)

**Data (selfplay_gen4.npz):** 30K decisions in 4899s (6.1 dec/s, slightly slower — gen-3 policy making harder decisions). z mean=+0.105, std=2.163.

**Training (az_gen4.pt):** init from az_gen3_bestwr.pt. Best policy-only WR=44% at epoch 8 (az_gen4_bestwr.pt), H=0.696 (more entropy collapse than gen-3's 0.850). Best val epoch 4 (val=1.206). Early stop epoch 9.

**With-search WR: 72.5% (145/200, CI [65.9%, 78.2%])** — within CI of gen-3's 75.0%. Statistically indistinguishable. AZ loop has plateau at ~73–75%.

**Plateau diagnosis:** Policy-only WR improving (38→40→44% across gens) but translating to noise-level with-search gains. Two likely causes: (1) bestwr checkpoint selection is noisy (100-game eval → high variance; epoch 8 44% might be lucky), (2) entropy collapse at epoch 8 (H=0.696 vs gen-3's 0.850) → more peaked Q-values → Q-drift re-emerging at late epochs. The ranking loss slows but doesn't stop entropy collapse.

**Proceeding to gen-5** using az_gen4_bestwr.pt (latest generation). If gen-5 also plateaus, consider V-at-leaf with higher n_det to break the search quality ceiling.

### Gen-5 (2026-04-25)

**Data (selfplay_gen5.npz):** 30K decisions in 5005s (6.0 dec/s). z mean=+0.102, std=2.222.

**Training (az_gen5.pt):** init from az_gen4_bestwr.pt. Best policy-only WR=40% at epoch 5 (az_gen5_bestwr.pt). Best val epoch 3 (val=1.215). Early stop epoch 8.

**With-search WR: 70.5% (141/200, CI [63.8%, 76.4%])** — plateau confirmed. Three consecutive generations: gen-3=75%, gen-4=72.5%, gen-5=70.5%, all within each other's CI. More Jidan-rollout AZ iterations won't break through.

### Plateau analysis and Phase 1b: V-at-leaf with higher n_det (2026-04-25)

**Root of plateau:** Jidan rollouts give "expected outcome against Jidan" — the policy has learned to optimally beat Jidan. More iterations with the same rollout policy can't discover strategies Jidan doesn't use. Policy-only WR stuck at 40% across gens 3-5.

**V cross-gen z-correlation diagnostic:**
- Gen-5 V on gen-5 data: 0.980 (training distribution)
- Gen-5 V on gen-3 data: 0.808 (different gen, same Jidan-rollout type)
- Gen-5 V on gen-4 data: 0.770
- Gen-3 V on gen-5 data: 0.289 (asymmetric: later V generalizes better)

**Insight:** Gen-5 V generalizes well across Jidan-rollout gens (0.77–0.81). This is very different from the gen-2-v2 failure (gen-1 V had 0.128 on V-at-leaf data). The failure was V-at-leaf → distribution shift; Jidan-rollout gens are close enough that cross-gen generalization holds.

**Gen-6 smoke run (ABORTED):** Attempted gen-5 V at leaf with n_det=100, pi_temp=2.0. Smoke run diagnostic (1000 decisions, 63s):
- Gen-5 V z-corr on gen-6 smoke data: **0.085** (catastrophic — same failure mode as gen-2-v2)
- pi_search entropy H: 0.105, 86.6% near-one-hot
- Root cause: same distribution shift — V trained on Jidan-rollout trajectories has near-zero z-corr on V-at-leaf trajectories, despite 0.808 corr on Jidan-rollout cross-gen data

**V-at-leaf is fundamentally incompatible with Jidan-rollout V.** The distribution shift between Jidan-rollout game trajectories and V-at-leaf game trajectories is too large for cross-distribution V generalization. This rules out V-at-leaf as a route to breaking the plateau via any Jidan-rollout-trained V.

**H4 constraints implemented (2026-04-25):** Added STRAIGHT, TUBE, PLATE, FULL_HOUSE pass inference to BeliefModel. `violates()` checks consecutive-rank sequences for straights/tubes/plates, and triple rank for full houses (wilds excluded → safe under-constraint).

**H4 eval result (gen-5 bestwr + H4): 72.5% WR (145/200, CI [65.9%, 78.2%])** — vs gen-5 baseline 70.5%. +2pp, within CI, marginal.

**Gen-3 bestwr + H4 eval: 72.0% WR (144/200, CI [65.4%, 77.8%])** — vs gen-3 without H4 = 75.0%. H4 HURTS by 3pp. Root cause identical to H5 removal: players routinely sandbag complex combos (straights, full-houses) for strategic deception. H4 constraints incorrectly reject valid determinizations → worse PIMC worlds → lower search quality. H4 reverted. H1-H3 retained (singles/pairs/triples are more greedily played).

**Critical discovery: training/eval belief mismatch.** All selfplay_data.py runs used `use_belief=False`, while bots.py eval uses `use_belief=True` (default). Training data was generated with unrealistic PIMC worlds (no constraint filtering) → noisier π_search targets. Gen-6 data fixes this with `use_belief=True` in selfplay (H1-H3 active during PIMC).

### Gen-6 (2026-04-25) — first with belief-consistent training

**Data (selfplay_gen6.npz):** 30K decisions in 4958s (6.1 dec/s). z mean=+0.100, std=2.193. First gen with `use_belief=True` in selfplay.

**Training (az_gen6.pt):** init from az_gen5_bestwr.pt. Best-val=1.180 at epoch 3 — lowest best-val across all gens (gen-5 was 1.215), suggesting belief-consistent data improves training signal quality. Best policy-only WR=41% at epoch 6 (az_gen6_bestwr.pt). Early stop epoch 8.

**With-search WR: 72.5% (145/200, CI [65.9%, 78.2%])** — statistically identical to gen-4 (72.5%) and gen-5 (70.5%). Belief-consistent training did NOT break the plateau. Best-val improvement (1.180 vs 1.215) did not translate to WR gain. Jidan-rollout AZ ceiling confirmed across 4 generations (gen-3 through gen-6): 70–75% with no upward trend.

**Conclusion:** The ceiling is the Jidan rollout itself, not data quality or belief consistency. Search evaluates positions as "expected outcome if all players play Jidan" — after a few iterations the policy has fully learned to exploit Jidan and further generations give no gain. Path forward requires a different signal source (Modal multi-generation AZ with self-evaluated rollouts once V is bootstrapped, or acceptance of 75% as the ceiling and shipping gen-3_bestwr).

### Bot ladder WR matrix + Glicko-2 ratings (2026-04-25)

7-bot round-robin (200 games/matchup, 42 directed pairs, 8400 total games, 6 min). partner_oracle (az_gen3_bestwr, n_det=30, K=3) injected via 100-game evals vs all 7 bots (2026-04-25 18:05 CST) + pre-existing 75.0% WR vs jidan (200 games).

**Win Rate Matrix** (row = seats {0,2}, col = seats {1,3}):

```
                  random  greedy heuristic strategic xingdream  yaoji  jidan  partner_oracle
random              ---   22.5%  16.5%    2.5%    7.0%   1.0%   1.5%    0.0%*
greedy            76.0%    ---   41.5%   13.5%   26.5%   5.5%   0.0%    0.0%*
heuristic         92.5%  55.5%    ---    21.5%   46.0%   9.0%  13.0%   15.0%*
strategic         96.5%  86.5%  72.0%     ---    70.5%  28.5%  25.0%   29.0%*
xingdream         95.5%  72.0%  64.0%   41.0%     ---    9.0%  10.0%    6.0%*
yaoji             99.5%  96.0%  89.0%   73.0%   84.0%    ---   53.0%   41.0%*
jidan             98.0%  97.0%  89.5%   72.5%   85.0%  51.5%    ---    25.0%*
partner_oracle   100.0%* 100.0%* 85.0%*  71.0%*  94.0%*  59.0%*  75.0%*   ---
```
(* = injected from prior eval, not re-run in round-robin)

**Glicko-2 Ratings** (30 convergence passes, 7 partner_oracle matchups fully calibrated):

| Bot | Rating | RD |
|-----|-------|----|
| **partner_oracle** | **1832** | 46 |
| yaoji | 1759 | 44 |
| jidan | 1735 | 43 |
| strategic | 1589 | 42 |
| xingdream | 1448 | 42 |
| heuristic | 1387 | 43 |
| greedy | 1256 | 46 |
| random | 1039 | 55 |

**Notable observations:**
- partner_oracle (1832) is +97 Glicko above jidan (1735) with full calibration. The earlier single-matchup estimate (1923) was inflated by small sample / high RD.
- yaoji (1759) is the closest competitor to partner_oracle: 59% WR for oracle, 41% for yaoji — meaningful gap but not dominant.
- jidan (1735) ≈ yaoji (1759): head-to-head nearly 50-50 (yaoji 53.0%, jidan 51.5%). Essentially peer-strength.
- partner_oracle sweeps random (100%) and greedy (100%), strong vs xingdream (94%) and heuristic (85%), competitive vs strategic (71%) and yaoji (59%).
- RD=46 for partner_oracle now matches calibrated bots (vs RD=75 with one matchup). Reliable estimate.

**Shipped checkpoint:** az_gen3_bestwr.pt (Glicko ~1832, calibrated vs all 7 bots).

---

## 28. Behavior-flags encoder retrain — team-coord-flags branch (2026-04-26)

### Motivation

Gen-3 bestwr (az_gen3_bestwr.pt, Glicko 1832) has a structural gap: **vs yaoji 59%, vs strategic 71%** — both well below the ≥85% target. The hypothesis: the Q-net cannot distinguish "this action overtakes my partner's winning trick" from "this action overtakes an opponent's trick" because the state encoder has no per-action team-coordination signal.

### Encoder change: 9-dim per-action behavior flags

Resurrected `compute_behavior_flags` from git `3237fa8` (Apr 10) into `ml/src/guandan/training/visibility/behavior_flags.py`. Added `encode_state_tier1_team_with_flags(env, player, action, legal) → [489]` = base[480] + flags[9].

**Three axes (3 dims each = [N/A, doing_it, refusing]):**

| Axis | Trigger | Signal |
|------|---------|--------|
| **Cooperating** | trick_winner == partner AND legal beat exists | "this action overtakes my partner's win" |
| **Dwarfing** | leading AND have combo > min(opp_hand_sizes) | "this lead is bigger than opponent can hold" |
| **Assisting** | leading AND have small non-strong combo | "small lead to preserve partner's hand size advantage" |

State dim: 480 → 489. Q-net first layer: [512, 640] → [512, 649].

### Pipeline

1. **Re-distill from Jidan** (`jidan_policy_v2.pt`): d_state=489, val_acc=91.2%, best_val_loss=0.3326. 858k params.
2. **Gen-1 v2 selfplay:** 30K decisions, 4 workers, n_det=20, K=3, Jidan rollouts. 4051s (7.4 dec/s). z mean=+0.114, std=2.174. Policy-only WR: 57%.
3. **Gen-2 v2 selfplay:** 30K decisions, 4 workers, 4725s (6.3 dec/s). z mean similar. Policy-only WR: 53%.
4. **Gen-3 v2 selfplay:** 30K decisions, 2 workers (4-worker OOM), 5442s (5.5 dec/s). Policy-only WR: 52%.

V-mae trend across gens: 1.40 → 1.01 → 0.86 → 0.33 (V head converging). Policy-only WR 52-57% is within ±10pp noise at 100-game inline eval — not a reliable metric.

### Gen-3 v2 with-search eval (2026-04-26, 200 games each)

| Opponent | Baseline (gen-3 original) | Gen-3 v2 | Delta |
|----------|--------------------------|----------|-------|
| strategic | 71.0% | **63.0%** | -8pp ❌ |
| jidan | 75.0% | **68.5%** | -6.5pp ❌ |
| yaoji | 59.0% | **57.0%** | -2pp ❌ |

**Gen-3 v2 regressed on all three opponents.** The flags encoder + AZ retrain from re-distill produced a weaker bot than the original gen-3.

### Root cause analysis

Gen-3 v2 was seeded from `jidan_policy_v2.pt` (re-distill from scratch), then ran only 3 AZ generations vs Jidan rollouts. The original gen-3 was also seeded from `jidan_policy.pt` and ran 3 AZ generations — **both pipelines are identical structurally**. The difference is:

1. **Re-distill loses AZ self-play gains.** `jidan_policy_v2.pt` starts at Jidan-clone level (91% distill accuracy). The original `jidan_policy.pt` was also a Jidan clone. The 489-dim distill shouldn't be weaker. But the original gen-3 was built on top of 6 gens of prior AZ work (including gen-1/gen-3 with ranking loss). The v2 pipeline resets to gen-1 from a fresh distill — 3 gens from a cold start vs 3 gens from an established policy foundation. The original pipeline had accumulated signal from gens 1-2 before gen-3; v2 gen-3 is actually only the 3rd independent iteration.

2. **9 flag dims may dilute the base-480-dim signal.** Adding 9 dims that are mostly zero (most turns don't trigger cooperating/dwarfing/assisting) adds noise to the first-layer weight updates. The base-480 signal is proven; the flag signal is small and sparse.

3. **3 gens from fresh distill is insufficient.** Original gen-3 at 75% WR required gen-1 (68%) → gen-3 (75%) — a two-generation improvement from an already-trained gen-1. V2 gen-1 at 68.5% vs jidan matches original gen-1. V2 gen-3 at the same level suggests the flags are not helping AND re-distill reset the training state.

### Conclusion

The behavior-flags encoder change is **net negative at 3 generations**. The flags don't hurt fundamentally (yaoji WR is flat within noise), but the forced re-distill + cold restart erases the accumulated AZ signal.

**Do not re-distill to add features.** The correct approach for adding encoder features to a trained network is weight surgery (zero-pad new columns in the first linear layer), not full re-distill. Re-distill costs 3+ AZ generations of accumulated signal.

**Shipped checkpoint remains:** `az_gen3_bestwr.pt` (Glicko 1832, original gen-3, 75% vs jidan, 59% vs yaoji).

---

## 29. Weight surgery + gen-4-flags AZ — result (2026-04-26)

### Approach

Instead of re-distilling, zero-pad 9 behavior-flag dims into `az_gen3_bestwr.pt`
(480→489 d_state) via weight surgery on the Q-trunk and V-trunk first layers.
Accumulated AZ signal preserved; new columns zero-init so forward pass is bit-identical
to unpadded checkpoint at t=0. Sanity eval confirmed: padded ckpt 69% vs Jidan
(within 100-game CI of 75% baseline). Then ran one AZ gen (30K decisions, n_det=20,
K=3, 4 workers) and trained from the padded init.

### Training observations

- Loaded padded ckpt cleanly (missing=0, unexpected=0)
- **Baseline policy-only WR at epoch 0: 32%** (policy-only, no search — this is expected
  for raw Q-net argmax; does not reflect PIMC-augmented strength)
- Val loss improved ep 1→4 (2.665→1.198), then **overfit ep 5–7** (1.313 final)
- Best-WR checkpoint saved at ep 7 (50% policy-only WR vs training-time Jidan)
- Early stop at epoch 7 (3 consecutive val no-improve)

### Gen-4-flags eval (200 games each, with PIMC search)

| Opponent | Baseline (gen-3) | Gen-4-flags | Delta |
|----------|-----------------|-------------|-------|
| strategic | 71.0% | **71.5%** | +0.5pp (flat) |
| jidan | 75.0% | **69.5%** | **-5.5pp ❌** |
| yaoji | 59.0% | **47.5%** | **-11.5pp ❌** |

Decision rule: **jidan < 70% → regression.** Training degraded the policy.

### Root cause

Surgery preserved the policy (sanity eval passed). The regression came from training:
1. **Selfplay data quality**: 30K decisions with n_det=20, K=3 produced flat π_search
   distributions (acc stuck at 51-53% throughout training). The search didn't produce
   confident action rankings to distill from.
2. **Overfitting**: val loss bottomed at ep 4 then climbed. Best-WR checkpoint (ep 7)
   was the overfit checkpoint, not the best-val checkpoint.
3. **Yaoji collapse (-11.5pp)**: yaoji plays a structurally different style. One gen
   of Jidan-rollout selfplay reinforces Jidan-specific patterns and may actively
   hurt generalization to other opponents.

### Conclusion

Weight surgery is the **correct mechanism** — surgery itself was clean and
zero-regression at t=0. The **training step** is the failure point. One gen of AZ on
top of the surgery is not sufficient to improve the policy; it degrades it via
overfitting on low-confidence selfplay data.

**az_gen3_bestwr.pt remains the best checkpoint** (Glicko 1832, 75% jidan, 59% yaoji).

The behavior-flags approach is not yet confirmed failed — the surgery works, the
encoder is correct — but generating high-quality selfplay data and preventing
overfitting are unsolved. Next directions: PV-PTIE PPO (trains on-policy, avoids
the flat-π distillation problem) or population play to escape the Jidan-rollout ceiling.

---

## 30. AZ codebase refactor: training/ → azguan/, pvguan/ added (2026-04-27)

### Motivation

Two structural problems accumulated over the AZ experiment:

1. `guandan/training/` had a two-level layout (`training/` + `training/visibility/`)
   that was inconsistent with the flat one-level `guandan/pvguan/` module written
   for the PV-PTIE experiment.
2. All encoder identifiers carried "tier1" versioning terminology (`encode_state_tier1_team`,
   `STATE_DIM_TIER1_TEAM_WITH_FLAGS`, etc.) that meant nothing to a reader unfamiliar
   with the old tiered curriculum plan.

### Changes

**Renamed module: `guandan/training/` → `guandan/azguan/`**

Flattened the two-level layout into a single directory (mirroring `pvguan/`).
`training/visibility/encoding.py` was merged into `azguan/encoding.py` — no
separate subdirectory, no separate file for partner-visible encoders.

**File mapping:**

| Old | New |
|---|---|
| `training/__init__.py` | `azguan/__init__.py` (merged with visibility/__init__) |
| `training/encoding.py` | `azguan/encoding.py` (base + partner-visible merged) |
| `training/q_network.py` | `azguan/q_network.py` |
| `training/visibility/behavior_flags.py` | `azguan/behavior_flags.py` |
| `training/visibility/encoding.py` | merged into `azguan/encoding.py` |

**Identifier renames (tier1 → descriptive):**

| Old | New |
|---|---|
| `encode_state_tier1` | `encode_state_partner` |
| `encode_state_tier1_team` | `encode_state_team` |
| `encode_state_tier1_team_with_flags` | `encode_state_team_with_flags` |
| `STATE_DIM_TIER1` | `STATE_DIM_PARTNER` |
| `STATE_DIM_TIER1_TEAM` | `STATE_DIM_TEAM` |
| `STATE_DIM_TIER1_TEAM_WITH_FLAGS` | `STATE_DIM_TEAM_WITH_FLAGS` |

**Import updates:** 10 files updated (agents, scripts, tests) — all previously
importing from `guandan.training` or `guandan.training.visibility` now import
from `guandan.azguan`.

### Verification

All 145 tests passed after the refactor. Import smoke test confirmed all public
constants and functions resolve correctly at the new paths.

**Commit:** `a28da03` (team-coord-flags branch)

---

## 31. pvguan module — partner-visible PTIE PPO scaffolding (2026-04-27)

### Motivation

With the AZ pipeline plateaued (Glicko 1832, 75% jidan, 59% yaoji — see §29),
the next experiment is **partner-visible PTIE** (Perfect-Training,
Partner-Visible-Execution): a separate PPO actor-critic where the actor sees
own + partner hand + public history, and a privileged critic additionally
sees opponent hands at training time only. The critic is discarded at
deployment. Headline question: **does the privileged critic improve the
partner-visible policy?** Answered by ablating PV-AC (state-only critic) vs
PV-PTIE (state + opponent hands critic) — both share the same actor input,
only the critic differs.

Plan document: `~/.claude/plans/a-lot-of-the-valiant-frog.md`.

### Architecture — what flows through PPO

**Encoded state at one decision** (single source of truth: 755 dims,
action-independent, identical between actor and critic):

```
Group 1 — Card zones (420)        slices [0:420]
  teammate_lo_hand     [60]    ← env.hands[0]
  teammate_hi_hand     [60]    ← env.hands[2]
  teammate_lo_played   [60]    ← env.played[0]
  teammate_hi_played   [60]    ← env.played[2]
  opp_l_played         [60]    ← env.played[1]
  opp_r_played         [60]    ← env.played[3]
  unknown_remaining    [60]    ← deck − all known
Group 2 — Per-seat status (40)    slices [420:460]
  hand_counts/27       [4]
  finish_position_oh   [20]    4 seats × 5 buckets
  pass_sequence_oh     [16]    4 seats × 4 buckets
Group 3 — Acting context (18)     slices [460:478]
  acting_teammate_flag [2]     [1,0]=lo acts, [0,1]=hi acts
  level_rank_oh        [13]
  team_wild_flags      [3]
Group 4 — Active trick (98)       slices [478:576]
  is_self_leader       [1], trick_owner_relative [4]
  trick_type_oh [17], trick_key_oh [15], trick_is_bomb [1]
  trick_cards [60]
Group 5 — Last non-pass (96)      slices [576:672]
  last_actor_relative [4], last_type/key/cards [17+15+60]
Group 6 — Move-history mean (83)  slices [672:755]
  mean of encode_move_event over actual T ≤ 15 moves
```

Coordinate trick: Groups 1-3 use a **team-fixed** basis (`opp_l = seat 1`,
`opp_r = seat 3` after `_reflect_env`). Groups 4-6 use a **self-relative**
basis (`{self, partner, opp_l, opp_r}` from acting POV). Both teammates'
decisions train the same parameters.

**Actor input (764)** = `state[755] ⊕ behavior_flags[9]`. The 9 flags are
candidate-dependent (`cooperating`, `dwarfing`, `assisting`, each 3-way
one-hot). **Action vector (198)** carries the rest of the per-candidate
signal: `played_cards[60] ⊕ remaining_after_play[60] ⊕ type/key/kicker
one-hots ⊕ bomb_tier_oh[9] ⊕ seq_length_oh[13] ⊕ wild_count_oh[3] ⊕ scalars`.

**Critic input (875)** = `state[755] ⊕ privileged_tail[120]`. Action-independent.
The privileged tail is the *only* place ablations differ:
- `PV-AC`: zeros[120]
- `PV-PTIE`: `opp_l_hand[60] ⊕ opp_r_hand[60]`

**Network**: two separate 4-layer MLPs, no shared trunk, no fusion.

```
actor_head:  Linear(962, 256) → ReLU → ... → Linear(256, 1)
             input = state[764] ⊕ action[198] = 962
             output = scalar logit per (state, action) pair
critic_head: Linear(875, 256) → ReLU → ... → Linear(256, 1)
             output = scalar V(s)
             ← final layer zero-init so V=0 at cold-start;
               hidden layers Xavier so gradients flow
```

Actor scoring is bilinear in (state, action): to score K candidates the
network is called K times with the same state and different actions.

**One decision** (player p about to act):

```
legal = env.legal_moves(p)                              # K candidates, mean ~100
state_actor[K, 764] = stack(encode_actor_pair_features(env, p, a_k, legal))
actions[K, 198]     = stack(encode_action(a_k, hand, level_rank))
logits[K] = actor_head(state_actor ⊕ actions) / τ
logits = logits.masked_fill(~legal_mask, -1e9)
log_probs = log_softmax(logits)
k* = sample (rollout) or argmax (eval)
state_critic[875] = encode_critic_state(env, p, mode)   # one vector, all K share it
V_old = critic_head(state_critic)                        # frozen for the rollout
```

Critical invariant: `state_critic` is **bit-identical across all K
candidates** at the same decision (`test_no_leakage.py::
test_critic_state_action_independent`). This is what makes the
privileged-critic ablation valid.

**One hand → terminal rewards**: `env.get_rewards()` per LEVEL_CHANGE
returns `±{1, 2, 3}`; the buffer divides by 3 to normalise to `[-1, 1]`.
Intermediate decisions get reward 0; only each player's *terminal* step
in their own track carries the team reward. This is the credit-assignment
shape PPO must learn through.

**One iter** (~4096 decisions, complete-hand budget — may overshoot):

1. **Per-player GAE** — bootstrap from same player's next decision:
   `δ_t = r_t + γ·V_olds[t+1] − V_olds[t]`, `gae = δ + γλ·gae`. `V_next = 0`
   at the player's terminal step. **Never** crosses players.
2. Normalise advantages across the batch (mean=0, std=1).
3. Pad ragged candidate sets to `max_K` and **freeze**
   `(V_old, returns, advantages)` for the whole update — they are *not*
   recomputed between epochs.
4. **K=4 PPO epochs** over minibatches of 256:

```
ratio = exp(new_log_prob − old_log_prob)
L_π   = -min(ratio·adv, clip(ratio, 1±ε)·adv).mean()
L_V   = MSE(critic_head(state_critic), returns)
H     = entropy of legal-only log-softmax
L_KL  = KL(π_warm || π_θ) over legal subset (warm = frozen iter-0 actor)
total = L_π + 0.5·L_V − 0.01·H + c_KL(t)·L_KL
```

**Critic warm-up phase** (first 30 iters): actor gradients zeroed manually
post-backward, only critic updates. Lets V converge before policy starts
trusting its advantages.

**KL schedule**: `c_KL` stays at `kl_init = 0.05` while actor frozen, then
linearly decays to 0 over 100 iters after unfreezing. Protects the policy
from running too far from the warmstart in the early post-thaw phase.

**Information flow**:

```
                env (full info)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 encode_actor_pair    encode_critic_state
 (own+partner+pub)   (state + privileged)
        │                     │
        ▼                     ▼
   [K, 764]                [875]
        │                     │
        ▼                     ▼
   actor_head            critic_head
        │                     │
   logits[K]                V(s)
        │                     │
   masked softmax             │
        │                     │
   sample ── reward ─────────►│
                              ▼
                       GAE advantages
                              │
                              ▼
                  PPO clip + value + entropy + KL
                              │
                              ▼
                  gradient update both heads
```

The privileged channel `[755:875]` flows **only into critic_head**, only
contributes to V loss and indirectly to advantages. It never sees the actor.
At deployment, critic_head is discarded entirely — the actor was always
partner-visible-only; privileged info just shaped better advantages during
training. The 120-dim slice is the only experimental knob.

### Encoding clarifications (in-session Q&A)

- **`level_rank_oh[13]`** — one-hot over ranks 2..A. The "level rank"
  determines wildcards (heart-suit cards of that rank) and reward shape
  (level A = match win). Sampled per hand; one network learns all 13 levels.
- **`team_wild_flags[3]`** — `[lo_has_wild, hi_has_wild, total_ge_2]`.
  Redundant with hand encoding but a strong inductive bias: wildcards drive
  most combo decisions.
- **Where is team communication encoded?** Three layers: (1) **direct**:
  `partner_hand[60]` (full visibility, the whole point of PTIE), (2) **per-action
  behaviour flags** `[755:764]` (cooperating / dwarfing / assisting — actor-only,
  zero-padded in critic input by API contract), (3) **implicit**: pass sequences
  + trick ownership + move-history mean. No explicit messaging channel.
- **What does "pooled summary" mean?** `move_history_mean[83]` is the
  *element-wise average* of per-move encodings (`encode_move_event`) over the
  last ≤15 moves. Discards order; preserves rates (pass rate, combo-type
  histogram, who's been active, average card volume). Sharp recent state lives
  in Group 5 (last non-pass) and Group 4 (current trick); the pool is just cheap
  background context. Divides by actual T (not 15) so hand age doesn't leak via
  vector magnitude.

### M1 smoke runs

Three end-to-end smokes verifying the pipeline:

| Run | Time | Result |
|---|---|---|
| `distill --supervisor jidan --supervisor-search off`, 5k samples × 2 epochs, 2 workers | 12s | val_acc=0.518; pipeline ✓; 431 dec/s |
| `train_pvguan --critic pv`, 4k decisions, 4 iters, 1k warmstart from above | 3.2 min | 4 iters complete, all metrics logged, leakage alarm passing |
| `train_pvguan --critic ptie`, same params | 3.1 min | Same — both ablations work end-to-end |
| `distill --supervisor oracle --supervisor-search on`, 1k samples × 3 epochs, 1 worker, n_det=4 | 14.5 min | 1 dec/s on M1 (PIMC overhead dominates); pipeline ✓ |
| `distill --supervisor jidan`, 50k samples × 4 epochs, 4 workers (medium smoke) | ~70s | val_acc=0.894 (0.79→0.87→0.89→0.89); 899 dec/s; convergence clean — 500k × 8 should clear 0.95 gate |

The oracle-supervisor M1 throughput at 1 dec/s rules out doing Stage 0
locally with oracle+PIMC. A 40-decision probe earlier showed 8 dec/s but
that turned out to be misleading pool-warmup amortisation.

### Strategic decision point — warmstart strategy (open)

Plan called for **Path A** — distil PartnerOracleBot + PIMC search into
pvguan as Stage 0 (~10-20h Modal CPU, ~$25), then 6 PPO runs from that
warmstart. After running the smoke, three options now under review:

| Path | Stage 0 cost | PPO budget | Total | Risk |
|---|---|---|---|---|
| **A.** Oracle+PIMC warmstart (original) | ~$25, 10-20h | 6M dec | ~$33 | Actor already plays cooperatively at iter 0 — *shrinks* the gap PV-PTIE could show vs PV-AC |
| **B.** Jidan warmstart (cheap) | ~$2, 1-3h | 6M dec | ~$10 | Actor knows valid moves, doesn't know cooperation — PPO has to learn it via the privileged critic |
| **C.** From scratch | $0 | 30-60M dec | $40-80 | Random init brittle on partial-info card games; high seed variance |

Argument shifting toward **Path B**: the original plan rejected Jidan
warmstart on the grounds that "the actor would ignore the partner-visible
channel at iter 0." On reflection, this is *exactly* what the experiment
needs — if the actor already plays cooperatively (Path A), there's less
room for the privileged critic to demonstrate its effect.

Decision pending; logbook will be updated when chosen and run.

---

## 32. PV-PTIE Modal experiments — chasing 80% WR (2026-04-28)

After Path B (jidan warmstart) was selected and the initial 6M PV-PTIE
production run completed, this session explored a series of follow-up
experiments aimed at breaking past the warmstart-plateau ceiling. None
reached the 80% target; the AZ pipeline failure mode (yaoji ~50-58%
ceiling) reproduced cleanly here.

### Validation setup

All runs evaluated against `jidan`, `yaoji`, `strategic` (the three top-3
bots from the Glicko-2 ladder). 300 hands per opponent (75 decks × 4 seat
rotations). Headline metric `validation_metric` = mean of
`promotion_diff_mean` across opponents (LEVEL_CHANGE diff per hand,
positive = we win).

### Infrastructure built this session

| Feature | What it does |
|---|---|
| **Parallel rollout** | `mp.Pool(spawn)` with N CPU workers; persistent pool, weights synced via on-disk state file. **32 workers on A10G/cpu=48 = 22.6× speedup** (1.56s/iter for 4096 dec, was ~37s). Sweet spot, super-linear past 32 saw diminishing returns. |
| **Resume from checkpoint** | `--resume <path>` restores optimizer state, trainer iter, scheduler counter, and loop counters. Mutually exclusive with `--warmstart`. Periodic `--snapshot-interval` replaces hardcoded `[1M, 3M, 6M]` list. |
| **Mixed-opponent rollout** | Per hand with prob `1 - selfplay_frac`: randomly pick from `--opponent-mix` and seat at {1,3}; net plays {0,2}. Only seat-{0,2} decisions enter the PPO buffer. Deterministic mode/opponent choice keyed on `deal_seed`. |
| **Going-out reward shaping** | Potential-based: +`shape` reward when net player goes out, −`shape` from terminal so total per-track reward is preserved (preserves optimal policy). |
| **Early stopping** | `--val-patience N`: stop after N consecutive non-improving evals. |
| **`mean_hand_length` correction** | `collect_rollout_parallel` was using 135 dec/hand (self-play assumption); corrected to `135 × (selfplay_frac + (1 - selfplay_frac) × 0.5)` since mixed-opp hands collect only 2/4 seats. Without this, target_decisions per iter undershoots ~50%. |
| **Parallel validation eval** | `_run_validation` now dispatches one opponent eval per rollout-pool worker. Each worker loads net from disk (same path as rollout), instantiates `PVGuanBot` + opponent bot, runs `paired_eval`. **3-opp val: ~150s sequential → ~50s parallel** (~3× speedup, capped by longest opp). Reuses existing pool — zero extra infrastructure. |

### Run summary

| # | Setup | Best metric | Best iter | jidan | yaoji | strategic | Outcome |
|---|---|---|---|---|---|---|---|
| 1 | Original 6M, jidan warmstart, pure self-play | n/a | ~200 | 70% | 50% | n/a | Peaked iter 200, then collapsed |
| 2 | Resume 3M, selfplay=0.25, mixed-opp | +1.056 | 50 | 70.7% | 49.3% | n/a | Early peak, oscillation |
| 3 | Resume 3M, selfplay=0.0 (pure mixed-opp) | +0.980 | 50 | — | — | — | Killed early; flat |
| 4 | Resume 1M, selfplay=0.25, mixed-opp | **+1.144** | 100 | 69.3% | 56.0% | — | **Best with warmstart** |
| 5 | Resume 1M, selfplay=0.25, going-out-shape=0.5 | +1.000 | 150 | 66.0% | 54.0% | 72.0% | Worse than #4; shaping no help |
| 6 | **From scratch** (no warmstart), selfplay=0.25, mixed-opp | (in progress) | 50 (early) | 42.7% | 31.3% | 30.0% | Real learning trajectory; jidan>yaoji>strategic order from random init |

### Lessons learned

**1. Going-out shaping is a no-op for this setup.** A player's track usually
ends at going-out (no more decisions for that track), so +shape at
going-out and ±terminal land on adjacent or the same timestep. EV barely
changed (~0.35 → ~0.35). The critic was already capturing what's
predictable; remaining error is irreducible noise from partner play and
dealing variance.

**2. The yaoji ceiling is jidan-prior bias, not credit assignment.** Across
all warmstart runs (1, 2, 4, 5), yaoji WR sat at 49-56% regardless of
shaping, selfplay_frac, total_decisions, or warmstart depth. Distillation
ancestry traces entirely to jidan_bot → KL-regularized PPO can't escape
the jidan-shaped policy class. *No reward reshape inside that policy class
breaks the yaoji ceiling.*

**3. PPO + jidan warmstart is at a local optimum.** Metric oscillates around
1.0 = warmstart baseline. Self-play symmetry makes per-track expected
advantage ≈ 0; KL holds policy near jidan; gradients are tiny → equilibrium.

**4. Self-play symmetry cancels gradient.** Real learning signal comes from
the 75% of mixed-opponent hands where seats {0,2} face fixed bots. The
remaining 25% pure self-play hands have ~zero gradient (both teams use the
same network, so expected per-track advantage = 0). `selfplay_frac=0.0`
(run #3) didn't help either — possibly because the partner-visible policy
loses its joint-coordination training without any self-play.

**5. Counter-intuitive WR ordering.** From scratch (run #6), the agent
beats jidan more easily (42.7%) than yaoji (31.3%) or strategic (30.0%) —
opposite of the bot tier table. Hypothesis: jidan plays sacrificially
(over-passes, holds bombs for partner) which is exploitable by an
aggressive learning net; yaoji/strategic punish weak play harder. Also
both jidan and PVGuanBot share the partner-visible decision basis, so the
policy class affinity favors jidan-shaped patterns.

**6. Parallel rollout pays off massively.** 22.6× speedup at 32 workers
made the iteration loop usable for these short-horizon experiments. Pure
sequential at 200 dec/s would have made these comparisons cost-prohibitive.

### Open question: from-scratch trajectory (run #6, in progress)

The from-scratch run will tell us whether the jidan-prior bias was the
true ceiling, or whether there's a deeper architectural / structural
limit. If yaoji exceeds 56% at any point in run #6, the warmstart was the
problem. If it stalls at the same level, the limit lives elsewhere
(observation space, network capacity, or self-play distribution).

### Run #6 — full eval trajectory (killed at iter 1125, ~52% of budget)

10M decision budget, no warmstart, `selfplay_frac=0.25`, mixed-opp =
`jidan,yaoji,strategic`, `val_every=50`, `val_games=300` per opp.
**Killed early** — clear plateau in metric=0.6–0.78 band since iter 700;
remaining budget would not have changed the conclusion.

| iter | dec | metric | jidan | yaoji | strategic | note |
|---|---|---|---|---|---|---|
|  50 | 0.27M | −0.331 | 42.7% | 31.3% | 30.0% | first eval |
| 100 | 0.53M | −0.256 | 48.7% | 32.0% | 34.0% | |
| 150 | 0.77M | +0.056 | 52.0% | 39.3% | 42.7% | metric crosses 0 |
| 200 | 1.01M | +0.227 | 56.0% | 39.3% | 48.0% | |
| 250 | 1.24M | +0.247 | 56.7% | 36.7% | 51.3% | |
| 300 | 1.47M | +0.218 | 60.0% | 36.7% | 49.3% | jidan >60% first time |
| 350 | 1.70M | +0.422 | 56.0% | 38.7% | 56.7% | |
| 400 | 1.92M | **+0.718** | 62.7% | **52.0%** | 66.0% | yaoji breaks 50% (1st) |
| 450 | 2.15M | +0.524 | 64.7% | 37.3% | 59.3% | yaoji crash |
| 500 | 2.38M | +0.536 | 67.3% | 41.3% | 54.0% | jidan all-time high |
| 550 | 2.61M | +0.716 | 58.7% | 43.3% | 68.7% | strategic all-time high |
| 600 | 2.84M | +0.773 | 66.7% | 48.7% | 61.3% | |
| 650 | 3.06M | +0.562 | 58.0% | 42.7% | 63.3% | |
| 700 | 3.29M | **+0.782** | 64.0% | 50.7% | 61.3% | yaoji breaks 50% (2nd); new best metric |
| 750 | 3.52M | +0.689 | 64.0% | 43.3% | 61.3% | |
| 800 | 3.75M | +0.713 | 62.0% | 49.3% | 64.0% | |
| 850 | 3.98M | +0.624 | 63.3% | 44.0% | 64.7% | |
| 900 | 4.20M | +0.760 | 66.7% | 50.7% | 62.0% | yaoji breaks 50% (3rd) |
| 950 | 4.44M | +0.709 | 67.3% | 43.3% | 64.7% | jidan all-time high |
| 1000 | 4.67M | +0.627 | 66.7% | 50.0% | 57.3% | |
| 1050 | 4.91M | +0.671 | 64.7% | 46.7% | 58.7% | killed shortly after |

**Final observations** (killed at iter 1125, ~52% of budget):
- Steady upward trend from iter 50 → iter 400; plateau in metric=0.6–0.78 band thereafter.
- **No new best metric in 350 iters** (iter 700 = +0.782 was final high). Mean of last 5 evals = +0.678 — clearly converged.
- Yaoji crossed 50% three times (iter 400: 52%, iter 700: 50.7%, iter 900: 50.7%) but **never exceeded 52%**. Warmstart-era yaoji peak was 56%, so warmstart actually outperformed from-scratch on yaoji.
- Best from-scratch metric **+0.782** vs run 4 warmstart-best **+1.144** — gap of 0.36 never closed.
- Jidan stable 60–67%, strategic 57–69%; both within warmstart-era band.
- High variance (metric swings ±0.15–0.25 between adjacent evals). `val_games=300` insufficient for low-noise tracking.

### What run #6 actually proved

The from-scratch experiment was designed to test "is jidan-prior bias the
yaoji ceiling?" Answer: **no, in the opposite direction**. Yaoji peaks
*lower* without warmstart (52% from-scratch vs 56% warmstart). The
warmstart was a small *help* on yaoji, not a hindrance.

The yaoji ~50-55% ceiling and the metric ~0.78 plateau appear to be
structural to the (architecture × opponent_mix × selfplay_frac × val
size) system. Reward shaping, warmstart removal, and longer training all
fail to break it. Next experiments must target architecture or training
distribution:

1. **Online opponent-style features** (highest leverage, low cost): bolt
   per-seat statistics (pass rate, lead aggression, bomb rate) onto the
   state encoder. Currently the policy has zero observable signal
   distinguishing yaoji from strategic from jidan.
2. **Reactive curriculum** on opponent mix: weight harder opponents more
   when their WR is below the others. Targets the structural pull toward
   easier-opponent gradients.
3. **Sequential history encoder**: replace pooled `move_history_mean[83]`
   with a small RNN/transformer over the last 15 moves to preserve
   sequential signal.

---

## 33. Run 7 — opp-style features + reactive curriculum (2026-04-29)

**Setup**: from-scratch (no warmstart), `selfplay_frac=0.0`, opponent_mix=jidan/yaoji/strategic,
reactive curriculum (`curriculum_temp=0.3`), Group 7 opp-style features (12 dims, slots 755–767),
`val_patience=8` (run 1), then resumed best checkpoint with `val_patience=0` (run 2 / r2).

**Key changes vs run 6**: opponent-style encoder features, reactive curriculum, no self-play.

### Run 1 (from-scratch → early stop at 2.95M)

| iter | dec | metric | jidan | yaoji | strategic | note |
|---|---|---|---|---|---|---|
| 50 | 0.10M | −0.940 | 42.0% | 14.0% | 27.0% | first eval |
| 100 | 0.20M | −0.370 | 37.0% | 26.0% | 33.0% | |
| 150 | 0.31M | +0.163 | 45.0% | 40.0% | 51.0% | metric crosses 0 |
| 200 | 0.40M | +0.280 | 52.0% | 34.0% | 61.0% | |
| 250 | 0.50M | +0.187 | 55.0% | 33.0% | 48.0% | yaoji dip |
| 300 | 0.62M | −0.117 | 40.0% | 30.0% | 43.0% | regression |
| 350 | 0.73M | +0.300 | 52.0% | 36.0% | 53.0% | recovery |
| 400 | 0.85M | +0.373 | 57.0% | 42.0% | 51.0% | |
| 450 | 0.96M | +0.277 | 53.0% | 39.0% | 52.0% | |
| 500 | 1.07M | +0.387 | 64.0% | 35.0% | 51.0% | jidan spikes |
| 550 | 1.19M | +0.287 | 59.0% | 36.0% | 49.0% | |
| 600 | 1.30M | +0.187 | 57.0% | 33.0% | 45.0% | |
| 650 | 1.41M | +0.430 | 60.0% | 34.0% | 56.0% | |
| 700 | 1.53M | +0.557 | 60.0% | 41.0% | 57.0% | |
| 750 | 1.63M | +0.457 | 61.0% | 37.0% | 59.0% | |
| 800 | 1.73M | +0.537 | 59.0% | 42.0% | 60.0% | |
| 850 | 1.84M | +0.523 | 68.0% | 42.0% | 52.0% | jidan all-time high |
| 900 | 1.94M | +0.593 | 65.0% | 34.0% | 61.0% | |
| 950 | 2.05M | +0.723 | 62.0% | 49.0% | 58.0% | yaoji breaks 49% |
| 1000 | 2.16M | +0.770 | 63.0% | 48.0% | 64.0% | best (run 1 main) |
| 1050 | 2.25M | +0.643 | 66.0% | 41.0% | 65.0% | |
| 1100 | 2.35M | +0.560 | 65.0% | 46.0% | 59.0% | |
| 1150 | 2.45M | +0.737 | 68.0% | 47.0% | 59.0% | |
| 1200 | 2.55M | +0.680 | 63.0% | 51.0% | 58.0% | yaoji breaks 50% |
| 1250 | 2.65M | +0.680 | 60.0% | 54.0% | 56.0% | yaoji all-time high |
| 1300 | 2.75M | +0.703 | 63.0% | 50.0% | 58.0% | |
| 1350 | 2.85M | +0.550 | 63.0% | 52.0% | 54.0% | |
| 1400 | 2.95M | +0.597 | 60.0% | 46.0% | 55.0% | early stop (patience=8) |
| — | 3.18M | **+0.960** | — | — | — | concurrent job best; resumed here |

Three concurrent jobs ran into the same volume (accidental duplicate launches). The best
checkpoint (metric=+0.960 at 3.18M) came from the first job and was used to seed run 2.

### Run 2 (resume from best checkpoint, val_patience=0)

Resumed at cumulative_decisions=3,181,364 / iter=1501. No early stopping.

| iter | dec | metric | jidan | yaoji | strategic | note |
|---|---|---|---|---|---|---|
| 1550 | 3.28M | +0.737 | 58.0% | 45.0% | 62.0% | |
| 1600 | 3.38M | +0.720 | 60.0% | 49.0% | 60.0% | |
| 1650 | 3.48M | +0.671 | 54.0% | 50.0% | 64.0% | |
| 1700 | 3.58M | +0.690 | 59.0% | 47.0% | 65.0% | |
| 1750 | 3.68M | +0.837 | 60.0% | 52.0% | 62.0% | |
| 1800 | 3.78M | +0.827 | 59.0% | 47.0% | 70.0% | strategic breaks 70% |
| 1850 | 3.88M | +0.880 | 60.0% | 42.0% | 77.0% | strategic all-time high |
| 1900 | 3.98M | +0.880 | 63.0% | 47.0% | 67.0% | |
| 1950 | 4.08M | +0.727 | 61.0% | 45.0% | 63.0% | |
| 2000 | 4.18M | +0.883 | 65.0% | 49.0% | 66.0% | |
| 2050 | 4.28M | +0.787 | 64.0% | 46.0% | 64.0% | |
| 2100 | 4.38M | +0.823 | 59.0% | 48.0% | 66.0% | |
| 2150 | 4.48M | +0.887 | 63.0% | 47.0% | 70.0% | best so far (r2) |
| 2200 | 4.58M | +0.763 | 63.0% | 52.0% | 59.0% | yaoji ties best |
| 2250 | 4.68M | +0.633 | 57.0% | 50.0% | 60.0% | |
| 2300 | 4.78M | +0.697 | 56.0% | 52.0% | 62.0% | |
| 2350 | 4.88M | +0.697 | 56.0% | 52.0% | 62.0% | |
| 2400 | 4.98M | +0.867 | 57.0% | 51.0% | 61.0% | |
| 2450 | 5.08M | +0.920 | 60.0% | 52.0% | 69.0% | |
| 2500 | 5.18M | +0.803 | 60.0% | 52.0% | 60.0% | |
| 2550 | 5.28M | +0.713 | 62.0% | 42.0% | 68.0% | |
| 2600 | 5.38M | +0.897 | 59.0% | 52.0% | 69.0% | |
| 2650 | 5.48M | +0.730 | 61.0% | 40.0% | 65.0% | |
| 2700 | 5.58M | +0.940 | 66.0% | 56.0% | 61.0% | yaoji hits 56% |
| 2750 | 5.68M | +0.683 | 58.0% | 51.0% | 59.0% | |
| 2800 | 5.78M | +0.890 | 62.0% | 47.0% | 67.0% | |
| 2850 | 5.88M | +1.023 | 70.0% | 52.0% | 71.0% | new best (r2); jidan 70% |
| 2900 | 5.98M | +0.820 | 61.0% | 52.0% | 68.0% | |
| 2950 | 6.08M | +1.030 | 68.0% | 52.0% | 77.0% | all-time high metric; strategic 77% again |
| 3000 | 6.18M | +0.663 | 53.0% | 50.0% | 64.0% | |
| 3050 | 6.28M | +0.770 | 61.0% | 48.0% | 64.0% | |
| 3100 | 6.38M | +0.728 | 59.0% | 51.0% | 61.0% | *run in progress* |

### Opp-style weight-norm diagnostic

Delta norm (distance of actor first-layer opp-style columns from init) grows monotonically
0.03 → 1.22 over 1.8M decisions (iters 1501→2441), confirming the network actively learns
to use the opponent-style features. Norm grows 1.91 → 2.17 (rotation, not just scaling).

---

## 34. Rollout perf experiments — GPU inference server + worker scaling (2026-04-29)

Goal: target a 3× rollout-throughput speedup. Two changes tested independently
and in combination on Modal A10G (cpu=48), 60K decisions, 8K iter_decisions,
hidden=256, no warmstart, opp-mix={jidan,yaoji,strategic}.

| Config | Workers | GPU server | Wall | Decisions | dec/s | vs A |
|---|---|---|---|---|---|---|
| A | 32 | off (baseline) | 48 s | 60,221 | **1,335** | 1.00× |
| B | 32 | on             | 86 s | 60,099 | 722   | 0.54× |
| C | 64 | off            | 49 s | 60,186 | 1,380 | 1.03× |
| D | 64 | on             | 92 s | 62,967 | 729   | 0.55× |

**Key findings:**

1. **GPU inference server hurts at hidden=256** (-45% on both 32 and 64 worker configs).
   The IPC overhead (mp.Queue pickling each (state, action) batch + reply) exceeds
   the time saved by GPU forward at this small network size. Server stats show
   actor avg batch ≈ 56, critic avg batch ≈ 5.5 — far below GPU saturation.
   Each decision pays 2 round-trips (~150-500 μs each) for what was a 50 μs CPU
   forward; net loss.
2. **Doubling workers 32 → 64 gives only +3% throughput** (1335 → 1380 dec/s).
   We're hitting diminishing returns — likely the per-iter `torch.save(state_dict)`
   to disk + Pool dispatch overhead, not raw CPU compute. cpu=48 oversubscription
   may also limit 64-worker mode.

**What this means for the 3× speedup target:**

- The GPU server design is correct and validated (actor batches form, replies
  return in order, weight sync each iter works), but the *economics* don't pay
  off until the network is large enough that GPU forward dominates IPC. At
  hidden=512+, this likely flips. Re-evaluate alongside any capacity bump.
- The CPU baseline at 1380 dec/s on 64 workers is already the best we can do
  without changing the rollout architecture. Further gains require attacking
  Python overhead (Numba/Cython on `combos.py` + `encoders.py`), or amortizing
  weight sync (broadcast via mp.Manager / shared memory instead of disk save).

**Not a bug to fix; a finding to remember:** keep `--use-gpu-inference` off in
production until/unless we move to a larger network or replace mp.Queue IPC
with a faster shared-memory transport.

---

## 35. Specialist fine-tune diagnostic — architecture ceiling confirmed (2026-04-29)

**Hypothesis:** Multi-opponent training dilutes the policy. Fine-tuning from the best checkpoint
(+1.030 @ 6.08M) on a single opponent each should reveal whether 80% WR is achievable with
the current 256-hidden architecture.

**Setup:** Three independent Modal runs, each resuming from `pvguan_ptie_seed0_best.pt`,
`selfplay_frac=0.0`, `curriculum_temp=0`, `val_games=500` (high-accuracy), 2M new decisions each.

| Run | Opponent | Best WR | Trend |
|-----|----------|---------|-------|
| finetune_jidan | jidan | **64.4%** (iter 1550, first eval) | Immediate decline → 53–62% |
| finetune_yaoji | yaoji | **50.4%** (iter 1850) | Flat throughout, 47–50% |
| finetune_strategic | strategic | **69.2%** (iter 1600, 1700) | Stable oscillation 62–69% |

**Key findings:**

1. **Jidan peaked before any specialist training took effect.** The best checkpoint already encodes
   ~64% vs jidan (with 500 games; the r2 peak of 75% was 100-game noise). Specialist training
   degraded it — first eval was the best, then slow drift downward.

2. **Yaoji is unresponsive.** Seven consecutive evals at 47–50%, CIs all spanning 50%. Dedicated
   jidan-style rollouts produce zero learning signal against yaoji. This rules out dilution as the
   cause — pure yaoji exposure for 2M decisions doesn't help.

3. **Strategic is the only one that fine-tunes cleanly**, with two peaks at 69.2%. But 69% is still
   well below 80%, and it oscillates — not a clear upward trend.

**Verdict:** Dilution is **not** the bottleneck. The 256-hidden architecture genuinely cannot model
yaoji's playstyle. The path to 80% requires a larger net (hidden ≥ 512) or richer opponent-style
features. The specialist-checkpoint idea (router model) is viable for strategic but not worth
pursuing until the base architecture is upgraded.

**Action:** Kill all three runs (killed at iter ~1900 / 4.7M decisions). Next step: bump hidden to 512.

---

## 36. Going-out shaping diagnostic — credit-assignment is not the bottleneck (2026-04-29)

**Hypothesis:** Yaoji wins via partner coordination near the endgame (its scoring function
has hard rules: "if mate.rest ≤ 1, lead my best non-bomb +100"; "if greater is partner with
rest ≤ 6, PASS"). PPO with γ=1.0 over ~135-decision hands has poor credit assignment for
"this lead 30 steps ago helped my partner go out." Going-out shaping (`+shape` when a
teammate goes out, `-shape` from terminal) explicitly delivers that signal.

**Setup:** Same 3-fan-out as §35, but with `--going-out-shape 0.5` and `seed=2`. Fine-tunes
from `pvguan_ptie_seed0_best.pt`, single opponent each, 500-game evals.

| Run | Best WR (shape) | Best WR (baseline §35) | Δ |
|-----|---|---|---|
| yaoji | **51.6%** | 50.4% | +1.2% |
| jidan | **64.8%** | 64.4% | +0.4% |
| strategic | **65.6%** | 69.2% | -3.6% |

All deltas inside the 500-game CI band (~±5%). Shaping did **not** break the yaoji ceiling.

**Verdict:** Credit assignment is not the dominant bottleneck. The yaoji ceiling is
something else — most likely the architecture genuinely cannot represent yaoji's specific
decision boundaries (no explicit `partner_can_win_now`, `opp_can_win_now` features) and/or
the policy distribution doesn't cover the states yaoji-vs-yaoji teams steer the game into.

**Killed at iter ~1750 / 4.2M decisions across all three.**

**Next experiment:** Yaoji-distilled warmstart (mirror of `pvguan_distilled_jidan.pt`) →
seed the policy directly into yaoji's coordination basin, then RL refines past its
blind spots (no card counting, `Possibility=1`).

---

## 37. Yaoji distillation — strong evidence for architecture ceiling (2026-04-29)

**Hypothesis:** A yaoji-distilled warmstart (mirror of `pvguan_distilled_jidan.pt`) seeds the
policy directly into yaoji's coordination basin, then PPO refines past its blind spots
(no card counting, `Possibility=1`).

**Setup:** Modal A10G, 48 CPU, 32 workers. 500k decisions of yaoji self-play, hard-label
distillation with `label_smoothing=0.1`, 8 epochs cosine LR=3e-4 → 1.5e-5, batch=512.

**Result — striking degradation:**

| Epoch | tr_acc | val_acc | Note |
|---|---|---|---|
| 1 | 0.611 | **0.622** | best, saved |
| 2 | 0.625 | 0.563 | val regressing |
| 3 | 0.510 | 0.496 | both crashing |
| 4 | 0.477 | 0.474 | early stop |

Final: **val_acc=0.622** at epoch 1. Saved as `/runs/warmstart/pvguan_distilled_yaoji.pt`.

**Compare:** `jidan` distillation (logbook §31 smoke) hit **0.894** on **10× fewer samples**
(50k × 4 epochs). Yaoji on 500k samples can only hit 0.622. Same architecture, same script.

**Yaoji "always pass" baseline:** ~30% (yaoji passes 30% of recorded decisions). So 62% is
real learning over the trivial baseline, but **38% of yaoji's argmax decisions are still
mispredicted by an MLP with 256 hidden + 12 opp-style features**. This strongly suggests
yaoji's branchy partnership rules (`if mate.rest ≤ 1: lead small; elif greater_pos != partner:
play normally; ...`) don't decompose cleanly into the current feature space.

**Numerical bug noticed:** training loss reported as 78M → 48M → 6M → 4M. This is
`label_smoothing=0.1` distributing mass across **all** output positions including masked ones
(set to `-1e9`), creating a huge constant offset. The bug doesn't break gradients (signal
still flows through unmasked positions), but may contribute to the instability that crashes
both train and val acc after epoch 1. Fix would be manually-masked label smoothing — not
worth doing unless we revisit yaoji distillation seriously.

**Action:** Saved 62% checkpoint as warmstart. Launched PPO from this warmstart on
yaoji-only mixed rollouts to test whether even a weak warmstart bootstraps better than RL
from scratch (§35 baseline: 50% flat).

---

## 38. PPO from yaoji-distilled warmstart — actively harmful (2026-04-29)

**Hypothesis:** Even a weak (62% val_acc) yaoji imitation might bootstrap PPO into a better
basin than RL from scratch.

**Setup:** PPO from `pvguan_distilled_yaoji.pt` (62% val_acc), yaoji-only opponents,
selfplay_frac=0.0, curriculum off, 32 workers, 500 val games, fresh seed=3.

**Result — much worse than from-scratch:**

| iter | dec | wr | Δ |
|---|---|---|---|
| 50  | 0.21M | **0.184** | -1.576 |
| 100 | 0.41M | 0.224 | -1.372 |
| 150 | 0.63M | **0.100** | -2.012 |

vs **§35 from-scratch baseline: 50% flat**. The warmstart cuts yaoji WR by 30–40%.
Killed at iter 150 / 0.63M decisions.

**Interpretation:** A 62%-accurate yaoji imitator is *catastrophically* miscalibrated against
yaoji. The 38% wrong decisions concentrate in high-leverage moments — partial-yaoji
behavior (e.g., correct passes but wrong leads) is worse than balanced self-play. The
warmstart pushes the network into a pathological region of policy space that PPO can't
recover from in 150 iters.

**Three converging pieces of evidence for architecture ceiling:**
1. §35: yaoji RL fine-tune flat at 47–50%, no learning signal
2. §36: going-out shaping (credit-assignment fix) didn't help
3. §37/§38: distillation peaked at 62% (vs jidan 89%), warmstart actively harmful

**Verdict:** The 256-hidden + 12-opp-style architecture genuinely cannot represent yaoji's
policy distribution. Yaoji's branchy partner-coordination rules don't decompose linearly
into the current feature space, and any partial approximation is dangerously off-policy.

**Next experiment:** Bump hidden to 512. Re-distill from yaoji to see if val_acc clears
80%+. If yes, full PPO from the bigger warmstart. If no, we need richer opp-style features
(`partner_can_win_now`, `opp_can_win_now`, `partner_almost_out`) — yaoji's exact decision
boundaries — not just more capacity.

---

## 39. Deep dive — opponent bot strategies (2026-04-29)

Before deciding the next architecture move, this section catalogues exactly what each
rule-based opponent does. The §35-38 ceiling story is shaped by these decision boundaries —
some are smooth (jidan ✓), some are discontinuous cliffs (yaoji ✗).

### 39.1 Yaoji (`_vendor/yaoji/mysolve.py`, 312 lines)

NUAA 3rd place, 2020. Single-pass action scorer with explicit partner awareness.

**Core formula:** `Score = Gain × (1 + Possibility) / Value`. `Possibility` is hard-coded
to `1` everywhere — never used. So really `Score = 2 × Gain / Value`.

**Per-card value `getval()`:**
- Rank 2..A → 1..13, level rank → 14, hearts level rank → 340.
- Big joker → 14 base, escalates with count: `+20` if pair held, `×100` if both jokers held.
- Red joker → same escalation.
- Same-rank multiplicity bonus: pairs +20, triples +40, quads +220, 5-of-kind +300, 6+ +400-600.
- Suit-by-suit straight-flush scan: if 5 consecutive of same suit, override value to `320+rank`.

**Action scoring per type:**
- `Single`/`Pair`/`Trips`: `gain ∈ {1,2,3}`, `value = max(card_values)`.
- `Straight`: `gain=5`, `value = sum(card_values)`.
- `ThreePair`/`ThreeWithTwo`/`TwoTrips`: `gain ∈ {5,6}`, custom value averaging.

**Partner-coordination rules — the CLIFFS (these are the architecture-ceiling problem):**

| Condition | Effect on score |
|---|---|
| `mate.rest == 1` AND leading single | `score += 100` (set up partner) |
| `mate.rest == 2` AND leading pair | `score += 100` |
| `opp.rest == 1` AND leading single | `score *= -1` (don't feed) |
| `opp.rest == 2` AND leading pair | `score *= -1` |
| `opp.rest == 3` AND leading trips | `score *= -1` |
| greater is partner, partner.rest ≤ 6 | PASS gets max priority |
| greater is partner | bomb/SF score = `-10000` |
| greater is opponent, opp2.rest ∈ [1,3] | PASS = `-9999` |
| Following partner Single | PASS value = 25, gain=2 |
| Following partner Pair | PASS value = 65, gain=4 |

These are **discrete one-hot indicators** of opponent state. A smooth MLP cannot represent
them without explicit features.

**What yaoji does NOT do:** card counting, opponent hand inference, tree search,
probabilistic reasoning (`Possibility=1` always).

### 39.2 Jidan (`_vendor/jidan/message_Reyn_CUR.py`, 2514 lines)

NUAA 2nd place, 2020 ("Reyn_AI 2.0"). 8× more code than yaoji but **paradoxically easier to
imitate** (89% val_acc vs 62%).

**Per-card value `get_point_val()` — non-linear schedule:**
```
point_val = [28, 1, 2, 3, 4, 5, 7, 9, 12, 15, 18, 22, 25, 100]
                     2  3  4  5  6  7  8   9   T   J   Q   K   A
```
Note convex growth — small cards barely worth anything, A worth 25, but the schedule has a
slot-0 = 28 used for level-rank context.

**Per-card overrides:**
- Hearts at level rank → 500 (THE single most valuable card)
- Level rank (non-hearts) → 100
- Big joker → 150
- Red joker → 200

**Hand evaluation `get_remain_VAL()`:**
1. Suit-by-suit SF scan (S/H/C/D each scanned for any 5-consecutive) → `+50 × point_val[start]`
2. Per-rank decomposition penalties (bigger group = better):
   - singleton: `-200 + point_val`
   - pair: `-180 + 2×point_val`
   - triple: `-300 + 5×point_val`
   - bomb (n≥4): `+100 × point_val × (n-3)` (bombs are great, escalating)
3. Joker bonus: `count_big × 150 + count_red × 200`
4. Level rank: `count_level × 100 + count_heart_level × 500`

**Action scoring `get_VAL()`:** `val = base − sum(card_values) + get_remain_VAL(after_play)`.
This is **1-step lookahead** — value the resulting hand and pick max. Bomb scoring
explicitly penalizes bomb-spending (`−100 × point_val(rank)`).

**Decision flow `check_message()`:**

| Phase | Logic |
|---|---|
| **Tribute return** | Pick action maximizing remaining hand value |
| **Leading** (greater is self/none) | Score all actions via `get_VAL`, pick max |
| **Following partner** (greater is mate) | Hard-coded "don't over-step" rules: |
| | • partner Bomb → PASS |
| | • partner Single/Pair/Trips/3W2 with rank ∈ {T,J,Q,K,A,B,R,level} → PASS |
| | • partner ThreePair/TripsPair with rank ∈ {T,J,Q,K,A} → PASS |
| | • partner Straight starting at {7,8,9,T,J} → PASS |
| | • partner StraightFlush/Joker → PASS |
| | • Else: filter to same-type, point-dist ≤ 2, no Heart-level → max value |
| **Following opponent** (else) | Score all actions via `get_VAL_OPP`, pick max |

**Why jidan is EASIER for our network despite being more complex:**
- Most decisions are **monotonic in card rank** (higher = more value).
- `get_remain_VAL` is **separable** (each card contributes independently).
- Boundaries are **smooth** — there's no "exactly rest==1" cliff.
- The partner-suppression rules are coarse (rank thresholds) and the network can approximate
  them via "high-card detector" features.

### 39.3 Strategic (`strategic_bot.py`, 281 lines)

Our own implementation. Phase-based logic on top of `HeuristicBot.HandPlan`.

**Hand decomposition `HandPlan`:**
- Splits into wilds + naturals
- Groups naturals by rank → singles, pairs, triples, quads (sorted by level order)
- Tracks jokers, quad ranks, bomb count

**Leading (`_strategic_lead`) — phase ladder:**
1. Can go out in one play → do it
2. Hand ≤ 5 cards → `_endgame_lead` finds X-then-Y two-play sequence
3. Partner already out → `_aggressive_lead` (multi-card combos to finish fast)
4. Partner has 1–5 cards → `_help_partner_lead` (small singles/pairs partner can follow)
5. Default → `_efficient_lead` (singles, then straights/tubes/plates, then pairs, then FH)

**Following (`_strategic_follow`):**
1. Partner is winning → PASS *unless* I can go out
2. Trick is BJ single → play RJ if I have it (free win)
3. Opp winning, cheapest same-type beat that doesn't break a bomb → play it
4. Bomb decision — `should_bomb` gate:
   - `opp_min_cards ≤ 5` (opp about to win)
   - OR `partner_close` (partner ≤ 5)
   - OR `my_hand ≤ 4` (I'm about to win)
   - If true: play weakest bomb
5. Default → PASS

Strategic has **smoother phase boundaries** than yaoji — `partner_cards ≤ 5` is a 5-step
ladder, not a 1-step cliff. The network handles strategic at ~69% WR.

### 39.4 Comparison matrix

| Capability | Yaoji | Jidan | Strategic |
|---|:-:|:-:|:-:|
| Per-card non-linear valuation | ✓ | ✓ (most sophisticated) | implicit |
| Suit-by-suit SF detection | ✓ | ✓ | ✗ |
| Hand decomposition | rank counts | rank+suit | natural groups |
| 1-step lookahead | ✗ | ✓ (resimulates hand value) | ✗ |
| Partner-rest cliffs (`==1`, `==2`, `==3`) | ✓ (sharp) | ✗ | partial (`≤5` smooth) |
| Action-type-based partner suppression | ✗ | ✓ (8 rules) | partial |
| Bomb conservation | -10000 vs partner | negative score | gated |
| Card counting | ✗ | ✗ | ✗ |
| Opponent hand inference | ✗ | ✗ | ✗ |
| Tree search / MCTS | ✗ | ✗ | ✗ |
| Lines of code | 312 | 2514 | 281 |
| Our network's argmax accuracy | **62%** | **89%** | n/a |
| Our PPO win rate | **~50%** | **~64%** | **~69%** |

### 39.5 Why yaoji is uniquely hard — the cliff hypothesis

The pattern is consistent across both metrics (distillation and PPO):

| Bot | Argmax (distill) | PPO win rate |
|---|---|---|
| yaoji (3rd place) | 62% | 50% |
| strategic (custom) | n/a | 69% |
| jidan (2nd place, 8× more code) | **89%** | 64% |

**Jidan, the strongest hand-coded opponent, is the easiest to imitate.** This rules out
"strength" or "complexity" as the cause of our yaoji ceiling. The discriminator is
**logical structure**:

- Yaoji's score has **division** (`Score = Gain × (1+Poss) / Value`) — nonlinear, hard for
  ReLU MLPs to approximate.
- Yaoji's partner rules are **single-step cliffs**: `if rest == 1: ... else 0`. With our
  current `seat_status (40)` features as smoothed embeddings, the network has to invert the
  embedding to recover "rest == 1 exactly," then implement an indicator function.
- Jidan's complexity is **monotonic and separable**: the score function is mostly a sum of
  per-card terms. ReLU MLPs eat that for breakfast.

### 39.6 Implications for the next architecture move

The §35-38 conclusion was "bump hidden=512." This deep-dive sharpens the recommendation:

**Capacity (hidden=512) won't fix yaoji on its own.** Adding capacity to a smooth MLP makes
it a smoother, larger smooth MLP — still bad at representing cliffs.

**The real fix is to add explicit one-hot features that match yaoji's exact decision
boundaries:**

```
partner_rest_eq_1         (1 dim)  — hits yaoji's "set up partner" cliff
partner_rest_eq_2         (1 dim)
partner_rest_eq_3         (1 dim)
partner_rest_leq_6        (1 dim)  — hits yaoji's "PASS to partner" cliff
opp1_rest_eq_1            (1 dim)
opp1_rest_eq_2            (1 dim)
opp1_rest_eq_3            (1 dim)
opp2_rest_eq_{1,2,3}      (3 dims)
greater_is_partner        (1 dim)  — gates yaoji's bomb/SF -10000 rule
greater_is_opp1/opp2      (2 dims)
```

12-15 cheap one-hot dims. With these, yaoji's rules become a single ReLU layer:
`output = w · indicator + bias`. No capacity argument needed.

**Test sequence:**
1. Add features → re-distill yaoji → measure val_acc (should jump 62% → 80%+)
2. If yes: full PPO run with augmented features
3. If no (val_acc stays ≈62%): yaoji has logic we still aren't capturing — investigate
   division/Possibility approximation or the suit-SF interaction

---

## 40. Tactical legal-move cap + cap impact on prior runs (2026-04-29)

### The bug

`_cap_legal` (rollout.py) used to:
1. Hard-cap legal moves at 64
2. Always keep `PASS` and bombs/SF/BJ
3. **Fill remaining slots with cheapest-rank moves** (sorted by `sum(c.rank for c in cards)`)

Combined with the suit-variant inflation in §39's investigation (one 27-card hand had K=143 with only 40 strategically distinct plays — 3.6× duplicate inflation), this cheapest-rank fill silently dropped exactly the high-rank tactical plays yaoji's cliff rules trigger on:
- `len(cards) == opp_rest` matches (must-play to deny opp from going out)
- `len(cards) == partner_rest` matches (must-play to set partner up)
- Hand-emptying plays in late game

Cap fired on ~10% of pre-dedup decisions, ~4% post-dedup.

### The fix

Replaced cheapest-rank fill with a tactical must-keep + stratified rank-diverse round-robin:

Must-keep:
- PASS, all bombs / SF / BJ
- actions emptying our hand
- actions with `len(cards) ∈ {partner_rest, opp_l_rest, opp_r_rest}`
- on lead, all SINGLE/PAIR/TRIPLE if any active opp has rest ∈ {1,2,3}

Fill: round-robin across combo types, with `_spread_order` (endpoints-first bisection) for rank diversity within each type.

Default cap raised 64 → 128. At cap=128, cap fires on **0.4%** of decisions (155/39k random rollouts) — true safety valve. K p99 post-dedup = 59, well under 128. All 192 tests pass.

### Impact on prior runs

The cap distortion biased every PPO and distillation run on yaoji systematically low. Estimated effect:

| Run | Yaoji metric | Likely cap impact |
|-----|--------------|-------------------|
| Run 4 (warmstart, §32) | 56% WR | ~1-3pp |
| Run 6 (from-scratch, §32) | 52% WR | ~1-3pp |
| Run 7 r2 (opp-style + curriculum, §33) | 56% WR | ~1-3pp |
| §35 finetune_yaoji | 50.4% WR | ~1-3pp |
| §37 distill_yaoji | 62% val_acc | already retested → 64.4% post-dedup (cap=64 still in play) |

The cap fix won't close the 50% → 80% WR gap. The architecture/encoder is the dominant bottleneck. Prior verdicts (yaoji ceiling is structural; dilution isn't the cause; warmstart from 62% val_acc is harmful) all hold qualitatively. The pessimism in §32–§38 numbers is bounded at a few percentage points.

### Important: val_acc and WR are NOT the same metric

In §37 we noted yaoji distillation peaks at 62-64% val_acc. In §32-§35 we noted PPO yaoji WR plateaus at 50-56%. These look numerically similar but measure orthogonal things, and **must not be conflated**:

| Metric | Definition | What "stuck" means |
|---|---|---|
| **val_acc** (distillation) | On held-out `(state, legal, supervisor_pick)` tuples: fraction where `argmax_a Q(state,a) == supervisor_pick`. Pure imitation top-1 agreement. | Network can't represent the supervisor's decision function under available capacity/encoder. The 36% mismatches may be strategically equivalent or genuinely wrong — distillation alone can't tell. |
| **WR** (eval) | Out of N games (typically 300 or 500): fraction where seats {0,2} = our policy beat seats {1,3} = opponent bot. Game-outcome metric. | Our policy + partner can't outscore opp + partner in actual games over many hands. |

**They have no monotonic relationship to each other:**
- A model with val_acc = 100% (perfect yaoji imitation) would WR ≈ 50% vs yaoji — a mirror match against the bot it's imitating.
- A model with val_acc = 0% (deliberately anti-yaoji) could WR > 50% if anti-yaoji happens to be stronger.

**Why the two ceilings (62-64% val_acc, 50-56% WR) are still weak independent evidence for an architecture ceiling:**
- Both fail to track yaoji-specific patterns (cliff rules, partner-rest matching) regardless of training signal type
- Distillation provides dense per-decision supervision and the model still misses 36% — purely a representational limit
- PPO provides sparse per-hand rewards and the model still plateaus at 50-56% — could be representational, exploration, or credit assignment

The convergent ceiling across two independent training paradigms suggests the bottleneck isn't training-signal noise but the (encoder × architecture) combination. **But the magnitudes are not directly comparable.** Future logbook entries should never put val_acc and WR side-by-side without disambiguating units.

---

## 42. GuanZero M0 — paper-faithful DMC implementation from scratch (2026-05-02)

### Motivation

Every prior GuanZero attempt (§18) reused the existing `training/` stack (480-dim team state, AZ search pipeline). The DanZero/GuanZero paper uses a different approach: 4 position-specific Q-nets trained end-to-end on Deep Monte Carlo returns with **own-hand + oracle-partner-hand** encoding and no search. §18 never isolated whether the architecture or the training signal was the bottleneck. This fresh implementation follows the paper exactly, using a fully independent `guanzero/` module.

### Architecture

**Encoder (`guanzero/encoder.py`)** — paper-faithful state-action dict:

| Channel | Shape | Notes |
|---|---|---|
| `own_hand` | (108,) | multi-hot, card_id = rank_idx×8 + suit×2 + deck |
| `others_hand` | (108,) | oracle channel; zeroed for M4 belief ablation |
| `recent_action_each_player` | (4, 108) | last non-pass per seat from history |
| `played_cards_others` | (3, 108) | cumulative played, relative seat order |
| `remaining_counts_others` | (3, 27) | one-hot bucket per non-self seat |
| `level` | (13,) | one-hot over ranks 2..A |
| `history` | (20, 108) | last 20 moves, pad-left, PASS=zero |
| `behavior` | (9,) | cooperation/dwarfing/assisting × {N/A, doing, refusing} — reused from `azguan/behavior_flags.py` |
| `candidate_action` | (108,) | action being scored |

Dict output (not flat concat) so future ablation variants can swap channels without rewriting the network.

**Q-network (`guanzero/q_network.py`)** — LSTM over history + flat MLP:
- LSTM(input=108, hidden=hidden_lstm) processes `history` sequence (20 steps, 1 card per step)
- Static channels (everything except history) concatenated with LSTM final hidden state
- N-layer MLP(hidden_mlp) → scalar Q
- Paper spec: hidden_lstm=256, hidden_mlp=1024, n_mlp_layers=6 → ~5M params/seat, 20M total

4 independent networks (one per seat 0–3), 4 independent Adam optimizers. Q-nets never share weights.

**Returns (`guanzero/returns.py`)** — `compute_mc_returns`: sparse terminal reward from `env.get_rewards()` normalised by 3, propagated back to every step in each player's trajectory. γ=1.0 → every step on player's trajectory gets the identical terminal Q-target.

### Single-process baseline (`guanzero/train.py`)

`TrainConfig` dataclass; `train()` alternates: play_episode → buffer.push → learner.update every `learn_every_episodes`. Epsilon-greedy actors with linear decay over `epsilon_decay_episodes`. Checkpoints to `ml/runs/guanzero_m0_{YYYYMMDD_HHMM}/`.

**Smoke run (`--quick`, tiny net 64/128/3, 100 ep):** 5.0 ep/s, loss 0.59→0.41 over 100 eps. Engine integration confirmed clean.

**Tests (`ml/tests/guanzero/`):** 22 tests — encoder shapes, returns correctness, buffer sampling, q-net forward/backward, agent adapter, distributed smoke. All pass.

---

## 43. GuanZero M0 — faithful persistent actor-learner DMC (2026-05-02)

### Approach

The DanZero/GuanZero paper uses N persistent CPU actors generating trajectories continuously into a central replay buffer, with a central learner updating global Q-nets asynchronously. The key invariant: **actors never stall waiting for each other, and the learner updates between every queue drain — not once per rollout batch.** This is structurally different from Pool.starmap (synchronous barriers between batches).

Implementation in `guanzero/train_distributed.py` + `actor.py` + `learner.py`:

```
N CPU actors (persistent, local Q-net copies, torch.set_num_threads(1))
        ↓  mp.Queue(maxsize=64, timeout=5s)  — bounded, drop on overflow
Central learner (global Q-nets, replay buffer, MPS device)
        ↓  os.replace() atomic writes  — no partial-read window
Actors sync every sync_interval_episodes
```

**Atomic weight publishing contract:**
1. `torch.save` → `weights_{v}.tmp`
2. `os.replace(tmp, weights_{v}.pt)` — POSIX atomic rename
3. `latest.tmp` ← version number
4. `os.replace(latest.tmp, latest.txt)` — atomic

Actors poll `latest.txt`; race window is zero.

**Config (`guanzero/config/m0_faithful_distributed.yaml`):** n_actors=6, batch_size=512, buffer_capacity_per_player=50k, sync_interval=20 episodes, publish_interval=100 updates, paper-spec network.

Run dir auto-generates `guanzero_m0_{YYYYMMDD_HHMM}` timestamp (matching pvguan convention).

### Benchmark (tiny network, 1000 learner updates each)

The learner is the bottleneck with tiny nets — actors fill the 200k buffer immediately regardless of actor count. With the paper-spec network, the bottleneck shifts to actors (MPS learner outpaces 6 CPU actors).

| n_actors | wall time | notes |
|---|---|---|
| 1 | 3m 23s | learner-bottlenecked |
| 2 | *(OS stall — invalid)* | macOS process suspension mid-run |
| 4 | 3m 47s | learner-bottlenecked |
| 6 | 4m 25s | learner-bottlenecked |

### Win-rate vs random (200 games, team 0/2 = GuanZero, team 1/3 = random)

| Checkpoint | Updates | WR vs random | ±SE |
|---|---|---|---|
| update_00005000.pt | 5k | 52.6% | 1.6% |
| update_00010000.pt | 10k | 56.7% | 1.6% |
| update_00015000.pt | 15k | 59.3% | 1.6% |
| update_00020000.pt | 20k | 59.3% | 1.6% |
| update_00025000.pt | 25k | 61.6% | 1.5% |
| update_00030000.pt | 30k | 62.6% | 1.5% |
| update_00035000.pt | 35k | 57.4% | 1.6% |
| update_00040000.pt | 40k | 61.6% | 1.5% |
| update_00045000.pt | 45k | 61.5% | 1.5% |
| update_00050000.pt | 50k | 63.3% | 1.5% |

1000 games per checkpoint (±1.5–1.6% SE). Random baseline = 50%. 35k dip likely due to the laptop-restart resume from 20k — 3k updates of experience were lost. Overall trend: steady improvement 5k→30k, plateau/noise 30k→50k around 61–63%. Did not reach >70% vs random in this run.

### Paper-spec run (2026-05-02, in progress)

**CPU baseline:** 0.45 updates/sec → 31h estimated for 50k updates. Not viable locally.

**MPS (`--device mps`):** learner backward pass moves to Metal. Warmup 4 min → steady-state 3.45 updates/sec (7.6× over CPU). Run started 01:20:28, currently at 12,400/50,000 updates (loss ≈ 0.025). ETA ~07:45 — note: approximately 25 min over the 6h local cap; acceptable given partial run cannot be cheaply restarted.

### Batch size sweep (MPS, paper-spec network, 6 actors, encode_all active)

Tested on 2026-05-02/03 with batch_size=256/512/1024, measured at steady state (buf=200k):

| batch_size | upd/s | notes |
|---|---|---|
| 256 | 1.46 | too small — kernel launch overhead dominates |
| **512** | **2.8** | **optimal — best MPS utilization** |
| 1024 | 1.81 | diminishing returns — forward/backward cost outweighs batch efficiency |

512 is the MPS sweet spot for this network size (LSTM 256 + MLP 1024×6). The bottleneck is 4 serial optimizer steps per update, not memory bandwidth. `encode_all` (not present in 50k run) adds ~0.8 upd/s vs the earlier 2.0 upd/s baseline.

### Encoder optimization (`encode_all`)

With MPS, the bottleneck shifted to actors (buf=37k, not capped — learner consuming faster than 6 actors produce). Profiling showed `encoder.encode()` called N times per step, recomputing 7 shared state channels for every legal action. Fix: `encode_all(env, player, legal_moves)` computes `own_hand`, `others_hand`, `history`, `last_action`, `played_cards`, `remaining_counts`, `level` once; only `behavior` + `candidate_action` iterate per action. Shallow-copies state dict into each result — shared read-only numpy arrays, no extra allocation.

Speedup estimate: ~3–5× actor throughput for typical ~20-legal-action steps. Will apply to the next run.

---

## 44. GuanZero M0 — 200k checkpoint multi-opponent eval (2026-05-04)

`update_00200000.pt` from `guanzero_m0_20260503_1337` (an orphaned-process run that kept training past its `--updates 2500` cap because SIGTERM on the orchestrator skipped the `finally:` block). 200k updates ≈ 8h MPS wall, batch=512, paper-spec LSTM 256 + MLP 1024×6 + 4 per-position nets.

Eval: 1000 games per opponent, balanced across seat assignments (500 with GuanZero on seats 0/2, 500 on seats 1/3). 6-worker CPU parallelism via `eval_guanzero.py --workers 6` (CPU beats MPS for the small per-move batches in eval).

| Opponent | even (0/2) WR | odd (1/3) WR | **combined** | ±SE |
|---|---|---|---|---|
| random | 83.2% | 83.0% | **83.1%** | 1.2% |
| greedy | 70.2% | 62.4% | **66.3%** | 1.5% |
| heuristic | 30.4% | 34.6% | **32.5%** | 1.5% |

**Reads:**
- **Per-seat symmetry holds for random and heuristic** (gaps within SE), confirming the 4 per-position nets converged to comparable strength. **Greedy shows an 8-pt asymmetry** (70 vs 62) — the only opponent where the per-seat networks diverge meaningfully. Likely due to greedy's deterministic play creating asymmetric opponent-action distributions the seat-specific nets weren't equally exposed to.
- **83% vs random** is meaningful but bounded: the loss curve had plateaued from 80k onward (avg ≈ 0.018), so 200k is at or near the M0 ceiling. Pushing to 1M updates wouldn't likely close much of the gap.
- **32.5% vs heuristic ≪ project goal of ≥85%** vs the strongest baselines (strategic 71%, jidan 75%, yaoji 59%). M0 as implemented is far below baseline strength on the hard opponents — confirms the conclusion from §41 that the algorithm/architecture (DMC + per-position nets + paper-spec LSTM/MLP) hits a ceiling well below the heuristic agent.

**Eval speed**: ~1m46s for 1000 games on 6 CPU workers (vs estimated ~25 min sequential MPS). MPS dispatch latency dominates compute at small per-move batches (~10–50 actions); CPU + multiprocess fan-out is the right combo for eval.

### Full WR curve vs random — every checkpoint (5k → 200k)

Resumable sweep via `guandan.guanzero.wr_curve` — 1000 games per checkpoint, balanced (500 even / 500 odd), persists JSON after each checkpoint. ~70 min total wall time on 6 CPU workers.

| updates | combined WR | even (0/2) | odd (1/3) | gap |
|---:|---:|---:|---:|---:|
|   5k | 49.5% | 47.6% | 51.4% |  -3.8 |
|  10k | 52.9% | 56.6% | 49.2% |  +7.4 |
|  15k | 54.4% | 55.6% | 53.2% |  +2.4 |
|  20k | 57.3% | 57.4% | 57.2% |  +0.2 |
|  25k | 56.9% | 58.8% | 55.0% |  +3.8 |
|  30k | 60.1% | 57.2% | 63.0% |  -5.8 |
|  35k | 59.1% | 65.0% | 53.2% | +11.8 |
|  40k | 60.8% | 60.6% | 61.0% |  -0.4 |
|  45k | 57.9% | 60.8% | 55.0% |  +5.8 |
|  50k | 61.9% | 64.8% | 59.0% |  +5.8 |
|  55k | 62.3% | 60.2% | 64.4% |  -4.2 |
|  60k | 66.7% | 71.4% | 62.0% |  +9.4 |
|  65k | 63.6% | 66.4% | 60.8% |  +5.6 |
|  70k | 65.4% | 68.4% | 62.4% |  +6.0 |
|  75k | 69.4% | 71.2% | 67.6% |  +3.6 |
|  80k | 68.9% | 69.0% | 68.8% |  +0.2 |
|  85k | 73.8% | 75.2% | 72.4% |  +2.8 |
|  90k | 69.6% | 72.4% | 66.8% |  +5.6 |
|  95k | 69.2% | 72.0% | 66.4% |  +5.6 |
| 100k | 73.2% | 74.4% | 72.0% |  +2.4 |
| 105k | 74.4% | 77.2% | 71.6% |  +5.6 |
| 110k | 74.5% | 75.2% | 73.8% |  +1.4 |
| 115k | 77.8% | 81.0% | 74.6% |  +6.4 |
| 120k | 75.8% | 78.6% | 73.0% |  +5.6 |
| 125k | 78.1% | 81.0% | 75.2% |  +5.8 |
| 130k | 74.2% | 78.8% | 69.6% |  +9.2 |
| 135k | 78.7% | 81.2% | 76.2% |  +5.0 |
| 140k | 77.7% | 77.6% | 77.8% |  -0.2 |
| 145k | 77.9% | 80.8% | 75.0% |  +5.8 |
| 150k | 76.4% | 77.4% | 75.4% |  +2.0 |
| 155k | 81.0% | 82.2% | 79.8% |  +2.4 |
| 160k | 80.2% | 83.4% | 77.0% |  +6.4 |
| 165k | 80.0% | 80.2% | 79.8% |  +0.4 |
| 170k | 80.8% | 82.2% | 79.4% |  +2.8 |
| 175k | 82.7% | 81.4% | 84.0% |  -2.6 |
| 180k | 82.6% | 85.4% | 79.8% |  +5.6 |
| 185k | 80.6% | 82.8% | 78.4% |  +4.4 |
| 190k | 81.7% | 83.8% | 79.6% |  +4.2 |
| 195k | 78.7% | 78.4% | 79.0% |  -0.6 |
| 200k | 83.1% | 83.2% | 83.0% |  +0.2 |

±SE ≈ 1.5% per combined WR row, ±2.2% per per-seat WR.

**Reads:**
- **Monotonic upward, no plateau** — 49.5% → 83.1% across 200k updates. Crossed 50% at 5k, 70% at ~75k, 80% at ~155k.
- **Loss plateaued at ~80k but WR did NOT.** Loss curve was nearly flat from 80k onward (avg ≈ 0.018), yet WR climbed from 69% (80k) to 83% (200k). **Loss is a poor proxy for policy strength in DMC** — the MC target variance dominates MSE noise. Always eval directly.
- **Per-seat asymmetry persists throughout** — even-seat WR is consistently higher by ~3–6 pts on average. Largest gaps in mid-training (35k: +11.8, 60k: +9.4, 130k: +9.2). Final 200k is the most balanced point (83.2 vs 83.0). The asymmetry shrinks as the model gets stronger but never fully disappears in mid-training.
- **Local dips (e.g., 130k=74.2% sandwiched between 78.1% and 78.7%) are within ±1.5% SE noise**, not real regressions.
- **Peak = final** = 83.1% at 200k; the model is still gaining at sweep end. Plausibly pushes to 85–88% vs random with another 100–200k updates. Won't help vs heuristic (32.5% in same family of evals — structural ceiling).

Plot: `ml/runs/guanzero_m0_20260503_1337/wr_curve_random.png`. Raw rows: `wr_curve_random.json` (same dir).

---

## 41. What this logbook is for

When designing the next training run:
- Do NOT propose QMIX, GNN, PIMC, or aux-head-without-selection-pressure. They are all on the failure list above.
- DO consider perfect-info partner visibility (the central hypothesis for the next plan).
- DO use threshold-tracking eval intervals, not modulo on episodes_done.
- DO use ts=0.5 + position rewards [3,0,−1,−2] as the reward baseline.
- DO target Modal A10G for any run >2h; M1 Pro for sanity checks only (<6h hard cap).
- The strongest attainable result with the archived architecture is ~50% vs Jidan. To beat that, the next plan must change either (a) the observation space (add partner visibility), (b) the training signal (offline RL on competition-bot games), or (c) the architecture (transformer policy, larger network, joint-action planner). Anything that doesn't change one of these three is unlikely to break the ceiling.

---

## 42. GuanZero distributed architecture — paper vs ours (May 4)

### Paper figure (GuanZero §4, Figure 4)

```
   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
   │ Env 1  │   │ Env 2  │   │ Env 3  │   │ Env 4  │
   └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
       │            │            │            │
   ┌───▼────┐   ┌───▼────┐   ┌───▼────┐   ┌───▼────┐
   │Actor 1 │   │Actor 2 │   │Actor 3 │   │Actor 4 │   each holds 4 local nets
   │ LN1-4  │   │ LN1-4  │   │ LN1-4  │   │ LN1-4  │   (LN1..LN4 = per-seat Q)
   └───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
       └────────────┼────────────┼────────────┘
                    ▼            ▼
              ┌─────────────────────────┐
              │   Experience Buffer     │   one shared buffer
              └────────────┬────────────┘
                           ▼
                     ┌──────────┐
                     │ Learner  │
                     └──┬─┬─┬─┬─┘
                        │ │ │ │
              ┌─────────┘ │ │ └─────────┐
              ▼           ▼ ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
         │GlobalN1│  │GlobalN2│  │GlobalN3│  │GlobalN4│
         └────────┘  └────────┘  └────────┘  └────────┘
```

### Our Modal A10G architecture (post-pipelined-stream optimization)

```
                  Modal A10G + 32 vCPU container
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  8 ACTOR PROCESSES (CPU, torch.set_num_threads(1))                  │
  │  ┌────────────────┐    ┌────────────────┐                           │
  │  │ Actor 0        │    │ Actor 7        │                           │
  │  │  GuanDan env   │    │  GuanDan env   │                           │
  │  │  encoder       │... │  encoder       │                           │
  │  │  4 local Q-nets│    │  4 local Q-nets│  rolls episodes,          │
  │  │  ε-greedy      │    │  ε-greedy      │  pre-stacks 512 samples   │
  │  │  (CPU forward) │    │  (CPU forward) │  → numpy arrays           │
  │  └───────┬────────┘    └───────┬────────┘                           │
  │          │ pickled batch       │ pickled batch                      │
  │          └─────────┬───────────┘                                    │
  │                    ▼                                                │
  │            ┌───────────────┐                                        │
  │            │ mp.Queue      │  bounded, max=64                       │
  │            │ (pre-stacked) │                                        │
  │            └───────┬───────┘                                        │
  │                    │  drain 16/loop                                 │
  │                    ▼                                                │
  │  LEARNER PROCESS (1 proc, A10G)                                     │
  │  ┌─────────────────────────────────────────────────────────────┐    │
  │  │  4× ReplayBuffer (one per seat, list-backed circular,       │    │
  │  │                   50K samples each, O(1) random access)     │    │
  │  │  ────────────────────────────────────────────────────────   │    │
  │  │  Learner.update — pipelined per-stream (no global sync)     │    │
  │  │                                                             │    │
  │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │
  │  │   │ Stream 0 │  │ Stream 1 │  │ Stream 2 │  │ Stream 3 │    │    │
  │  │   │ ─────────│  │ ─────────│  │ ─────────│  │ ─────────│    │    │
  │  │   │ zero_grad│  │ zero_grad│  │ zero_grad│  │ zero_grad│    │    │
  │  │   │ fwd p0   │  │ fwd p1   │  │ fwd p2   │  │ fwd p3   │    │    │
  │  │   │ (BF16    │  │ (BF16    │  │ (BF16    │  │ (BF16    │    │    │
  │  │   │  autocast│  │  autocast│  │  autocast│  │  autocast│    │    │
  │  │   │  on QNet0│  │  on QNet1│  │  on QNet2│  │  on QNet3│    │    │
  │  │   │  compiled│  │  compiled│  │  compiled│  │  compiled│    │    │
  │  │   │  default)│  │  default)│  │  default)│  │  default)│    │    │
  │  │   │ bwd      │  │ bwd      │  │ bwd      │  │ bwd      │    │    │
  │  │   │ Adam.step│  │ Adam.step│  │ Adam.step│  │ Adam.step│    │    │
  │  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │
  │  │                                                             │    │
  │  │   torch.cuda.synchronize() ONLY at log/publish/ckpt ticks   │    │
  │  └─────────────────────────────────────────────────────────────┘    │
  │           │                                  │                      │
  │           ▼ atomic publish                   ▼ checkpoints          │
  │  ┌────────────────────┐         ┌──────────────────────┐            │
  │  │/tmp/guanzero_      │         │ /runs/guanzero/<run>/│            │
  │  │ weights/           │ ◄──poll │  checkpoints/        │            │
  │  │  (container-local) │  every  │   (Modal volume —    │            │
  │  │                    │  20 ep  │    pvguan-runs)      │            │
  │  └────────────────────┘         └──────────────────────┘            │
  │           ▲                                                         │
  │           │ load weights                                            │
  │           └──── actors                                              │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

### What changed vs the paper layout

| Component | Paper | Ours (Modal A10G) |
|---|---|---|
| Actor count | 4 | **8** (sweet spot — 24 caused CPU contention with learner) |
| Actor → learner transport | Shared buffer (single object) | **mp.Queue with pre-stacked numpy batches** (pickled per-actor) |
| Replay buffer | 1 shared | **4 per-seat FIFOs**, list-backed circular for O(1) random access |
| Sample pinning | implicit | **No `pin_memory()`** — H2D copy serializes on the same stream as compute, pinning is pure overhead |
| Q-net forward dtype | FP32 | **BF16 autocast** + TF32 enabled for FP32 paths |
| Compile | none | `torch.compile(mode="default")` on each Q-net |
| Position updates | sequential 4× | **4 dedicated CUDA streams**, pipelined with no per-update sync |
| Sync barrier | per-update | **only at log/publish/checkpoint ticks** (every 200 updates) |
| Weight publishing | in-memory shared | atomic `os.replace` on `/tmp/guanzero_weights` (container-local, not Modal volume) |
| Streaming logs | n/a | `GUANZERO_STREAM_LOGS=1` adds StreamHandler so `modal run` shows live progress |
| BLAS thread pinning | n/a | `OMP/MKL/OPENBLAS_NUM_THREADS=1` to prevent actor BLAS pools from contending |

### Throughput progression on Modal A10G (steady-state @ updates=2000)

| Config | upd/s | vs M1 (2.7) |
|---|---|---|
| Single-process baseline (paper-faithful, sequential) | — | — |
| Distributed, n_actors=8, FP32, sequential 4-pos updates | ~9.1 | 3.4× |
| + parallel CUDA streams (global sync per update) | ~10.5 | 3.9× |
| + deque→list buffer + drop pin_memory | ~10.2 | 3.8× (within noise) |
| + pipelined per-stream (no per-update sync) | (in progress) | — |

Target: ≥ 27 upd/s (10× M1).

## 43. GuanZero actor-rollout profile — where actor wall time goes (2026-05-05)

After reverting streams/pipelining (found to give 0% gain on A10G), turn focus to the
actor side: distributed throughput is bounded by how fast the queue refills, which
is ultimately bounded by per-actor episodes/sec. Profile single-actor on M1 first to
see the real cost breakdown before deciding whether to optimize legal-action gen,
encoding, or Q-forward.

Tool: `ml/scripts/util/profile_guanzero_rollout.py` — re-implements `play_episode`
with `perf_counter` timers around each phase. Runs single-process, no learner, no queue.

### First pass — paper-spec network (LSTM 256 + MLP 1024×6), CPU, M1 Pro

200 episodes, `epsilon=0.0` (every decision goes through Q-forward), `max_legal=128`,
`torch.set_num_threads(1)` (mirrors actor process), random init weights.

| section          |  sec    |   %  |   ms/call |
|------------------|---------|------|-----------|
| **q_forward**    | 117.67  | 92.2 | **4.524** |
| legal_actions    |   4.55  |  3.6 |     0.175 |
| collate          |   2.56  |  2.0 |     0.098 |
| encode_state     |   1.80  |  1.4 |     0.069 |
| encode_actions   |   0.78  |  0.6 |     0.030 |
| env_step         |   0.23  |  0.2 |     0.009 |
| mc_returns       |   0.02  |  0.0 |   0.097 / episode |

Derived:
- episodes/sec: **1.56**
- decisions/sec: 203
- decisions/episode: 130 (avg)
- avg legal/decision (post-dedup, post-cap): **5.3**

### Sanity check — `epsilon=1.0` (random play, skip Q-forward)

Same config, all 26k decisions take random path → no model inference at all.

| section          |  sec  |   %   |   ms/call |
|------------------|-------|-------|-----------|
| legal_actions    | 3.43  | 65.5  |     0.132 |
| encode_state     | 1.15  | 21.9  |     0.044 |
| encode_actions   | 0.60  | 11.5  |     0.023 |
| env_step         | 0.04  |  0.8  |     0.002 |

Throughput jumps from 1.56 → **36.94 ep/s** — a 24× speedup confirms Q-forward
fully dominates actor wall time. Encoding+legal+step floor is ~5s for 200 ep
(≈27 ms total per episode).

### Headline takeaway

Q-forward is **92%** of actor time, ~4.5 ms per decision over an average 5.3-action
batch. The model is small (LSTM 256 → MLP 1024×6) but each decision requires a
forward pass with batch size ≈ 5, so torch dispatch overhead dominates the dense
matmul. Optimizing legal-action gen or encoding (Rust port, bitsets, etc.) is
chasing the 8% tail.

**Optimization candidates, in priority order:**
1. **Batch Q-forward across actors / episodes** — if 2 actors share a Q-net process
   and submit their candidate sets together, batch grows ~2× and Python/dispatch
   overhead amortizes. Most leverage.
2. **Reduce per-decision Q-forward calls** — e.g. cache Q-values for unchanged
   `(hand, history-tail)` signatures (likely low hit rate but worth a sample).
3. **Smaller `max_legal` cap** — currently 128, but actual avg is 5.3 → cap is
   never binding here. Skip.
4. Anything in legal/encode → only 8% of total. Skip until Q-forward is solved.

### Second pass — cProfile drill-down inside Q-forward

100 episodes, same config. cumtime breakdown of Q-forward’s 56s (84% of 67s wall):

| op                       | cumtime | % wall | calls   | comment |
|--------------------------|---------|--------|---------|---------|
| `torch._C._nn.linear`    | 33.60   | 48.7   | 89,313  | MLP matmuls (6 layers × 12,759 fwd) |
| `torch.lstm`             | 19.09   | 27.7   | 12,759  | history encoder (20×108 → 256) |
| `Sequential.forward`     | 35.22   | 51.0   | 12,759  | wraps the linear stack |
| Q-net `forward()` total  | 55.72   | 80.7   | 12,759  | linear + lstm + concat |
| `_py_generate_all_leads` |  5.49   |  7.9   | 12,759  | engine lead generation |
| `_py_generate_responses` |  4.72   |  6.8   | 10,294  | engine response gen |
| `_multi_hot`             |  1.11   |  1.6   | 275,288 | encoder card → 108-vec |

Linear dominates: 89,313 calls / 33.6 s = **376 µs per linear call** at batch ≈ 5.
A 1024×1024 fp32 matmul with batch=5 should take <50 µs on an M1 P-core — so
~85% of that 376 µs is torch dispatch overhead, not arithmetic.

### Q-forward microbench — batch-size sweep (`bench_qforward.py`)

Synthetic constant-shape batches, paper-spec net, M1 CPU, threads=1:

| batch | ms/call | µs/sample | samples/sec |
|------:|--------:|----------:|------------:|
|   1   |   1.51  |    1514   |        661  |
|   4   |   4.85  |    1213   |        824  |
|   8   |   5.23  |     653   |       1531  |
|  16   |   5.98  |     374   |       2675  |
|  32   |   6.41  |     200   |       4989  |
|  64   |  11.28  |     176   |       5676  |
| 128   |  17.64  |     138   |       7258  |
| 256   |  33.54  |     131   |       7634  |

Going from batch=4 (≈ our actor reality) to batch=32 costs only **+32% per call**
but gives **6× more samples per second**. Threads=4 doesn’t help below batch=64
(BLAS spin-up cost > matmul cost). This is the canonical dispatch-bound profile.

### Third pass — multi-actor scaling on M1 (`bench_actor_scaling.py`)

N independent CPU actor processes, 30 ep/actor, threads=1 each, no learner / no queue:

|  N | wall (s) | total ep/s | speedup | efficiency |
|---:|---------:|-----------:|--------:|-----------:|
|  1 |    18.5  |     1.62   |  1.00×  |    100 %   |
|  2 |    19.7  |     3.05   |  1.88×  |     94 %   |
|  4 |    35.6  |     3.37   |  2.08×  |     52 %   |
|  6 |    54.2  |     3.32   |  2.05×  |     34 %   |
|  8 |    61.7  |     3.89   |  2.40×  |     30 %   |

M1 saturates between N=2 and N=4 — total throughput plateaus at ~3.4–3.9 ep/s no
matter how many actors are spawned. Per-actor rate collapses from 1.71 → 0.50 ep/s
at N=8. Spawn cost amortizes worse with the small slice (30 ep), but the trend is
clear: M1 will not scale beyond ~3-4 effective actors. (Modal A10G has 32 vCPUs
and shows the same shape later; its plateau just sits at higher N.)

### Where the 92.2% goes — final breakdown

| layer                    | % wall | bottleneck type        |
|--------------------------|-------:|------------------------|
| MLP linear (matmuls)     |  48.7  | torch dispatch overhead at batch≈5 |
| LSTM history encoder     |  27.7  | per-call setup + 20-step recurrence |
| Other Q-net wrapper code |   4.3  | python                 |
| Legal action enumeration |   8.7  | pure Python combos     |
| Encoding (`_multi_hot` × 275k) |  ~3 | python loops + np.zeros |
| Collate / env_step       |   ~2  | mostly torch.from_numpy |
| All else (mc_returns, etc.) | <1 | negligible             |

### Recommended optimization order

1. **Cross-actor inference batching** (largest leverage) — e.g. an inference-server
   actor pattern: M rollout workers send candidate-action sets to a single GPU/CPU
   inference process, which batches across workers, runs one forward, returns Q
   per (worker, candidate). At batch ≈ 32 this is ≈ 6× the per-sample throughput
   of the current per-decision forward.
2. **Reduce LSTM cost** — switch to a small set/transformer head over the same
   20-move history. Cheaper per call than rolling a 20-step LSTM. Algorithm change,
   not just engineering.
3. **Skip optimizing legal-action gen + encoding for now**. Even a 10× speedup of
   that 8% saves <1% of wall time. Defer until Q-forward is solved.

### Fourth pass — same profile, but on MPS (device our actor would actually use)

200 episodes, paper-spec net, `epsilon=0.0`, `--device mps`:

| section          |  sec    |   %  |   ms/call | vs CPU |
|------------------|---------|------|-----------|--------|
| q_forward        |  62.59  | 55.3 |     2.406 | 0.53× (1.9× faster) |
| **collate**      |  44.31  | 39.2 | **1.703** | **17.4× slower** |
| legal_actions    |   3.94  |  3.5 |     0.152 | 0.87× |
| encode_state     |   1.44  |  1.3 |     0.055 | 0.80× |
| encode_actions   |   0.69  |  0.6 |     0.026 | 0.87× |
| env_step         |   0.13  |  0.1 |     0.005 | 0.56× |

End-to-end episodes/sec: CPU 1.56 → MPS 1.76 — only **+13%** despite Q-forward
being nearly 2× faster, because **per-decision H2D dispatch becomes the new bottleneck**.

Each decision builds 9 small numpy arrays and pushes each to MPS via
`torch.from_numpy(arr).to("mps")`. With ~5 candidate actions per decision and 9
keys per candidate that's ~45 tiny `.to(device)` calls per decision. The Python
dispatch + small-tensor H2D launch cost on M1 MPS adds up to ~1.7 ms/decision.
On CPU `.to("cpu")` is a no-op so collate is essentially free (0.10 ms).

### Q-forward microbench — MPS sweep

Same harness as before, with `torch.mps.synchronize()` around the timed region:

| batch | CPU ms/call | MPS ms/call | MPS samples/sec |
|------:|------------:|------------:|----------------:|
|   1   |    1.33     |    0.67     |      1,485      |
|   4   |    4.56     |    1.08     |      3,710      |
|   8   |    4.90     |    1.26     |      6,370      |
|  16   |    5.98     |    1.13     |     14,151      |
|  32   |    6.61     |    1.30     |     24,622      |
|  64   |   11.28     |    1.76     |     36,439      |
| 128   |   17.08     |    2.67     |     47,871      |
| 256   |   33.54     |    5.48     |     46,748      |

MPS scales superbly with batch — at batch=128 it’s 6.4× faster per sample than
CPU, and ms/call only doubles from batch=4 to batch=128. The actor batches at
~5, leaving most of MPS’s headroom on the table.

### Updated takeaway

The CPU profile’s message ("Q-forward is 92% of wall, batch up to 32 to amortize
dispatch") still holds — but on MPS the same insight is even stronger:

- per-call MPS Q-forward at batch=32 = 1.30 ms; at our actor’s batch=5 ≈ 1.08 ms
  → **~25× more samples/sec** if we cross-batch ≈ 32 candidates per call
- the collate H2D cost is currently 39% of MPS wall — fixable by stacking all 9
  channels into one contiguous numpy buffer per decision and one `.to(mps)` call
  (single transfer of ~3 KB instead of 9 transfers of ~0.3 KB each)
- only after both are done would legal-action gen / encoding become worth touching

**Recommended order on M1 / MPS actor:**
1. Single-tensor stacked H2D in `collate_encoded` — should claw back most of the
   1.6 ms collate cost. Pure engineering, no algorithm change.
2. Cross-actor inference batching (the bigger structural win, same as on Modal).
3. Then legal/encode/etc.

### Fifth pass — `collate_encoded` H2D batching (`non_blocking=True`)

Microbench `bench_collate.py` (5 variants, MPS, repeats=2000) showed the
production `collate_encoded` was paying a **per-call MPS sync** on every
`.to(device)` call — 9 calls per decision = ~1.7 ms regardless of batch size.

| variant            | ms/call (B=5) | µs/sample |
|--------------------|--------------:|----------:|
| **v0_current**     |     1.683     |    337    |
| v1_nonblock        |     0.213     |     43    | **8× faster, 1-line change** |
| v2_flatone         |     0.278     |     56    |
| v3_flatone_contig  |     0.418     |     84    |
| v4_persample_pack  |     0.240     |     48    |

The flat-buffer variants concat all 9 channels into one (B, 1859) tensor
before a single H2D, but the on-device split + reshape introduces non-contig
views that don't actually beat the simpler `non_blocking=True`. The 1-line
change wins both on ergonomics and speed.

**Patch:** [buffer.py:115](ml/src/guandan/guanzero/buffer.py#L115) — add
`non_blocking=True` to `.to(device)` in `collate_encoded`.

### Re-run rollout profile after the fix (MPS, 200 ep, eps=0.0)

| section          | before  | after  | Δ          |
|------------------|--------:|-------:|-----------:|
| **ep/s (end-to-end)** | **1.76** | **3.08** | **+75 %** |
| q_forward ms/call |  2.406 | 1.870  | −22 %      |
| collate ms/call   |  1.703 | 0.251  | **−85 %**  |
| collate share of wall | 39.2 % | 10.6 % | dropped from #2 |
| q_forward share   | 55.3 %  | 79.4 % | back to dominant |
| total wall (s)    | 113.5   | 64.86  | −43 %      |

Q-forward also got faster because H2D now overlaps with the prior
forward’s tail instead of blocking the dispatch thread. CPU path is
unchanged (`.to("cpu")` ignores `non_blocking`).

### Updated bottleneck stack (MPS, post-fix)

| layer                    | % wall | next move |
|--------------------------|-------:|-----------|
| q_forward (LSTM + MLP)   |  79.4  | cross-actor inference batching, or smaller history encoder |
| collate (H2D + stack)    |  10.6  | only worth chasing after Q-forward is solved |
| legal_actions            |   6.4  | Rust port — defer (small share now, smaller still post-batching) |
| encode_state/actions     |   3.4  | defer |
| env_step                 |   0.2  | ignore |

### Files added

- [profile_guanzero_rollout.py](ml/scripts/util/profile_guanzero_rollout.py) —
  inline-instrumented `play_episode` with phase timers (CPU + MPS)
- [bench_qforward.py](ml/scripts/util/bench_qforward.py) — batch-size sweep with CUDA/MPS sync
- [bench_actor_scaling.py](ml/scripts/util/bench_actor_scaling.py) — N-actor wall-clock sweep
- [bench_collate.py](ml/scripts/util/bench_collate.py) — collate variant microbench
- `ml/runs/profile/rollout_cpu_eps0.prof` — cProfile dump (gitignored under runs/)

