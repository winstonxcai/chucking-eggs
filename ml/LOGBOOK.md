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

## 24. What this logbook is for

When designing the next training run:
- Do NOT propose QMIX, GNN, PIMC, or aux-head-without-selection-pressure. They are all on the failure list above.
- DO consider perfect-info partner visibility (the central hypothesis for the next plan).
- DO use threshold-tracking eval intervals, not modulo on episodes_done.
- DO use ts=0.5 + position rewards [3,0,−1,−2] as the reward baseline.
- DO target Modal A10G for any run >2h; M1 Pro for sanity checks only (<6h hard cap).
- The strongest attainable result with the archived architecture is ~50% vs Jidan. To beat that, the next plan must change either (a) the observation space (add partner visibility), (b) the training signal (offline RL on competition-bot games), or (c) the architecture (transformer policy, larger network, joint-action planner). Anything that doesn't change one of these three is unlikely to break the ceiling.
