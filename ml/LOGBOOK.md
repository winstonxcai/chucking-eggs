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

## 33. What this logbook is for

When designing the next training run:
- Do NOT propose QMIX, GNN, PIMC, or aux-head-without-selection-pressure. They are all on the failure list above.
- DO consider perfect-info partner visibility (the central hypothesis for the next plan).
- DO use threshold-tracking eval intervals, not modulo on episodes_done.
- DO use ts=0.5 + position rewards [3,0,−1,−2] as the reward baseline.
- DO target Modal A10G for any run >2h; M1 Pro for sanity checks only (<6h hard cap).
- The strongest attainable result with the archived architecture is ~50% vs Jidan. To beat that, the next plan must change either (a) the observation space (add partner visibility), (b) the training signal (offline RL on competition-bot games), or (c) the architecture (transformer policy, larger network, joint-action planner). Anything that doesn't change one of these three is unlikely to break the ceiling.
