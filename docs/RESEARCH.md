# DART Research Note

**DART** means **Dynamic Action-Relative Routing for Tricks**. This document
is the scientific description of the project; implementation details and
reproduction commands are collected in [Reproducibility](REPRODUCIBILITY.md)
and [Architecture](ARCHITECTURE.md).

## Abstract

This project studies cooperative reinforcement learning in Guan Dan, a
four-player, two-team trick-taking game with a 108-card double deck, changing
wild cards, large legal-action sets, and sparse team-level outcomes. DART uses
a shared Deep Monte Carlo Q-network whose output is routed by the acting
player's relative role in the current trick. The central hypothesis is that
role-relative sharing provides a compact representation of tactical position
while retaining enough specialization for leading, responding, and closing a
trick.

The primary empirical study is a four-cell comparison of canonical DART and
GuanZero agents under a common Modal L4 runtime, ten-hour wall-clock budget,
effective learner-batch semantics, seed, replay policy, and evaluation
protocol. The result is an equal-resource comparison of two algorithmic
formulations, not a parameter-matched study.

## Problem Setting

Guan Dan combines a large action space with partnership incentives. A move can
be valuable because it preserves a teammate's control, forces an opponent to
spend a resource, or changes the order in which partners finish. The tactical
meaning of a legal action therefore depends on the actor's position in the
current trick, not only on absolute seat identity.

The training objective follows the Deep Monte Carlo lineage of DouZero and
GuanZero. Actors generate complete games, terminal team rewards are assigned
to the decisions in those games, and a learner fits Q-values from replay. The
agent scores legal actions directly; there is no separate policy head.

## Method

### Action-relative routing

DART uses one shared network trunk and four small Q-heads. A deterministic
router selects the head corresponding to the actor's current trick role:
leading, first responder, across from the leader, or last responder. State and
candidate-action features are encoded relative to the current actor, so the
same network can be applied across absolute seats.

This design makes parameter sharing part of the algorithm. It is not a tuning
choice made only for the ablation. The expected benefit is lower duplicated
capacity and cheaper actor-side inference; the risk is that role-encoding
errors affect the shared policy rather than a single seat-specific module.

### Information setting

Partner visibility is an explicit experimental axis. In the visible condition,
the actor receives its partner's hand. In the hidden condition, the partner
hand channel is removed. Both conditions retain public game information and a
derived `others_hand` union representing the two opposing hands collectively.
That union is a deck-complement deduction; it does not identify which
opponent holds any particular card and is not per-opponent oracle information.

Partner visibility is therefore a research assumption about cooperative
coordination, not a claim that the released model is directly deployable under
strict hidden-information play.

## Controlled Ablation

The study is a 2×2 factorial design:

| Algorithm | Partner visibility |
|---|---|
| DART | visible / hidden |
| GuanZero | visible / hidden |

All cells use the same L4 hardware class, actor and queue settings, total
learner batch size, replay and optimizer settings, exploration schedule,
duration, seed, checkpoint policy, and evaluation opponents. GuanZero's 4,096
total learner samples are divided into four 1,024-sample seat batches per
update.

The architectural capacity is intentionally not equalized. DART contains
approximately 4.7M shared parameters. GuanZero contains four independent
approximately 7.1M seat networks, or approximately 28.4M parameters in total.
The comparison answers which canonical formulation is stronger under equal
hardware, time, and effective learner samples; it does not establish
parameter-matched superiority.

## Findings

The ten-hour, single-seed endpoint results are reported in the root
[README](../README.md) and the machine-readable
[ablation artifact](../ml/results/ablation_l4_10h/summary.json). DART finishes
ahead of GuanZero in both visibility conditions. Partner visibility itself
does not produce a positive endpoint effect in this run, so the result should
not be read as evidence that hidden information is generally better.

The historical two-week GuanZero training run is not a controlled comparison
to this ten-hour budget. It is useful context for compute-to-convergence
questions, but it cannot be substituted for the equal-budget result.

## Intended Use and Release Status

The released checkpoint is intended for research and engineering exploration of
partner-visible Guan Dan agents, offline evaluation against the bundled rule
bots, and local inference through the DART runtime. It is not presented as a
human-deployable strict hidden-information policy. The partner-visible model
also receives `others_hand`, which is the union of the two opposing hands
inferred by deck subtraction; it is not a per-opponent ownership oracle.

Training data is generated online by self-play actors. The production release
uses 32 actors, an L4 learner with BF16 autocast, a 400,000-sample replay
capacity, and fixed exploration at 0.01. The selected release checkpoint is
[`update_01250000.pt`](../ml/results/release_1_25m/update_01250000.pt), with
its configuration, evaluation outputs, and SHA-256 manifest in the same
directory. Full checkpoints contain optimizer, replay, and RNG state for exact
process-level resume; weight-only checkpoints are intended for evaluation and
inspection.

The release evidence is complete for the stated single-seed, rule-bot protocol.
Vendored competition bots remain separately attributed in [`NOTICE`](../NOTICE)
and may have licensing conditions distinct from the repository code.

## Limitations and Open Questions

- The study uses one training seed. Evaluation confidence intervals describe
  game-sampling uncertainty, not run-to-run training variance.
- Parameter sharing creates a substantial capacity asymmetry. A separate
  parameter-matched control would answer a different question.
- The ten-hour budget measures a compute-constrained regime, not convergence.
- Evaluation runs occur within the measured wall-clock lifecycle, although
  the policy evaluation protocol is identical across cells.
- The partner-visible setting is not strict hidden-information deployment.
- Evaluation is against bundled rule-based opponents; learned-opponent and
  human studies remain open.

The highest-value follow-ups are multi-seed replication, longer matched-budget
runs, a parameter-matched control, and an ablation of the derived
`others_hand` feature.

## Supporting Evidence

- [Ablation artifacts](../ml/results/ablation_l4_10h/): evaluation JSONs,
  aligned metric summaries, plots, and provenance manifest.
- [Architecture](ARCHITECTURE.md): process model, state channels, replay, and
  checkpoint invariants.
- [Evaluation](EVALUATION.md): paired-deck protocol and confidence intervals.
- [Reproducibility](REPRODUCIBILITY.md): environment, configurations, and
  artifact provenance.
