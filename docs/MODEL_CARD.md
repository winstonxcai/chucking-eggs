# DART Model Card

This model card describes the 1.25M-update DART checkpoint referenced by the
README. The checkpoint and release evaluation evidence are stored in
`ml/results/release_1_25m/`.

## Model

- Name: DART partner-visible Guan Dan RL agent
- Architecture: role-aware shared Q-network with four trick-position heads
- Training algorithm: Deep Monte Carlo control with distributed actors
- Training config: `ml/src/guandan/dart/configs/dart_l4.yaml`
- Intended checkpoint artifact: `ml/results/release_1_25m/update_01250000.pt`

## Intended Use

DART is intended for research and engineering exploration of partner-visible
Guan Dan agents. It is suitable for offline evaluation against the bundled
rule-based bots and for local inference through `DartBot.load()`.

It is not a human-deployable hidden-information policy. The reported checkpoint
uses partner-visible state and includes an `others_hand` channel that holds
the *union* of the two opponent hands — i.e., the deck-complement of own +
partner, not per-opponent oracle information. See
[TRADEOFFS.md](TRADEOFFS.md) §10 for the full discussion.

## Training Data and Procedure

The training data is generated online by self-play actors. The production run
uses 32 actors on Modal with an L4 learner, BF16 learner autocast, a replay
capacity of 400k role-aware samples, and fixed low exploration
(`epsilon_start == epsilon_final == 0.01`).

Full checkpoints contain model weights, optimizer state, replay state, learner
RNG state, and the latest actor RNG states drained into replay. Weight-only
periodic checkpoints are faster to write but are not exact-resume artifacts.

## Evaluation

The README reports 5000-game paired fixed-deck evaluations against rule-based
bots. The reported Glicko-2 table reuses the existing 5000-game rule-bot
round-robin and injects the 1.25M DART matchup results instead of rerunning the
entire matrix. Reproduce a single matchup with:

```bash
uv run guandan-eval-dart \
  --checkpoint ml/results/release_1_25m/update_01250000.pt \
  --opponent strategic \
  --games 1000 \
  --out results.json
```

## Limitations

- Single training seed; there are no cross-seed confidence intervals.
- Rule-bot-only evaluation; no human study and no comparison against another
  openly released learned partner-visible Guan Dan agent.
- Partner-visible observation with a deck-complement `others_hand` feature;
  not strict hidden information, but also not per-opponent oracle. See
  [TRADEOFFS.md](TRADEOFFS.md) §10.
- Vendored competition bots need original-author license confirmation before a
  broad public release.

## Release Checklist

Before publishing, verify `ml/results/release_1_25m/manifest.json` checksums and
attach the same bundle to a GitHub Release or Hugging Face model repo if you want
a release asset outside git. The raw learner metrics JSONL is intentionally not
committed because it is hundreds of megabytes; summarize or host it separately if
needed.
