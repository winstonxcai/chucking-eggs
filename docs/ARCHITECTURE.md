# Architecture

**DART** is **Dynamic Action-Relative Routing for Tricks**. The production
training path uses a role-aware encoder and batched local actor inference.
Centralized GPU inference-server variants were tested, but local actor-side
batching was faster for this model because it amortizes Q-forward work without
IPC.

Key entry points:

- `ml/src/guandan/dart/runtime/train.py`: process orchestration, signal handling, checkpoint eval watcher.
- `ml/src/guandan/dart/runtime/learner.py`: learner loop, replay draining, checkpointing, weight publishing.
- `ml/src/guandan/dart/runtime/worker.py`: persistent actor loop and sample queue producer.
- `ml/src/guandan/dart/runtime/actor/rollout.py`: batched self-play rollouts.
- `ml/src/guandan/dart/model/q_network.py`: GuanZero and DART Q-net definitions.

## Process Model

The main process spawns the learner first, waits for initial weights, then
starts actor processes. It owns lifecycle handling, signal handling, progress
display, and checkpoint-eval requests.

The learner process owns the authoritative Q-net, optimizer, replay buffer,
checkpoint state, and weight publishing. Its loop drains actor sample batches,
throttles when replay usage outruns fresh samples, takes gradient steps when
the buffer is warm, and periodically logs metrics, publishes weights, and
checkpoints.

Actor processes each hold an eval-mode local copy of the Q-net. They run
batched self-play lanes, score the full deduplicated legal-action set, append
Monte Carlo samples to a local accumulator, and push pre-stacked batches to the
learner queue.

## Communication

Actors send samples to the learner through a bounded `multiprocessing.Queue`.
Messages are pre-stacked numpy arrays rather than per-sample dictionaries, which
keeps queue serialization overhead low. If the queue remains full for the actor
put timeout, training stops rather than dropping samples silently.

The learner publishes weights through the filesystem. It writes
`weights_<version>.tmp`, atomically renames it to `weights_<version>.pt`, then
atomically updates `latest.txt`. Actors poll that metadata and load newer
weights when their configured update lag threshold is reached.

Shutdown uses a shared `mp.Event`. Checkpoint evaluation uses a learner-to-main
request queue plus a done event; both sides enforce `eval.max_wait_s` to avoid
deadlock.

## Training Targets

Game terminal rewards are team-level level changes: `+3/+2/+1` for the winning
team and the negated value for the losing team. DART normalizes those values by
3 for training. For each player turn, the Monte Carlo target is:

```text
G_t = gamma^(remaining_turns_for_that_player) * normalized_terminal_reward
```

With the production `gamma: 1.0`, every decision by a player in a completed
game receives that player's normalized terminal reward.

## Encoded State

The role-aware encoder emits structured numpy channels for own hand, partner
hand, opponent cards, public player blocks, global features, behavior features,
recent history, candidate action, and `trick_head_id`. The Q-network routes
each candidate through one of four trick-position heads: leading, first
responder, across, and last responder.

Card channels are 108-dimensional because Guan Dan uses two full decks plus
four jokers. The structured dict remains intact until collate time so channels
can be swapped or ablated independently.

## Replay

Actors generate complete-game samples; the learner samples uniformly from the
replay buffer for gradient updates. Episodes and updates are intentionally
decoupled. A replay sample remains valid after policy updates because the target
is the realized game outcome, not a bootstrapped estimate from the current net.

The L4 config uses `buffer_capacity: 400000`, approximately 100k samples per
seat. Periodic checkpoints may be weight-only for storage efficiency; clean
shutdown writes `final.pt` as a full resume checkpoint including optimizer,
replay, learner RNG, and actor RNG state.

## Operational invariants

- The bounded actor queue never silently drops samples; a persistent full
  queue stops the run.
- The learner owns the authoritative model, optimizer, replay, and checkpoint
  state.
- `batch_size` is interpreted through the persisted batch-size semantics. The
  ablation protocol compares effective learner samples, not raw update count.
- A duration-only run exits through the same clean-shutdown path as an
  update-limited run and writes a full `final.pt`.
- Checkpoint evaluation uses a bounded handoff with a timeout, so evaluation
  cannot deadlock training indefinitely.

## Supporting documents

- [Research note](RESEARCH.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Evaluation](EVALUATION.md)
