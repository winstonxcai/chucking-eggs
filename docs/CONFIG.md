# DART Configuration

The production config is `ml/src/guandan/dart/configs/dart_l4.yaml`; local MPS and lane-count smoke configs live in the same directory.

## Core Training

- `model_type`: `Dart` is the production role-aware shared Q-network. `GuanZero` keeps the older per-seat comparison path.
- `seed`: base seed. The learner uses `seed + 1`; actor `i` starts from `seed + i * 10000` and restores actor RNG state from full checkpoints when available.
- `batch_size`: learner samples per update. The L4 config uses `4096` to keep BF16 updates stable.
- `lr`: Adam learning rate. The L4 config uses `3e-5`, tuned with `batch_size=4096`.
- `gamma`: Monte Carlo return discount. Production uses `1.0`.

## Actor/Learner Runtime

- `n_actors`: number of CPU actor processes.
- `actor_batch_lanes`: simultaneous games inside each actor. Higher values improve CPU inference batching but increase latency.
- `actor_push_batch_size`: samples per queue message.
- `sample_queue_maxsize`: bounded queue capacity. If this fills for 5 seconds, actors stop the run to avoid silent sample loss.
- `sync_interval_updates` and `sync_jitter_updates`: actor weight refresh cadence.
- `publish_interval_updates`: learner weight publication cadence.
- `checkpoint_every_updates`: periodic checkpoint interval.
- `checkpoint_save_type`: `weight` writes only model/config; `full` also writes optimizer, replay, learner RNG, and actor RNG states.

## Replay

- `buffer_capacity`: total role-aware replay capacity for `Dart`.
- `buffer_capacity_per_player`: legacy per-seat capacity for `GuanZero`.
- `buffer_min_size`: minimum warmup before learner updates.
- `target_replay_ratio`, `max_replay_ratio`, `max_throttle_sleep_s`: keep gradient updates from outrunning fresh actor samples.
- `replay_mix`: optional weighted bucket sampling for role-aware replay.
- `max_forced_k1_replay_frac`: cap or remove forced-move samples where only one legal action exists.

## Evaluation

- `eval.enabled`: request checkpoint evaluation during training.
- `eval.opponents`: `all` or a list of supported rule-bot names.
- `eval.n_eval_games_per_opponent`: paired fixed-deck games per opponent; must be even.
- `eval.every_updates`: `0` means evaluate at `checkpoint_every_updates`.
- `eval.workers` and `eval.lanes`: `0` means inherit training actor counts/lanes.
- `eval.max_wait_s`: total timeout for eval handoff and subprocess execution.

## Smoke Configs

- `dart_l4_lanes1_smoke.yaml`: actor-limited baseline.
- `dart_l4_lanes64_smoke.yaml`: production-like lane batching.
- `dart_l4_lanes128_smoke.yaml`: high-lane throughput probe.
