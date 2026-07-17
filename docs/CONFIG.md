# DART Configuration

The production config is `ml/src/guandan/dart/configs/dart_l4.yaml`; local MPS and lane-count smoke configs live in the same directory.

YAML configs may set `base_config` to a relative YAML path. The base is loaded
recursively, nested mappings are merged, and child values take precedence.
The resolved configuration is what gets serialized into run metadata and
checkpoints.

## Core Training

- `model_type`: `Dart` is the production role-aware shared Q-network. `GuanZero` keeps the older per-seat comparison path.
- `config_schema_version`: current schema version is `1`. Older configs that omit it load as version 1.
- `seed`: base seed. The learner uses `seed + 1`; actor `i` starts from `seed + i * 10000` and restores actor RNG state from full checkpoints when available.
- `coordination_bucket_card_threshold` and `coordination_bucket_final_fraction`: actor-side boundaries for tagging hard-bot coordination-endgame samples.
- `batch_size`: learner samples per update under the configured semantics. The L4 config uses `4096` to keep BF16 updates stable.
- `batch_size_semantics`: `total` means `batch_size` is the total samples consumed by one optimizer update; `per_seat` preserves legacy GuanZero behavior and samples that many examples for each of four seats. DART always treats `batch_size` as total.
- `max_train_seconds`: duration limit for a run; `0` disables duration stopping.
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
- `max_forced_pass_replay_frac`: cap or remove forced-move samples where only one legal action exists.

Compatibility note: old configs using `max_forced_k1_replay_frac` are migrated
to `max_forced_pass_replay_frac` on load. Old `opponents.latest_team_odd_probability`
keys are migrated to `opponents.latest_learner_team_odd_probability`. The
unused `updates_per_learner_step` key is accepted and ignored for older YAML
files.

## Evaluation

- `eval.enabled`: request checkpoint evaluation during training.
- `eval.opponents`: `all` or a list of supported rule-bot names.
- `eval.n_eval_games_per_opponent`: paired fixed-deck games per opponent; must be even.
- `eval.every_updates`: `0` means evaluate at `checkpoint_every_updates`.
- `eval.workers` and `eval.lanes`: `0` means inherit training actor counts/lanes.
- `eval.max_wait_s`: total timeout for eval handoff and subprocess execution.
- `--checkpoint-every-updates`: optional CLI override for the periodic checkpoint interval; useful for short smoke runs without changing the YAML.

Learner metrics are written to `metrics_learner.jsonl`. Each row includes
`wall_clock_s` (elapsed wall time from the training start, including checkpoint
evaluation pauses), `learner_samples_total`, and `samples_per_sec`; these fields
can be joined to checkpoint evaluation JSON by update count for wall-clock and
matched-sample learning curves.

## Smoke and Throughput Configs

- `ablation_l4_common.yaml`: shared 10-hour L4 recipe for the four-cell DART/GuanZero comparison.
- `ablation_l4_*_10h.yaml`: minimal model/partner-visibility overrides of the shared L4 recipe.

The four-cell recipe is runtime- and sample-matched, not parameter-count-matched:
DART uses one shared network, while GuanZero maintains one network per seat.
That algorithmic difference is intentional and should be reported alongside both
wall-clock and matched-learner-sample results.

- `dart_cpu_smoke.yaml`: tiny CPU-only actor/learner/checkpoint smoke for CI and local setup checks.
- `dart_mps.yaml`: Apple Silicon local training/smoke config.
- `dart_l4_lanes1_throughput.yaml`: CUDA actor-limited throughput baseline.
- `dart_l4_lanes64_throughput.yaml`: CUDA production-like lane batching throughput probe.
- `dart_l4_lanes128_throughput.yaml`: CUDA high-lane throughput probe.
