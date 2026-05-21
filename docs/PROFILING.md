# Profiling

DART has two profiler toggles, one per runtime component:

- `DART_ACTOR_PROFILE=1`, `true`, `yes`, or `on` writes actor phase timings and legal-set K buckets.
- `DART_LEARNER_PROFILE=1` writes learner sample, forward, backward, and sync timings.

Typical local smoke:

```bash
DART_ACTOR_PROFILE=1 DART_LEARNER_PROFILE=1 \
  uv run python -m guandan.dart --config ml/src/guandan/dart/configs/dart_mps.yaml \
  --updates 100 --run-dir ml/runs/profile_smoke
```

Actor profile snapshots are written under the run directory as
`actor_<id>_profile.txt`. Learner profile rows are logged in `learner.log`.

For raw actor-throughput profiling without a learner or replay buffer:

```bash
uv run python ml/scripts/util/profile_actor_throughput.py \
  --workers 8 --episodes 32 --lanes 64
```

To run the same benchmark on Modal-sized CPU/GPU containers:

```bash
modal run ml/scripts/modal/profile_actor_throughput_modal.py \
  --workers 32 --episodes 64 --lanes 64
```
