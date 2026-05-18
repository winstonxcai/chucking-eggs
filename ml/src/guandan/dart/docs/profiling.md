# Dart Profiling

Dart has three profiler toggles, one per runtime component:

- `DART_ACTOR_PROFILE=1` writes actor phase timings and legal-set K buckets.
- `DART_SERVER_PROFILE=1` writes inference-server batching and forward timings.
- `DART_LEARNER_PROFILE=1` writes learner sample, forward, backward, and sync timings.

Typical local smoke:

```bash
DART_ACTOR_PROFILE=1 DART_LEARNER_PROFILE=1 \
  uv run python -m guandan.dart --config <config.yaml>
```

When running the inference-server systems ablation, add:

```bash
DART_SERVER_PROFILE=1
```

Actor profile snapshots are written under the run directory as
`actor_<id>_profile.txt`. Learner and server profile rows are logged in
`learner.log` and `inference_server.log`.
