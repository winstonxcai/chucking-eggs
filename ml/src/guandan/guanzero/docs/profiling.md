# GuanZero Profiling

GuanZero has three profiler toggles, one per runtime component:

- `GUANZERO_ACTOR_PROFILE=1` writes actor phase timings and legal-set K buckets.
- `GUANZERO_SERVER_PROFILE=1` writes inference-server batching and forward timings.
- `GUANZERO_LEARNER_PROFILE=1` writes learner sample, forward, backward, and sync timings.

Typical local smoke:

```bash
GUANZERO_ACTOR_PROFILE=1 GUANZERO_LEARNER_PROFILE=1 \
  uv run python -m guandan.guanzero.train --quick
```

When the inference server is enabled, add:

```bash
GUANZERO_SERVER_PROFILE=1
```

Actor profile snapshots are written under the run directory as
`actor_<id>_profile.txt`. Learner and server profile rows are logged in
`learner.log` and `inference_server.log`.
