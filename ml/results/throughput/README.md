# Actor–Learner Throughput Evidence

This directory records the published throughput scope for the DART actor–learner
runtime. The measurements are post-warmup means from 1,000-update Modal probes
using 32 actors and a 4,096-sample learner batch. The reported rate is accepted
fresh actor samples per second, measured at the learner boundary.

| Hardware | Lanes per actor | Accepted samples/s | Status |
|---|---:|---:|---|
| NVIDIA L4 | 1 | 4,640 | measured |
| NVIDIA L4 | 64 | — | canonical production-like configuration; retained for reproduction |
| NVIDIA L4 | 128 | 19,765 | measured |
| NVIDIA A10G | 1 | 4,889 | measured |
| NVIDIA A10G | 128 | 22,383 | measured |

The 32×128 configuration is 4.3× faster than 32×1 on L4 and 4.6× faster on
A10G. The 32×1 setting is actor-limited, while 32×128 keeps the learner
supplied with work. The 32×64 setting is retained because it is the production-
like operating point used by the controlled L4 ablations; no unreported rate is
imputed for it.

## Provenance

The reproducible configurations are:

- `ml/src/guandan/dart/configs/sweep_l4_dart_actor32_lanes1_throughput.yaml`
- `ml/src/guandan/dart/configs/sweep_l4_dart_actor32_lanes64_throughput.yaml`
- `ml/src/guandan/dart/configs/sweep_l4_dart_actor32_lanes128_throughput.yaml`

The benchmark entry points are `ml/scripts/modal/bench_actor_throughput_modal.py`
and `ml/scripts/util/bench_actor_throughput.py`. The original raw Modal logs are
not included in the public artifact; the values above are the curated reported
measurements and should not be interpreted as a complete lane or batch sweep.
