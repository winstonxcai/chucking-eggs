"""Modal launcher for the Phase 3 shared-inference smoke.

Spins up the inference server on an A10G with N synthetic actors (no learner)
and reports throughput + per-actor inference wait times. Matches the same
inference config that train_guanzero_modal.py uses for paper-spec.

Usage:
    modal run --detach ml/scripts/modal/bench_shared_inference_modal.py \\
        --n-actors 32 --duration-s 30 --use-bf16
"""

from __future__ import annotations

from pathlib import Path

import modal


app = modal.App("guanzero-bench-shared-inference")
vol = modal.Volume.from_name("pvguan-runs", create_if_missing=True)
RUN_VOL = "/runs"

_root = Path(__file__).resolve().parent.parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pyyaml", "tqdm")
    .add_local_dir(str(_root / "ml" / "src"),     remote_path="/root/ml/src")
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=32,
    memory=32 * 1024,
    timeout=3600,
    volumes={RUN_VOL: vol},
)
def bench_remote(
    n_actors:       int,
    duration_s:     float,
    device:         str,
    num_slots:      int,
    max_actions:    int,
    max_requests:   int,
    max_action_rows: int,
    timeout_ms:     float,
    hidden_lstm:    int,
    hidden_mlp:     int,
    n_mlp_layers:   int,
    use_bf16:       bool,
) -> dict:
    import os
    import sys

    os.environ["PYTHONPATH"] = "/root/ml/src:" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, "/root/ml/src")

    # argparse-compatible namespace for run_bench
    class Args:
        pass
    args = Args()
    args.n_actors        = n_actors
    args.device          = device
    args.duration_s      = duration_s
    args.num_slots       = num_slots
    args.max_actions     = max_actions
    args.max_requests    = max_requests
    args.max_action_rows = max_action_rows
    args.timeout_ms      = timeout_ms
    args.hidden_lstm     = hidden_lstm
    args.hidden_mlp      = hidden_mlp
    args.n_mlp_layers    = n_mlp_layers
    args.use_bf16        = use_bf16
    args.out             = None

    sys.path.insert(0, "/root/ml/scripts/util")
    from bench_shared_inference import run_bench

    print(f"=== Phase 3 shared-inference smoke (Modal A10G) ===")
    print(f"  n_actors={n_actors}  device={device}  duration={duration_s}s")
    print(f"  paper-spec net: lstm={hidden_lstm}  mlp={hidden_mlp}x{n_mlp_layers}")
    print(f"  batch caps: max_requests={max_requests}  max_action_rows={max_action_rows}")
    print(f"  timeout_ms={timeout_ms}  use_bf16={use_bf16}")
    print()

    result = run_bench(args)

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s} = {v}")
    return result


@app.local_entrypoint()
def main(
    n_actors:        int   = 32,
    duration_s:      float = 30.0,
    device:          str   = "cuda",
    num_slots:       int   = 512,
    max_actions:     int   = 320,
    max_requests:    int   = 32,
    max_action_rows: int   = 4096,
    timeout_ms:      float = 1.0,
    hidden_lstm:     int   = 256,
    hidden_mlp:      int   = 1024,
    n_mlp_layers:    int   = 6,
    use_bf16:        bool  = False,
) -> None:
    out = bench_remote.remote(
        n_actors=n_actors,
        duration_s=duration_s,
        device=device,
        num_slots=num_slots,
        max_actions=max_actions,
        max_requests=max_requests,
        max_action_rows=max_action_rows,
        timeout_ms=timeout_ms,
        hidden_lstm=hidden_lstm,
        hidden_mlp=hidden_mlp,
        n_mlp_layers=n_mlp_layers,
        use_bf16=use_bf16,
    )
    print(f"\nDone. Result: {out}")
