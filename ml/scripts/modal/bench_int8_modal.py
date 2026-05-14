"""Modal CPU A/B: fp32 vs int8 dynamic-quantized Q-net forward.

Builds the SharedTrickHeadQNet from m5_clean_baseline_l4.yaml on a Modal CPU
worker (x86 Linux → FBGEMM backend) and times forward_grouped for a range of
candidate-action counts that match real actor decisions.

Run:
    modal run ml/scripts/modal/bench_int8_modal.py
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("guanzero-bench-int8")

_root = Path(__file__).resolve().parent.parent.parent.parent  # repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "curl", "ca-certificates")
    .run_commands(
        # Install rustup with retries (transient network resets possible).
        "curl --retry 5 --retry-delay 5 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain stable --profile minimal",
        "ln -s /root/.cargo/bin/cargo /usr/local/bin/cargo",
        "ln -s /root/.cargo/bin/rustc /usr/local/bin/rustc",
    )
    .pip_install("torch", "numpy", "pyyaml", "maturin")
    .add_local_dir(
        str(_root / "ml" / "src"),
        remote_path="/root/ml/src",
        # Skip Rust build artifacts (large; will rebuild on the worker)
        ignore=["**/target/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(_root / "ml" / "scripts"), remote_path="/root/ml/scripts")
)


@app.function(
    image=image,
    cpu=4,
    memory=4 * 1024,
    timeout=300,
)
def bench_remote() -> str:
    import os
    import platform
    import subprocess
    import sys
    import time
    from pathlib import Path

    # Match actor behaviour: single-threaded forward so we measure per-actor cost.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, "/root/ml/src")

    # Build guandan_rs (Rust movegen extension). combos.py hard-imports it.
    # Drop Cargo.lock — debian bookworm's cargo 1.65 can't parse lock v4 written
    # by newer rust; cargo will regenerate it during build.
    lock = Path("/root/ml/src/guandan_rs/Cargo.lock")
    if lock.exists():
        lock.unlink()
    print("Building guandan_rs ...", flush=True)
    subprocess.run(
        ["maturin", "build", "--release", "--out", "/tmp/wheels"],
        cwd="/root/ml/src/guandan_rs",
        check=True,
    )
    wheels = list(Path("/tmp/wheels").glob("guandan_rs-*.whl"))
    assert wheels, "no guandan_rs wheel produced"
    subprocess.run(["pip", "install", "--no-deps", str(wheels[0])], check=True)

    import torch
    import torch.ao.quantization as qao
    import yaml

    from guandan.guanzero.config import TrainConfig, shared_trick_head_qnet_config
    from guandan.guanzero.q_network import SharedTrickHeadQNet
    from guandan.cards import CARD_ID_DIM

    torch.set_num_threads(1)
    # FBGEMM is the x86 Linux quant backend.
    engines = torch.backends.quantized.supported_engines
    if "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"

    cfg_path = "/root/ml/src/guandan/guanzero/config/m5_clean_baseline_l4.yaml"
    with open(cfg_path) as f:
        cfg = TrainConfig.from_flat_dict(yaml.safe_load(f))
    qnet_cfg = shared_trick_head_qnet_config(cfg)

    net_fp32 = SharedTrickHeadQNet(qnet_cfg).eval()
    net_int8 = qao.quantize_dynamic(
        net_fp32, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )

    n_params = sum(p.numel() for p in net_fp32.parameters()) / 1e6

    def synth_batch(b_state: int, n_actions: int, hist_len: int = 30):
        torch.manual_seed(0)
        sb = {
            "player_blocks":     torch.randn(b_state, 4, 256),
            "history_actions":   torch.randn(b_state, hist_len, CARD_ID_DIM),
            "history_roles":     torch.randn(b_state, hist_len, 4),
            "history_is_pass":   torch.randn(b_state, hist_len, 1),
            "global_features":   torch.randn(b_state, 13),
            "own_hand":          torch.randn(b_state, CARD_ID_DIM),
            "partner_hand":      torch.randn(b_state, CARD_ID_DIM),
            "others_hand":       torch.randn(b_state, CARD_ID_DIM),
            "behavior":          torch.randn(b_state, 9),
        }
        n_rows = n_actions * b_state
        ab = {
            "candidate_action": torch.randn(n_rows, CARD_ID_DIM),
            "trick_head_id":    torch.randint(0, 4, (n_rows,)),
        }
        repeats = torch.tensor([n_actions] * b_state, dtype=torch.long)
        return sb, ab, repeats

    def bench(net, sb, ab, reps, n_iters=200, n_warmup=20):
        with torch.inference_mode():
            for _ in range(n_warmup):
                net.forward_grouped(sb, ab, reps)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                q = net.forward_grouped(sb, ab, reps)
            t = time.perf_counter() - t0
        return (t / n_iters) * 1e6, q

    lines: list[str] = []

    def L(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    L(f"Modal CPU bench — {platform.processor() or platform.machine()}")
    L(f"quantized engine: {torch.backends.quantized.engine}  (supported: {engines})")
    L(f"torch threads: {torch.get_num_threads()}, interop: {torch.get_num_interop_threads()}")
    L(f"Net: SharedTrickHeadQNet, ~{n_params:.2f}M fp32 params, trunk hidden={qnet_cfg.trunk_hidden}, layers={qnet_cfg.trunk_layers}")
    L("")

    for n_actions in (20, 50, 100, 200):
        sb1, ab1, r1 = synth_batch(b_state=1, n_actions=n_actions)
        sb2, ab2, r2 = synth_batch(b_state=2, n_actions=n_actions)
        fast_us, q_fast = bench(net_fp32, sb1, ab1, r1)
        slow_us, q_slow = bench(net_fp32, sb2, ab2, r2)
        int8_us, q_int8 = bench(net_int8, sb2, ab2, r2)
        slow_per = slow_us / 2
        int8_per = int8_us / 2
        cos = torch.nn.functional.cosine_similarity(q_slow.flatten(), q_int8.flatten(), dim=0).item()
        L(f"--- n_actions = {n_actions} ---")
        L(f"  fp32 fast path:        {fast_us:>9.1f} us/call")
        L(f"  fp32 slow /decision:   {slow_per:>9.1f} us")
        L(f"  int8 slow /decision:   {int8_per:>9.1f} us")
        L(f"  speedup int8 vs slow:  {slow_per/int8_per:>9.2f}x")
        L(f"  speedup int8 vs fast:  {fast_us/int8_per:>9.2f}x  (production gap)")
        L(f"  cosine_sim:            {cos:>9.4f}")
        L(f"  argmax match:          {(q_slow.argmax() == q_int8.argmax()).item()!s}")
        L("")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    out = bench_remote.remote()
    print("\n========== REMOTE OUTPUT ==========\n")
    print(out)
