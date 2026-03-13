"""Diagnostic script: verify LSTM training works correctly on this device.

Run this BEFORE starting any training run to catch MPS/CUDA issues early.

Usage:
    PYTHONPATH=src python scripts/test_mps_lstm.py

Tests:
    1. LSTM forward on device with full history (T=15)
    2. LSTM forward with single-move history (T=1) — packing edge case
    3. QNetworkLSTM forward + backward, no NaN gradients in LSTM params
    4. Mixed-length batch through QNetworkLSTM (B=8, lengths=[15,10,5,3,1,1,1,1])
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guandan.encoding import ACTION_DIM, D_MOVE, MAX_HISTORY, STATE_DIM
from guandan.q_network import QNetworkLSTM, get_device

LSTM_HIDDEN = 128
PASS = "✓ PASS"
FAIL = "✗ FAIL"


def run_test(name: str, fn) -> bool:
    try:
        fn()
        print(f"  {PASS}: {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}: {name}")
        print(f"         {type(e).__name__}: {e}")
        return False


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Dimensions: state={STATE_DIM}, action={ACTION_DIM}, d_move={D_MOVE}, max_history={MAX_HISTORY}")
    print()

    net = QNetworkLSTM(lstm_hidden=LSTM_HIDDEN).to(device)
    results = []

    # ─── Test 1: Full history (T=15) ─────────────────────────────────────
    def test_full_history():
        B = 4
        s = torch.randn(B, STATE_DIM, device=device)
        a = torch.randn(B, ACTION_DIM, device=device)
        h = torch.randn(B, MAX_HISTORY, D_MOVE, device=device)
        hl = torch.full((B,), MAX_HISTORY, dtype=torch.long)
        out = net(s, a, h, hl)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
        assert not out.isnan().any(), "NaN in output"

    results.append(run_test("LSTM forward, full history (T=15)", test_full_history))

    # ─── Test 2: Single-move history (T=1) ───────────────────────────────
    def test_single_move_history():
        B = 4
        s = torch.randn(B, STATE_DIM, device=device)
        a = torch.randn(B, ACTION_DIM, device=device)
        h = torch.randn(B, MAX_HISTORY, D_MOVE, device=device)
        hl = torch.full((B,), 1, dtype=torch.long)  # only 1 move in history
        out = net(s, a, h, hl)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
        assert not out.isnan().any(), "NaN in output with T=1"

    results.append(run_test("LSTM forward, single-move history (T=1)", test_single_move_history))

    # ─── Test 3: Backward pass, no NaN gradients ─────────────────────────
    def test_backward():
        net.zero_grad()
        B = 8
        s = torch.randn(B, STATE_DIM, device=device)
        a = torch.randn(B, ACTION_DIM, device=device)
        h = torch.randn(B, MAX_HISTORY, D_MOVE, device=device)
        hl = torch.full((B,), 8, dtype=torch.long)
        out = net(s, a, h, hl)
        loss = out.sum()
        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is None:
                raise ValueError(f"No gradient for {name}")
            if param.grad.isnan().any():
                raise ValueError(f"NaN gradient in {name}")

    results.append(run_test("Backward pass, no NaN gradients", test_backward))

    # ─── Test 4: Mixed-length batch ───────────────────────────────────────
    def test_mixed_lengths():
        lengths = [15, 10, 5, 3, 1, 1, 1, 1]
        B = len(lengths)
        s = torch.randn(B, STATE_DIM, device=device)
        a = torch.randn(B, ACTION_DIM, device=device)
        h = torch.randn(B, MAX_HISTORY, D_MOVE, device=device)
        hl = torch.tensor(lengths, dtype=torch.long)
        out = net(s, a, h, hl)
        assert out.shape == (B,), f"Expected ({B},), got {out.shape}"
        assert not out.isnan().any(), "NaN in mixed-length output"

    results.append(run_test(
        f"Mixed-length batch (B=8, lengths=[15,10,5,3,1,1,1,1])", test_mixed_lengths))

    # ─── Summary ─────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print()
    print(f"Results: {n_pass}/{len(results)} tests passed")

    if n_fail == 0:
        print(f"\n✓ All tests passed. Device '{device}' is ready for training.")
        print("  Next: PYTHONPATH=src python -m guandan.train --episodes 8000 --quick")
    else:
        print(f"\n✗ {n_fail} test(s) failed on device '{device}'.")
        if "mps" in str(device):
            print("  MPS LSTM support may be limited. Try:")
            print("  → Upgrade PyTorch to ≥ 2.1: pip install --upgrade torch")
            print("  → Or fall back to CPU: export PYTORCH_DEVICE=cpu")
        print("  Do NOT start a training run until all tests pass.")


if __name__ == "__main__":
    main()
