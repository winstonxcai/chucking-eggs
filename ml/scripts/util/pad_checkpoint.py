"""Pad QValueNet checkpoint to add N zero-init flag dims to the state input.

For the Q-trunk first layer (`net.0.weight` shape `[hidden, d_state + d_action]`),
insert `state_pad` zero columns at position `d_state` so the layout becomes
`[base_state | new_state_pad | action]`. For the V-trunk first layer
(`v_net.0.weight` shape `[hidden, d_state]`), append `state_pad` zero columns.

Zero-init means the padded checkpoint produces bit-exact identical outputs to
the original whenever the new flag dims are 0, so the surgery is a no-op at t=0
and gradient teaches the new columns from there.

Usage:
    PYTHONPATH=ml/src python ml/scripts/util/pad_checkpoint.py \\
        --in ml/checkpoints/az_gen3_bestwr.pt \\
        --out ml/checkpoints/az_gen3_bestwr_padded.pt \\
        --pad 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def pad_checkpoint(ckpt_path: Path, out_path: Path, state_pad: int) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    sd = dict(ckpt["state_dict"])

    old_d_state = cfg["d_state"]
    d_action = cfg["d_action"]

    # Q-trunk: net.0.weight shape [hidden, d_state + d_action]
    qw = sd["net.0.weight"]
    H = qw.shape[0]
    expected_q_in = old_d_state + d_action
    assert qw.shape[1] == expected_q_in, (
        f"net.0.weight shape mismatch: got {qw.shape[1]}, "
        f"expected {expected_q_in} (d_state={old_d_state} + d_action={d_action})"
    )
    state_part = qw[:, :old_d_state]
    action_part = qw[:, old_d_state:]
    pad = torch.zeros(H, state_pad, dtype=qw.dtype)
    sd["net.0.weight"] = torch.cat([state_part, pad, action_part], dim=1)

    # V-trunk: v_net.0.weight shape [hidden, d_state]
    if "v_net.0.weight" in sd:
        vw = sd["v_net.0.weight"]
        assert vw.shape[1] == old_d_state, (
            f"v_net.0.weight shape mismatch: got {vw.shape[1]}, expected {old_d_state}"
        )
        Hv = vw.shape[0]
        sd["v_net.0.weight"] = torch.cat(
            [vw, torch.zeros(Hv, state_pad, dtype=vw.dtype)], dim=1
        )

    cfg["d_state"] = old_d_state + state_pad
    ckpt["state_dict"] = sd
    ckpt["config"] = cfg
    torch.save(ckpt, out_path)
    print(
        f"Padded {ckpt_path} -> {out_path}\n"
        f"  d_state: {old_d_state} -> {cfg['d_state']}\n"
        f"  net.0.weight:   {qw.shape} -> {sd['net.0.weight'].shape}\n"
        f"  v_net.0.weight: {vw.shape} -> {sd['v_net.0.weight'].shape}"
    )


def verify_bit_exact(orig_path: Path, padded_path: Path, old_d_state: int) -> None:
    """Forward pass with zeroed flag dims must equal original output."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from guandan.training import QValueNet

    orig_ckpt = torch.load(orig_path, map_location="cpu", weights_only=True)
    pad_ckpt = torch.load(padded_path, map_location="cpu", weights_only=True)

    orig_net = QValueNet(
        d_state=orig_ckpt["config"]["d_state"],
        d_action=orig_ckpt["config"]["d_action"],
        hidden=orig_ckpt["config"]["hidden"],
    )
    orig_net.load_state_dict(orig_ckpt["state_dict"])
    orig_net.eval()

    pad_net = QValueNet(
        d_state=pad_ckpt["config"]["d_state"],
        d_action=pad_ckpt["config"]["d_action"],
        hidden=pad_ckpt["config"]["hidden"],
    )
    pad_net.load_state_dict(pad_ckpt["state_dict"])
    pad_net.eval()

    state_pad = pad_ckpt["config"]["d_state"] - old_d_state

    torch.manual_seed(0)
    s_orig = torch.randn(8, old_d_state)
    a = torch.randn(8, orig_ckpt["config"]["d_action"])
    s_pad = torch.cat([s_orig, torch.zeros(8, state_pad)], dim=-1)

    with torch.no_grad():
        q_orig = orig_net(s_orig, a)
        q_pad = pad_net(s_pad, a)
        v_orig = orig_net.value(s_orig)
        v_pad = pad_net.value(s_pad)

    q_diff = (q_orig - q_pad).abs().max().item()
    v_diff = (v_orig - v_pad).abs().max().item()
    print(f"\nBit-exact verification (8 random samples):")
    print(f"  Q max abs diff: {q_diff:.2e}")
    print(f"  V max abs diff: {v_diff:.2e}")
    assert q_diff < 1e-4 and v_diff < 1e-4, "Surgery broke output equivalence!"
    print("  OK: padded checkpoint produces identical outputs at zero flag dims")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pad", type=int, default=9, help="Number of state dims to add")
    p.add_argument("--no-verify", action="store_true", help="Skip bit-exact verification")
    args = p.parse_args()

    orig_d_state = torch.load(args.inp, map_location="cpu", weights_only=True)["config"]["d_state"]
    pad_checkpoint(args.inp, args.out, args.pad)
    if not args.no_verify:
        verify_bit_exact(args.inp, args.out, orig_d_state)


if __name__ == "__main__":
    main()
