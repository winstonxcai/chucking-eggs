"""Numerical-equivalence tests for the shared GPU inference server.

The server's chosen action index must match the local-CPU `argmax_q` path
when given the same q-net weights and the same encoded_list. These tests run
the shared-memory path in-process so we can validate batching logic and
per-request argmax slicing without subprocess scheduling noise.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest
import torch

from guandan.dart.runtime.actor import argmax_q, select_legal
from guandan.dart.model.encoding.base_encoder import StateActionEncoder
from guandan.dart.runtime.inference_server import (
    InferenceClient,
    InferenceServer,
    allocate_shared_buffers,
    release_shared_buffers,
)
from guandan.dart.config import QNetConfig
from guandan.dart.model.q_network import GuanZeroQNet, init_guanzero_nets
from guandan.game import GuanDanEnv


# ─── helpers ─────────────────────────────────────────────────


def _build_decisions(n: int, seed: int = 7) -> list[tuple[int, list[dict]]]:
    """Roll random episodes until we have n distinct (seat, encoded_list) decisions."""
    encoder = StateActionEncoder()
    decisions: list[tuple[int, list[dict]]] = []
    rng = np.random.default_rng(seed)
    ep = 0
    while len(decisions) < n:
        env = GuanDanEnv()
        env.reset(seed=int(rng.integers(0, 10_000_000)))
        steps = 0
        while not env.done and len(decisions) < n:
            p = env.current_player
            legal = select_legal(env, p)
            encoded = encoder.encode_all(env, p, legal)
            if len(encoded) >= 2:
                decisions.append((p, encoded))
            # take a random legal action to advance the env
            idx = int(rng.integers(0, len(legal)))
            env.step(legal[idx])
            steps += 1
            if steps > 200:
                break
        ep += 1
        if ep > 200:
            break
    return decisions


def _local_argmaxes(
    q_nets: Mapping[int, GuanZeroQNet],
    decisions: list[tuple[int, list[dict]]],
    device: str = "cpu",
) -> list[int]:
    """Reference path — what `actor.argmax_q` would pick for each decision."""
    out: list[int] = []
    dev = torch.device(device)
    for seat, encoded in decisions:
        idx, _ = argmax_q(q_nets[seat], encoded, dev)
        out.append(idx)
    return out


# ─── tests ───────────────────────────────────────────────────


def _shared_argmaxes(
    q_nets: Mapping[int, GuanZeroQNet],
    decisions: list[tuple[int, list[dict]]],
    device: str = "cpu",
    num_slots: int = 16,
    max_actions: int = 512,
    max_requests: int = 8,
    timeout_ms: float = 50.0,
) -> list[int]:
    """In-process shared-mem path: server in a thread, single in-process actor.

    Uses real shared-memory blocks so we exercise the pack/unpack offsets and
    response-slot wiring, but skips spawn-context multiprocessing since we
    don't need cross-process isolation for numerics validation.
    """
    ctx = mp.get_context("spawn")
    bufs, _meta = allocate_shared_buffers(
        num_slots=num_slots,
        max_actions=max_actions,
        n_actors=1,
        ctx=ctx,
    )
    stop_event = ctx.Event()

    server = InferenceServer(
        q_nets=q_nets,
        bufs=bufs,
        stop_event=stop_event,
        device=device,
        max_requests=max_requests,
        max_action_rows=max_requests * max_actions,
        timeout_ms=timeout_ms,
        use_bf16=False,
    )

    server_thread = threading.Thread(target=server.server_loop, daemon=True)
    server_thread.start()

    client = InferenceClient(
        actor_id=0,
        bufs=bufs,
        timeout_s=10.0,
        max_actions=max_actions,
    )

    out: list[int] = []
    try:
        for seat, encoded in decisions:
            chosen, _ver = client.submit(seat, encoded)
            out.append(chosen)
    finally:
        stop_event.set()
        server_thread.join(timeout=5.0)
        release_shared_buffers(bufs, unlink=True)
    return out


def test_shared_mem_cpu_equivalence():
    """Shared-memory wire format must produce identical argmax to local."""
    torch.manual_seed(0)
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=20, seed=42)
    expected = _local_argmaxes(q_nets, decisions, device="cpu")
    actual   = _shared_argmaxes(q_nets, decisions, device="cpu")

    assert actual == expected, (
        f"shared-mem server diverged from local argmax_q.\n"
        f"  expected={expected}\n"
        f"  actual  ={actual}"
    )


def test_shared_mem_batched_requests_match_serial():
    """Forced batching across multiple requests still matches serial argmax."""
    torch.manual_seed(1)
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=12, seed=99)
    expected = _local_argmaxes(q_nets, decisions, device="cpu")
    actual = _shared_argmaxes(
        q_nets, decisions, device="cpu",
        max_requests=12, timeout_ms=200.0,
    )

    assert actual == expected, (
        f"batched shared-mem argmax diverged.\n"
        f"  expected={expected}\n"
        f"  actual  ={actual}"
    )


def test_shared_mem_client_rejects_oversize_K():
    """K > inference_max_actions should fail loudly, not silently truncate."""
    torch.manual_seed(2)
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    ctx = mp.get_context("spawn")
    bufs, _ = allocate_shared_buffers(num_slots=4, max_actions=8, n_actors=1, ctx=ctx)
    try:
        client = InferenceClient(actor_id=0, bufs=bufs, max_actions=8)
        # Build a fake encoded_list of length 9 (over the 8-cap).
        encoder = StateActionEncoder()
        env = GuanDanEnv()
        env.reset(seed=0)
        legal = select_legal(env, env.current_player)
        # Force K=9 by replicating any encoded entry
        encoded = encoder.encode_all(env, env.current_player, legal[:1]) * 9
        with pytest.raises(AssertionError, match="exceeds inference_max_actions"):
            client.submit(env.current_player, encoded)
    finally:
        release_shared_buffers(bufs, unlink=True)


def test_shared_mem_actor_timeout_raises():
    """If no server is running, the actor's submit() must time out cleanly."""
    from guandan.dart.runtime.inference_server import InferenceTimeoutError
    torch.manual_seed(3)
    encoder = StateActionEncoder()
    ctx = mp.get_context("spawn")
    bufs, _ = allocate_shared_buffers(num_slots=4, max_actions=128, n_actors=1, ctx=ctx)
    try:
        client = InferenceClient(actor_id=0, bufs=bufs, timeout_s=0.5, max_actions=128)
        env = GuanDanEnv()
        env.reset(seed=0)
        legal = select_legal(env, env.current_player)
        encoded = encoder.encode_all(env, env.current_player, legal)
        with pytest.raises(InferenceTimeoutError):
            client.submit(env.current_player, encoded)
    finally:
        release_shared_buffers(bufs, unlink=True)


def test_server_shared_weight_reload_respects_local_version():
    """Shared-memory refresh skips equal versions and loads newer versions."""
    torch.manual_seed(4)
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    ctx = mp.get_context("spawn")
    bufs, _ = allocate_shared_buffers(num_slots=1, max_actions=8, n_actors=1, ctx=ctx)
    stop_event = ctx.Event()

    name, param = next(iter(q_nets[0].state_dict().items()))
    weights_buf = torch.full((param.numel(),), 7.0, dtype=param.dtype)
    weights_version = SimpleNamespace(value=5)
    spec = SimpleNamespace(seat=0, name=name, offset=0, numel=param.numel(), shape=param.shape)

    try:
        server = InferenceServer(
            q_nets=q_nets,
            bufs=bufs,
            stop_event=stop_event,
            device="cpu",
            weights_buf=weights_buf,
            weights_version=weights_version,
            weights_lock=nullcontext(),
            weight_specs=[spec],
        )
        original = server.q_nets[0].state_dict()[name].clone()

        server._local_version = 5
        server._maybe_reload_weights()
        assert torch.equal(server.q_nets[0].state_dict()[name], original)

        weights_version.value = 6
        server._maybe_reload_weights()
        assert server._local_version == 6
        assert torch.all(server.q_nets[0].state_dict()[name] == 7.0)
    finally:
        release_shared_buffers(bufs, unlink=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tolerance_argmax_matches_local_or_near_tie():
    """On CUDA, FP differences from kernel choice may flip near-ties.
    Allow either an exact match or a top-2 pick when the gap to the best
    Q-value is below 1e-5.
    """
    torch.manual_seed(0)
    q_nets = init_guanzero_nets(QNetConfig(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2))
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=10, seed=42)

    expected_cpu = _local_argmaxes(q_nets, decisions, device="cpu")
    actual_cuda  = _shared_argmaxes(q_nets, decisions, device="cuda")

    # For each decision where they differ, verify the gap is below tolerance
    # by computing both Q-vectors on CPU and checking near-tie.
    for i, (e, a) in enumerate(zip(expected_cpu, actual_cuda)):
        if e == a:
            continue
        seat, encoded = decisions[i]
        from guandan.dart.data.buffer import collate_base_encoded
        with torch.no_grad():
            state_batch, action_batch, repeats = collate_base_encoded(
                [encoded],
                device="cpu",
            )
            q = q_nets[seat].forward_grouped(
                state_batch,
                action_batch,
                repeats,
            ).cpu().numpy()
        gap = float(q[e] - q[a])
        assert abs(gap) < 1e-5, (
            f"decision {i}: server picked {a} but local picked {e}; "
            f"q[expected]-q[actual]={gap:.2e} exceeds tie tolerance"
        )
