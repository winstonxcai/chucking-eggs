"""Numerical-equivalence tests for the shared GPU inference server.

The server's chosen action index must match the local-CPU `_argmax_q` path
when given the same q-net weights and the same encoded_list. Phase 1 tests
the simple in-process path (no subprocess fork) so we can validate the
batching logic and per-request argmax slicing without IPC complexity.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from typing import Mapping

import numpy as np
import pytest
import torch

from guandan.guanzero.actor import _argmax_q, _select_legal
from guandan.guanzero.encoder import StateActionEncoder
from guandan.guanzero.inference_server import (
    InferenceClient,
    InferenceServer,
)
from guandan.guanzero.q_network import GuanZeroQNet, init_position_nets
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
            legal = _select_legal(env, p, max_legal=128)
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
    """Reference path — what `actor._argmax_q` would pick for each decision."""
    out: list[int] = []
    dev = torch.device(device)
    for seat, encoded in decisions:
        out.append(_argmax_q(q_nets[seat], encoded, dev))
    return out


def _server_argmaxes(
    q_nets: Mapping[int, GuanZeroQNet],
    decisions: list[tuple[int, list[dict]]],
    device: str = "cpu",
    max_requests: int = 8,
    timeout_ms: float = 50.0,
) -> list[int]:
    """In-process server: run the InferenceServer in a background thread,
    submit decisions one at a time as if from a single actor.

    Using a thread (not subprocess) for the test keeps it deterministic and
    avoids spawning fresh torch processes for every test run.
    """
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    stop_event = ctx.Event()

    server = InferenceServer(
        q_nets=q_nets,
        request_queue=request_queue,
        response_queues={0: response_queue},
        stop_event=stop_event,
        device=device,
        max_requests=max_requests,
        max_action_rows=8192,
        timeout_ms=timeout_ms,
        use_bf16=False,
    )

    # Run server in a daemon thread so we can submit from main and then signal stop.
    server_thread = threading.Thread(target=server.server_loop, daemon=True)
    server_thread.start()

    client = InferenceClient(
        actor_id=0,
        request_queue=request_queue,
        response_queue=response_queue,
        timeout_s=10.0,
    )

    out: list[int] = []
    try:
        for seat, encoded in decisions:
            chosen, _ver = client.submit(seat, encoded)
            out.append(chosen)
    finally:
        stop_event.set()
        server_thread.join(timeout=5.0)
    return out


# ─── tests ───────────────────────────────────────────────────


def test_cpu_equivalence_argmax_matches_local():
    """For matching weights, the server's argmax must equal the local-CPU
    `_argmax_q` path for every decision. CPU FP is deterministic so this
    must be exact."""
    torch.manual_seed(0)
    # Tiny net for fast test; equivalence is architecture-independent.
    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=20, seed=42)
    assert len(decisions) == 20

    expected = _local_argmaxes(q_nets, decisions, device="cpu")
    actual   = _server_argmaxes(q_nets, decisions, device="cpu")

    assert actual == expected, (
        f"server argmax diverged from local _argmax_q.\n"
        f"  expected={expected}\n"
        f"  actual  ={actual}"
    )


def test_batched_requests_match_serial():
    """The server batches across decisions; argmax over each request's K-row
    slice must still pick the same index as a per-decision local forward."""
    torch.manual_seed(1)
    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=12, seed=99)
    assert len(decisions) == 12

    expected = _local_argmaxes(q_nets, decisions, device="cpu")
    # Force batching by setting max_requests high — server will drain the
    # whole load before the timeout hits, exercising the per-seat slicing path.
    actual = _server_argmaxes(
        q_nets, decisions, device="cpu",
        max_requests=12, timeout_ms=200.0,
    )

    assert actual == expected, (
        f"batched server argmax diverged from per-request local argmax.\n"
        f"  expected={expected}\n"
        f"  actual  ={actual}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_tolerance_argmax_matches_local_or_near_tie():
    """On CUDA, FP differences from kernel choice may flip near-ties.
    Allow either an exact match or a top-2 pick when the gap to the best
    Q-value is below 1e-5.
    """
    torch.manual_seed(0)
    q_nets = init_position_nets(hidden_lstm=16, hidden_mlp=32, n_mlp_layers=2)
    for net in q_nets.values():
        net.eval()

    decisions = _build_decisions(n=10, seed=42)

    expected_cpu = _local_argmaxes(q_nets, decisions, device="cpu")
    actual_cuda  = _server_argmaxes(q_nets, decisions, device="cuda")

    # For each decision where they differ, verify the gap is below tolerance
    # by computing both Q-vectors on CPU and checking near-tie.
    for i, (e, a) in enumerate(zip(expected_cpu, actual_cuda)):
        if e == a:
            continue
        seat, encoded = decisions[i]
        from guandan.guanzero.buffer import collate_encoded
        with torch.no_grad():
            q = q_nets[seat](collate_encoded(encoded, device="cpu")).cpu().numpy()
        gap = float(q[e] - q[a])
        assert abs(gap) < 1e-5, (
            f"decision {i}: server picked {a} but local picked {e}; "
            f"q[expected]-q[actual]={gap:.2e} exceeds tie tolerance"
        )
