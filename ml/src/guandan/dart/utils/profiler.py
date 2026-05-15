"""Per-phase timing profiler for Dart actors, learner, and inference server.

A single ``PhaseProfiler`` class handles all three use cases via two optional
extensions:

* **K-bucket breakdown** — actors populate bucket counts via ``add_count``
  (e.g. ``"decisions_k=1"``, ``"decisions_k=2-5"``). Server and learner don't,
  so ``report_k_buckets`` returns an empty string for them automatically.

* **CUDA synchronization** — pass ``device`` on construction; the context
  manager inserts ``torch.cuda.synchronize()`` before stopping the clock when
  ``sync=True``. Required for accurate GPU timing; a no-op on CPU/MPS.

Usage::

    prof = PhaseProfiler(enabled=True, device=torch.device("cuda"))
    with prof.time("forward", sync=True):
        output = net(batch)
    prof.add_count("decisions_k>20", 1)
    print(prof.report(wall_s=elapsed, n_events=n_episodes, event_label="episodes"))
    print(prof.report_k_buckets(n_decisions=n_dec))
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

import torch

from ..data.sample_tags import K_BUCKET_NAMES, k_bucket


def k_bucket_label(K: int) -> str:
    """Map a legal-action count K to its bucket label (e.g. 'k=2-5')."""
    return K_BUCKET_NAMES[k_bucket(K)]


class PhaseProfiler:
    """Phase-level wall-time profiler.

    All methods are no-ops when ``enabled=False``, so callers need no
    ``if prof:`` guards.
    """

    def __init__(
        self,
        enabled: bool,
        device: torch.device | None = None,
    ) -> None:
        self.enabled = enabled
        self._device = device
        self._times: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def time(self, name: str, sync: bool = False) -> Generator[None, None, None]:
        """Context manager that records wall time spent in ``name``.

        ``sync=True`` inserts a CUDA synchronize before stopping the clock —
        use this for GPU phases so async kernels are flushed before measurement.
        """
        if not self.enabled:
            yield
            return
        import time
        if sync and self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            self._times[name] += time.perf_counter() - t0
            self._counts[name] += 1

    def add_count(self, name: str, value: int) -> None:
        """Accumulate an integer counter (e.g. decision counts, K-bucket tallies)."""
        if self.enabled:
            self._counts[name] += value

    def report(
        self,
        wall_s: float,
        n_events: int,
        event_label: str = "events",
    ) -> str:
        """Return a formatted per-phase timing table.

        ``wall_s`` — total elapsed wall seconds (from the caller's clock).
        ``n_events`` — number of top-level events (episodes, updates, etc.).
        ``event_label`` — label for ``n_events`` in the header line.
        """
        if not self.enabled or not self._times:
            return ""
        total_timed = sum(self._times.values())
        lines: list[str] = [
            "",
            f"Profile  |  {n_events} {event_label}  |  {wall_s:.2f}s wall",
            f"{'phase':22s} {'sec':>9s} {'%':>7s} {'calls':>8s} {'ms/call':>10s}",
            "-" * 62,
        ]
        for name, t in sorted(self._times.items(), key=lambda x: -x[1]):
            c = max(1, self._counts.get(name, 1))
            pct = 100.0 * t / total_timed if total_timed > 0 else 0.0
            lines.append(
                f"{name:22s} {t:9.3f} {pct:7.1f} {c:8d} {1000*t/c:10.3f}"
            )
        lines.append("-" * 62)
        pct_wall = 100.0 * total_timed / wall_s if wall_s > 0 else 0.0
        lines.append(
            f"{'TOTAL TIMED':22s} {total_timed:9.3f}  ({pct_wall:.1f}% of wall)"
        )
        return "\n".join(lines)

    def report_k_buckets(self, n_decisions: int) -> str:
        """Return a K-bucket breakdown table, or '' if no bucket counts present.

        Actors populate ``decisions_k=1``, ``decisions_k=2-5``, etc. via
        ``add_count``; server and learner don't, so this returns '' for them.
        """
        if not self.enabled:
            return ""
        bucket_counts = {b: self._counts.get(f"decisions_{b}", 0) for b in K_BUCKET_NAMES.values()}
        if not any(bucket_counts.values()):
            return ""
        n = n_decisions or 1
        shortcut_K1 = self._counts.get("shortcut_K1", 0)
        lines: list[str] = [
            "",
            "K distribution + q_net_forward per bucket:",
            f"  {'bucket':>8s} {'decisions':>10s} {'%':>6s} "
            f"{'fwd_calls':>10s} {'fwd_total_s':>12s} {'fwd_ms/call':>12s}",
        ]
        for b in K_BUCKET_NAMES.values():
            d = bucket_counts[b]
            pct = 100.0 * d / n
            fwd_calls = self._counts.get(f"q_net_forward_{b}", 0)
            fwd_total = self._times.get(f"q_net_forward_{b}", 0.0)
            ms_per_call = 1000.0 * fwd_total / fwd_calls if fwd_calls else 0.0
            tag = " (shortcut)" if b == K_BUCKET_NAMES[0] and shortcut_K1 else ""
            lines.append(
                f"  {b:>8s} {d:>10d} {pct:>5.1f}% "
                f"{fwd_calls:>10d} {fwd_total:>12.3f} {ms_per_call:>12.3f}{tag}"
            )
        if shortcut_K1:
            lines.append(f"  (K=1 shortcut fired {shortcut_K1}× — no forward needed)")
        return "\n".join(lines)


__all__ = ["PhaseProfiler", "k_bucket_label"]
