"""Eval a Dart checkpoint against a fixed opponent (fixed-deck / paired).

For the requested ``--games`` total ``n``, we draw ``n/2`` unique decks (one per
seed) and play each deck twice: once with Dart on seats (0,2), once on
(1,3). Both playthroughs start from identical hands and starting player, so the
even- vs. odd-seating win rates are directly comparable on the same deals —
the standard variance-reduction trick for card-game eval.

By default games run on CPU and fan out across N worker processes. Within each
worker, multiple environment lanes are stepped together so Dart Q-forwards are
batched across active games, matching the actor rollout shape used in training.

Usage:
    uv run python -m guandan.scripts.eval.eval_dart \\
        --checkpoint ml/runs/<run>/checkpoints/update_00200000.pt \\
        --opponent random --games 1000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from guandan.agents import AGENT_REGISTRY
from guandan.scripts.eval._eval_worker import play_n_games_dart_lanes

DEFAULT_OPPONENTS = [
    name for name in AGENT_REGISTRY
    if name not in {"noai"}
]


def _tqdm_disabled() -> bool:
    setting = os.environ.get("DART_TQDM", "").strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return True
    if setting in {"1", "true", "yes", "on"}:
        return False
    return not sys.stderr.isatty()


# ─── Worker (top-level so spawn can pickle it) ───────────────────────────────

# Per-worker-process cache: load the bot once via initializer, then handle many
# (opponent, seeds) tasks against the same bot. Saves N opponents * spawn cost.
_BOT = None


def _set_torch_worker_threads() -> None:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_bot(checkpoint: Path, device: str) -> None:
    global _BOT
    from guandan.dart.agent import DartBot
    _BOT = DartBot.load(checkpoint, device=device)


def _init_worker(checkpoint: Path | None, device: str) -> None:
    _set_torch_worker_threads()
    if checkpoint is not None:
        _load_bot(checkpoint, device)
    elif _BOT is None:
        raise RuntimeError("fork worker started without an inherited DartBot")


def _worker(args: tuple) -> tuple[str, int, int, int, int]:
    """Play a slice of games against one opponent.

    Returns (opponent_name, wins_even, n_even, wins_odd, n_odd).
    """
    opponent_name, even_seeds, odd_seeds, lanes = args

    r_even = play_n_games_dart_lanes(
        _BOT,
        opponent_name,
        len(even_seeds),
        seeds=even_seeds,
        dart_seats=(0, 2),
        lanes=lanes,
    )
    r_odd = play_n_games_dart_lanes(
        _BOT,
        opponent_name,
        len(odd_seeds),
        seeds=odd_seeds,
        dart_seats=(1, 3),
        lanes=lanes,
    )
    return opponent_name, r_even["wins"], len(even_seeds), r_odd["wins"], len(odd_seeds)


# ─── Driver ──────────────────────────────────────────────────────────────────


def _split(seq: list, n: int) -> list[list]:
    """Split seq into n roughly-equal contiguous chunks."""
    if n <= 0:
        return [seq]
    n = min(n, len(seq))
    if n == 0:
        return []
    k, r = divmod(len(seq), n)
    out = []
    i = 0
    for w in range(n):
        size = k + (1 if w < r else 0)
        out.append(seq[i : i + size])
        i += size
    return out


def run_eval(
    checkpoint: Path,
    opponents: list[str],
    n_games: int,
    device: str,
    workers: int,
    seed: int,
    chunks_per_worker: int = 1,
    lanes: int = 8,
    start_method: str = "auto",
    on_opponent_done: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """Evaluate `checkpoint` against each opponent in `opponents`.

    Returns {opponent_name: {"even": {...}, "odd": {...}}}. Workers load the
    model once via initializer and reuse it across all opponents.
    """
    if n_games % 2 != 0:
        raise ValueError(f"--games must be even for paired fixed-deck eval, got {n_games}")

    # Same seed list for both seatings: each deck is played once with Dart
    # on (0,2) and once on (1,3). reset(seed=N) is fully deterministic, so both
    # playthroughs start from identical hands and starting player.
    half = n_games // 2
    deck_seeds = list(range(seed, seed + half))

    if chunks_per_worker < 1:
        raise ValueError(f"chunks_per_worker must be >= 1, got {chunks_per_worker}")
    if lanes < 1:
        raise ValueError(f"lanes must be >= 1, got {lanes}")
    if start_method == "auto":
        start_method = (
            "fork"
            if device == "cpu" and "fork" in mp.get_all_start_methods()
            else "spawn"
        )

    # Workers control process concurrency. Lanes control useful batch width
    # inside each task. Build lane-sized chunks per opponent, then let the
    # process pool schedule all opponent chunks globally. This keeps lanes full
    # even when workers is high relative to per-opponent game count.
    target_chunk_size = max(1, math.ceil(lanes / chunks_per_worker))
    n_chunks = max(1, math.ceil(half / target_chunk_size))
    even_chunks = _split(deck_seeds, n_chunks)
    odd_chunks  = _split(deck_seeds, n_chunks)
    tasks = [
        (opp, e, o, lanes)
        for opp in opponents
        for e, o in zip(even_chunks, odd_chunks, strict=False)
    ]

    chunks_per_opp = len(even_chunks)
    agg = {opp: {"we": 0, "ne": 0, "wo": 0, "no": 0, "done": 0} for opp in opponents}
    ctx = mp.get_context(start_method)
    desc = (
        f"{len(opponents)} opp x {n_games} games "
        f"({workers}w/{chunks_per_opp}c/{lanes}l/{len(tasks)} tasks)"
    )
    init_checkpoint: Path | None = checkpoint
    if start_method == "fork":
        if device != "cpu":
            raise ValueError("--start-method fork is only supported with --device cpu")
        _set_torch_worker_threads()
        _load_bot(checkpoint, device)
        init_checkpoint = None
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(init_checkpoint, device),
    ) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=desc,
            unit="chunk",
            disable=_tqdm_disabled(),
        ):
            opp, we, ne, wo, no = f.result()
            agg[opp]["we"] += we
            agg[opp]["ne"] += ne
            agg[opp]["wo"] += wo
            agg[opp]["no"] += no
            agg[opp]["done"] += 1
            if agg[opp]["done"] == chunks_per_opp and on_opponent_done is not None:
                on_opponent_done(opp, _result_from_agg(agg[opp]))

    results: dict[str, dict] = {}
    for opp, a in agg.items():
        results[opp] = _result_from_agg(a)
    return results


def _result_from_agg(a: dict) -> dict:
    out: dict = {}
    if a["ne"]:
        wr = a["we"] / a["ne"]
        se = math.sqrt(wr * (1 - wr) / a["ne"])
        out["even"] = {"games": a["ne"], "wins": a["we"], "win_rate": wr, "se": se}
    if a["no"]:
        wr = a["wo"] / a["no"]
        se = math.sqrt(wr * (1 - wr) / a["no"])
        out["odd"] = {"games": a["no"], "wins": a["wo"], "win_rate": wr, "se": se}
    return out


def _summarize_result(result: dict[str, dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for opp_name, by_seat in result.items():
        total_w = sum(r["wins"] for r in by_seat.values())
        total_n = sum(r["games"] for r in by_seat.values())
        wr = total_w / total_n if total_n else 0.0
        summary[opp_name] = {
            "n_games": total_n,
            "wins": total_w,
            "wr": round(wr, 4),
            "se": round(math.sqrt(wr * (1 - wr) / total_n), 4) if total_n else 0.0,
            **{f"games_{label}": r["games"] for label, r in by_seat.items()},
            **{f"wins_{label}": r["wins"] for label, r in by_seat.items()},
            **{f"wr_{label}": round(r["win_rate"], 4) for label, r in by_seat.items()},
        }
    return summary


def _seat_wins(row: dict, label: str) -> int | None:
    wins = row.get(f"wins_{label}")
    if wins is not None:
        return int(wins)
    wr = row.get(f"wr_{label}")
    n_games = row.get("n_games")
    if wr is None or n_games is None:
        return None
    return round(float(wr) * (int(n_games) // 2))


def _seat_games(row: dict, label: str) -> int:
    games = row.get(f"games_{label}")
    if games is not None:
        return int(games)
    return int(row.get("n_games", 0)) // 2


def _merge_row(existing: dict | None, extra: dict) -> dict:
    if not existing:
        return extra

    old_n = int(existing.get("n_games", 0))
    old_w = int(existing.get("wins", round(float(existing.get("wr", 0.0)) * old_n)))
    new_n = int(extra.get("n_games", 0))
    new_w = int(extra.get("wins", 0))
    total_n = old_n + new_n
    total_w = old_w + new_w
    wr = total_w / total_n if total_n else 0.0

    merged = {
        **existing,
        "n_games": total_n,
        "wins": total_w,
        "wr": round(wr, 4),
        "se": round(math.sqrt(wr * (1 - wr) / total_n), 4) if total_n else 0.0,
    }

    for label in ("even", "odd"):
        old_seat_w = _seat_wins(existing, label)
        new_seat_w = _seat_wins(extra, label)
        if old_seat_w is None or new_seat_w is None:
            continue
        old_seat_n = _seat_games(existing, label)
        new_seat_n = _seat_games(extra, label)
        seat_n = old_seat_n + new_seat_n
        seat_w = old_seat_w + new_seat_w
        merged[f"games_{label}"] = seat_n
        merged[f"wins_{label}"] = seat_w
        merged[f"wr_{label}"] = round(seat_w / seat_n, 4) if seat_n else 0.0
    return merged


def _load_existing_results(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    results = data.get("results", {})
    if not isinstance(results, dict):
        raise ValueError(f"{path} has no object-valued 'results' field")
    return results


def _write_results(path: Path, checkpoint: Path, opponents: list[str], summary: dict) -> None:
    out_data = {
        "checkpoint": str(checkpoint),
        "opponents": opponents,
        "results": {opp: summary[opp] for opp in opponents if opp in summary},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out_data, indent=2))
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--opponent", nargs="+", default=["random"],
                   choices=DEFAULT_OPPONENTS,
                   help="One or more opponents. Workers load the model once and "
                        "play against each opponent in sequence.")
    p.add_argument("--games", type=int, default=200,
                   help="Total games (must be even). n/2 unique decks, each played twice "
                        "— once with Dart on (0,2), once on (1,3).")
    p.add_argument("--workers", type=int, default=8,
                   help="Worker processes (default 8, sized for local CPU eval throughput).")
    p.add_argument("--chunks-per-worker", type=int, default=1,
                   help="Chunk subdivision factor. 1 means chunks are sized to fill the "
                        "requested lanes; higher values create smaller chunks to reduce "
                        "tail latency at the cost of task overhead.")
    p.add_argument("--lanes", type=int, default=8,
                   help="Environment lanes per worker chunk. Dart decisions across active "
                        "lanes are batched into grouped Q-forwards.")
    p.add_argument("--start-method", default="auto", choices=["auto", *mp.get_all_start_methods()],
                   help="Multiprocessing start method. 'auto' uses fork for CPU evals when "
                        "available, avoiding repeated Torch imports/checkpoint loads.")
    p.add_argument("--device", default="cpu",
                   help="Torch device for the Dart forward pass. Default cpu — for the "
                        "small per-move batches in eval, CPU beats MPS due to dispatch latency.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for env.reset.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to save structured JSON results.")
    args = p.parse_args()

    existing = _load_existing_results(args.out)
    requested = list(dict.fromkeys(args.opponent))
    final_summary = {opp: dict(existing[opp]) for opp in requested if opp in existing}
    groups: dict[tuple[int, int], list[str]] = {}
    for opp in requested:
        old_n = int(existing.get(opp, {}).get("n_games", 0))
        if old_n >= args.games:
            print(f"  {args.checkpoint.name}  vs {opp}: already has {old_n} games; skipping")
            continue
        missing = args.games - old_n
        if missing % 2 != 0:
            raise ValueError(
                f"Need an even number of missing games for {opp}: "
                f"requested {args.games}, existing {old_n}"
            )
        groups.setdefault((missing, args.seed + old_n // 2), []).append(opp)

    all_results: dict[str, dict] = {}
    for (missing, seed), opponents in sorted(groups.items()):
        print(f"Evaluating {missing} missing games from seed {seed}: {', '.join(opponents)}")
        def flush_opponent(opp_name: str, result: dict) -> None:
            row = _summarize_result({opp_name: result})[opp_name]
            final_summary[opp_name] = _merge_row(existing.get(opp_name), row)
            if args.out is not None:
                _write_results(args.out, args.checkpoint, requested, final_summary)
                print(f"\n  flushed {opp_name} to {args.out}")

        all_results.update(
            run_eval(
                args.checkpoint,
                opponents,
                missing,
                args.device,
                args.workers,
                seed,
                args.chunks_per_worker,
                args.lanes,
                args.start_method,
                on_opponent_done=flush_opponent,
            )
        )

    new_summary = _summarize_result(all_results)
    print()
    for opp_name in requested:
        result = all_results.get(opp_name)
        if result is not None:
            for label, r in result.items():
                print(f"  {args.checkpoint.name}  vs {opp_name}  ({label}, new): "
                      f"{r['win_rate']:.1%} ± {r['se']:.1%}  ({r['wins']}/{r['games']})")
            final_summary[opp_name] = _merge_row(existing.get(opp_name), new_summary[opp_name])

        row = final_summary.get(opp_name)
        if row is None:
            continue
        total_n = int(row["n_games"])
        total_w = int(row["wins"])
        for label in ("even", "odd"):
            wr_label = row.get(f"wr_{label}")
            if wr_label is None:
                continue
            seat_n = _seat_games(row, label)
            seat_w = _seat_wins(row, label)
            if seat_w is None:
                continue
            wr = seat_w / seat_n if seat_n else 0.0
            se = math.sqrt(wr * (1 - wr) / seat_n) if seat_n else 0.0
            print(f"  {args.checkpoint.name}  vs {opp_name}  ({label}): "
                  f"{wr:.1%} ± {se:.1%}  ({seat_w}/{seat_n})")
        wr = total_w / total_n if total_n else 0.0
        se = math.sqrt(wr * (1 - wr) / total_n) if total_n else 0.0
        print(f"  {args.checkpoint.name}  vs {opp_name}  (combined): "
              f"{wr:.1%} ± {se:.1%}  ({total_w}/{total_n})")

    if args.out is not None:
        _write_results(args.out, args.checkpoint, requested, final_summary)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
