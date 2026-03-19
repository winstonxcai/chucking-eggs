"""Supervised distillation: train Q-networks by imitating teacher bots."""

from __future__ import annotations

import logging
import multiprocessing as mp
from queue import Empty

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..game import GuanDanEnv
from .encoding import ACTION_DIM, D_MOVE, MAX_HISTORY, encode_action, encode_history, encode_state

log = logging.getLogger(__name__)


# ── Worker (CPU-only, runs in separate process) ────────────────────────────

def _worker(queue: mp.Queue, teacher_cls, level_rank: int, n_games: int) -> None:
    """Simulate games, encode decisions, push to shared queue."""
    teacher = teacher_cls(level_rank)
    env = GuanDanEnv(level_rank)

    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            legal = env.legal_moves(player)

            if len(legal) <= 1:
                env.step(legal[0])
                continue

            expert_action = teacher.act(env, player)
            expert_idx = _find_action_index(legal, expert_action)
            if expert_idx is None:
                env.step(expert_action)
                continue

            state_enc = encode_state(env, player)
            action_encs = np.array(
                [encode_action(m, env.hands[player], level_rank) for m in legal],
                dtype=np.float32,
            )
            history, hist_len = encode_history(env, player, level_rank)
            is_leading = env.current_trick is None

            queue.put({
                "state": state_enc,
                "action_encs": action_encs,
                "history": history,
                "hist_len": hist_len,
                "expert_idx": expert_idx,
                "is_leading": is_leading,
            })
            env.step(expert_action)

        queue.put("GAME_DONE")
    queue.put(None)  # sentinel


# ── Training (GPU, main process) ───────────────────────────────────────────

def _train_decision(item, q_lead, q_follow, opt_lead, opt_follow, batch_size, device,
                    lead_batch_count, follow_batch_count, stats):
    """Train on a single decision from a worker. Returns updated batch counts."""
    is_leading = item["is_leading"]
    q_net = q_lead if is_leading else q_follow
    action_encs = item["action_encs"]
    B = len(action_encs)

    s = (
        torch.tensor(item["state"], dtype=torch.float32)
        .unsqueeze(0).expand(B, -1).contiguous().to(device)
    )
    a = torch.tensor(action_encs).to(device)
    h = (
        torch.tensor(item["history"], dtype=torch.float32)
        .unsqueeze(0).expand(B, -1, -1).contiguous().to(device)
    )
    hl = torch.tensor([item["hist_len"]], dtype=torch.long).expand(B)

    q_values = q_net(s, a, h, hl)
    target = torch.tensor(item["expert_idx"], dtype=torch.long, device=device)
    loss = F.cross_entropy(q_values.unsqueeze(0), target.unsqueeze(0))
    (loss / batch_size).backward()

    predicted = q_values.argmax().item()

    if is_leading:
        stats["lead_losses"].append(loss.item())
        stats["lead_correct"] += predicted == item["expert_idx"]
        stats["lead_total"] += 1
        lead_batch_count += 1
        if lead_batch_count >= batch_size:
            torch.nn.utils.clip_grad_norm_(q_lead.parameters(), max_norm=1.0)
            opt_lead.step()
            opt_lead.zero_grad()
            lead_batch_count = 0
    else:
        stats["follow_losses"].append(loss.item())
        stats["follow_correct"] += predicted == item["expert_idx"]
        stats["follow_total"] += 1
        follow_batch_count += 1
        if follow_batch_count >= batch_size:
            torch.nn.utils.clip_grad_norm_(q_follow.parameters(), max_norm=1.0)
            opt_follow.step()
            opt_follow.zero_grad()
            follow_batch_count = 0

    return lead_batch_count, follow_batch_count


def _flush_grads(q_lead, q_follow, opt_lead, opt_follow, lead_batch_count, follow_batch_count):
    """Flush remaining accumulated gradients."""
    if lead_batch_count > 0:
        torch.nn.utils.clip_grad_norm_(q_lead.parameters(), max_norm=1.0)
        opt_lead.step()
        opt_lead.zero_grad()
    if follow_batch_count > 0:
        torch.nn.utils.clip_grad_norm_(q_follow.parameters(), max_norm=1.0)
        opt_follow.step()
        opt_follow.zero_grad()


def _log_epoch(epoch, epochs, stats):
    """Log epoch summary."""
    lead_acc = stats["lead_correct"] / stats["lead_total"] if stats["lead_total"] > 0 else 0
    follow_acc = stats["follow_correct"] / stats["follow_total"] if stats["follow_total"] > 0 else 0
    lead_avg = sum(stats["lead_losses"]) / len(stats["lead_losses"]) if stats["lead_losses"] else 0
    follow_avg = sum(stats["follow_losses"]) / len(stats["follow_losses"]) if stats["follow_losses"] else 0

    log.info(
        "Epoch %d/%d — lead: loss=%.4f acc=%.1f%% (%d) | follow: loss=%.4f acc=%.1f%% (%d)",
        epoch + 1, epochs,
        lead_avg, lead_acc * 100, stats["lead_total"],
        follow_avg, follow_acc * 100, stats["follow_total"],
    )


def train_supervised(
    q_lead,
    q_follow,
    opt_lead,
    opt_follow,
    teacher,
    n_games: int,
    level_rank: int,
    epochs: int = 3,
    batch_size: int = 32,
    device: str = "mps",
    n_workers: int = 0,
) -> None:
    """Train Q-networks by imitating teacher via cross-entropy.

    n_workers=0: single-threaded streaming (original path).
    n_workers>0: producer-consumer with N CPU workers generating games in parallel.
    """
    if n_workers > 0:
        _train_parallel(q_lead, q_follow, opt_lead, opt_follow,
                        type(teacher), n_games, level_rank,
                        epochs, batch_size, device, n_workers)
    else:
        _train_sequential(q_lead, q_follow, opt_lead, opt_follow,
                          teacher, n_games, level_rank,
                          epochs, batch_size, device)


def _train_batch(decisions, q_net, optimizer, device, stats, role):
    """Batched forward pass: N decisions with variable action counts, one GPU pass.

    Uses LSTM deduplication: encode history once per decision, expand to actions.
    """
    N = len(decisions)
    B_sizes = [len(d["action_encs"]) for d in decisions]
    max_B = max(B_sizes)

    # Pad actions to [N, max_B, ACTION_DIM], build mask [N, max_B]
    action_pad = torch.zeros(N, max_B, ACTION_DIM, dtype=torch.float32)
    mask = torch.zeros(N, max_B, dtype=torch.bool)
    targets = torch.zeros(N, dtype=torch.long)

    for i, d in enumerate(decisions):
        Bi = B_sizes[i]
        action_pad[i, :Bi] = torch.tensor(d["action_encs"], dtype=torch.float32)
        mask[i, :Bi] = True
        targets[i] = d["expert_idx"]

    # Stack states: [N, STATE_DIM]
    states = torch.stack([torch.tensor(d["state"], dtype=torch.float32) for d in decisions])

    # Pad histories to [N, MAX_HISTORY, D_MOVE] (variable T per decision)
    hists = torch.zeros(N, MAX_HISTORY, D_MOVE, dtype=torch.float32)
    hlens = torch.tensor([d["hist_len"] for d in decisions], dtype=torch.long)
    for i, d in enumerate(decisions):
        h = d["history"]
        T = min(len(h), MAX_HISTORY)
        hists[i, :T] = torch.tensor(h[:T], dtype=torch.float32)

    # Move to device
    states = states.to(device)
    hists = hists.to(device)
    action_pad = action_pad.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

    # LSTM once per decision: [N, T, 83] → [N, lstm_hidden]
    hist_emb = q_net.encode_history(hists, hlens)

    # Expand to match actions: [N, max_B, ...]
    states_exp = states.unsqueeze(1).expand(N, max_B, -1).reshape(N * max_B, -1)
    actions_flat = action_pad.reshape(N * max_B, -1)
    hist_exp = hist_emb.unsqueeze(1).expand(N, max_B, -1).reshape(N * max_B, -1)

    # Single MLP forward pass
    q_flat = q_net.forward_from_embedding(states_exp, actions_flat, hist_exp)
    q_matrix = q_flat.reshape(N, max_B)

    # Mask invalid actions
    q_matrix = q_matrix.masked_fill(~mask, float("-inf"))

    # Batched cross-entropy
    loss = F.cross_entropy(q_matrix, targets)
    loss.backward()

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    # Stats
    predictions = q_matrix.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    key = "lead" if role == "lead" else "follow"
    stats[f"{key}_losses"].append(loss.item())
    stats[f"{key}_correct"] += correct
    stats[f"{key}_total"] += N


def _train_parallel(q_lead, q_follow, opt_lead, opt_follow,
                    teacher_cls, n_games, level_rank,
                    epochs, batch_size, device, n_workers):
    """Producer-consumer with batched GPU training."""
    for epoch in range(epochs):
        stats = {
            "lead_losses": [], "follow_losses": [],
            "lead_correct": 0, "lead_total": 0,
            "follow_correct": 0, "follow_total": 0,
        }
        opt_lead.zero_grad()
        opt_follow.zero_grad()

        # Distribute games across workers
        games_per_worker = [n_games // n_workers] * n_workers
        for i in range(n_games % n_workers):
            games_per_worker[i] += 1

        queue = mp.Queue(maxsize=500)
        workers = []
        for i in range(n_workers):
            p = mp.Process(target=_worker, args=(queue, teacher_cls, level_rank, games_per_worker[i]))
            p.start()
            workers.append(p)

        sentinels = 0
        lead_buf, follow_buf = [], []

        with tqdm(total=n_games, desc=f"Epoch {epoch + 1}/{epochs}", unit="game", leave=True) as pbar:
            while sentinels < n_workers:
                try:
                    item = queue.get(timeout=60)
                except Empty:
                    alive = sum(1 for p in workers if p.is_alive())
                    if alive == 0 and sentinels < n_workers:
                        log.warning("All workers died before sending sentinels")
                        break
                    continue

                if item is None:
                    sentinels += 1
                    continue
                if item == "GAME_DONE":
                    pbar.update(1)
                    pbar.set_postfix(
                        l_acc=f"{stats['lead_correct'] / stats['lead_total']:.1%}" if stats["lead_total"] else "n/a",
                        f_acc=f"{stats['follow_correct'] / stats['follow_total']:.1%}" if stats["follow_total"] else "n/a",
                    )
                    continue

                if item["is_leading"]:
                    lead_buf.append(item)
                else:
                    follow_buf.append(item)

                if len(lead_buf) >= batch_size:
                    _train_batch(lead_buf[:batch_size], q_lead, opt_lead, device, stats, "lead")
                    lead_buf = lead_buf[batch_size:]
                if len(follow_buf) >= batch_size:
                    _train_batch(follow_buf[:batch_size], q_follow, opt_follow, device, stats, "follow")
                    follow_buf = follow_buf[batch_size:]

        # Flush remaining buffered decisions
        if lead_buf:
            _train_batch(lead_buf, q_lead, opt_lead, device, stats, "lead")
        if follow_buf:
            _train_batch(follow_buf, q_follow, opt_follow, device, stats, "follow")

        for p in workers:
            p.join(timeout=10)

        _log_epoch(epoch, epochs, stats)


def _train_sequential(q_lead, q_follow, opt_lead, opt_follow,
                      teacher, n_games, level_rank,
                      epochs, batch_size, device):
    """Original single-threaded streaming path."""
    env = GuanDanEnv(level_rank)

    for epoch in range(epochs):
        stats = {
            "lead_losses": [], "follow_losses": [],
            "lead_correct": 0, "lead_total": 0,
            "follow_correct": 0, "follow_total": 0,
        }

        opt_lead.zero_grad()
        opt_follow.zero_grad()
        lead_batch_count = 0
        follow_batch_count = 0

        with tqdm(total=n_games, desc=f"Epoch {epoch + 1}/{epochs}", unit="game", leave=True) as pbar:
            for _ in range(n_games):
                env.reset()

                while not env.done:
                    player = env.current_player
                    legal = env.legal_moves(player)

                    if len(legal) <= 1:
                        env.step(legal[0])
                        continue

                    expert_action = teacher.act(env, player)
                    expert_idx = _find_action_index(legal, expert_action)
                    if expert_idx is None:
                        env.step(expert_action)
                        continue

                    history, hist_len = encode_history(env, player, level_rank)
                    item = {
                        "state": encode_state(env, player),
                        "action_encs": np.array(
                            [encode_action(m, env.hands[player], level_rank) for m in legal],
                            dtype=np.float32,
                        ),
                        "history": history,
                        "hist_len": hist_len,
                        "expert_idx": expert_idx,
                        "is_leading": env.current_trick is None,
                    }

                    lead_batch_count, follow_batch_count = _train_decision(
                        item, q_lead, q_follow, opt_lead, opt_follow,
                        batch_size, device, lead_batch_count, follow_batch_count, stats,
                    )
                    env.step(expert_action)

                pbar.update(1)
                pbar.set_postfix(
                    l_acc=f"{stats['lead_correct'] / stats['lead_total']:.1%}" if stats["lead_total"] else "n/a",
                    f_acc=f"{stats['follow_correct'] / stats['follow_total']:.1%}" if stats["follow_total"] else "n/a",
                )

        _flush_grads(q_lead, q_follow, opt_lead, opt_follow, lead_batch_count, follow_batch_count)
        _log_epoch(epoch, epochs, stats)


def _find_action_index(legal_moves, chosen_action) -> int | None:
    """Find index of chosen_action in legal_moves by matching type + cards."""
    chosen_cards = {(c.rank, c.suit, c.deck) for c in chosen_action.cards}
    chosen_type = chosen_action.type
    for i, m in enumerate(legal_moves):
        if m.type != chosen_type:
            continue
        if {(c.rank, c.suit, c.deck) for c in m.cards} == chosen_cards:
            return i
    return None
