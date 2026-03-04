"""Circular replay buffer for DMC training."""

from __future__ import annotations

import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.pos = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward))
        else:
            self.buffer[self.pos] = (state, action, reward)
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards = zip(*batch)
        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.float32)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        if device is not None:
            s, a, r = s.to(device), a.to(device), r.to(device)
        return s, a, r

    def __len__(self) -> int:
        return len(self.buffer)
