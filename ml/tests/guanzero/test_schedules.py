from __future__ import annotations

import pytest

from guandan.guanzero.config import EpsilonConfig
from guandan.guanzero.schedules import epsilon_linear


def test_epsilon_linear_interpolates_and_clamps():
    cfg = EpsilonConfig(start=1.0, final=0.1, decay_updates=10)

    assert epsilon_linear(0, cfg) == 1.0
    assert epsilon_linear(5, cfg) == 0.55
    assert epsilon_linear(10, cfg) == pytest.approx(0.1)
    assert epsilon_linear(20, cfg) == pytest.approx(0.1)


def test_epsilon_linear_nonpositive_decay_returns_final():
    assert epsilon_linear(0, EpsilonConfig(start=1.0, final=0.2, decay_updates=0)) == 0.2
