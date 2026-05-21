from __future__ import annotations

import pytest

from guandan.dart.config import EpsilonConfig
from guandan.dart.utils.schedules import epsilon_linear


def test_epsilon_linear_interpolates_and_clamps():
    cfg = EpsilonConfig(start=1.0, final=0.1, decay_updates=10)

    assert epsilon_linear(0, cfg) == 1.0
    assert epsilon_linear(5, cfg) == 0.55
    assert epsilon_linear(10, cfg) == pytest.approx(0.1)
    assert epsilon_linear(20, cfg) == pytest.approx(0.1)


def test_epsilon_linear_nonpositive_decay_returns_final():
    assert epsilon_linear(0, EpsilonConfig(start=1.0, final=0.2, decay_updates=0)) == 0.2


def test_epsilon_linear_zero_decay_is_constant_for_fixed_epsilon():
    """M4 sets decay_updates=0 with start==final to lock ε at a constant; verify
    no division-by-zero or drift across the full update range."""
    cfg = EpsilonConfig(start=0.02, final=0.02, decay_updates=0)
    for u in (0, 1, 100, 5_000, 100_000, 10_000_000):
        assert epsilon_linear(u, cfg) == 0.02


def test_epsilon_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="epsilon.start"):
        EpsilonConfig(start=1.1, final=0.1)
    with pytest.raises(ValueError, match="epsilon.start must be >= epsilon.final"):
        EpsilonConfig(start=0.1, final=0.2)
    with pytest.raises(ValueError, match="epsilon.decay_updates"):
        EpsilonConfig(decay_updates=-1)
