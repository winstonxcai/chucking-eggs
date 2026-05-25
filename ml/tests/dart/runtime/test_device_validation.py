from __future__ import annotations

import pytest
import torch
from guandan.dart.config import TrainConfig
from guandan.dart.runtime.train import _validate_config_devices


def test_cuda_config_fails_fast_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="requires CUDA"):
        _validate_config_devices(TrainConfig(device="cuda"))


def test_mps_config_fails_fast_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="requires Apple MPS"):
        _validate_config_devices(TrainConfig(device="mps"))


def test_eval_device_is_validated_when_eval_enabled(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = TrainConfig.from_flat_dict({
        "device": "cpu",
        "eval": {"enabled": True, "device": "cuda"},
    })

    with pytest.raises(ValueError, match="eval.device=.*requires CUDA"):
        _validate_config_devices(cfg)
