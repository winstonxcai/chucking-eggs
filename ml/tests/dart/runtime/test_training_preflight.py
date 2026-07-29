from __future__ import annotations

import dataclasses

import pytest
from guandan.dart.utils.training_preflight import (
    PreflightExpectations,
    PreflightReport,
    validate_report,
)


def _valid_report() -> PreflightReport:
    return PreflightReport(
        torch_version="2.10.0",
        torch_cuda="12.8",
        cuda_available=True,
        cuda_device_count=1,
        gpu_name="NVIDIA A800-SXM4-80GB",
        gpu_vram_mib=81_920,
        compute_capability=(8, 0),
        bf16_supported=True,
        smi_index=0,
        driver_version="580.65.06",
        persistence_mode="Enabled",
        compute_mode="Default",
        mig_mode="Disabled",
        cpu_count=128,
        ram_gib=768,
        native_move_generator=True,
        run_dir="/runs",
        run_dir_writable=True,
    )


def test_valid_a800_report_passes():
    assert validate_report(_valid_report(), PreflightExpectations()) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cuda_available": False}, "cannot access CUDA"),
        ({"cuda_device_count": 2}, "exactly 1 visible CUDA device"),
        ({"gpu_name": "NVIDIA L4"}, "expected GPU"),
        ({"gpu_vram_mib": 40_960}, "at least 80000 MiB VRAM"),
        ({"compute_capability": (7, 5)}, "Ampere compute capability"),
        ({"bf16_supported": False}, "does not support BF16"),
        ({"torch_cuda": "13.0"}, "expected PyTorch CUDA 12.8"),
        ({"smi_index": 1}, "device index 0"),
        ({"driver_version": "570.00.00"}, "driver >= 580.65.06"),
        ({"persistence_mode": "Disabled"}, "persistence mode Enabled"),
        ({"compute_mode": "Exclusive_Process"}, "compute mode Default"),
        ({"mig_mode": "Enabled"}, "MIG mode Disabled"),
        ({"cpu_count": 64}, "at least 128 CPUs"),
        ({"ram_gib": 96}, "at least 128 GiB RAM"),
        ({"native_move_generator": False}, "native move generator"),
        ({"run_dir_writable": False}, "run directory is not writable"),
    ],
)
def test_invalid_a800_report_explains_failure(changes, message):
    report = dataclasses.replace(_valid_report(), **changes)

    errors = validate_report(report, PreflightExpectations())

    assert any(message in error for error in errors)


def test_newer_driver_is_accepted():
    report = dataclasses.replace(_valid_report(), driver_version="590.10.2")

    assert validate_report(report, PreflightExpectations()) == []
