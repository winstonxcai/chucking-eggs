"""Validate a Docker host before starting the A800 DART training recipe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

import guandan_rs
import torch

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class PreflightExpectations:
    gpu_name: str = "NVIDIA A800-SXM4-80GB"
    min_vram_mib: int = 80_000
    min_driver: str = "580.65.06"
    torch_cuda: str = "12.8"
    min_cpus: int = 128
    min_ram_gib: int = 128


@dataclass(frozen=True)
class PreflightReport:
    torch_version: str
    torch_cuda: str | None
    cuda_available: bool
    cuda_device_count: int
    gpu_name: str | None
    gpu_vram_mib: int | None
    compute_capability: tuple[int, int] | None
    bf16_supported: bool
    smi_index: int | None
    driver_version: str | None
    persistence_mode: str | None
    compute_mode: str | None
    mig_mode: str | None
    cpu_count: int
    ram_gib: int
    native_move_generator: bool
    run_dir: str
    run_dir_writable: bool


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _ram_gib() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return 0
    return (pages * page_size) // GIB


def _directory_is_writable(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".dart-preflight-"):
            pass
    except OSError:
        return False
    return True


def _query_nvidia_smi() -> dict[str, str | int | None]:
    fields = (
        "index,name,memory.total,driver_version,persistence_mode,"
        "compute_mode,mig.mode.current"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(StringIO(result.stdout), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"unexpected nvidia-smi response: {result.stdout!r}")
    row = [value.strip() for value in rows[0]]
    return {
        "smi_index": int(row[0]),
        "driver_version": row[3],
        "persistence_mode": row[4],
        "compute_mode": row[5],
        "mig_mode": row[6],
    }


def collect_report(run_dir: Path) -> PreflightReport:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name: str | None = None
    gpu_vram_mib: int | None = None
    capability: tuple[int, int] | None = None
    bf16_supported = False
    smi: dict[str, str | int | None] = {
        "smi_index": None,
        "driver_version": None,
        "persistence_mode": None,
        "compute_mode": None,
        "mig_mode": None,
    }

    if cuda_available and device_count:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_vram_mib = props.total_memory // MIB
        capability = torch.cuda.get_device_capability(0)
        bf16_supported = torch.cuda.is_bf16_supported()
        try:
            smi = _query_nvidia_smi()
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError):
            pass

    return PreflightReport(
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        gpu_name=gpu_name,
        gpu_vram_mib=gpu_vram_mib,
        compute_capability=capability,
        bf16_supported=bf16_supported,
        smi_index=int(smi["smi_index"]) if smi["smi_index"] is not None else None,
        driver_version=(
            str(smi["driver_version"]) if smi["driver_version"] is not None else None
        ),
        persistence_mode=(
            str(smi["persistence_mode"])
            if smi["persistence_mode"] is not None
            else None
        ),
        compute_mode=(
            str(smi["compute_mode"]) if smi["compute_mode"] is not None else None
        ),
        mig_mode=str(smi["mig_mode"]) if smi["mig_mode"] is not None else None,
        cpu_count=os.cpu_count() or 0,
        ram_gib=_ram_gib(),
        native_move_generator=bool(getattr(guandan_rs, "HAS_NATIVE", False)),
        run_dir=str(run_dir),
        run_dir_writable=_directory_is_writable(run_dir),
    )


def validate_report(
    report: PreflightReport,
    expected: PreflightExpectations,
) -> list[str]:
    errors: list[str] = []
    if not report.cuda_available:
        errors.append("PyTorch cannot access CUDA")
    if report.cuda_device_count != 1:
        errors.append(f"expected exactly 1 visible CUDA device, got {report.cuda_device_count}")
    if report.gpu_name is None or expected.gpu_name not in report.gpu_name:
        errors.append(f"expected GPU {expected.gpu_name!r}, got {report.gpu_name!r}")
    if report.gpu_vram_mib is None or report.gpu_vram_mib < expected.min_vram_mib:
        errors.append(
            f"expected at least {expected.min_vram_mib} MiB VRAM, "
            f"got {report.gpu_vram_mib!r}"
        )
    if report.compute_capability is None or report.compute_capability < (8, 0):
        errors.append(
            f"expected Ampere compute capability >= (8, 0), got {report.compute_capability!r}"
        )
    if not report.bf16_supported:
        errors.append("CUDA device does not support BF16")
    if report.torch_cuda != expected.torch_cuda:
        errors.append(
            f"expected PyTorch CUDA {expected.torch_cuda}, got {report.torch_cuda!r}"
        )
    if report.smi_index != 0:
        errors.append(f"expected nvidia-smi device index 0, got {report.smi_index!r}")
    if report.driver_version is None:
        errors.append("could not query the NVIDIA driver with nvidia-smi")
    elif _version_tuple(report.driver_version) < _version_tuple(expected.min_driver):
        errors.append(
            f"expected NVIDIA driver >= {expected.min_driver}, got {report.driver_version}"
        )
    if report.persistence_mode != "Enabled":
        errors.append(f"expected persistence mode Enabled, got {report.persistence_mode!r}")
    if report.compute_mode != "Default":
        errors.append(f"expected compute mode Default, got {report.compute_mode!r}")
    if report.mig_mode != "Disabled":
        errors.append(f"expected MIG mode Disabled, got {report.mig_mode!r}")
    if report.cpu_count < expected.min_cpus:
        errors.append(f"expected at least {expected.min_cpus} CPUs, got {report.cpu_count}")
    if report.ram_gib < expected.min_ram_gib:
        errors.append(f"expected at least {expected.min_ram_gib} GiB RAM, got {report.ram_gib}")
    if not report.native_move_generator:
        errors.append("guandan_rs native move generator is unavailable")
    if not report.run_dir_writable:
        errors.append(f"run directory is not writable: {report.run_dir}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/runs"))
    parser.add_argument("--expected-gpu", default="NVIDIA A800-SXM4-80GB")
    parser.add_argument("--min-vram-mib", type=int, default=80_000)
    parser.add_argument("--min-driver", default="580.65.06")
    parser.add_argument("--expected-torch-cuda", default="12.8")
    parser.add_argument("--min-cpus", type=int, default=128)
    parser.add_argument("--min-ram-gib", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expected = PreflightExpectations(
        gpu_name=args.expected_gpu,
        min_vram_mib=args.min_vram_mib,
        min_driver=args.min_driver,
        torch_cuda=args.expected_torch_cuda,
        min_cpus=args.min_cpus,
        min_ram_gib=args.min_ram_gib,
    )
    report = collect_report(args.run_dir)
    print(json.dumps(asdict(report), indent=2))
    errors = validate_report(report, expected)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("A800 training preflight passed")


if __name__ == "__main__":
    main()
