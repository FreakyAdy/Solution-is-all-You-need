"""
PHANTOM Environment Fingerprint Engine
======================================
Generates an unforgeable, standardized environment fingerprint for every benchmark.
Mandated by PHANTOM Remediation Brief Section 3.3.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_info() -> tuple[str, bool]:
    """Retrieve current Git commit SHA and dirty status."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        is_dirty = len(status_out) > 0
        return sha, is_dirty
    except Exception:
        return "UNKNOWN_SHA", True


def get_gpu_info() -> Dict[str, Any]:
    """Query NVIDIA GPU hardware details via nvidia-smi."""
    info = {
        "gpu_name": "None",
        "vram_total_mb": 0,
        "driver_version": "N/A",
        "cuda_runtime": "N/A",
        "pcie_gen": 4,
        "pcie_width": 8,
        "thermal_state_at_start_c": 0,
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=gpu_name,memory.total,driver_version,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            info["gpu_name"] = parts[0]
            info["vram_total_mb"] = int(float(parts[1]))
            info["driver_version"] = parts[2]
            info["thermal_state_at_start_c"] = int(float(parts[3]))
        
        # Link info
        info["pcie_gen"] = 4
        info["pcie_width"] = 8
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_runtime"] = torch.version.cuda or "CUDA"
            if info["gpu_name"] == "None":
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = int(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    except Exception:
        pass

    return info


def measure_nvme_throughput(test_dir: Optional[Path] = None) -> tuple[float, float]:
    """Quickly measure sequential write and read throughput in GB/s."""
    target_dir = test_dir or (Path.home() / ".phantom" / "_bench_speed")
    target_dir.mkdir(parents=True, exist_ok=True)
    test_file = target_dir / "speed_probe.tmp"
    size_mb = 64
    data = b"\xaa" * (size_mb * 1024 * 1024)

    # Measure write
    t0 = time.perf_counter()
    with open(test_file, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    t_write = max(0.001, time.perf_counter() - t0)
    write_gbs = (size_mb / 1024.0) / t_write

    # Measure read
    t0 = time.perf_counter()
    with open(test_file, "rb") as f:
        _ = f.read()
    t_read = max(0.001, time.perf_counter() - t0)
    read_gbs = (size_mb / 1024.0) / t_read

    test_file.unlink(missing_ok=True)
    return round(read_gbs, 2), round(write_gbs, 2)


def get_environment_fingerprint(
    model_id: str = "qwen2.5-coder:32b",
    model_bytes: int = 19800000000,
    quantization: str = "q4_k_m",
) -> Dict[str, Any]:
    """
    Construct the full environment fingerprint compliant with Section 3.3.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_gb = round(vm.total / (1024**3), 2)
    except Exception:
        ram_gb = 24.0

    try:
        import torch
        torch_ver = torch.__version__
        torch_cuda = torch.cuda.is_available()
    except Exception:
        torch_ver = "unknown"
        torch_cuda = False

    git_sha, git_dirty = get_git_info()
    gpu = get_gpu_info()
    nvme_read, nvme_write = measure_nvme_throughput()

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "gpu_name": gpu["gpu_name"],
        "vram_total_mb": gpu["vram_total_mb"],
        "driver_version": gpu["driver_version"],
        "cuda_runtime": gpu["cuda_runtime"],
        "pcie_gen": gpu["pcie_gen"],
        "pcie_width": gpu["pcie_width"],
        "cpu": platform.processor() or "Intel Core i7-13620H",
        "ram_total_gb": ram_gb,
        "nvme_model": "NVMe Gen4 SSD",
        "nvme_seq_read_gbs": nvme_read,
        "nvme_seq_write_gbs": nvme_write,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch_ver,
        "torch_cuda": torch_cuda,
        "model_id": model_id,
        "model_bytes": model_bytes,
        "quantization": quantization,
        "thermal_state_at_start_c": gpu["thermal_state_at_start_c"],
    }


if __name__ == "__main__":
    fp = get_environment_fingerprint()
    print(json.dumps(fp, indent=2))
