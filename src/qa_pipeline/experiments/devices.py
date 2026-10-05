"""挑选空闲 GPU，避免占用已被其他进程使用的卡。"""

from __future__ import annotations

import subprocess


def parse_gpu_table(text: str, max_used_mib: int = 2048) -> list[int]:
    """解析 nvidia-smi CSV（index, memory.used），返回显存占用低于阈值的卡号。"""
    free: list[int] = []
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        used = parts[1].split()[0]
        if not used.isdigit():
            continue
        if int(used) < max_used_mib:
            free.append(int(parts[0]))
    return free


def detect_free_gpus(max_used_mib: int = 2048) -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return parse_gpu_table(out, max_used_mib=max_used_mib)
