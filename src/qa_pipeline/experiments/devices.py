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


MIN_TRAIN_FREE_MIB = 18432


def parse_trainable_gpus(text: str, min_free_mib: int = MIN_TRAIN_FREE_MIB, max_used_mib: int = 2048) -> list[int]:
    """index, memory.used, memory.free。占用低只表示没有其他大进程，空闲显存还要够 7B 训练。"""
    chosen: list[int] = []
    for line in (text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        used = parts[1].split()[0]
        free = parts[2].split()[0]
        if not used.isdigit() or not free.isdigit():
            continue
        if int(used) < max_used_mib and int(free) >= min_free_mib:
            chosen.append(int(parts[0]))
    return chosen


def query_gpu_table() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def select_trainable_gpus(
    requested: list[int] | None = None,
    min_free_mib: int = MIN_TRAIN_FREE_MIB,
    max_used_mib: int = 2048,
) -> tuple[list[int], str]:
    table = query_gpu_table()
    if not table.strip():
        return [], "无法查询 GPU 显存。占用低于 2GB 只表示没有其他大进程，不代表够训练 7B。"
    free = parse_trainable_gpus(table, min_free_mib=min_free_mib, max_used_mib=max_used_mib)
    if requested is None:
        if not free:
            return [], f"没有空闲显存达到 {min_free_mib} MiB 的 GPU。占用低于 2GB 只表示没有其他大进程。"
        return free, ""
    chosen = [index for index in requested if index in free]
    lacking = [index for index in requested if index not in free]
    if lacking or not chosen:
        return [], (
            f"GPU {lacking or requested} 空闲显存不足 {min_free_mib} MiB，或已有其他进程占用。"
            "占用低于 2GB 只表示卡上没有其他大进程。"
        )
    return chosen, ""


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
