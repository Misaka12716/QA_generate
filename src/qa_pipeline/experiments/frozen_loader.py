"""按冻结清单读取说明书正文。不改写子集，也不把元数据 JSON 当正文。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..textutil import CORPUS_METADATA_FILENAMES


class FrozenLoadError(Exception):
    """清单、路径或哈希无法支持本次读取。"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_frozen_subset(subset_dir: str | Path, manifest_path: str | Path) -> list[dict]:
    """读取 subset_manifest 指向的 txt。路径不存在或哈希不符时失败。"""
    subset = Path(subset_dir)
    manifest_file = Path(manifest_path)
    if not subset.is_dir():
        raise FrozenLoadError(f"missing_subset:{subset}")
    if not manifest_file.is_file():
        raise FrozenLoadError(f"missing_manifest:{manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise FrozenLoadError("manifest_not_list")
    rows = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise FrozenLoadError("manifest_entry_invalid")
        stem = str(entry.get("stem") or "")
        order = entry.get("order")
        if order is None or not stem:
            raise FrozenLoadError("manifest_missing_identity")
        name = f"{int(order):03d}_{stem}.txt"
        if name in CORPUS_METADATA_FILENAMES or name.endswith(".json"):
            raise FrozenLoadError(f"metadata_as_body:{name}")
        path = subset / name
        if not path.is_file():
            raise FrozenLoadError(f"missing_file:{name}")
        data = path.read_bytes()
        digest = _sha256(data)
        expected = entry.get("source_hash") or entry.get("content_sha256")
        if expected and expected != digest:
            raise FrozenLoadError(f"hash_mismatch:{name}")
        rows.append(
            {
                "order": int(order),
                "stem": stem,
                "source_id": stem,
                "source": entry.get("source") or "",
                "chars": entry.get("chars"),
                "path": str(path),
                "content_sha256": digest,
                "text": data.decode("utf-8", errors="replace"),
            }
        )
    return rows
