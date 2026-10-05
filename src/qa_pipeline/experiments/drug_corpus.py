"""校医院说明书的清洗、分区和冻结子集。不把子集写成全库结论。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..pipeline import document_from_file
from ..plugins.chunking import HeadingWindowChunking
from ..textutil import sentences


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def raw_dir() -> Path:
    return repo_root() / "data" / "campus_hospital_drug_instructions" / "raw"


def frozen_dir() -> Path:
    return repo_root() / "data" / "campus_hospital_drug_instructions" / "frozen"


def _is_pdf(path: Path) -> bool:
    try:
        head = path.read_bytes()[:8]
    except OSError:
        return True
    return head.startswith(b"%PDF")


def _split_name(stem: str) -> str:
    bucket = int(hashlib.sha256(stem.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        return "locked_test"
    if bucket == 1:
        return "dev"
    return "train"


def inventory() -> dict:
    rows = []
    for path in sorted(raw_dir().glob("*.txt")):
        kind = "unparsed_pdf" if _is_pdf(path) else "parsed_text"
        text = "" if kind != "parsed_text" else path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "stem": path.stem,
                "path": str(path),
                "status": kind,
                "chars": len(text),
                "split": None if kind != "parsed_text" else _split_name(path.stem),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        key = row["split"] or row["status"]
        counts[key] = counts.get(key, 0) + 1
    return {"documents": rows, "counts": counts, "total": len(rows)}


def select_subset(rows: list[dict], size: int = 60) -> list[dict]:
    train = [row for row in rows if row["split"] == "train"]
    train.sort(key=lambda row: row["chars"])
    if not train:
        return []
    quartiles = [[] for _ in range(4)]
    for index, row in enumerate(train):
        quartiles[min(3, index * 4 // len(train))].append(row)
    picked = []
    per = max(1, size // 4)
    for group in quartiles:
        step = max(1, len(group) // per)
        picked.extend(group[::step][:per])
    return picked[:size]


def write_subset(picked: list[dict]) -> Path:
    subset = frozen_dir() / "subset"
    if subset.exists():
        for child in subset.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
    subset.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(picked):
        link = subset / f"{index:03d}_{row['stem']}.txt"
        target = Path(row["path"])
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        manifest.append({"order": index, "stem": row["stem"], "chars": row["chars"], "source": row["path"]})
    (frozen_dir() / "subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return subset


def write_catalog(subset: Path, limit_chunks: int = 100) -> Path:
    docs = [document_from_file(path) for path in sorted(subset.glob("*.txt"))]
    chunks = HeadingWindowChunking(max_tokens=512, overlap=0.1).run(docs, None)[:limit_chunks]
    catalog = []
    for chunk in chunks:
        for sentence in sentences(chunk.text):
            if len(sentence) < 24:
                continue
            unit_id = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:16]
            catalog.append(
                {
                    "catalog_version": "drug-subset-v1",
                    "catalog_role": "dev_eval",
                    "source_group": chunk.source_doc,
                    "split": "train",
                    "reference_unit_id": f"ref_{unit_id}",
                    "description": sentence,
                    "evidence_location": chunk.chunk_id,
                    "business_weight": 1,
                    "verification_status": "substring_checked",
                }
            )
            if len(catalog) >= 80:
                break
        if len(catalog) >= 80:
            break
    path = frozen_dir() / "coverage_catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in catalog:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256("\n".join(chunk.chunk_id for chunk in chunks).encode("utf-8")).hexdigest()
    (frozen_dir() / "chunk_freeze.json").write_text(
        json.dumps(
            {
                "chunk_count": len(chunks),
                "catalog_units": len(catalog),
                "manifest_hash": digest,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "note": "目录由切块中的原文句子独立建立，并做了子串核对。不是人工金标，也不是路线 K 的 unit_id。各生成臂使用同一份切块参数，并停在这份清单的前 100 块。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_heldout(rows: list[dict], size: int = 16) -> Path:
    locked = [row for row in rows if row["split"] == "locked_test" and row["chars"] > 400]
    locked.sort(key=lambda row: row["stem"])
    items = []
    for row in locked:
        if len(items) >= size:
            break
        text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        if text.startswith("%PDF"):
            continue
        usable = [sent for sent in sentences(text) if 30 <= len(sent) <= 180]
        if not usable:
            continue
        answer = usable[len(usable) // 2]
        start = max(0, text.find(answer) - 200)
        context = text[start : start + 900]
        if answer not in context:
            context = answer
        other = locked[(locked.index(row) + 3) % len(locked)]
        distractor = Path(other["path"]).read_text(encoding="utf-8", errors="replace")[:900]
        if answer in distractor:
            distractor = "本品说明书未提供与问题对应的内容。"
        items.append(
            {
                "id": f"drug-heldout-{len(items)+1}",
                "question": "根据给定资料，说明这一句表述的具体内容。",
                "answer": answer,
                "context": context,
                "distractor": distractor,
                "source": row["stem"],
            }
        )
    path = frozen_dir() / "heldout.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def find_conflicts(rows: list[dict]) -> list[dict]:
    """同一文件名标题下用法用量不一致才记为已确认冲突。不据此自动出题。"""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] != "train" or row["status"] != "parsed_text":
            continue
        text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("URL:")]
        title = lines[0][:40] if lines else row["stem"]
        dosage = ""
        for line in lines:
            if "用法" in line or "用量" in line:
                dosage = line[:120]
                break
        if title and dosage:
            grouped.setdefault(title, []).append({"stem": row["stem"], "dosage": dosage})
    conflicts = []
    for title, items in grouped.items():
        doses = {item["dosage"] for item in items}
        if len(items) >= 2 and len(doses) >= 2:
            conflicts.append({"title": title, "items": items})
    return conflicts


def prepare() -> dict:
    frozen_dir().mkdir(parents=True, exist_ok=True)
    data = inventory()
    picked = select_subset(data["documents"], 60)
    subset = write_subset(picked)
    catalog = write_catalog(subset)
    heldout = write_heldout(data["documents"])
    conflicts = find_conflicts(data["documents"])
    (frozen_dir() / "conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "counts": data["counts"],
        "subset_docs": len(picked),
        "confirmed_conflicts": len(conflicts),
        "subset": str(subset),
        "catalog": str(catalog),
        "heldout": str(heldout),
        "heldout_n": sum(1 for _ in heldout.read_text(encoding="utf-8").splitlines() if _.strip()),
    }
    (frozen_dir() / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (frozen_dir() / "inventory.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in data["documents"]) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    os.chdir(repo_root())
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
