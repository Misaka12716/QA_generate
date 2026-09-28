"""把管线产物写成可复现的 run 目录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import PipelineResult
from .schemas import QAPair


def json_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_line(row))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_result(result: PipelineResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "chunks.jsonl", [c.model_dump() for c in result.chunks])
    write_jsonl(out / "questions.jsonl", [q.model_dump() for q in result.questions])
    write_jsonl(out / "qa.kept.jsonl", [p.model_dump() for p in result.pairs])
    write_jsonl(out / "qa.rejected.jsonl", [p.model_dump() for p in result.rejected])
    write_jsonl(out / "qa.raw.jsonl", [p.model_dump() for p in result.raw_pairs])
    (out / "stats.json").write_text(
        json.dumps(result.stats.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "recipe": result.recipe_name,
                "kept": len(result.pairs),
                "rejected": len(result.rejected),
                "chunks": len(result.chunks),
                "questions": len(result.questions),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def load_pairs(path: str | Path) -> list[QAPair]:
    return [QAPair.model_validate(row) for row in read_jsonl(Path(path))]
