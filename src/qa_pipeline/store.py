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
    _write_events(result, out)
    return out


def _write_events(result: PipelineResult, out: Path) -> None:
    from .observability import stage_count_balanced

    events: list[dict[str, Any]] = []
    seq = 0
    funnel = getattr(result.stats, "funnel", None) or {}
    for name, row in funnel.items():
        seq += 1
        inn = row.get("in") if isinstance(row, dict) else None
        out_n = row.get("out") if isinstance(row, dict) else None
        removed = row.get("dropped") if isinstance(row, dict) else None
        events.append(
            {
                "event_id": f"evt_{seq:04d}",
                "seq": seq,
                "event_type": "stage_count",
                "stage_key": name,
                "input_n": inn,
                "created_n": 0,
                "output_n": out_n,
                "removed_n": removed,
                "count_unit": "qa",
                "balance_ok": stage_count_balanced(inn, 0, out_n, removed),
                "duration_sec": None,
                "itemized_reasons": False,
            }
        )
    kept_ids = {pair.qa_id for pair in result.pairs}
    for pair in [*result.pairs, *result.rejected]:
        seq += 1
        reasons = []
        trace = pair.filter_trace or {}
        if isinstance(trace, dict):
            for key, payload in trace.items():
                if not isinstance(payload, dict):
                    continue
                action = payload.get("action")
                reason = payload.get("reason")
                if reason or action not in {None, "pass"}:
                    reasons.append({"filter": key, "action": action, "reason": reason})
        events.append(
            {
                "event_id": f"evt_{seq:04d}",
                "seq": seq,
                "event_type": "sample_disposition",
                "entity_id": pair.qa_id,
                "terminal_disposition": "kept" if pair.qa_id in kept_ids else "rejected",
                "reason_codes": reasons,
                "reason_source": "filter_trace",
            }
        )
    with (out / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json_line(event))


def load_pairs(path: str | Path) -> list[QAPair]:
    return [QAPair.model_validate(row) for row in read_jsonl(Path(path))]
