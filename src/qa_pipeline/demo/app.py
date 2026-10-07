"""对照台 HTTP 服务。只读实验目录，不触发新的生成。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
import uuid
from datetime import datetime, timezone

class _Static(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


STATIC = Path(__file__).resolve().parent / "static"
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CLIP_KEYS = ("question", "answer", "evidence_span", "chunk_text")
_SAMPLE_KEYS = (
    "qa_id",
    "question",
    "answer",
    "evidence_span",
    "chunk_text",
    "grade",
    "q_type",
    "intent_primary",
    "evidence_state",
    "expected_action",
    "selection_role",
    "goal",
    "nli_score",
    "judge_overall",
    "kb_gain",
    "action",
    "filter_trace",
    "evolution_type",
)

COMPARE_GROUPS = {
    "e1": ["E1_baseline", "E1_ours_full", "E1_ours_s_only"],
    "e2": ["E2_a1", "E2_a2", "E2_a3", "E2_a4", "E2_a5"],
    "e3": ["E3_b1", "E3_b2", "E3_b3", "E3_b4"],
    "e4": ["E4_c1", "E4_c2", "E4_c3", "E4_c4", "E4_c5"],
}


def create_app(run_dir: str | Path) -> FastAPI:
    root = Path(run_dir).resolve()
    app = FastAPI(title="QA 方案对照台", docs_url=None, redoc_url=None)
    app.mount("/static", _Static(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/results")
    def results_page() -> FileResponse:
        return FileResponse(STATIC / "results.html")

    @app.get("/api/suite")
    def suite() -> dict[str, Any]:
        return _load_suite(root)

    @app.get("/api/experiments/{exp_id}")
    def experiment(
        exp_id: str,
        status: str = Query(default="kept", pattern="^(kept|rejected)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=12, ge=1, le=50),
        clip: int = Query(default=480, ge=80, le=4000),
    ) -> dict[str, Any]:
        _require_id(exp_id)
        payload = _load_suite(root)
        if not any(row.get("id") == exp_id for row in payload.get("experiments") or []):
            raise HTTPException(status_code=404, detail=f"未知实验 {exp_id}")
        path = root / exp_id / ("qa.kept.jsonl" if status == "kept" else "qa.rejected.jsonl")
        rows = _read_jsonl(path)
        page = rows[offset : offset + limit]
        return {
            "id": exp_id,
            "status": status,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "samples": [_project(row, clip) for row in page],
        }

    @app.get("/api/compare")
    def compare(
        group: str = Query(default="e1", pattern="^(e1|e2|e3|e4)$"),
        status: str = Query(default="kept", pattern="^(kept|rejected)$"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=4, ge=1, le=20),
        clip: int = Query(default=360, ge=80, le=4000),
    ) -> dict[str, Any]:
        payload = _load_suite(root)
        purposes = {row.get("id"): row.get("purpose") or "" for row in payload.get("experiments") or []}
        ids = COMPARE_GROUPS[group]
        align = "question" if group == "e3" else ("overlap" if group == "e1" else "chunk")
        per_exp = {
            exp_id: [_project(row, clip) for row in _read_jsonl(root / exp_id / f"qa.{status}.jsonl")]
            for exp_id in ids
        }
        rows = align_rows(per_exp, align)
        page = rows[offset : offset + limit]
        return {
            "group": group,
            "align": align,
            "status": status,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "columns": [{"id": exp_id, "purpose": purposes.get(exp_id, "")} for exp_id in ids],
            "rows": [
                {
                    "source": _clip_text(row["source"], clip),
                    "cells": row["cells"],
                }
                for row in page
            ],
        }

    @app.get("/api/v1/state")
    def state_v1() -> Any:
        path = root / "state_snapshot.json"
        if not path.is_file():
            return _api_error("missing_state", "没有 state_snapshot.json", 404)
        return {"data": json.loads(path.read_text(encoding="utf-8")), "meta": _api_meta(total=1)}

    @app.get("/api/v1/runs")
    def runs_v1(cursor: str = Query(default="0"), limit: int = Query(default=20, ge=1, le=100)) -> Any:
        start, error = _cursor_index(cursor)
        if error is not None:
            return error
        names = sorted(path.name for path in root.iterdir() if path.is_dir() and _looks_like_run(path)) if root.is_dir() else []
        page = names[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(names) else None
        return {
            "data": [{"run_id": name, "availability": "present" if (root / name / "events.jsonl").is_file() else "historical_partial"} for name in page],
            "meta": _api_meta(total=len(names), next_cursor=next_cursor),
        }

    @app.get("/api/v1/review-batches")
    def review_batches(cursor: str = Query(default="0"), limit: int = Query(default=20, ge=1, le=100)) -> Any:
        path = root / "review_aggregates.jsonl"
        if not path.is_file():
            meta = _api_meta(total=0)
            meta["availability"] = "historical_partial"
            meta["missing_fields"] = ["review_aggregates.jsonl"]
            return {"data": [], "meta": meta}
        start, error = _cursor_index(cursor)
        if error is not None:
            return error
        rows, total = _stream_page(path, start, limit)
        next_cursor = str(start + limit) if start + limit < total else None
        meta = _api_meta(total=total, next_cursor=next_cursor)
        meta["availability"] = "present"
        return {"data": rows, "meta": meta}

    @app.get("/api/v1/result-cases")
    def result_cases(
        pattern: str = "",
        rule: str = "",
        review: str = "",
        block: str = "",
    ) -> Any:
        from ..experiments.result_view import load_result_cases, public_case, select_cases

        if not (root / "cases.jsonl").is_file():
            return _api_error("missing_results", "没有 cases.jsonl", 404)
        cases, meta = load_result_cases(root)
        selected = select_cases(cases, pattern=pattern, rule=rule, review=review, block=block)
        return {
            "data": [public_case(case, full=False) for case in selected],
            "meta": {**_api_meta(total=len(selected)), "view": meta, "availability": "present"},
        }

    @app.get("/api/v1/result-cases/{case_id}")
    def result_case(case_id: str) -> Any:
        from ..experiments.result_view import load_result_cases, public_case

        _require_id(case_id)
        if not (root / "cases.jsonl").is_file():
            return _api_error("missing_results", "没有 cases.jsonl", 404)
        cases, meta = load_result_cases(root)
        found = next((case for case in cases if case.get("case_id") == case_id), None)
        if found is None:
            return _api_error("unknown_case", f"没有用例 {case_id}", 404)
        return {"data": public_case(found, full=True), "meta": {**_api_meta(total=1), "view": meta}}

    @app.get("/api/v1/comparisons")
    def comparisons(task_mode: str = "rag_grounded", batch: str = "historical") -> Any:
        from ..experiments.workbench import comparison_view

        payload = comparison_view(root, task_mode=task_mode, batch=batch)
        if payload.get("status") == "error" and payload.get("error", {}).get("code") == "unknown_task_mode":
            return _api_error("unknown_task_mode", payload["error"]["message"], 422)
        if payload.get("availability") == "missing":
            return _api_error("missing_metrics", payload.get("headline") or "没有指标", 404)
        return {"data": payload, "meta": _api_meta(total=1)}

    @app.get("/api/v1/comparisons/{batch_id}")
    def comparison_one(batch_id: str, task_mode: str = "rag_grounded") -> Any:
        from ..experiments.workbench import comparison_view

        _require_id(batch_id)
        payload = comparison_view(root, task_mode=task_mode, batch=batch_id)
        if payload.get("availability") == "missing":
            return _api_error("missing_metrics", payload.get("headline") or "没有指标", 404)
        return {"data": payload, "meta": _api_meta(total=1)}

    @app.get("/api/v1/comparisons/{batch_id}/cases")
    def comparison_case_list(
        batch_id: str,
        task_mode: str = "rag_grounded",
        relation: str = "",
        subset: str = "",
    ) -> Any:
        from ..experiments.workbench import comparison_cases

        _require_id(batch_id)
        payload = comparison_cases(root, task_mode=task_mode, batch=batch_id, relation=relation, subset=subset)
        if payload.get("availability") == "error":
            return _api_error(payload.get("error", {}).get("code") or "bad_request", payload.get("error", {}).get("message") or "无法读取案例", 422)
        return {"data": payload.get("data") or [], "meta": {**_api_meta(total=len(payload.get("data") or [])), **payload}}

    return app


def serve(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(run_dir), host=host, port=port, log_level="info")


def _api_meta(total: int | None = None, next_cursor: str | None = None) -> dict[str, Any]:
    return {
        "request_id": uuid.uuid4().hex[:16],
        "schema_version": "v1",
        "next_cursor": next_cursor,
        "total": total,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


def _api_error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": {}, "retryable": False}, "request_id": uuid.uuid4().hex[:16]},
    )


def _cursor_index(cursor: str) -> tuple[int, JSONResponse | None]:
    if not str(cursor).isdigit():
        return 0, _api_error("invalid_cursor", "cursor 必须是非负整数", 422)
    return int(cursor), None


def _looks_like_run(path: Path) -> bool:
    return any((path / name).is_file() for name in ("stats.json", "report.md", "meta.json", "metrics.json"))


def _stream_page(path: Path, start: int, limit: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if total >= start and len(rows) < limit:
                rows.append(json.loads(line))
            total += 1
    return rows, total


def _require_id(exp_id: str) -> None:
    if not _ID_RE.fullmatch(exp_id or ""):
        raise HTTPException(status_code=400, detail="实验编号不合法")


def _load_suite(root: Path) -> dict[str, Any]:
    path = root / "metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="缺少 metrics.json，请先运行 qa-pipeline experiment")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="metrics.json 格式不正确")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _project(row: dict[str, Any], clip: int) -> dict[str, Any]:
    clipped: list[str] = []
    out: dict[str, Any] = {}
    for key in _SAMPLE_KEYS:
        value = row.get(key)
        if key in _CLIP_KEYS and isinstance(value, str) and len(value) > clip:
            out[key] = value[:clip] + "…"
            clipped.append(key)
        else:
            out[key] = value
    out["clipped"] = clipped
    return out


def _clip_text(value: str, clip: int) -> str:
    text = value or ""
    if len(text) > clip:
        return text[:clip] + "…"
    return text


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _bucket_samples(samples: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for sample in samples:
        if mode == "question":
            key = _norm(str(sample.get("question") or ""))
        else:
            key = _norm(str(sample.get("chunk_text") or "")) or _norm(str(sample.get("evidence_span") or ""))
        key = key or str(sample.get("qa_id") or len(order))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(sample)
    return [{"key": key, "samples": groups[key]} for key in order]


def _source_text(samples: list[dict[str, Any]], mode: str) -> str:
    for sample in samples:
        if mode == "question" and sample.get("question"):
            return str(sample["question"])
        if sample.get("chunk_text"):
            return str(sample["chunk_text"])
    for sample in samples:
        if sample.get("evidence_span"):
            return str(sample["evidence_span"])
        if sample.get("question"):
            return str(sample["question"])
    return ""


def _related(left: list[str], right: list[str]) -> bool:
    for item in left:
        if len(item) < 8:
            continue
        for other in right:
            if len(other) < 8:
                continue
            if item in other or other in item:
                return True
    return False


def align_rows(per_exp: dict[str, list[dict[str, Any]]], mode: str) -> list[dict[str, Any]]:
    """把不同方法的样本排成行。chunk/question 用同一文本做键，overlap 用证据互含。"""
    if mode == "overlap":
        return _align_overlap(per_exp)
    buckets = {exp_id: _bucket_samples(samples, mode) for exp_id, samples in per_exp.items()}
    order: list[str] = []
    seen: set[str] = set()
    for exp_id in per_exp:
        for bucket in buckets[exp_id]:
            if bucket["key"] not in seen:
                seen.add(bucket["key"])
                order.append(bucket["key"])
    rows = []
    for key in order:
        cells = {exp_id: [] for exp_id in per_exp}
        source = ""
        for exp_id, grouped in buckets.items():
            for bucket in grouped:
                if bucket["key"] != key:
                    continue
                cells[exp_id].extend(bucket["samples"])
                if not source:
                    source = _source_text(bucket["samples"], mode)
        rows.append({"source": source, "cells": cells})
    return rows


def _align_overlap(per_exp: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for exp_id, samples in per_exp.items():
        for bucket in _bucket_samples(samples, "chunk"):
            texts = []
            for sample in bucket["samples"]:
                texts.append(_norm(str(sample.get("chunk_text") or "")))
                texts.append(_norm(str(sample.get("evidence_span") or "")))
            blocks.append({"exp": exp_id, "samples": bucket["samples"], "texts": texts})
    parent = list(range(len(blocks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if blocks[i]["exp"] != blocks[j]["exp"] and _related(blocks[i]["texts"], blocks[j]["texts"]):
                union(i, j)
    clusters: dict[int, list[dict[str, Any]]] = {}
    for index, block in enumerate(blocks):
        clusters.setdefault(find(index), []).append(block)
    rows = []
    for cluster in clusters.values():
        cells = {exp_id: [] for exp_id in per_exp}
        source = ""
        for block in cluster:
            cells[block["exp"]].extend(block["samples"])
            text = _source_text(block["samples"], "chunk")
            if len(text) > len(source):
                source = text
        rows.append({"source": source, "cells": cells})
    return rows
