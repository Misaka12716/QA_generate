"""对照台 HTTP 服务。只读实验目录，不触发新的生成。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

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
    "nli_score",
    "judge_overall",
    "kb_gain",
    "action",
    "filter_trace",
    "evolution_type",
)


def create_app(run_dir: str | Path) -> FastAPI:
    root = Path(run_dir).resolve()
    app = FastAPI(title="QA 方案对照台", docs_url=None, redoc_url=None)
    app.mount("/static", _Static(directory=STATIC), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

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

    return app


def serve(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(run_dir), host=host, port=port, log_level="info")


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
