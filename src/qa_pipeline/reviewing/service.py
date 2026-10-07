"""教师审核运行器。dry-run 不调用模型；失败、分歧和过期哈希不放行。"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from .cache import JsonlStore
from .identity import (
    evidence_snapshot_hash,
    inference_config_hash,
    prompt_for,
    prompt_hash,
    review_cache_key,
    subject_hash,
)
from .policy import ReviewPolicy, build_aggregate, normalize_judges
from .schemas import ReviewRecord
from .teacher import (
    build_messages,
    call_teacher,
    client_for,
    make_record,
    new_id,
    parse_teacher_payload,
    teacher_credentials_configured,
    technical_reason,
    utc_now,
)

LLMFactory = Callable[[dict[str, Any]], Any]


class CallLedger:
    def __init__(self, max_calls: int, max_tokens: int) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens = 0
        self._lock = threading.Lock()

    def try_reserve(self, tokens: int) -> bool:
        with self._lock:
            if self.calls >= self.max_calls or self.tokens >= self.max_tokens:
                return False
            self.calls += 1
            self.tokens += tokens
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "tokens": self.tokens,
                "max_calls": self.max_calls,
                "max_tokens": self.max_tokens,
                "estimated_cost": None,
                "currency": "unknown",
                "pricing_version": "unknown",
            }


def load_policy(path: str | Path) -> ReviewPolicy:
    return ReviewPolicy.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_judges(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("judges") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("models 必须是非空裁判列表")
    return normalize_judges(rows)


def load_subjects(path: str | Path, batch_id: str) -> list[dict[str, Any]]:
    file = Path(path)
    rows = []
    if file.suffix == ".json":
        payload = json.loads(file.read_text(encoding="utf-8"))
        source = payload.get("subjects") if isinstance(payload, dict) else payload
        rows = list(source or [])
    else:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [normalize_subject(row, batch_id) for row in rows]


def normalize_subject(raw: dict[str, Any], batch_id: str) -> dict[str, Any]:
    subject = dict(raw)
    if not subject.get("subject_type"):
        subject["subject_type"] = "protocol_case" if subject.get("case_id") or subject.get("question") else "training_sample"
    subject["subject_id"] = str(subject.get("subject_id") or subject.get("case_id") or subject.get("qa_id") or "")
    subject["subject_version"] = str(subject.get("subject_version") or "1")
    subject["batch_id"] = str(subject.get("batch_id") or batch_id)
    if "student_context" not in subject:
        subject["student_context"] = subject.get("context") or ""
    subject["task_mode"] = subject.get("task_mode") or subject.get("goal") or "rag_grounded"
    subject["risk"] = subject.get("risk") or ("high" if subject["subject_type"] == "protocol_case" else "low")
    return subject


def deterministic_issues(subject: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not str(subject.get("subject_id") or "").strip():
        issues.append("missing_id")
    if subject.get("subject_type") not in {"training_sample", "protocol_case", "prediction"}:
        issues.append("invalid_subject_type")
    declared = subject.get("declared_subject_hash")
    if declared and declared != subject_hash(subject):
        issues.append("declared_hash_mismatch")
    text = str(subject.get("question") or "") + str(subject.get("student_context") or "")
    if len(text) > 200_000:
        issues.append("input_too_long")
    kind = subject.get("subject_type")
    if kind in {"protocol_case", "training_sample"} and not str(subject.get("question") or "").strip():
        issues.append("empty_question")
    if kind == "protocol_case" and not str(subject.get("expected_action") or "").strip():
        issues.append("missing_expected_action")
    if kind == "prediction" and not str(subject.get("answer_text") or subject.get("prediction") or "").strip():
        issues.append("empty_prediction")
    return issues


def assert_safe_review_out(out_dir: Path) -> None:
    if (out_dir / "review" / "batches.json").is_file() or (out_dir / "review" / "import_result.json").is_file():
        raise FileExistsError("refuse_overwrite_human_review_run")


def run_teacher_review(
    *,
    manifest: str | Path,
    policy_path: str | Path,
    models_path: str | Path,
    out_dir: str | Path,
    max_calls: int,
    max_tokens: int,
    concurrency: int = 2,
    dry_run: bool = False,
    resume: bool = False,
    fake: bool = False,
    batch_id: str = "teacher_batch",
    llm_factory: LLMFactory | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    assert_safe_review_out(out)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(policy_path)
    judges = load_judges(models_path)
    subjects = load_subjects(manifest, batch_id)
    credentials = teacher_credentials_configured()
    prompt_names = {}
    prompt_digests = {}
    for kind in {item["subject_type"] for item in subjects} or {"protocol_case"}:
        name, text = prompt_for(kind)
        prompt_names[kind] = name
        prompt_digests[kind] = prompt_hash(text)
    snapshot = {
        "policy": policy.model_dump(),
        "frozen_at": utc_now(),
        "credentials": credentials,
        "concurrency": concurrency,
        "max_calls": max_calls,
        "max_tokens": max_tokens,
        "pricing_version": "unknown",
        "currency": "unknown",
        "estimated_cost": None,
    }
    existing_policy = out / "policy_snapshot.json"
    if existing_policy.is_file():
        previous = json.loads(existing_policy.read_text(encoding="utf-8"))
        if previous.get("policy") != snapshot["policy"]:
            return {"status": "blocked", "reason": "policy_changed", "executed": False, "model_called": False}
    else:
        existing_policy.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "prompt_manifest.json").write_text(
        json.dumps({"prompts": prompt_names, "hashes": prompt_digests}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    planned_calls = len(subjects) * len([item for item in judges if item.get("judge_role") != "arbiter"])
    if dry_run or (not fake and not credentials["configured"]):
        status = "dry_run" if dry_run and (fake or credentials["configured"]) else "blocked"
        reason = None if status == "dry_run" else "teacher_credentials_missing"
        if dry_run and not fake and not credentials["configured"]:
            status = "blocked"
            reason = "teacher_credentials_missing"
        payload = {
            "status": status,
            "reason": reason,
            "executed": False,
            "model_called": False,
            "planned_subjects": len(subjects),
            "planned_calls": planned_calls,
            "input_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "budget": {"max_calls": max_calls, "max_tokens": max_tokens, "concurrency": concurrency},
            "pricing": {"currency": "unknown", "estimated_cost": None, "pricing_version": "unknown"},
            "credentials": credentials,
        }
        (out / "dry_run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    review_set_id = new_id("set_")
    reviews = JsonlStore(out / "reviews.jsonl")
    events = JsonlStore(out / "review_events.jsonl")
    usage = JsonlStore(out / "usage.jsonl")
    ledger = CallLedger(max_calls, max_tokens)
    infer_digest = inference_config_hash()
    cached = reviews.valid_rows() if resume else []
    workers = max(1, concurrency)

    def cached_success(key: str, digest: str) -> dict[str, Any] | None:
        for row in cached:
            if row.get("cache_key") == key and row.get("execution_status") == "succeeded" and row.get("subject_hash") == digest:
                return row
        return None

    def review_subject(subject: dict[str, Any]) -> list[ReviewRecord]:
        issues = deterministic_issues(subject)
        if issues:
            record = make_record(
                subject=subject,
                policy=policy,
                judge={"provider_id": "deterministic", "model_id": "none", "model_revision": "none", "judge_role": "gold_reviewer", "independence_level": "single"},
                review_set_id=review_set_id,
                prompt_digest=prompt_digests[subject["subject_type"]],
                inference_digest=infer_digest,
                decision="reject",
                execution_status="succeeded",
                reason_codes=["deterministic:" + item for item in issues],
            )
            _commit(reviews, events, record, "deterministic_reject")
            return [record]
        written: list[ReviewRecord] = []
        primary = [item for item in judges if item.get("judge_role") != "arbiter"]
        subject_set = review_set_id
        for judge in primary:
            digest = subject_hash(subject)
            key = review_cache_key(
                subject_hash_value=digest,
                review_kind=str(subject["subject_type"]),
                model_id=str(judge.get("model_id") or ""),
                model_revision=str(judge.get("model_revision") or "unknown"),
                prompt_digest=prompt_digests[subject["subject_type"]],
                rubric_version=policy.rubric_version,
                policy_version=policy.policy_version,
                evidence_hash=evidence_snapshot_hash(subject),
                judge_role=str(judge.get("judge_role") or "gold_reviewer"),
            )
            hit = cached_success(key, digest)
            if hit is not None:
                subject_set = str(hit.get("review_set_id") or subject_set)
                break
        for judge in primary:
            record = _judge_once(
                subject,
                judge,
                policy=policy,
                ledger=ledger,
                reviews=reviews,
                events=events,
                usage=usage,
                review_set_id=subject_set,
                prompt_digest=prompt_digests[subject["subject_type"]],
                infer_digest=infer_digest,
                resume_hit=cached_success,
                fake=fake,
                llm_factory=llm_factory,
            )
            written.append(record)
        aggregate = build_aggregate(
            written,
            subject_id=str(subject["subject_id"]),
            expected_hash=subject_hash(subject),
            policy=policy,
            task_mode=str(subject.get("task_mode") or "rag_grounded"),
            risk=str(subject.get("risk") or "high"),
        )
        arbiter = next((item for item in judges if item.get("judge_role") == "arbiter"), None)
        if aggregate.teacher_review_status == "disputed" and aggregate.quarantine_reason == "judge_disagreement" and arbiter is not None:
            record = _judge_once(
                subject,
                arbiter,
                policy=policy,
                ledger=ledger,
                reviews=reviews,
                events=events,
                usage=usage,
                review_set_id=subject_set,
                prompt_digest=prompt_digests[subject["subject_type"]],
                infer_digest=infer_digest,
                resume_hit=cached_success,
                fake=fake,
                llm_factory=llm_factory,
            )
            written.append(record)
        return written

    if workers == 1 or len(subjects) <= 1:
        produced = [review_subject(subject) for subject in subjects]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            produced = list(pool.map(review_subject, subjects))
    flat = [record for group in produced for record in group]
    aggregates = []
    by_id: dict[str, list[ReviewRecord]] = {}
    for record in flat:
        by_id.setdefault(record.subject_id, []).append(record)
    subjects_by_id = {item["subject_id"]: item for item in subjects}
    for subject_id, group in by_id.items():
        subject = subjects_by_id[subject_id]
        aggregate = build_aggregate(
            group,
            subject_id=subject_id,
            expected_hash=subject_hash(subject),
            policy=policy,
            task_mode=str(subject.get("task_mode") or "rag_grounded"),
            risk=str(subject.get("risk") or "high"),
        )
        aggregates.append(aggregate.model_dump())
    agg_store = JsonlStore(out / "review_aggregates.jsonl")
    for row in aggregates:
        agg_store.append(row)
    summary = {
        "status": "succeeded",
        "executed": True,
        "model_called": ledger.calls > 0,
        "planned_subjects": len(subjects),
        "record_n": len(flat),
        "aggregate_n": len(aggregates),
        "accepted_n": sum(1 for row in aggregates if row["accepted_by_policy"]),
        "usage": ledger.snapshot(),
        "formal_ready_n": sum(1 for row in aggregates if row["ready_for_formal_human_eval"]),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _commit(reviews: JsonlStore, events: JsonlStore, record: ReviewRecord, event_type: str) -> None:
    payload = record.model_dump()
    reviews.append(payload)
    events.append(
        {
            "event_id": new_id("evt_"),
            "event_type": event_type,
            "entity_id": record.subject_id,
            "review_id": record.review_id,
            "before_hash": None,
            "after_hash": record.subject_hash,
            "reason_code": ",".join(record.reason_codes),
            "execution_status": record.execution_status,
            "decision": record.decision,
            "created_at": record.created_at,
        }
    )


def _judge_once(
    subject: dict[str, Any],
    judge: dict[str, Any],
    *,
    policy: ReviewPolicy,
    ledger: CallLedger,
    reviews: JsonlStore,
    events: JsonlStore,
    usage: JsonlStore,
    review_set_id: str,
    prompt_digest: str,
    infer_digest: str,
    resume_hit,
    fake: bool,
    llm_factory: LLMFactory | None,
) -> ReviewRecord:
    digest = subject_hash(subject)
    key = review_cache_key(
        subject_hash_value=digest,
        review_kind=str(subject["subject_type"]),
        model_id=str(judge.get("model_id") or ""),
        model_revision=str(judge.get("model_revision") or "unknown"),
        prompt_digest=prompt_digest,
        rubric_version=policy.rubric_version,
        policy_version=policy.policy_version,
        evidence_hash=evidence_snapshot_hash(subject),
        judge_role=str(judge.get("judge_role") or "gold_reviewer"),
    )
    hit = resume_hit(key, digest)
    if hit is not None:
        record = ReviewRecord.model_validate(hit)
        events.append({"event_id": new_id("evt_"), "event_type": "cache_reuse", "entity_id": record.subject_id, "review_id": record.review_id, "created_at": utc_now()})
        return record
    _name, text = prompt_for(str(subject["subject_type"]))
    messages = build_messages(subject, prompt_text=text)
    client = llm_factory(judge) if llm_factory else client_for(judge, fake=fake)
    attempt = 0
    last_reason = "technical_failed"
    response = None
    while True:
        if not ledger.try_reserve(1):
            record = make_record(
                subject=subject,
                policy=policy,
                judge=judge,
                review_set_id=review_set_id,
                prompt_digest=prompt_hash(text),
                inference_digest=infer_digest,
                decision="abstain",
                execution_status="budget_stopped",
                reason_codes=["budget_stopped"],
                attempt=attempt + 1,
                cache_key=key,
            )
            _commit(reviews, events, record, "budget_stopped")
            usage.append({"request_id": record.request_id, "status": "budget_stopped", "estimated_cost": None, "currency": "unknown", "created_at": record.created_at})
            return record
        response = call_teacher(client, messages, str(judge.get("model_id") or client.default_model))
        usage.append(
            {
                "request_id": response.request_id,
                "status": response.status,
                "model_id": judge.get("model_id"),
                "attempt": attempt + 1,
                "estimated_cost": None,
                "currency": "unknown",
                "pricing_version": "unknown",
                "created_at": utc_now(),
            }
        )
        if response.status == "ok":
            parsed, error = parse_teacher_payload(response.data)
            if parsed is not None:
                record = make_record(
                    subject=subject,
                    policy=policy,
                    judge=judge,
                    review_set_id=review_set_id,
                    prompt_digest=prompt_hash(text),
                    inference_digest=infer_digest,
                    decision=parsed["decision"],
                    execution_status="succeeded",
                    reason_codes=parsed["reason_codes"],
                    parsed=parsed,
                    attempt=attempt + 1,
                    cache_key=key,
                    request_id=response.request_id,
                    usage={"status": "ok"},
                )
                _commit(reviews, events, record, "review_committed")
                return record
            last_reason = error or "schema_invalid"
        else:
            last_reason = "timeout" if "timeout" in (response.error or "") else technical_reason(response.status)
        if attempt >= policy.max_technical_retries:
            record = make_record(
                subject=subject,
                policy=policy,
                judge=judge,
                review_set_id=review_set_id,
                prompt_digest=prompt_hash(text),
                inference_digest=infer_digest,
                decision="abstain",
                execution_status="technical_failed",
                reason_codes=[last_reason],
                attempt=attempt + 1,
                cache_key=key,
                request_id=response.request_id,
            )
            _commit(reviews, events, record, "technical_failed")
            return record
        attempt += 1
    raise AssertionError("unreachable")
