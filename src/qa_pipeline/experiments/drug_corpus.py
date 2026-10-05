"""校医院说明书的清洗、来源族分区、只读快照和可审核测试协议。"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

from ..pipeline import document_from_file
from ..plugins.chunking import HeadingWindowChunking
from ..plugins.question_gen import review_gold
from ..schemas import content_id
from ..textutil import (
    balanced_take,
    char_ngram_jaccard,
    compact,
    heading_sections,
    parse_quality,
    sentences,
)

AMBIGUOUS_QUESTION = "根据给定资料，说明这一句表述的具体内容。"
CLEAN_VERSION = "drug-clean-v2"
CHUNK_PARAMS = {"strategy": "heading_window", "max_tokens": 512, "overlap": 0.1, "tokenizer_id": "approx-v1"}
# 第五节改写方向只作回归说明，不是双人审核后的正式金标。
REGRESSION_DIRECTIONS = [
    "5g 规格对糖尿病及严重慢性病患者的医师指导要求，不把 2.5g 与 5g 黏成同一装量。",
    "急性毒性试验中小鼠口服 LD50 的数值、单位和原文不确定性表述。",
    "160/7.2/4.8 组给药后 5 分钟 ETHOS 与 KRONOS 的 FEV1 变化，不改答死亡率。",
    "说明书列出的药品通用名称和汉语拼音。",
    "体温超过 38.5℃并出现扁桃体肿大时的处理要求。",
    "坦索罗辛用于肌酐清除率小于 10ml/分钟的重度肾功能障碍患者时的谨慎理由。",
    "吡诺克辛滴眼液对早产儿、新生儿及哺乳期儿童安全性证据的原文边界。",
    "麻黄的植物来源和药用部位，不改答贮藏条件。",
    "替硝唑乳汁排泄描述及哺乳期限制。",
    "大面积注射部位不良反应与既往含无细胞百日咳疫苗剂次的关系。",
    "【不良反应】栏目列出的反应，需补回药品身份。",
    "维生素 D2 疗程监测项目与治疗量血钙范围。",
    "煅赭石炮制步骤顺序。",
    "治疗期间 LDL-C 变化与剂量的关系，需补回药品身份。",
    "瑞舒伐他汀合用蛋白酶抑制剂的谨慎理由和剂量处理前提。",
    "美国以外开放研究的患者数，以及莫西沙星的对照方案。",
]

_FIELDS = {
    "generic_name": re.compile(r"(?:通用名称|药品名称)\s*[:：]\s*([^\n【]{2,40})"),
    "strength": re.compile(r"规格\s*[:：]\s*([^\n【]{1,60})"),
    "dosage_form": re.compile(r"剂型\s*[:：]\s*([^\n【]{1,20})"),
    "manufacturer": re.compile(r"(?:生产企业|生产单位|厂家|上市许可持有人)\s*[:：]\s*([^\n【]{2,80})"),
    "approval": re.compile(r"(?:批准文号|药品批准文号)\s*[:：]\s*([A-Za-z0-9\u4e00-\u9fff（）()]{4,40})"),
    "version": re.compile(r"(?:修订日期|修改日期|版本号)\s*[:：]\s*([^\n【]{4,40})"),
}
_POPULATION = re.compile(r"(儿童|老年人|孕妇|哺乳|肾功能|肝功能|新生儿|早产儿)")
_ROUTE = re.compile(r"(口服|外用|静脉|肌内|滴眼|吸入|皮下)")


class SnapshotInvalid(Exception):
    """来源内容与冻结清单不一致。"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def raw_dir() -> Path:
    return repo_root() / "data" / "campus_hospital_drug_instructions" / "raw"


def frozen_dir() -> Path:
    return repo_root() / "data" / "campus_hospital_drug_instructions" / "frozen"


def question_is_ambiguous(question: str) -> bool:
    text = question or ""
    if any(flag in text for flag in ("这一句", "这句话", "该句", "上述句子")):
        return True
    if re.match(r"^\s*[它其这那]", text):
        return True
    return False


def admits_main_metric(item: dict) -> bool:
    if question_is_ambiguous(str(item.get("question") or "")):
        return False
    if item.get("validity_status") != "self_contained":
        return False
    if item.get("review_status") not in {"agreed", "dual_agreed"}:
        return False
    if item.get("gold_mismatch"):
        return False
    return True


def extract_identity(text: str) -> dict[str, str]:
    found = {key: "" for key in _FIELDS}
    for key, pattern in _FIELDS.items():
        match = pattern.search(text or "")
        if match:
            found[key] = re.sub(r"\s+", "", match.group(1)).strip("。；;，,")
    if not found["generic_name"]:
        for path, body in heading_sections(text or ""):
            if path and path[-1] in {"药品名称", "通用名称"}:
                line = next((line.strip() for line in body.splitlines() if line.strip()), "")
                found["generic_name"] = re.sub(r"\s+", "", line.split("汉语拼音")[0])[:40]
                break
    population = _POPULATION.search(text or "")
    route = _ROUTE.search(text or "")
    found["population"] = population.group(1) if population else ""
    found["route"] = route.group(1) if route else ""
    return found


def _family_key(identity: dict[str, str], mode: str = "source") -> str:
    if mode == "generic":
        return content_id("gen_", {"generic_name": identity.get("generic_name") or ""})
    return content_id(
        "src_",
        {
            "generic_name": identity.get("generic_name") or "",
            "strength": identity.get("strength") or "",
            "dosage_form": identity.get("dosage_form") or "",
            "manufacturer": identity.get("manufacturer") or "",
            "approval": identity.get("approval") or "",
            "version": identity.get("version") or "",
        },
    )


def cluster_documents(rows: list[dict], near_threshold: float = 0.92) -> list[dict]:
    """同一说明书族只进入一个分区。精确重复、近重复和通用名隔离分开记录。"""
    families: list[dict] = []
    for row in rows:
        text = row.get("text") or ""
        identity = row.get("identity") or extract_identity(text)
        row["identity"] = identity
        exact = hashlib.sha256(compact(text).encode("utf-8")).hexdigest()
        row["exact_hash"] = exact
        placed = None
        dup_kind = "unique"
        for family in families:
            if exact and exact in family["exact_hashes"]:
                placed = family
                dup_kind = "exact"
                break
            if text and family["rep_text"] and char_ngram_jaccard(text, family["rep_text"]) >= near_threshold:
                placed = family
                dup_kind = "near"
                break
            same_name = identity.get("generic_name") and identity.get("generic_name") == family["identity"].get("generic_name")
            same_product = (
                same_name
                and identity.get("strength") == family["identity"].get("strength")
                and identity.get("manufacturer") == family["identity"].get("manufacturer")
                and identity.get("approval") == family["identity"].get("approval")
            )
            if same_product:
                placed = family
                dup_kind = "identity"
                break
        if placed is None:
            placed = {
                "family_id": _family_key(identity, "source"),
                "identity": identity,
                "exact_hashes": set(),
                "rep_text": text,
                "members": [],
            }
            families.append(placed)
        if exact:
            placed["exact_hashes"].add(exact)
        placed["members"].append(row.get("stem") or "")
        row["dup_kind"] = dup_kind
        row["source_family_id"] = placed["family_id"]
        row["generic_family_id"] = _family_key(identity, "generic")
    for row in rows:
        row["split"] = split_for_family(row["source_family_id"]) if row.get("parse_ok") else None
    return rows


def split_for_family(family_id: str) -> str:
    bucket = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        return "locked_test"
    if bucket == 1:
        return "dev"
    return "train"


def _read_text(path: Path) -> tuple[str, bool, list[str]]:
    try:
        data = path.read_bytes()
    except OSError:
        return "", False, ["unreadable"]
    if data[:8].startswith(b"%PDF"):
        return "", False, ["pdf_header"]
    text = data.decode("utf-8", errors="replace")
    reasons = ["replacement_char"] if "\ufffd" in text else []
    return text, True, reasons


def inventory(directory: Path | None = None) -> dict:
    root = directory or raw_dir()
    rows = []
    readable_n = 0
    parse_n = 0
    total = 0
    if root.is_dir():
        paths = sorted(root.glob("*.txt"))
    else:
        paths = []
    for path in paths:
        total += 1
        text, readable, read_reasons = _read_text(path)
        quality = parse_quality(text) if readable else {"readable": False, "parse_ok": False, "reasons": read_reasons}
        quality["reasons"] = list(dict.fromkeys([*read_reasons, *quality.get("reasons", [])]))
        if quality["readable"]:
            readable_n += 1
        if quality["parse_ok"]:
            parse_n += 1
        rows.append(
            {
                "stem": path.stem,
                "path": str(path),
                "text": text,
                "status": "parsed_text" if readable else "unreadable",
                "chars": len(text),
                "readable": quality["readable"],
                "parse_ok": quality["parse_ok"],
                "quality_reasons": quality["reasons"],
                "identity": extract_identity(text),
            }
        )
    cluster_documents(rows)
    return {
        "documents": rows,
        "counts": _split_counts(rows),
        "total": total,
        "readable_rate": round(readable_n / total, 4) if total else None,
        "parse_quality_pass_rate": round(parse_n / total, 4) if total else None,
        "readable_n": readable_n,
        "parse_ok_n": parse_n,
        "isolation": {
            "primary": "source_family",
            "strict_generic_name": "reported_separately",
        },
    }


def _split_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("split") or row.get("status") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def select_subset(rows: list[dict], size: int = 60, seed: int = 42) -> list[dict]:
    """按来源族轮转，并按篇幅分层。每个来源族最多一篇进入子集。"""
    train = [row for row in rows if row.get("split") == "train" and row.get("parse_ok", True)]
    seen: set[str] = set()
    unique = []
    for row in train:
        family = row.get("source_family_id") or row.get("stem")
        if family in seen:
            continue
        seen.add(family)
        unique.append(row)
    if not unique:
        return []
    unique.sort(key=lambda row: (row.get("chars") or 0, row.get("stem") or ""))
    quartiles: list[list[dict]] = [[], [], [], []]
    for index, row in enumerate(unique):
        quartiles[min(3, index * 4 // len(unique))].append(row)
    rng = random.Random(seed)
    picked: list[dict] = []
    cursors = [0, 0, 0, 0]
    while len(picked) < size and any(cursors[i] < len(quartiles[i]) for i in range(4)):
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        for bucket in order:
            if cursors[bucket] < len(quartiles[bucket]) and len(picked) < size:
                picked.append(quartiles[bucket][cursors[bucket]])
                cursors[bucket] += 1
    return picked


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_subset(picked: list[dict], dest: Path | None = None) -> Path:
    subset = dest or (frozen_dir() / "subset")
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for row in picked:
        source = Path(row["path"])
        name = f"{row.get('source_family_id', 'fam')[:12]}_{source.name}"
        target = subset / name
        shutil.copyfile(source, target)
        os.chmod(target, 0o444)
        digest = file_sha256(target)
        manifest_files.append(
            {
                "stem": row.get("stem"),
                "source_family_id": row.get("source_family_id"),
                "snapshot_path": str(target),
                "source_hash": digest,
                "clean_version": CLEAN_VERSION,
                "chunk_params": CHUNK_PARAMS,
                "chars": row.get("chars"),
            }
        )
    manifest = {
        "clean_version": CLEAN_VERSION,
        "chunk_params": CHUNK_PARAMS,
        "files": manifest_files,
        "manifest_hash": hashlib.sha256(
            json.dumps(manifest_files, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    (subset / "snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(subset / "snapshot_manifest.json", 0o444)
    return subset


def verify_snapshot(manifest_path: Path) -> None:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for row in data.get("files") or []:
        path = Path(row["snapshot_path"])
        if not path.is_file() or file_sha256(path) != row["source_hash"]:
            raise SnapshotInvalid(f"快照失效: {row.get('stem')}")


def catalog_units_from_text(text: str, family_id: str, stem: str = "") -> list[dict]:
    units = []
    for path, body in heading_sections(text or ""):
        section = path[-1] if path else ""
        for sent in sentences(body):
            sent = sent.strip()
            if len(sent) < 12:
                continue
            if any(token in sent for token in ("除外", "除非", "禁用", "尚不明确", "尚未", "不得")):
                kind = "exception"
            elif any(token in sent for token in ("如果", "若", "应当", "应在", "慎用", "不宜")):
                kind = "condition"
            elif any(token in sent for token in ("依次", "然后", "先", "步骤")):
                kind = "procedure"
            else:
                kind = "fact"
            units.append(
                {
                    "unit_id": content_id("unit_", {"family": family_id, "section": section, "sent": sent}),
                    "family_id": family_id,
                    "source": stem,
                    "kind": kind,
                    "points": [sent],
                    "evidence": sent,
                    "location": section,
                    "applicable_conditions": [],
                    "weight": 1,
                    "stage": "candidate",
                }
            )
    return units


def coverage_report(units: list[dict], answers: list[dict], stage: str = "accepted") -> dict:
    """同一家族只计一次。证据包含只作辅助，主口径是要点覆盖。"""
    eligible = [unit for unit in units if unit.get("stage", "candidate") in {stage, "candidate", "accepted", "released"}]
    by_family: dict[str, dict] = {}
    for unit in eligible:
        family = unit.get("family_id") or unit["unit_id"]
        slot = by_family.setdefault(family, {"units": [], "hit": False, "auxiliary": False})
        slot["units"].append(unit)
    for answer in answers:
        blob = "\n".join(
            [
                str(answer.get("answer") or ""),
                "\n".join(answer.get("answer_points") or []),
                str(answer.get("evidence") or ""),
            ]
        )
        for family, slot in by_family.items():
            if slot["hit"]:
                continue
            points = [point for unit in slot["units"] for point in unit.get("points") or []]
            if points and all(point and point in blob for point in points):
                slot["hit"] = True
            elif any(unit.get("evidence") and unit["evidence"] in blob for unit in slot["units"]):
                slot["auxiliary"] = True
    hit = sum(1 for slot in by_family.values() if slot["hit"])
    auxiliary = sum(1 for slot in by_family.values() if slot["auxiliary"])
    denominator = len(by_family)
    return {
        "stage": stage,
        "weighting": "equal_family",
        "numerator": hit,
        "denominator": denominator,
        "coverage": round(hit / denominator, 4) if denominator else None,
        "evidence_contains_auxiliary": auxiliary,
        "unresolved": denominator - hit,
        "reason": None if denominator else "zero_denominator",
    }


def _topic_from_sentence(sentence: str) -> str:
    topic = re.sub(r"\d+(?:\.\d+)?", "某数值", sentence)
    return topic[:18]


def build_question_case(text: str, stem: str, family_id: str, unit: dict | None = None) -> dict:
    identity = extract_identity(text)
    name = identity.get("generic_name") or ""
    units = [unit] if unit else catalog_units_from_text(text, family_id, stem)
    chosen = units[0] if units else None
    if not name or not chosen:
        return {
            "id": content_id("case_", {"stem": stem, "reason": "needs_identity"}),
            "source": stem,
            "source_family_id": family_id,
            "question": "",
            "answer_points": [],
            "validity_status": "needs_identity",
            "review_status": "unreviewed",
            "main_metric": False,
            "expected_action": "clarify",
        }
    section = chosen.get("location") or "正文"
    point = chosen["points"][0]
    question = f"所给{name}资料的【{section}】如何表述与「{_topic_from_sentence(point)}」相关的事实？"
    status = "ambiguous_referent" if question_is_ambiguous(question) else "self_contained"
    item = {
        "id": content_id("case_", {"family": family_id, "section": section, "point": point}),
        "family_id": content_id("qfam_", {"family": family_id, "section": section, "point": point}),
        "task_variant_id": "",
        "source": stem,
        "source_family_id": family_id,
        "question": question,
        "answer_points": [point],
        "acceptable_variants": [],
        "context": _window_around(text, point),
        "support_spans": [point, chosen.get("evidence") or point],
        "evidence_state": "sufficient",
        "expected_action": "answer",
        "validity_status": status,
        "review_status": "unreviewed",
        "layer": "T3",
        "distractor_type": "",
    }
    mismatch = review_gold(question, point, [point])
    item["gold_mismatch"] = bool(mismatch)
    if mismatch:
        item["review_status"] = "rejected"
        item["validity_status"] = "gold_mismatch"
    item["main_metric"] = admits_main_metric(item)
    return item


def _window_around(text: str, point: str, size: int = 900) -> str:
    idx = text.find(point) if point else -1
    if idx < 0:
        return text[:size]
    start = max(0, idx - 200)
    return text[start : start + size]


def apply_evidence_condition(item: dict, condition: str, visible: str, support_spans: list[str] | None = None) -> dict:
    """由最终可见输入重算证据状态。保留等价支持时不机械标成缺失。"""
    spans = [span for span in (support_spans if support_spans is not None else item.get("support_spans") or []) if span]
    points = [point for point in item.get("answer_points") or [] if point]
    found = [span for span in spans if span in (visible or "")]
    leaked = [point for point in points if point in (visible or "")]
    out = dict(item)
    out["condition"] = condition
    out["context"] = visible
    out["equivalent_support"] = bool(found)
    out["answer_leak"] = bool(leaked)
    if condition in {"sufficient", "sufficient_distractor"}:
        out["evidence_state"] = "sufficient" if found or leaked else "missing"
        out["expected_action"] = "answer" if found or leaked else "state_insufficient"
    elif condition == "missing":
        if found:
            out["evidence_state"] = "sufficient"
            out["expected_action"] = "answer"
        elif leaked:
            out["evidence_state"] = "partial"
            out["expected_action"] = "partial_answer"
        else:
            out["evidence_state"] = "missing"
            out["expected_action"] = "state_insufficient"
    elif condition == "partial":
        out["task_variant_id"] = out.get("task_variant_id") or f"{out.get('id')}:partial"
        if found or leaked:
            out["evidence_state"] = "partial"
            out["expected_action"] = "partial_answer"
        else:
            out["evidence_state"] = "missing"
            out["expected_action"] = "state_insufficient"
    elif condition == "ambiguous":
        out["task_variant_id"] = out.get("task_variant_id") or f"{out.get('id')}:ambiguous"
        out["evidence_state"] = "ambiguous"
        out["expected_action"] = "clarify"
    else:
        out["evidence_state"] = "missing"
        out["expected_action"] = "state_insufficient"
    out["main_metric"] = admits_main_metric(out)
    return out


def dosage_section(text: str) -> str:
    for path, body in heading_sections(text or ""):
        if any(("用法" in part) or ("用量" in part) for part in path):
            body = body.strip()
            if len(body) >= 12:
                return body
    return ""


def find_conflicts(rows: list[dict]) -> dict:
    """证据不足记为未确认。不把栏目名当成剂量，也不编造冲突。"""
    unconfirmed = []
    grouped: dict[tuple, list] = {}
    for row in rows:
        if row.get("split") not in {None, "train"} and row.get("split") != "train":
            continue
        if row.get("split") not in {None, "train"}:
            continue
        identity = row.get("identity") or extract_identity(row.get("text") or "")
        body = dosage_section(row.get("text") or "")
        if not identity.get("generic_name") or not body:
            unconfirmed.append({"stem": row.get("stem"), "status": "unconfirmed", "reason": "insufficient_dosage_evidence"})
            continue
        key = (
            identity.get("generic_name"),
            identity.get("dosage_form"),
            identity.get("strength"),
            identity.get("population"),
            identity.get("route"),
            identity.get("version"),
        )
        grouped.setdefault(key, []).append({"stem": row.get("stem"), "dosage": body[:180], "status": "unconfirmed"})
    candidates = []
    for key, items in grouped.items():
        doses = {item["dosage"] for item in items}
        if len(items) >= 2 and len(doses) >= 2:
            candidates.append(
                {
                    "key": list(key),
                    "status": "unconfirmed",
                    "reason": "dosage_text_differs_without_human_label",
                    "items": items,
                }
            )
    return {
        "conflicts": [],
        "candidates": candidates,
        "unconfirmed": unconfirmed,
        "note": "未确认不等于无冲突。本轮不执行冲突臂。",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def archive_legacy_heldout(directory: Path | None = None) -> Path | None:
    root = directory or frozen_dir()
    current = root / "heldout.jsonl"
    if not current.is_file():
        return None
    legacy = root / "heldout_legacy.jsonl"
    rows = []
    for line in current.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = str(row.get("question") or "")
        row["validity_status"] = "ambiguous_referent" if question_is_ambiguous(question) or question == AMBIGUOUS_QUESTION else row.get("validity_status")
        row["main_metric"] = False
        row["historical"] = True
        row["note"] = "题干歧义、不可用于主效果结论"
        rows.append(row)
    _write_jsonl(legacy, rows)
    return legacy


def write_catalog(subset: Path, limit_units: int | None = None, seed: int = 42) -> Path:
    docs = [document_from_file(path) for path in sorted(subset.glob("*.txt"))]
    chunker = HeadingWindowChunking(max_tokens=512, overlap=0.1)
    chunks = chunker.run(docs, None)
    chunks = balanced_take(chunks, limit_units, lambda chunk: chunk.source_family_id or chunk.doc_id, seed)
    catalog = []
    excluded = 0
    for chunk in chunks:
        units = catalog_units_from_text(chunk.text, chunk.source_family_id or chunk.doc_id, chunk.source_doc)
        if not units:
            excluded += 1
            continue
        catalog.extend(units[:4])
    root = subset.parent if subset.name == "subset" else subset
    path = root / "coverage_catalog.jsonl"
    _write_jsonl(path, catalog)
    freeze = {
        "clean_version": CLEAN_VERSION,
        "chunk_params": CHUNK_PARAMS,
        "chunk_count": len(chunks),
        "catalog_units": len(catalog),
        "excluded_empty": excluded,
        "content_ids": [chunk.chunk_id for chunk in chunks],
        "manifest_hash": hashlib.sha256("\n".join(chunk.chunk_id for chunk in chunks).encode("utf-8")).hexdigest(),
    }
    (root / "chunk_freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_heldout(rows: list[dict], size: int = 16, dest: Path | None = None) -> Path:
    """正式题必须自足。未审核或指代不明的题不进入主指标。"""
    root = dest or frozen_dir()
    locked = [row for row in rows if row.get("split") == "locked_test" and row.get("parse_ok", True)]
    locked = balanced_take(locked, max(size * 4, size), lambda row: row.get("source_family_id") or row.get("stem"), 42)
    cases = []
    seen_families: set[str] = set()
    for row in locked:
        family = row.get("source_family_id") or ""
        if family in seen_families:
            continue
        text = row.get("text") or ""
        if not text and row.get("path"):
            text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        case = build_question_case(text, row.get("stem") or "", family)
        case["distractor"] = ""
        case["distractor_type"] = ""
        cases.append(case)
        seen_families.add(family)
        if len(cases) >= size:
            break
    _write_jsonl(root / "heldout_candidates.jsonl", cases)
    main = [case for case in cases if admits_main_metric(case)]
    path = root / "heldout.jsonl"
    _write_jsonl(path, main)
    notes = {
        "status": "direction_only_not_gold",
        "items": REGRESSION_DIRECTIONS,
        "note": "这些方向尚未完成双人金标审核，不能升格为锁定测试答案。",
    }
    (root / "heldout_regression_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prepare(directory: Path | None = None) -> dict:
    data = inventory(directory)
    rows = data["documents"]
    archive_legacy_heldout()
    picked = select_subset(rows)
    subset = write_subset(picked) if picked else frozen_dir() / "subset"
    catalog = write_catalog(subset) if picked else None
    heldout = write_heldout(rows)
    conflicts = find_conflicts(rows)
    root = frozen_dir()
    (root / "conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "total": data["total"],
        "readable_rate": data["readable_rate"],
        "parse_quality_pass_rate": data["parse_quality_pass_rate"],
        "counts": data["counts"],
        "subset": len(picked),
        "catalog": str(catalog) if catalog else "",
        "heldout_main": str(heldout),
        "conflicts_confirmed": 0,
        "conflicts_unconfirmed": len(conflicts["unconfirmed"]) + len(conflicts["candidates"]),
    }
    (root / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    public_rows = [{key: value for key, value in row.items() if key != "text"} for row in rows]
    _write_jsonl(root / "inventory.jsonl", public_rows)
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
