"""题干对象与文档身份对照。引用存在不能抵消对象错误。"""

from __future__ import annotations

import re
from typing import Any

from .schemas import Document, DocumentIdentity

_DEICTIC = re.compile(r"本品|该药|本药|该药物|此药|该制剂|本制剂")
_PLANT = re.compile(r"植物([\u4e00-\u9fff]{2,12})")
_SUBJECT_ROLE = re.compile(r"药材|药用部位|制剂|药品|患者")


def identity_from_document(doc: Document) -> DocumentIdentity:
    """只接受显式身份。标题不自动当成标准对象，也不写入正文。"""
    meta = doc.metadata or {}
    raw = meta.get("document_identity")
    if isinstance(raw, DocumentIdentity):
        identity = raw.model_copy(deep=True)
    elif isinstance(raw, dict):
        identity = DocumentIdentity.model_validate(raw)
    else:
        identity = DocumentIdentity(
            document_title=doc.title or "",
            canonical_subject=str(meta.get("canonical_subject") or ""),
            entity_type=str(meta.get("entity_type") or "unknown"),
            aliases=[str(item) for item in (meta.get("aliases") or []) if str(item).strip()],
            dosage_form=str(meta.get("dosage_form") or meta.get("identity", {}).get("dosage_form") or ""),
            strength=str(meta.get("strength") or ""),
            version=doc.source_version or str(meta.get("version") or ""),
            source_id=doc.doc_id,
            source_family_id=doc.source_family_id or "",
        )
    if not identity.document_title:
        identity.document_title = doc.title or ""
    if not identity.source_id:
        identity.source_id = doc.doc_id
    if not identity.source_family_id:
        identity.source_family_id = doc.source_family_id or ""
    notes = list(identity.identity_notes)
    if identity.document_title and not any(note.get("role") == "title" for note in notes):
        notes.append({"role": "title", "text": identity.document_title})
    if identity.canonical_subject and not any(note.get("role") == "entity" for note in notes):
        notes.append({"role": "entity", "text": identity.canonical_subject, "entity_type": identity.entity_type})
    identity.identity_notes = notes
    return identity


def source_plants(text: str) -> list[str]:
    found = []
    for match in _PLANT.finditer(text or ""):
        name = match.group(1).split("的", 1)[0]
        if name and name not in found:
            found.append(name)
    return found


def subject_status(question: str, identity: DocumentIdentity | dict[str, Any] | None, chunk_text: str = "") -> str:
    """返回空字符串表示未发现对象问题。"""
    if isinstance(identity, dict):
        identity = DocumentIdentity.model_validate(identity)
    identity = identity or DocumentIdentity()
    text = str(question or "")
    subject = identity.canonical_subject.strip()
    aliases = {item.strip() for item in identity.aliases if item and item.strip()}
    named = subject in text or any(alias in text for alias in aliases)
    plants = source_plants(chunk_text)
    for plant in plants:
        if plant == subject or plant in aliases:
            continue
        if plant in text and not named and _SUBJECT_ROLE.search(text):
            return "object_mismatch"
        if plant in text and subject and subject not in text and "药材" in text:
            return "object_mismatch"
    if subject and _DEICTIC.search(text) and not named:
        return "unresolved_anaphora"
    if not subject and _DEICTIC.search(text):
        return "unresolved_subject"
    if identity.document_title and subject and identity.document_title not in {subject, *aliases}:
        title_name = re.sub(r"[（(].*?[）)]", "", identity.document_title).strip()
        if title_name and title_name != subject and title_name in chunk_text and subject not in chunk_text[: max(1, len(title_name) + 8)]:
            return "title_body_conflict"
    return ""
