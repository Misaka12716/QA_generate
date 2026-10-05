"""质量分层：sab / binary / none。"""

from __future__ import annotations

from ..registry import register
from ..schemas import QAPair
from ..textutil import is_substring


_S_DIMS = ("accuracy", "relevancy", "completeness", "quality")


def _overall(p: QAPair) -> float:
    if p.judge_overall is not None:
        return p.judge_overall
    if p.nli_score is not None:
        return p.nli_score * 5
    return 3.5 if p.action == "pass" else 2.0


def _dims_at_least(p: QAPair, floor: float) -> bool:
    if not p.judge_scores:
        return False
    values = []
    for name in _S_DIMS:
        if name not in p.judge_scores:
            return False
        values.append(float(p.judge_scores[name]))
    return all(v >= floor for v in values)


@register("grading", "none")
class NoGrade:
    name = "none"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        for p in pairs:
            p.grade = None
            p.filter_trace["grading"] = {"mode": "none"}
        return pairs


@register("grading", "binary")
class BinaryGrade:
    name = "binary"

    def __init__(self, min_overall: float = 2.5, **_: object) -> None:
        self.min_overall = float(min_overall)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        out = []
        for p in pairs:
            score = _overall(p)
            if p.action == "reject" or score < self.min_overall:
                p.grade = "reject"
                p.action = "reject"
            else:
                p.grade = "A"
            p.filter_trace["grading"] = {"mode": "binary", "score": score, "grade": p.grade}
            p.log("grading", self.name, grade=p.grade, score=score)
            out.append(p)
        return out


@register("grading", "sab")
class SABGrade:
    name = "sab"

    def __init__(
        self,
        s_overall: float = 4.5,
        a_overall: float = 3.5,
        b_overall: float = 2.5,
        s_nli: float = 0.85,
        a_nli: float = 0.7,
        **_: object,
    ) -> None:
        self.s_overall = float(s_overall)
        self.a_overall = float(a_overall)
        self.b_overall = float(b_overall)
        self.s_nli = float(s_nli)
        self.a_nli = float(a_nli)

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        out = []
        for p in pairs:
            score = _overall(p)
            nli = p.nli_score if p.nli_score is not None else 0.75
            gain_ok = (p.kb_gain is None) or (p.kb_gain >= 0.2)
            if p.action == "reject" or score < self.b_overall:
                p.grade = "reject"
                p.action = "reject"
            elif (
                score >= self.s_overall
                and _dims_at_least(p, 4.0)
                and nli >= self.s_nli
                and gain_ok
                and p.action != "downgrade"
            ):
                p.grade = "S"
            elif score >= self.a_overall and nli >= self.a_nli:
                p.grade = "A"
            else:
                p.grade = "B"
            if p.action == "downgrade" and p.grade == "S":
                p.grade = "A"
            p.filter_trace["grading"] = {"mode": "sab", "score": score, "nli": nli, "grade": p.grade}
            p.log("grading", self.name, grade=p.grade, score=score)
            out.append(p)
        return out


_PASS_CLAIM = {"supported"}


@register("grading", "validity_tier")
class ValidityTier:
    """共同有效性门槛之上区分 S/A。未解决样本标为 quarantine，不发布。"""

    name = "validity_tier"

    def __init__(self, **_: object) -> None:
        pass

    def run(self, pairs: list[QAPair], ctx) -> list[QAPair]:
        out = []
        for pair in pairs:
            if pair.evidence_state == "missing" and pair.expected_action == "state_insufficient" and pair.action == "pass":
                pair.grade = "A"
                pair.selection_role = "behavior"
            elif pair.action == "needs_escalation":
                pair.grade = "quarantine"
                pair.action = "quarantine"
            elif pair.action in {"reject", "quarantine"} or pair.grade == "quarantine":
                if pair.action == "reject":
                    pair.grade = "reject"
                else:
                    pair.grade = "quarantine"
                    pair.action = "quarantine"
            elif pair.claims and any(str(item.get("status")) not in _PASS_CLAIM for item in pair.claims):
                pair.grade = "quarantine"
                pair.action = "quarantine"
            elif (
                pair.evidence_state == "sufficient"
                and pair.chunk_text
                and not is_substring(pair.evidence_span, pair.chunk_text)
            ):
                pair.grade = "quarantine"
                pair.action = "quarantine"
            elif pair.claims and all(str(item.get("status")) in _PASS_CLAIM for item in pair.claims):
                pair.grade = "S"
                pair.selection_role = pair.selection_role or "learning"
            else:
                pair.grade = "A"
                pair.selection_role = pair.selection_role or "learning"
            pair.filter_trace["grading"] = {
                "mode": "validity_tier",
                "grade": pair.grade,
                "role": pair.selection_role,
            }
            pair.log("grading", self.name, grade=pair.grade)
            out.append(pair)
        return out
