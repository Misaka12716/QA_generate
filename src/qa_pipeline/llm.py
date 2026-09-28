"""OpenAI 兼容教师客户端。支持多模型名、JSON 输出、用量统计与 FakeLLM。"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from .textutil import approx_tokens

logger = logging.getLogger(__name__)

_PRICE = {
    "default": (0.14 / 1e6, 0.28 / 1e6),
    "deepseek-chat": (0.14 / 1e6, 0.28 / 1e6),
    "deepseek-reasoner": (0.55 / 1e6, 2.19 / 1e6),
    "gpt-4o-mini": (0.15 / 1e6, 0.60 / 1e6),
    "gpt-4o": (2.50 / 1e6, 10.00 / 1e6),
}


def _read_gpt_api_file() -> tuple[str, str]:
    key, base = "", "https://api.deepseek.com"
    for candidate in (
        Path.cwd() / "gpt_api",
        Path("/home/mmc/workspace/data_governance/gpt_api"),
        Path(__file__).resolve().parents[2] / "gpt_api",
    ):
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"OPENAI_API_KEY\s*=\s*(\S+)", text)
        if m:
            key = m.group(1).strip().strip("\"'")
        if "deepseek" in text.lower():
            base = "https://api.deepseek.com"
        break
    return key, base


def default_credentials() -> tuple[str, str, str]:
    file_key, file_base = _read_gpt_api_file()
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QA_PIPELINE_API_KEY") or file_key
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("QA_PIPELINE_BASE_URL") or file_base
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("QA_PIPELINE_MODEL") or "deepseek-chat"
    return key, base, model


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        key, base, model = default_credentials()
        self.api_key = api_key if api_key is not None else key
        self.base_url = base_url if base_url is not None else base
        self.default_model = default_model or model
        self.timeout = timeout
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0
        self._client = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI

                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        max_retries=0,
                    )
        return self._client

    def _charge(self, model: str, pin: int, pout: int) -> None:
        self.calls += 1
        self.prompt_tokens += pin
        self.completion_tokens += pout
        pin_p, pout_p = _PRICE.get(model, _PRICE["default"])
        self.estimated_cost_usd += pin * pin_p + pout * pout_p

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        model = model or self.default_model
        client = self._get()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pin = getattr(usage, "prompt_tokens", None) or sum(approx_tokens(m["content"]) for m in messages)
        pout = getattr(usage, "completion_tokens", None) or approx_tokens(text)
        self._charge(model, pin, pout)
        return text

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> dict[str, Any] | None:
        model = model or self.default_model
        client = self._get()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            pin = getattr(usage, "prompt_tokens", None) or sum(approx_tokens(m["content"]) for m in messages)
            pout = getattr(usage, "completion_tokens", None) or approx_tokens(raw)
            self._charge(model, pin, pout)
            return json.loads(raw)
        except Exception as exc:
            logger.warning("chat_json failed: %s", exc)
            return None

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "default_model": self.default_model,
        }


class FakeLLM(LLMClient):
    """确定性假教师：按 system prompt 关键词返回结构化结果，供单测与离线 smoke。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake", base_url="http://fake", default_model="fake-teacher")
        self.script: list[dict[str, Any]] = []
        self.history: list[list[dict[str, str]]] = []

    def chat(self, messages, model=None, temperature=0.3, max_tokens=2000) -> str:
        self.history.append(messages)
        blob = " ".join(m.get("content", "") for m in messages)
        user = messages[-1]["content"] if messages else ""
        pin = sum(approx_tokens(m["content"]) for m in messages)
        if "遮住" in blob or "不要给出参考文本" in blob or "无上下文" in blob:
            text = "根据已有常识无法确定具体参数与规程，需要查阅给定材料。"
        elif "逐步推理" in blob or "Chain of Thought" in blob:
            text = json.dumps(
                {
                    "answer": _pick_sentence(user),
                    "reasoning": "先定位参考文本中的相关句子，再归纳结论。",
                    "evidence_span": _pick_sentence(user),
                    "confidence": 0.86,
                },
                ensure_ascii=False,
            )
        else:
            text = _pick_sentence(user) or "材料中给出了对应说明。"
        self._charge(model or self.default_model, pin, approx_tokens(text))
        return text

    def chat_json(self, messages, model=None, temperature=0.2, max_tokens=3000):
        self.history.append(messages)
        blob = " ".join(m.get("content", "") for m in messages)
        user = messages[-1]["content"] if messages else ""
        pin = sum(approx_tokens(m["content"]) for m in messages)
        data = _fake_json(blob, user)
        raw = json.dumps(data, ensure_ascii=False)
        self._charge(model or self.default_model, pin, approx_tokens(raw))
        return data


def _pick_sentence(text: str) -> str:
    from .textutil import sentences

    body = text
    for marker in ("文本块：", "参考文本：", "资料：", "原文："):
        if marker in text:
            body = text.split(marker, 1)[-1]
            break
    sents = sentences(body)
    for s in sents:
        if 8 <= len(s) <= 80 and "问题" not in s[:4]:
            return s.strip()
    return (sents[0].strip() if sents else body[:80].strip())


def _extract_listed_items(user: str, label: str) -> list[str]:
    items = []
    if label in user:
        rest = user.split(label, 1)[-1]
        for line in rest.splitlines():
            line = line.strip(" -•\t")
            if line and not line.endswith("：") and len(line) < 80:
                items.append(re.sub(r"^[\d.、)]+\s*", "", line))
            if len(items) >= 4:
                break
    return items


def _fake_json(blob: str, user: str) -> dict[str, Any]:
    sent = _pick_sentence(user)
    if "锚点" in blob and "生成问题" not in blob and "questions" not in blob.lower():
        kws = _extract_listed_items(user, "关键词") or re.findall(r"[\u4e00-\u9fff]{2,8}", user)[:4]
        return {
            "anchors": [
                {"anchor_text": k, "anchor_type": "keyword", "position_in_chunk": i}
                for i, k in enumerate(kws[:3] or ["系统"])
            ]
        }
    if "生成" in blob and ("问题" in blob or "QA" in blob or "问答" in blob):
        anchors = _extract_listed_items(user, "锚点") or ["该模块"]
        qs = []
        for a in anchors[:3]:
            qs.append(
                {
                    "question": f"{a}的作用是什么？",
                    "anchor": a,
                    "q_type": "factual",
                    "evidence_span": sent,
                    "answer": sent,
                }
            )
            qs.append(
                {
                    "question": f"为什么需要{a}？",
                    "anchor": a,
                    "q_type": "explanatory",
                    "evidence_span": sent,
                    "answer": sent,
                }
            )
        return {"questions": qs[:4], "samples": [
            {"q": q["question"], "a": q.get("answer", sent), "evidence": sent, "kind": "事实问答", "block": 1}
            for q in qs[:3]
        ]}
    if "进化" in blob or "Evol" in blob or "改写问题" in blob:
        q = ""
        if "问题：" in user:
            q = user.split("问题：", 1)[-1].splitlines()[0].strip()
        evolved = q.replace("是什么", "在故障条件下应如何处理") if q else "该如何结合上下文分步处理？"
        return {"question": evolved, "evolution_type": "加约束", "q_type": "reasoning"}
    if "NLI" in blob or "蕴含" in blob or "entailment" in blob.lower():
        return {"entailment": 0.88, "label": "entailment", "supported": True}
    if "质检" in blob or "supported" in blob or "是否被" in blob:
        return {"supported": True, "reason": "答案可由原文支持"}
    if "打分" in blob or "Judge" in blob or "相关性" in blob:
        return {
            "relevance": 5,
            "correctness": 4,
            "completeness": 4,
            "fluency": 5,
            "reasoning": 4,
        }
    if "抽取答案片段" in blob or "evidence span" in blob.lower() or "答案跨度" in blob:
        return {"spans": [sent] if sent else []}
    return {
        "answer": sent,
        "reasoning": "根据参考文本归纳。",
        "evidence_span": sent,
        "confidence": 0.9,
        "supported": True,
    }
