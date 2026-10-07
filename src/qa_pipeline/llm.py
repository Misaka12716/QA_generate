"""OpenAI 兼容教师客户端。支持多模型名、JSON 输出、用量统计与 FakeLLM。"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from .schemas import request_id
from .textutil import approx_tokens

logger = logging.getLogger(__name__)

_PRICE = {
    "default": (0.14 / 1e6, 0.28 / 1e6),
    "qwen3.8-27b": (0.0, 0.0),
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
    file_key, _file_base = _read_gpt_api_file()
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("QA_PIPELINE_API_KEY") or file_key or "EMPTY"
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("QA_PIPELINE_BASE_URL")
        or "http://192.168.4.110:4000/v1"
    )
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("QA_PIPELINE_MODEL") or "qwen3.8-27b"
    return key, base, model


class LLMResponse:
    """一次教师调用的终态。失败不再伪装成空字典。"""

    def __init__(
        self,
        status: str,
        data: dict[str, Any] | None = None,
        *,
        model: str = "",
        request_id_value: str = "",
        finish_reason: str = "",
        raw_summary: str = "",
        error: str = "",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        token_source: str = "unavailable",
    ) -> None:
        self.status = status
        self.data = data
        self.model = model
        self.request_id = request_id_value or request_id("llm_")
        self.finish_reason = finish_reason
        self.raw_summary = raw_summary[:240]
        self.error = error
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.token_source = token_source

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "raw_summary": self.raw_summary,
            "error": self.error,
        }


def json_payload(result: LLMResponse | dict | None) -> dict[str, Any]:
    if isinstance(result, LLMResponse):
        if result.status == "ok" and isinstance(result.data, dict):
            return result.data
        return {}
    if isinstance(result, dict):
        return result
    return {}


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
        self.budget_cap_usd: float | None = None
        self.max_calls: int | None = None
        self.max_prompt_tokens: int | None = None
        self.budget_stops: list[str] = []
        self.call_log: list[dict[str, Any]] = []
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

    def over_budget(self) -> bool:
        cap = self.budget_cap_usd
        if cap is not None and self.estimated_cost_usd >= float(cap):
            if "budget_cap" not in self.budget_stops:
                self.budget_stops.append("budget_cap")
            return True
        if self.max_calls is not None and self.calls >= int(self.max_calls):
            if "call_cap" not in self.budget_stops:
                self.budget_stops.append("call_cap")
            return True
        if self.max_prompt_tokens is not None and self.prompt_tokens >= int(self.max_prompt_tokens):
            if "token_cap" not in self.budget_stops:
                self.budget_stops.append("token_cap")
            return True
        return False

    def _stopped(self, model: str) -> LLMResponse:
        return LLMResponse(status="budget_stopped", model=model or self.default_model, finish_reason="budget")

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
        if self.over_budget():
            self.call_log.append({"status": "budget_stopped", "model": model})
            return ""
        client = self._get()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
    ) -> LLMResponse:
        model = model or self.default_model
        if self.over_budget():
            result = self._stopped(model)
            self.call_log.append(result.as_dict())
            return result
        client = self._get()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            reported_pin = getattr(usage, "prompt_tokens", None) if usage is not None else None
            reported_pout = getattr(usage, "completion_tokens", None) if usage is not None else None
            if reported_pin is None or reported_pout is None:
                pin = sum(approx_tokens(m["content"]) for m in messages)
                pout = approx_tokens(raw)
                token_source = "approximated"
            else:
                pin = int(reported_pin)
                pout = int(reported_pout)
                token_source = "provider"
            self._charge(model, pin, pout)
            choice = resp.choices[0]
            finish = str(getattr(choice, "finish_reason", "") or "")
        except Exception as exc:
            logger.warning("chat_json failed: %s", exc)
            result = LLMResponse(status="transport_failed", model=model, error=str(exc), token_source="unavailable")
            self.call_log.append(result.as_dict())
            return result
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            result = LLMResponse(
                status="parse_failed",
                model=model,
                finish_reason=finish,
                raw_summary=raw,
                prompt_tokens=pin,
                completion_tokens=pout,
                token_source=token_source,
            )
            self.call_log.append(result.as_dict())
            return result
        if not isinstance(data, dict):
            result = LLMResponse(
                status="schema_failed",
                model=model,
                finish_reason=finish,
                raw_summary=raw,
                error=type(data).__name__,
                prompt_tokens=pin,
                completion_tokens=pout,
                token_source=token_source,
            )
            self.call_log.append(result.as_dict())
            return result
        result = LLMResponse(
            status="ok",
            data=data,
            model=model,
            finish_reason=finish,
            raw_summary=raw,
            prompt_tokens=pin,
            completion_tokens=pout,
            token_source=token_source,
        )
        self.call_log.append(result.as_dict())
        return result

    def reset_usage(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "default_model": self.default_model,
        }


def _scripted_usage(item: dict[str, Any]) -> tuple[int, int, str]:
    """脚本可声明本次用量。未声明时记 1+1，避免假调用把近似长度当成供应商用量。"""
    if "__prompt_tokens" in item or "__completion_tokens" in item:
        prompt = int(item.pop("__prompt_tokens", 0) or 0)
        completion = int(item.pop("__completion_tokens", 0) or 0)
        return prompt, completion, "provider"
    return 1, 1, "unspecified"


class FakeLLM(LLMClient):
    """确定性假教师：按 system prompt 关键词返回结构化结果，供单测与离线 smoke。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake", base_url="http://fake", default_model="fake-teacher")
        self.script: list[dict[str, Any]] = []
        self.history: list[list[dict[str, str]]] = []

    def chat(self, messages, model=None, temperature=0.3, max_tokens=2000) -> str:
        if self.over_budget():
            return ""
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
        model_name = model or self.default_model
        if self.over_budget():
            result = self._stopped(model_name)
            self.call_log.append(result.as_dict())
            return result
        self.history.append(messages)
        blob = " ".join(m.get("content", "") for m in messages)
        user = messages[-1]["content"] if messages else ""
        if getattr(self, "fail_mode", ""):
            result = LLMResponse(status=self.fail_mode, model=model_name, error=self.fail_mode, raw_summary=self.fail_mode)
            self.call_log.append(result.as_dict())
            self._charge(model_name, 1, 1)
            return result
        if self.script:
            item = self.script.pop(0)
            status = str(item.get("__status") or "")
            scripted = _scripted_usage(item)
            if status:
                result = LLMResponse(
                    status=status,
                    model=model_name,
                    error=str(item.get("__error") or status),
                    raw_summary=str(item.get("__raw") or ""),
                    prompt_tokens=scripted[0],
                    completion_tokens=scripted[1],
                    token_source=scripted[2],
                )
                self.call_log.append(result.as_dict())
                self._charge(model_name, scripted[0], scripted[1])
                return result
            raw = json.dumps(item, ensure_ascii=False)
            pin = sum(approx_tokens(m["content"]) for m in messages)
            self._charge(model_name, pin, approx_tokens(raw))
            result = LLMResponse(
                status="ok",
                data=item,
                model=model_name,
                finish_reason="stop",
                raw_summary=raw,
                prompt_tokens=scripted[0],
                completion_tokens=scripted[1],
                token_source=scripted[2],
            )
            self.call_log.append(result.as_dict())
            return result
        pin = sum(approx_tokens(m["content"]) for m in messages)
        data = _fake_json(blob, user)
        raw = json.dumps(data, ensure_ascii=False)
        self._charge(model_name, pin, approx_tokens(raw))
        result = LLMResponse(status="ok", data=data, model=model_name, finish_reason="stop", raw_summary=raw)
        self.call_log.append(result.as_dict())
        return result


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
    if "审核任务" in blob:
        return {
            "decision": "abstain",
            "dimensions": {},
            "claims": [],
            "reason_codes": ["fake_unscripted"],
            "evidence_summary": "未提供脚本裁定",
            "contradictions": [],
        }
    if "知识单元" in blob:
        return {
            "units": [
                {
                    "proposition": sent,
                    "conditions": [],
                    "exceptions": [],
                    "evidence": sent,
                    "intent_primary": "lookup_explain",
                }
            ],
            "samples": [
                {
                    "question": "请根据给定资料说明该规定的具体要求是什么？",
                    "candidate_answer": sent,
                    "answer_points": [sent],
                    "evidence": sent,
                    "intent_primary": "lookup_explain",
                }
            ],
        }
    if "planned_grounded" in blob:
        requested = "factual"
        match = re.search(r"requested_q_type=([a-z_]+)", user)
        action_match = re.search(r"expected_action=([a-z_]+)", user)
        action = action_match.group(1) if action_match else "answer"
        if match:
            requested = match.group(1)
        questions = {
            "factual": "示例制剂甲每片含有多少主成分？",
            "procedural": "使用示例制剂甲时需要按什么顺序完成哪些步骤？",
            "conditional": "肾功能不全者在什么条件下禁用示例制剂甲，有什么例外？",
            "comparative": "示例制剂甲和示例制剂乙在每片主成分含量上有什么不同？",
            "multihop": "结合示例制剂甲和示例制剂乙的含量，合并前需要确认哪些事实？",
        }
        behavior_questions = {
            "state_insufficient": "资料没有记载的患者体重换算剂量是多少？",
            "partial_answer": "示例制剂甲的含量和未记载的儿童剂量分别是多少？",
            "clarify": "该药指的是哪一个制剂？",
            "correct_premise": "示例制剂甲每片含主成分 100 mg，这个前提对吗？",
            "state_scope": "请推荐一种资料范围以外的新药，并说明资料里的含量边界。",
        }
        question = behavior_questions.get(action) or questions.get(requested) or questions["factual"]
        return {
            "samples": [
                {
                    "question": question,
                    "candidate_answer": sent,
                    "answer_points": [sent] if sent else [],
                    "evidence": sent,
                    "intent_primary": "lookup_explain",
                    "operations": ["抽取"],
                    "evidence_topology": "single",
                    "requested_q_type": requested,
                    "expected_action": action,
                }
            ]
        }
    if "候选答案" in blob or "direct_grounded" in blob:
        return {
            "samples": [
                {
                    "question": "请根据给定资料说明该规定的具体要求是什么？",
                    "candidate_answer": sent,
                    "answer_points": [sent],
                    "evidence": sent,
                    "intent_primary": "lookup_explain",
                    "operations": ["抽取"],
                    "evidence_topology": "single",
                }
            ]
        }
    if "主张" in blob or "升级验证" in blob:
        return {"claims": [{"text": sent or "答案要点", "status": "supported"}]}
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
        types = ["factual", "procedural", "conditional", "comparative", "multihop"]
        qs = []
        for i, a in enumerate(anchors[:3]):
            qs.append(
                {
                    "question": f"{a}的作用是什么？",
                    "anchor": a,
                    "q_type": types[i % len(types)],
                    "evidence_span": sent,
                    "answer": sent,
                }
            )
            qs.append(
                {
                    "question": f"操作{a}时需要遵循哪些步骤？",
                    "anchor": a,
                    "q_type": types[(i + 1) % len(types)],
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
        return {"question": evolved, "evolution_type": "加约束", "q_type": "multihop"}
    if "NLI" in blob or "蕴含" in blob or "entailment" in blob.lower():
        return {"entailment": 0.88, "label": "entailment", "supported": True}
    if "质检" in blob or "supported" in blob or "是否被" in blob:
        return {"supported": True, "reason": "答案可由原文支持"}
    if "打分" in blob or "Judge" in blob or "相关性" in blob:
        return {
            "accuracy": 5,
            "relevancy": 5,
            "completeness": 5,
            "quality": 5,
            "relevance": 5,
            "correctness": 5,
            "fluency": 5,
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


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


class LocalLLM(LLMClient):
    """用本地 CausalLM 作为教师。API 费用记 0，仍统计 token。"""

    def __init__(self, model_path: str, device: str | None = None, max_new_tokens: int = 512) -> None:
        super().__init__(api_key="local", base_url="local", default_model=str(model_path))
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._tokenizer = None

    def _charge(self, model: str, pin: int, pout: int) -> None:
        self.calls += 1
        self.prompt_tokens += int(pin)
        self.completion_tokens += int(pout)

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        use_cuda = torch.cuda.is_available()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map={"": 0} if use_cuda else None,
            trust_remote_code=True,
        )
        self._model.eval()

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        import torch

        self._ensure()
        tokenizer = self._tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        limit = min(int(max_tokens), self.max_new_tokens)
        do_sample = temperature > 0.05
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": limit,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pin = int(inputs["input_ids"].shape[1])
        self._charge(model or self.default_model, pin, max(1, out.shape[1] - pin))
        return text

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> LLMResponse:
        model_name = model or self.default_model
        text = self.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        if not text and self.over_budget():
            return self._stopped(model_name)
        data = _extract_json_obj(text)
        if data is not None:
            return LLMResponse(status="ok", data=data, model=model_name, finish_reason="stop", raw_summary=text)
        retry = self.chat(
            messages + [{"role": "user", "content": "上一次没有输出合法 JSON。请只输出一个 JSON 对象。"}],
            model=model,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        data = _extract_json_obj(retry)
        if data is not None:
            return LLMResponse(status="ok", data=data, model=model_name, finish_reason="stop", raw_summary=retry)
        return LLMResponse(status="parse_failed", model=model_name, raw_summary=text or retry)
