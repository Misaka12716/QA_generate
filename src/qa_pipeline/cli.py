"""qa-pipeline CLI：run / experiment / export / list-strategies。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .adapters.zhixun import ReleaseRejected, export_zhixun
from .config import load_recipe
from .experiments.runner import run_suite
from .llm import FakeLLM, LLMClient
from .pipeline import Pipeline
from .registry import list_strategies
from .store import load_pairs, save_result


def _devices(args) -> tuple[list[int] | None, str]:
    if getattr(args, "fake", False):
        return None, ""
    raw = getattr(args, "devices", None)
    requested = [int(part) for part in str(raw).split(",") if part.strip()] if raw else None
    if not getattr(args, "local_model", None) and not getattr(args, "sft", False):
        return requested, ""
    from .experiments.devices import select_trainable_gpus

    return select_trainable_gpus(requested)


def _client(args) -> LLMClient:
    if getattr(args, "fake", False):
        return FakeLLM()
    return LLMClient(
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
        default_model=getattr(args, "model", None),
    )


def cmd_run(args) -> int:
    recipe = load_recipe(args.recipe)
    pipe = Pipeline(recipe, llm=_client(args))
    result = pipe.run(args.input)
    out = Path(args.out or f"runs/{recipe.name}")
    save_result(result, out)
    export_zhixun(result.pairs, out / "zhixun.jsonl")
    print(json.dumps(result.stats.model_dump(), ensure_ascii=False, indent=2))
    print(f"kept={len(result.pairs)} rejected={len(result.rejected)} -> {out}")
    return 0


def cmd_experiment(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    devices, device_error = _devices(args)
    if device_error:
        print(device_error, file=sys.stderr)
        return 1
    if (getattr(args, "local_model", None) or args.sft) and not args.fake and not devices:
        print("没有空闲显存达到约 18GB 的 GPU。占用低于 2GB 只表示没有其他大进程。", file=sys.stderr)
        return 1
    payload = run_suite(
        args.suite,
        out_dir=args.out,
        fake=args.fake,
        skip_sft=not args.sft,
        llm=_client(args),
        base_model=getattr(args, "local_model", None),
        devices=devices,
    )
    if payload.get("aborted"):
        print(payload.get("note") or payload.get("reason"), file=sys.stderr)
        return 1
    print(json.dumps({"suite": payload.get("suite"), "n": len(payload.get("experiments") or [])}, ensure_ascii=False))
    out = Path(args.out) if args.out else None
    if out:
        print(f"report: {out / 'report.md'}")
    return 0


def cmd_export(args) -> int:
    pairs = load_pairs(args.run if args.run.endswith(".jsonl") else str(Path(args.run) / "qa.kept.jsonl"))
    dest = Path(args.out or Path(args.run).parent / "zhixun.jsonl")
    try:
        export_zhixun(pairs, dest, split=args.split)
    except ReleaseRejected as exc:
        print(f"拒绝导出：{exc}", file=sys.stderr)
        return 1
    print(f"exported {len(pairs)} rows -> {dest}")
    return 0


def cmd_demo(args) -> int:
    try:
        from .demo.app import serve
    except ImportError as exc:
        print("缺少 Demo 依赖。请执行: pip install -e \".[demo]\"", file=sys.stderr)
        raise SystemExit(1) from exc
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"实验目录不存在: {run_dir}", file=sys.stderr)
        return 1
    print(f"对照台: http://{args.host}:{args.port}  数据: {run_dir.resolve()}")
    serve(run_dir, host=args.host, port=args.port)
    return 0


def _formal_source_arguments(args):
    """正式模式读取冻结训练来源清单和来源宇宙。探索模式不借此改称正式结果。"""
    from .experiments.adapter_eval import (
        ProtocolError,
        load_frozen_train_families,
        load_source_universe,
        refuse_source_list,
        trace_source_list,
    )

    if args.mode != "formal":
        return {}, None
    if not getattr(args, "frozen_train_families", None):
        return {"frozen_train_families": None}, None
    frozen_sha = None
    universe_sha = None
    try:
        manifest = load_frozen_train_families(args.frozen_train_families)
        frozen_sha = manifest["sha256"]
        if not getattr(args, "source_inventory", None):
            raise ProtocolError("invalid_source_list:universe_missing")
        universe = load_source_universe(args.source_inventory)
        universe_sha = universe["sha256"]
        traced = trace_source_list(manifest, universe)
        if not traced["ok"]:
            raise ProtocolError(str(traced["reason"]))
    except ProtocolError as exc:
        reason = str(exc)
        report = refuse_source_list(
            args.out,
            reason,
            mode=args.mode,
            protocol_path=args.protocol,
            frozen_train_families_sha256=frozen_sha or getattr(exc, "sha256", None),
            source_universe_sha256=universe_sha or getattr(exc, "universe_sha256", None),
        )
        print(json.dumps({"status": report.get("status"), "executed": False, "reason": reason, "out": args.out}, ensure_ascii=False))
        return None, 1
    return {
        "frozen_train_families": manifest["ids"],
        "source_universe": universe["families"],
        "frozen_train_families_sha256": frozen_sha,
        "source_universe_sha256": universe_sha,
    }, None


def cmd_eval_adapter(args) -> int:
    import hashlib

    from .experiments.adapter_eval import ProtocolError, eval_adapter

    template_path = Path(args.base_model) / "tokenizer_config.json"
    if template_path.is_file():
        template_id = hashlib.sha256(template_path.read_bytes()).hexdigest()[:16]
    else:
        print(f"找不到 tokenizer 配置: {template_path}", file=sys.stderr)
        return 1
    infer = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "dtype": "bfloat16",
    }
    source_kwargs, source_code = _formal_source_arguments(args)
    if source_code is not None:
        return source_code
    try:
        report = eval_adapter(
            base_model=args.base_model,
            base_id=args.base_id,
            adapter=args.adapter,
            adapter_id=args.adapter_id,
            protocol_path=args.protocol,
            out_dir=args.out,
            infer_config=infer,
            scorer_version=args.scorer,
            mode=args.mode,
            template_id=template_id,
            device=args.device,
            **source_kwargs,
        )
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": report.get("status"), "executed": report.get("executed"), "out": args.out}, ensure_ascii=False))
    return 0 if report.get("executed") else 2


def cmd_import_review(args) -> int:
    from .experiments.review_io import import_run_reviews

    result = import_run_reviews(args.run, batch=args.batch or None)
    print(json.dumps({"error_n": result.get("error_n"), "pending_review_n": result.get("pending_review_n"), "adjudicated_n": result.get("adjudicated_n"), "ready_for_inference": result.get("ready_for_inference"), "out": args.run}, ensure_ascii=False))
    return 0 if not result.get("error_n") else 1


def cmd_list(args) -> int:
    data = list_strategies()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qa-pipeline", description="可拔插 QA 生成 / 蒸馏 / 知识过滤")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_llm(p):
        p.add_argument("--fake", action="store_true", help="使用 FakeLLM，不调用真实 API")
        p.add_argument("--local-model", dest="local_model", default=None, help="本地 HuggingFace 目录，仅作 LoRA 学生基座。教师走已配置的 API")
        p.add_argument("--model", default=None)
        p.add_argument("--api-key", dest="api_key", default=None)
        p.add_argument("--base-url", dest="base_url", default=None)

    run_p = sub.add_parser("run", help="按 recipe 跑完整管线")
    run_p.add_argument("--recipe", required=True)
    run_p.add_argument("--input", required=True, help="文档文件或目录")
    run_p.add_argument("--out", default=None)
    add_llm(run_p)
    run_p.set_defaults(func=cmd_run)

    exp_p = sub.add_parser("experiment", help="跑方案组合实验套件")
    exp_p.add_argument("--suite", required=True)
    exp_p.add_argument("--out", default=None)
    exp_p.add_argument("--sft", action="store_true", help="同时跑 LoRA SFT 对比（需 GPU 与 extras）")
    exp_p.add_argument(
        "--devices",
        default=None,
        help="逗号分隔的 GPU 编号。默认选用显存占用低于 2GB 的卡",
    )
    add_llm(exp_p)
    exp_p.set_defaults(func=cmd_experiment)

    exp_e = sub.add_parser("export", help="把 kept QA 导出为智训 JSONL")
    exp_e.add_argument("--run", required=True, help="run 目录或 qa.kept.jsonl")
    exp_e.add_argument("--out", default=None)
    exp_e.add_argument("--split", default=None)
    exp_e.set_defaults(func=cmd_export)

    demo_p = sub.add_parser("demo", help="打开方案对照台，读取已有实验结果")
    demo_p.add_argument("--run", default="runs/ablation", help="experiment 输出目录")
    demo_p.add_argument("--host", default="127.0.0.1")
    demo_p.add_argument("--port", type=int, default=8765)
    demo_p.set_defaults(func=cmd_demo)

    ev = sub.add_parser("eval-adapter", help="评测已有 adapter，不训练、不重写 train.jsonl")
    ev.add_argument("--base-model", required=True)
    ev.add_argument("--base-id", required=True)
    ev.add_argument("--adapter", required=True)
    ev.add_argument("--adapter-id", required=True)
    ev.add_argument("--protocol", required=True)
    ev.add_argument("--out", required=True)
    ev.add_argument("--mode", choices=("formal", "exploratory"), required=True)
    ev.add_argument("--scorer", default="aux-rules-v2")
    ev.add_argument("--max-new-tokens", type=int, default=512)
    ev.add_argument("--device", type=int, default=None)
    ev.add_argument("--frozen-train-families", default=None, help="正式模式使用的冻结训练来源清单 JSON。空清单无效")
    ev.add_argument("--source-inventory", default=None, help="正式模式用来核对来源家族是否可追溯的宇宙文件")
    ev.set_defaults(func=cmd_eval_adapter)

    review_p = sub.add_parser("import-review", help="导入人工审核 CSV，并校验 ID、内容哈希和审核完整性")
    review_p.add_argument("--run", required=True, help="含 review/batches.json 的运行目录")
    review_p.add_argument("--batch", default="", help="只导入指定批次；默认导入清单中的全部批次")
    review_p.set_defaults(func=cmd_import_review)

    ls = sub.add_parser("list-strategies", help="列出已注册策略")
    ls.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
