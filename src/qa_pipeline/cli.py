"""qa-pipeline CLI：run / experiment / export / list-strategies。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .adapters.zhixun import export_zhixun
from .config import load_recipe
from .experiments.runner import run_suite
from .llm import FakeLLM, LLMClient, LocalLLM
from .pipeline import Pipeline
from .registry import list_strategies
from .store import load_pairs, save_result


def _client(args) -> LLMClient:
    if getattr(args, "fake", False):
        return FakeLLM()
    local_model = getattr(args, "local_model", None)
    if local_model:
        return LocalLLM(local_model)
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
    payload = run_suite(
        args.suite,
        out_dir=args.out,
        fake=args.fake,
        skip_sft=not args.sft,
        llm=_client(args),
        base_model=getattr(args, "local_model", None),
    )
    print(json.dumps({"suite": payload.get("suite"), "n": len(payload.get("experiments") or [])}, ensure_ascii=False))
    out = Path(args.out) if args.out else None
    if out:
        print(f"report: {out / 'report.md'}")
    return 0


def cmd_export(args) -> int:
    pairs = load_pairs(args.run if args.run.endswith(".jsonl") else str(Path(args.run) / "qa.kept.jsonl"))
    dest = Path(args.out or Path(args.run).parent / "zhixun.jsonl")
    export_zhixun(pairs, dest, split=args.split)
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


def cmd_list(args) -> int:
    data = list_strategies()
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qa-pipeline", description="可拔插 QA 生成 / 蒸馏 / 知识过滤")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_llm(p):
        p.add_argument("--fake", action="store_true", help="使用 FakeLLM，不调用真实 API")
        p.add_argument("--local-model", dest="local_model", default=None, help="本地 HuggingFace 模型目录，同时作为教师和学生基座")
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

    ls = sub.add_parser("list-strategies", help="列出已注册策略")
    ls.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
