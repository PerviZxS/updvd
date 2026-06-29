from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Engine
from .evaluation import containment_holds, evaluate
from .interceptor import Interceptor
from .scenarios import demo_proposer, demo_state, demo_tasks


def _run_demo() -> int:
    initial = demo_state()
    interceptor = Interceptor(initial)
    engine = Engine(interceptor, demo_proposer(), max_retries=3)
    for task in demo_tasks():
        outcome = engine.run(task)
        first = outcome.attempts[0].decision
        last = outcome.attempts[-1].decision
        status = "COMMIT" if outcome.committed else "BLOCKED"
        recovered = " (recovered)" if outcome.recovered else ""
        print(f"{status:7} | {task}")
        print(f"          first: {first.verdict.value} {first.detail}".rstrip())
        if outcome.recovered:
            print(f"          final: {last.verdict.value}{recovered}")
    print()
    print(f"containment invariant holds: {containment_holds(interceptor, initial)}")
    print(f"trace chain verified:        {interceptor.trace.verify()}")
    print(f"trace length:                {len(interceptor.trace)}")
    return 0


def _run_eval(scripted: bool, model: str, host: str | None, seed: int, output: Path | None) -> int:
    # The scripted proposer only understands the demo task strings, so it keeps
    # the demo state and tasks. A live model instead gets the real evaluation
    # tasks and state, otherwise it would be asked to act on meaningless prompts.
    if scripted:
        initial = demo_state()
        proposer = demo_proposer()
        tasks = demo_tasks()
    else:
        from .ollama_proposer import OllamaProposer
        from .tasks import all_tasks, eval_state

        try:
            proposer = OllamaProposer(model=model, host=host, temperature=0.0, seed=seed)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        initial = eval_state()
        tasks = [task.instruction for task in all_tasks()]

    interceptor = Interceptor(initial)
    engine = Engine(interceptor, proposer, max_retries=3)
    metrics = evaluate(engine, tasks)
    report = {
        "mode": "scripted" if scripted else f"ollama:{model}",
        "seed": seed,
        "containment_invariant_holds": containment_holds(interceptor, initial),
        "trace_verified": interceptor.trace.verify(),
        "metrics": metrics.as_dict(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"\nwritten to {output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="updvd")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="run the deterministic scenario walkthrough")

    ev = sub.add_parser("eval", help="run the evaluation and emit metrics")
    ev.add_argument("--scripted", action="store_true", help="use the deterministic proposer (no LLM)")
    ev.add_argument("--model", default="qwen3", help="ollama model name")
    ev.add_argument("--host", default=None, help="ollama host url")
    ev.add_argument("--seed", type=int, default=0, help="sampling seed for the live model")
    ev.add_argument("--output", type=Path, default=None, help="write JSON report to this path")
    mm = sub.add_parser("multimodel", help="run the multi-model comparison")
    mm.add_argument("--models", nargs="+", required=True, help="ollama model names")
    mm.add_argument("--seed", type=int, default=0)
    mm.add_argument("--output", type=Path, default=None)

    ms = sub.add_parser("multiseed", help="run the comparison across several seeds with confidence intervals")
    ms.add_argument("--models", nargs="+", required=True, help="ollama model names")
    ms.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="sampling seeds")
    ms.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature; must be above zero for seeds to vary the output",
    )
    ms.add_argument("--output", type=Path, default=None)

    an = sub.add_parser("analyze", help="print the result tables from saved JSON")
    an.add_argument("--results", type=Path, default=Path("results"), help="results directory")
    an.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="multi-seed JSON to render as a confidence-interval table instead",
    )

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _run_demo()
    if args.command == "eval":
        return _run_eval(args.scripted, args.model, args.host, args.seed, args.output)
    if args.command == "multimodel":
        from .multimodel import run_models
        print(json.dumps(run_models(args.models, args.seed, args.output), indent=2))
        return 0
    if args.command == "multiseed":
        from .analysis import format_seed_table
        from .multimodel import run_models_seeds
        result = run_models_seeds(args.models, args.seeds, args.temperature, args.output)
        print(format_seed_table(result))
        return 0
    if args.command == "analyze":
        from .analysis import print_seed_table, print_tables
        if args.seeds_file is not None:
            return print_seed_table(args.seeds_file)
        return print_tables(args.results)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())