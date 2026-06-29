from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Where the consolidated capability scores live. analysis.py reads this file to
# add an MMLU column next to the substitution rates.
STORE = Path("results/mmlu/mmlu.json")


def read_mmlu_acc(output: Path) -> float:
    # lm-evaluation-harness writes results under the --output_path. With a
    # directory it creates <output>/<model>/results_<timestamp>.json. Accept
    # either a results JSON file or a directory and pull the aggregate MMLU acc.
    if output.is_file():
        files = [output]
    else:
        files = sorted(output.rglob("results*.json"))
    if not files:
        raise FileNotFoundError(f"no lm-eval results json found under {output}")
    data = json.loads(files[-1].read_text())
    results = data.get("results", {})
    mmlu = results.get("mmlu") or results.get("mmlu_str") or {}
    for key in ("acc,none", "acc"):
        if key in mmlu:
            return float(mmlu[key])
    raise KeyError(f"no MMLU accuracy in {files[-1]}")


def load_store(store: Path = STORE) -> dict:
    if store.exists():
        return json.loads(store.read_text())
    return {"generated": "", "harness": "lm-evaluation-harness", "num_fewshot": 0, "scores": {}}


def record(model: str, acc: float, store: Path = STORE) -> dict:
    data = load_store(store)
    data["scores"][model] = acc
    data["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(data, indent=2))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="updvd-mmlu")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="extract MMLU acc from an lm-eval run and store it")
    rec.add_argument("--model", required=True, help="model name, must match the eval results key")
    rec.add_argument("--from", dest="src", type=Path, required=True, help="lm-eval --output_path")
    rec.add_argument("--store", type=Path, default=STORE)

    sh = sub.add_parser("show", help="print the stored scores")
    sh.add_argument("--store", type=Path, default=STORE)

    args = parser.parse_args(argv)
    if args.command == "record":
        acc = read_mmlu_acc(args.src)
        record(args.model, acc, args.store)
        print(f"{args.model}: MMLU acc = {acc:.4f} ({acc * 100:.2f}%) -> {args.store}")
        return 0
    if args.command == "show":
        data = load_store(args.store)
        for model, acc in data["scores"].items():
            print(f"{model:14} {acc * 100:.2f}%")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
