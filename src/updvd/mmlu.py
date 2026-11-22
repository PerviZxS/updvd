# Reads lm-evaluation-harness output and records MMLU scores with provenance.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Consolidated MMLU store read by analysis.py (relative to the working directory).
STORE = Path("results/mmlu/mmlu.json")


# Extracts MMLU accuracy and provenance from a harness results file, or the last results*.json under a directory.
def read_mmlu_record(output: Path) -> dict:
    files = [output] if output.is_file() else sorted(output.rglob("results*.json"))
    if not files:
        raise FileNotFoundError(f"no lm-eval results json found under {output}")
    src = files[-1]
    data = json.loads(src.read_text())
    results = data.get("results", {})
    mmlu = results.get("mmlu") or results.get("mmlu_str") or {}
    if "acc,none" not in mmlu and "acc" not in mmlu:
        raise KeyError(f"no MMLU accuracy in {src}")
    cfg = data.get("config", {})
    model_args = cfg.get("model_args", {})
    if isinstance(model_args, str):
        # Older harness builds serialize model_args as "k=v,k=v".
        parsed = {}
        for part in model_args.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                parsed[k.strip()] = v.strip()
        model_args = parsed
    # Store the source path relative to results/ when possible.
    try:
        source = src.relative_to(Path("results")).as_posix()
    except ValueError:
        source = src.as_posix()
    return {
        "acc": float(mmlu.get("acc,none", mmlu.get("acc"))),
        # Left as None when the harness reports no standard error.
        "acc_stderr": float(mmlu["acc_stderr,none"]) if "acc_stderr,none" in mmlu else None,
        "pretrained": model_args.get("pretrained"),
        # Only the load_in_4bit flag is known here. With nothing else set, bitsandbytes uses fp4.
        "quantization": (
            "bitsandbytes-4bit-fp4"
            if model_args.get("load_in_4bit") in (True, "True")
            else "fp16"
        ),
        "limit": cfg.get("limit"),
        "num_fewshot": sorted(set(data.get("n-shot", {}).values())),
        # The harness API does not record this; scripts/run_mmlu.py saves its exact call alongside.
        "apply_chat_template": cfg.get("apply_chat_template", data.get("simple_evaluate_call", {}).get("apply_chat_template")),
        "n_samples": sum(v.get("effective", 0) for v in data.get("n-samples", {}).values()),
        "lm_eval_version": data.get("lm_eval_version"),
        "transformers_version": data.get("transformers_version"),
        "seed": cfg.get("random_seed"),
        "source": source,
    }

# Loads the MMLU store, or an empty skeleton if it does not exist yet.
def load_store(store: Path = STORE) -> dict:
    if store.exists():
        return json.loads(store.read_text())
    return {
        "generated": "",
        "harness": "lm-evaluation-harness",
        "num_fewshot": 0,
        "scores": {},
        "details": {},
    }


# Adds or overwrites one model's score in the store.
def record(model: str, src: Path, store: Path = STORE) -> dict:
    data = load_store(store)
    rec = read_mmlu_record(src)
    data.setdefault("scores", {})[model] = rec["acc"]      # flat score, read by analysis.py
    data.setdefault("details", {})[model] = rec            # full provenance
    data["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(data, indent=2))
    return rec


# `python -m updvd.mmlu record|show`.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="updvd-mmlu")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="extract MMLU acc + provenance from an lm-eval run and store it")
    rec.add_argument("--model", required=True, help="model name, must match the eval results key")
    rec.add_argument("--from", dest="src", type=Path, required=True, help="lm-eval --output_path")
    rec.add_argument("--store", type=Path, default=STORE)

    sh = sub.add_parser("show", help="print the stored scores")
    sh.add_argument("--store", type=Path, default=STORE)

    args = parser.parse_args(argv)
    if args.command == "record":
        r = record(args.model, args.src, args.store)
        # Plain "+/-", because a real plus-minus sign raises UnicodeEncodeError on a cp1252 Windows console.
        stderr = f"+/- {r['acc_stderr'] * 100:.2f}" if r["acc_stderr"] is not None else "+/- n/a"
        print(f"{args.model}: MMLU acc = {r['acc']:.4f} ({r['acc'] * 100:.2f}%) "
              f"{stderr}  n={r['n_samples']}  -> {args.store}")
        return 0
    if args.command == "show":
        data = load_store(args.store)
        details = data.get("details", {})
        for model, acc in data["scores"].items():
            d = details.get(model, {})
            stderr = d.get("acc_stderr")
            n = d.get("n_samples")
            extra = f" +/- {stderr * 100:.2f}  n={n}" if stderr is not None else ""
            print(f"{model:14} {acc * 100:.2f}%{extra}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
