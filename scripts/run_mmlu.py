"""Measure MMLU for the nine study models under one protocol, as stated in the paper.

lm-evaluation-harness 0.4.12, 0-shot, log-likelihood scoring, no chat template, a fixed 10%
of each subject (1430 items), every model loaded by the harness in 4-bit with only
load_in_4bit=True (the bitsandbytes default 4-bit type), batch size 4, all seeds 1234.
Run from the repository root after `pip install -r requirements-colab.txt` (see README).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lm_eval

from updvd import mmlu

# Longer Hub timeout so a slow dataset metadata call does not fail the run.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# Errors treated as transient and retried; anything else (e.g. OOM) fails immediately.
try:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    _TRANSIENT: tuple = (HfHubHTTPError, httpx.HTTPError, ConnectionError, TimeoutError)
except Exception:  # pragma: no cover
    _TRANSIENT = (ConnectionError, TimeoutError)

# Ollama tag -> Hugging Face repo. The keys match the behavioral results, so both join on one name.
# The Meta and Google repos are gated (403 without an accepted license).
MODELS = {
    "gemma2:2b": "google/gemma-2-2b-it",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma3:4b": "google/gemma-3-4b-it",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3:8b": "Qwen/Qwen3-8B",
}

# Protocol settings, the same for every model. Scores measured with other values are not comparable.
NUM_FEWSHOT = 0
LIMIT = 0.10
BATCH_SIZE = 4
SEED = 1234
OUTPUT_DIR = Path("results/mmlu")


# Harness output path for one model, e.g. results/mmlu/qwen3-8b/results.json.
def _output_file(name: str) -> Path:
    return OUTPUT_DIR / name.replace(":", "-") / "results.json"


# The exact arguments passed to lm_eval.simple_evaluate for one model.
def call_args(repo: str) -> dict:
    return {
        "model": "hf",
        "model_args": f"pretrained={repo},load_in_4bit=True",
        "tasks": ["mmlu"],
        "num_fewshot": NUM_FEWSHOT,
        "limit": LIMIT,
        "batch_size": BATCH_SIZE,
        "device": "cuda:0",
        "apply_chat_template": False,
        "random_seed": SEED,
        "numpy_random_seed": SEED,
        "torch_random_seed": SEED,
        "fewshot_random_seed": SEED,
    }


# Scores one model with the harness loading it (same path as the lm_eval command line).
def score(repo: str, attempts: int = 12) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            return lm_eval.simple_evaluate(**call_args(repo))
        except _TRANSIENT as exc:
            # Transient Hub errors are retried with linear backoff; cached subjects are not refetched.
            if attempt == attempts:
                raise
            wait = min(120, 15 * attempt)
            print(f"  transient Hugging Face error (attempt {attempt}/{attempts}); "
                  f"retrying in {wait}s. {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# Scores the selected models (default: all nine), skipping any that already have a result file.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure MMLU for the study models under the paper's protocol.")
    parser.add_argument("--model", action="append", metavar="NAME",
                        help=f"score only this model (repeatable); default is all nine. Choices: {', '.join(MODELS)}")
    args = parser.parse_args(argv)

    selected = args.model or list(MODELS)
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}; choices: {', '.join(MODELS)}", flush=True)
        return 2

    for name in selected:
        out = _output_file(name)
        if out.exists():
            print(f"=== {name}: already scored ({out}), skipping ===", flush=True)
        else:
            print(f"=== {name} ({MODELS[name]}) ===", flush=True)
            results = score(MODELS[name])
            out.parent.mkdir(parents=True, exist_ok=True)
            # Harness output as returned, minus the per-sample records, plus the exact call.
            payload = {key: value for key, value in results.items() if key != "samples"}
            payload["simple_evaluate_call"] = call_args(MODELS[name])
            out.write_text(json.dumps(payload, indent=1, default=str))
        # Read back from the saved file, so mmlu.json points at the file each score came from.
        rec = mmlu.record(name, out)
        print(f"{name}: MMLU = {rec['acc'] * 100:.2f}%", flush=True)

    print()
    mmlu.main(["show"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
