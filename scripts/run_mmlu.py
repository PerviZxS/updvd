"""Measure MMLU for the five models under one identical 4-bit setup.

This builds each model with an explicit BitsAndBytesConfig (the API current
transformers wants) and hands the live model to the lm-evaluation-harness, which
avoids the CLI's deprecated load_in_4bit path. Scores are comparable to each
other (same benchmark, shots, harness, quantization), not to model-card numbers.

    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install lm-eval "transformers>=4.55,<5" accelerate bitsandbytes
    pip install -e .
    python scripts/run_mmlu.py

transformers is pinned below 5: its loader crashes with a native access violation
when loading bitsandbytes 4-bit weights on Windows.
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from updvd import mmlu

# A slow link or a busy Hugging Face Hub can time out the dataset metadata call;
# give it longer before that counts as a failure.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# Transient Hub errors (a 504 gateway timeout while listing the MMLU dataset, or a
# dropped connection) are retried rather than killing the run. Anything else, such
# as an out-of-memory error, is left to surface immediately.
try:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    _TRANSIENT: tuple = (HfHubHTTPError, httpx.HTTPError, ConnectionError, TimeoutError)
except Exception:  # pragma: no cover - the retry just narrows if these are absent
    _TRANSIENT = (ConnectionError, TimeoutError)

# Ungated mirrors, so no Hugging Face login or license click is needed.
MODELS = {
    "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct",
    "gemma3:4b": "unsloth/gemma-3-4b-it",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen3:4b": "Qwen/Qwen3-4B",
    "qwen3:8b": "Qwen/Qwen3-8B",
}

# Tuned so the run is tractable on a small GPU (for example a 6 GB laptop card).
# Only the ordering of the five models is used, not the exact accuracy, so a
# zero-shot run over a fixed fraction of MMLU is enough and keeps the ordering
# stable while cutting both time and memory. Zero-shot also shortens every prompt,
# which is what keeps the 8B model from spilling heavily to CPU. Set NUM_FEWSHOT
# back to 5 and LIMIT to None for closer-to-published numbers at much higher cost.
NUM_FEWSHOT = 0
LIMIT = 0.10  # fraction of each MMLU subject to score; None runs the whole benchmark

# Models whose 4-bit weights plus the fp16 token embedding do not all fit in 6 GB
# VRAM. For these the fp16 embedding (a plain ~1.2 GB lookup table) is placed on the
# CPU while every 4-bit layer stays on the GPU. That frees enough VRAM for the rest
# to fit, and it deliberately avoids offloading any quantized layer: bitsandbytes
# 4-bit weights cannot be streamed back from CPU by accelerate (their quant_state
# lives on the meta device and raises "Cannot copy out of meta tensor" mid-forward),
# so a max_memory cap that spills whole 4-bit blocks to CPU does not work. The 4-bit
# weights and the resulting score are unchanged; only the embedding's placement and
# speed differ.
NEEDS_CPU_OFFLOAD = {"qwen3:8b"}


def _evaluate_mmlu(lm, limit: float | None = LIMIT, attempts: int = 12) -> dict:
    # The MMLU dataset is fetched from the Hub the first time it is seen, one
    # subject at a time, so a transient gateway timeout on any subject would
    # otherwise abort the whole run. The call is retried with a growing backoff;
    # subjects that already downloaded are read from the local cache on the retry,
    # so each attempt only has to get through the ones still missing. Anonymous
    # requests are rate-limited, which causes these timeouts, so logging in with a
    # Hugging Face token (huggingface-cli login) makes them far less likely.
    for attempt in range(1, attempts + 1):
        try:
            return lm_eval.simple_evaluate(
                model=lm, tasks=["mmlu"], num_fewshot=NUM_FEWSHOT, limit=limit
            )
        except _TRANSIENT as exc:
            if attempt == attempts:
                raise
            wait = min(120, 15 * attempt)
            print(
                f"  transient Hugging Face error (attempt {attempt}/{attempts}); "
                f"retrying in {wait}s. {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # the loop always returns or raises


def score(repo: str, embed_on_cpu: bool = False, limit: float | None = LIMIT) -> float:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # Required to place the fp16 embedding on the CPU below: bitsandbytes refuses
        # to dispatch any module to CPU unless this is set. The 3-4B models load
        # entirely on the GPU and never hit it.
        llm_int8_enable_fp32_cpu_offload=True,
    )
    if embed_on_cpu:
        # The 8B's 4-bit layers fit in 6 GB only once the fp16 token embedding is off
        # the card. Pin every 4-bit layer to the GPU and put just the embedding on the
        # CPU ("" is the default device, the more specific key overrides it). The
        # embedding is a plain tensor, so accelerate streams its output to the GPU
        # cleanly; no quantized layer is offloaded, so the bitsandbytes meta-tensor
        # failure never arises. The 4-bit weights and the MMLU score are unchanged.
        load_kwargs = dict(device_map={"": 0, "model.embed_tokens": "cpu"})
    else:
        # 3-4B models fit entirely on the GPU; keep them all on-card and fast.
        load_kwargs = dict(device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, **load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(repo)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    if embed_on_cpu:
        # accelerate offloads the CPU-pinned embedding by setting its parameter to the
        # meta device (the real weight lives in a CPU side-table its forward hook
        # restores), so model.device -- which lm-eval copies into lm.device -- reports
        # "meta". lm-eval then builds torch.autocast("meta", ...), which raises
        # "unsupported scalarType". Compute happens on the GPU and the hook routes the
        # embedding lookup through the CPU regardless, so point the harness at cuda:
        # input tensors are created there and every layer runs where it was placed.
        lm._device = torch.device("cuda")
    results = _evaluate_mmlu(lm, limit=limit)
    acc = float(results["results"]["mmlu"]["acc,none"])

    del lm, model
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure MMLU for the study models, one identical 4-bit setup."
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME",
        help=(
            "score only this model (repeatable); default is all five. Running one "
            "model per process keeps peak memory to a single model on a small machine. "
            f"Choices: {', '.join(MODELS)}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=LIMIT,
        metavar="FRACTION",
        help=(
            "fraction of each MMLU subject to score (default %(default)s). A smaller "
            "value is faster but the score is from a smaller, noisier sample and is "
            "not directly comparable to models scored at a different limit."
        ),
    )
    args = parser.parse_args(argv)

    selected = args.model or list(MODELS)
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}; choices: {', '.join(MODELS)}", flush=True)
        return 2

    # Each score is written as soon as it is produced, so a crash on a later model
    # never loses the ones already done, and a per-model rerun just overwrites that
    # one entry in results/mmlu/mmlu.json.
    for name in selected:
        print(f"=== {name} ({MODELS[name]}) ===", flush=True)
        # Models that do not fit in 6 GB VRAM as 4-bit keep their fp16 embedding on
        # the CPU so the 4-bit layers fit on the card. The 3-4B models fit and load
        # entirely on the GPU.
        acc = score(
            MODELS[name],
            embed_on_cpu=name in NEEDS_CPU_OFFLOAD,
            limit=args.limit,
        )
        mmlu.record(name, acc)
        print(f"{name}: MMLU = {acc * 100:.2f}%", flush=True)

    print()
    mmlu.main(["show"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
