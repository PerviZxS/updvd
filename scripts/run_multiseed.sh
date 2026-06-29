#!/usr/bin/env bash
# Run the multi-model comparison across several seeds and report substitution
# rates with confidence intervals.
#
#   pip install -e ".[llm,dev]"
#   ./scripts/run_multiseed.sh
#
# Needs a running Ollama server with the five models pulled. The temperature is
# above zero on purpose: at temperature 0 the decoding is greedy and every seed
# returns the same output, so the seeds would add no information. Each model is
# run on its own so an Ollama stall costs only that model; rerun that one line.
# Every model writes one JSON file under results/seeds/, and the final command
# renders them together as one table.

set -euo pipefail

seeds="0 1 2 3 4"
temperature="0.7"

models=(
    "llama3.2:3b results/seeds/llama3.2-3b.json"
    "gemma3:4b results/seeds/gemma3-4b.json"
    "phi4-mini results/seeds/phi4-mini.json"
    "qwen3:4b results/seeds/qwen3-4b.json"
    "qwen3:8b results/seeds/qwen3-8b.json"
)

for entry in "${models[@]}"; do
    name="${entry%% *}"
    out="${entry##* }"
    echo "=== $name ==="
    ollama pull "$name"
    updvd multiseed --models "$name" --seeds $seeds --temperature "$temperature" --output "$out"
done

echo "=== Multi-seed table ==="
updvd analyze --seeds-file results/seeds
