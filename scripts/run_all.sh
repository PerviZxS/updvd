#!/usr/bin/env bash
# Reproduce every result in results/ from a clean checkout.
#
#   pip install -e ".[llm,dev]"
#   ./scripts/run_all.sh
#
# Needs a running Ollama server with the five models pulled. Each model writes
# one JSON file (aggregates + per-task logs + environment) under results/.

set -euo pipefail

models=(
    "llama3.2:3b results/llama3.2-3b.json"
    "gemma3:4b results/gemma3-4b.json"
    "phi4-mini results/phi4-mini.json"
    "qwen3:4b results/qwen3-4b.json"
    "qwen3:8b results/qwen3-8b.json"
)

for entry in "${models[@]}"; do
    name="${entry%% *}"
    out="${entry##* }"
    echo "=== $name ==="
    ollama pull "$name"
    updvd multimodel --models "$name" --seed 0 --output "$out"
done

echo "=== Table 1 and Table 2 ==="
updvd analyze --results results
