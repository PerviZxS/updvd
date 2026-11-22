#!/usr/bin/env bash
# Run the nine-model comparison over seeds 0-4 at temperature 0.7, one model per call,
# then print the multi-seed table. Needs a running Ollama server and: pip install -e ".[llm,dev]"
# Temperature must be > 0, otherwise every seed produces the same output.

set -euo pipefail

seeds="0 1 2 3 4"
temperature="0.7"

models=(
    "gemma2:2b results/seeds/gemma2-2b.json"
    "llama3.2:3b results/seeds/llama3.2-3b.json"
    "phi3:mini results/seeds/phi3-mini.json"
    "gemma3:4b results/seeds/gemma3-4b.json"
    "phi4-mini results/seeds/phi4-mini.json"
    "qwen3:4b results/seeds/qwen3-4b.json"
    "qwen2.5:7b results/seeds/qwen2.5-7b.json"
    "llama3.1:8b results/seeds/llama3.1-8b.json"
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
