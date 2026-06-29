# Run the multi-model comparison across several seeds and report substitution
# rates with confidence intervals.
#
#   pip install -e ".[llm,dev]"
#   ./scripts/run_multiseed.ps1
#
# Needs a running Ollama server with the five models pulled. The temperature is
# above zero on purpose: at temperature 0 the decoding is greedy and every seed
# returns the same output, so the seeds would add no information. Each model is
# run on its own so an Ollama stall costs only that model; rerun that one line.
# Every model writes one JSON file under results/seeds/, and the final command
# renders them together as one table.

$ErrorActionPreference = "Stop"

$seeds = 0, 1, 2, 3, 4
$temperature = 0.7

$models = @(
    @{ name = "llama3.2:3b"; out = "results/seeds/llama3.2-3b.json" },
    @{ name = "gemma3:4b";   out = "results/seeds/gemma3-4b.json" },
    @{ name = "phi4-mini";   out = "results/seeds/phi4-mini.json" },
    @{ name = "qwen3:4b";    out = "results/seeds/qwen3-4b.json" },
    @{ name = "qwen3:8b";    out = "results/seeds/qwen3-8b.json" }
)

foreach ($m in $models) {
    Write-Host "=== $($m.name) ==="
    ollama pull $m.name
    updvd multiseed --models $m.name --seeds $seeds --temperature $temperature --output $m.out
}

Write-Host "=== Multi-seed table ==="
updvd analyze --seeds-file results/seeds
