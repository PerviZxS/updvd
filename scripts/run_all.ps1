# Reproduce every result in results/ from a clean checkout.
#
#   pip install -e ".[llm,dev]"
#   ./scripts/run_all.ps1
#
# Needs a running Ollama server with the five models pulled. Each model writes
# one JSON file (aggregates + per-task logs + environment) under results/.

$ErrorActionPreference = "Stop"

$models = @(
    @{ name = "llama3.2:3b"; out = "results/llama3.2-3b.json" },
    @{ name = "gemma3:4b";   out = "results/gemma3-4b.json" },
    @{ name = "phi4-mini";   out = "results/phi4-mini.json" },
    @{ name = "qwen3:4b";    out = "results/qwen3-4b.json" },
    @{ name = "qwen3:8b";    out = "results/qwen3-8b.json" }
)

foreach ($m in $models) {
    Write-Host "=== $($m.name) ==="
    ollama pull $m.name
    updvd multimodel --models $m.name --seed 0 --output $m.out
}

Write-Host "=== Table 1 and Table 2 ==="
updvd analyze --results results
