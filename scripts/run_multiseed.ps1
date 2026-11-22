# Run the nine-model comparison over seeds 0-4 at temperature 0.7, one model per call,
# then print the multi-seed table. Needs a running Ollama server and: pip install -e ".[llm,dev]"
# Temperature must be > 0, otherwise every seed produces the same output.

$ErrorActionPreference = "Stop"

$seeds = 0, 1, 2, 3, 4
$temperature = 0.7

$models = @(
    @{ name = "gemma2:2b";   out = "results/seeds/gemma2-2b.json" },
    @{ name = "llama3.2:3b"; out = "results/seeds/llama3.2-3b.json" },
    @{ name = "phi3:mini";   out = "results/seeds/phi3-mini.json" },
    @{ name = "gemma3:4b";   out = "results/seeds/gemma3-4b.json" },
    @{ name = "phi4-mini";   out = "results/seeds/phi4-mini.json" },
    @{ name = "qwen3:4b";    out = "results/seeds/qwen3-4b.json" },
    @{ name = "qwen2.5:7b";  out = "results/seeds/qwen2.5-7b.json" },
    @{ name = "llama3.1:8b"; out = "results/seeds/llama3.1-8b.json" },
    @{ name = "qwen3:8b";    out = "results/seeds/qwen3-8b.json" }
)

foreach ($m in $models) {
    Write-Host "=== $($m.name) ==="
    ollama pull $m.name
    updvd multiseed --models $m.name --seeds $seeds --temperature $temperature --output $m.out
}

Write-Host "=== Multi-seed table ==="
updvd analyze --seeds-file results/seeds
