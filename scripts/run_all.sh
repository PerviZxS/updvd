#!/usr/bin/env bash
# Runs the whole experiment for the paper: the behavioral runs with Ollama, then MMLU, then the checks.
# Run it on Google Colab with a Tesla T4 runtime, from the repository root:  bash scripts/run_all.sh
# If the session drops, run it again; models that already finished are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_VERSION="0.32.5"
PYTHON_VERSION="3.12.13"
# On local disk, not Drive: Google Drive cannot hold the symlinks a virtual environment needs.
VENV="$HOME/updvd-venv"
MODELS=(gemma2:2b llama3.2:3b phi3:mini gemma3:4b phi4-mini qwen3:4b qwen2.5:7b llama3.1:8b qwen3:8b)
SEEDS="0 1 2 3 4"
TEMPERATURE="0.7"

fail() { echo "STOP: $*" >&2; exit 1; }

# 1. Check the GPU, set up the paper's Python, and check that results/ belongs to this run.
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
[[ "$gpu" == *"Tesla T4"* ]] || fail "the paper's runs used a Tesla T4 GPU; this runtime has '${gpu:-no GPU}'. In Colab: Runtime > Change runtime type > T4 GPU."

# Colab's own Python can be newer than the paper's, so the run uses its own Python 3.12.13 environment.
# Colab sets UV_SYSTEM_PYTHON=1, which makes uv ignore virtual environments and install into the
# system Python instead, so it is cleared here and every uv pip call names the environment's Python.
unset UV_SYSTEM_PYTHON
command -v uv >/dev/null || pip install -q "uv==0.13.5"
if [[ ! -x "$VENV/bin/python" ]]; then
    uv python install "$PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi
source "$VENV/bin/activate"
[[ "$(python -c 'import platform; print(platform.python_version())')" == "$PYTHON_VERSION" ]] \
    || fail "could not set up Python $PYTHON_VERSION (got $(python --version 2>&1))."
mkdir -p results
if [[ -n "$(ls -A results)" && ! -f results/.run_all ]]; then
    fail "results/ already holds files from another run. Move them out of the repository first."
fi
touch results/.run_all

# 2. Pinned Python packages, then the package itself. unsafe-best-match lets the +cu128 torch
# builds come from the PyTorch index even though PyPI also has a package called torch.
echo "Installing the pinned packages into $VENV (several GB, a few minutes)..."
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r requirements-colab.txt
uv pip install --python "$VENV/bin/python" -q --no-deps -e .
python -c 'import numpy, torch; assert (torch.__version__, numpy.__version__) == ("2.11.0+cu128", "2.0.2"), (torch.__version__, numpy.__version__)' \
    || fail "the pinned torch/numpy did not end up in $VENV."
python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1 \
    || fail "log in to Hugging Face first (several MMLU models are gated): run  huggingface-cli login"

# 3. Record this session's environment (a resumed run gets its own folder).
env_dir="results/env/session-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$env_dir"
nvidia-smi > "$env_dir/nvidia-smi.txt"
python --version > "$env_dir/python.txt" 2>&1
uname -a > "$env_dir/uname.txt"
uv pip freeze --python "$VENV/bin/python" > "$env_dir/pip-freeze.txt"

# 4. Behavioral experiment: nine models, five seeds, temperature 0.7, pinned Ollama.
if ! ollama --version 2>/dev/null | grep -q "$OLLAMA_VERSION"; then
    # Ollama 0.32.5 ships only as .tar.zst, and the installer stops if zstd is missing.
    command -v zstd >/dev/null || { apt-get -qq update && apt-get -qq install -y zstd; }
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
fi
nohup ollama serve > "$env_dir/ollama-serve.log" 2>&1 &
ollama_pid=$!
for _ in $(seq 60); do curl -fs http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
curl -fs http://127.0.0.1:11434/api/version >/dev/null || fail "the Ollama server did not start (see $env_dir/ollama-serve.log)."
ollama --version > "$env_dir/ollama.txt" 2>&1

for model in "${MODELS[@]}"; do
    out="results/seeds/${model//:/-}.json"
    if [[ -f "$out" ]]; then
        echo "=== $model: already done ($out), skipping ==="
        continue
    fi
    echo "=== $model ==="
    ollama pull "$model"
    updvd multiseed --models "$model" --seeds $SEEDS --temperature "$TEMPERATURE" --output "$out"
done

# Shut Ollama down first; it and the MMLU models do not fit on one T4 together.
kill "$ollama_pid" 2>/dev/null || true
wait "$ollama_pid" 2>/dev/null || true
pkill -f "ollama" 2>/dev/null || true
sleep 5

# 5. MMLU for all nine models under one protocol.
python scripts/run_mmlu.py

# 6. Tables, and a check of every paper number against the new results.
updvd analyze --results results
updvd analyze --seeds-file results/seeds
updvd reproduce --results results \
    || echo "NOTE: the FAIL lines above are the paper numbers that changed in this run; update the paper from them."
echo "Done. Results are in results/."
