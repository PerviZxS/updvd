#!/usr/bin/env bash
# Measure MMLU for the five models under one identical 4-bit setup, so the scores
# are comparable to each other.
#
#   pip install torch  # CUDA build, e.g. --index-url https://download.pytorch.org/whl/cu124
#   pip install lm-eval "transformers>=4.55,<5" accelerate bitsandbytes
#   pip install -e .
#   ./scripts/run_mmlu.sh
#
# transformers is pinned below 5: its loader crashes with a native access
# violation when loading bitsandbytes 4-bit weights on Windows.
#
# The real work is in scripts/run_mmlu.py, which builds each model in 4-bit with
# an explicit BitsAndBytesConfig and hands it to the lm-evaluation-harness. This
# avoids the CLI's deprecated load_in_4bit path.

set -euo pipefail

python scripts/run_mmlu.py
updvd analyze --results results
