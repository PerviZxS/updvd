#!/usr/bin/env bash
# MMLU only, for all nine models under the paper's protocol (see scripts/run_mmlu.py).
# The full paper run is scripts/run_all.sh; this needs: pip install -r requirements-colab.txt && pip install -e .

set -euo pipefail

python scripts/run_mmlu.py
updvd analyze --results results
