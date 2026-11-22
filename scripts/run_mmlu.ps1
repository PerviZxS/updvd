# MMLU only, for all nine models under the paper's protocol (see scripts/run_mmlu.py).
# The full paper run is scripts/run_all.sh; this needs: pip install -r requirements-colab.txt && pip install -e .

$ErrorActionPreference = "Stop"

python scripts/run_mmlu.py
if ($LASTEXITCODE -ne 0) { throw "run_mmlu.py failed (exit $LASTEXITCODE)" }

updvd analyze --results results
