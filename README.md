# updvd Prototype

A Python prototype of a contract-checked execution layer for agentic AI, with a study of how it behaves as the proposing model changes.

## What this is

This repository contains a working prototype of UPVD (Untrusted Proposer, Verified Disposer), a deterministic layer that sits between an LLM agent and its tools and decides whether each proposed action is allowed to run.

The prototype is not a production system. It is a simulation that demonstrates one specific thing: that the model can be made an untrusted proposer, able only to suggest an action, while a separate deterministic component is the only thing allowed to change state. The same fixed layer is then held constant while the proposing model is swapped across five open models, to test whether the guarantee depends on the model behind it.

The accompanying paper, "Untrusted Proposer, Verified Disposer: Model-Independent Action Blocking and a Capability-Dependent Intent Gap", proposes the layer and reports two findings. This prototype instantiates the layer as executable code and produces the results used in the paper.

## What it demonstrates

The layer routes each proposed tool call through a sequence of checks and writes an audit log whose integrity can be independently verified.

Proposal. The model reads a plain-language task and emits a single structured tool call, a tool name and typed arguments. It cannot run anything; it can only propose.

Schema check. The call is validated with pydantic before any rule is checked. A malformed call (wrong type, missing field, blank data, unknown tool) is rejected here and never reaches the contract.

Precondition. The contract for that tool is checked against the current state. An update must target a record that exists, is not deleted, and belongs to the actor; a delete is allowed only for an admin. A failing call is rejected with a structured reason.

Apply and postcondition. The action is applied to a copy of the state and the postcondition is checked. The real state advances only if that copy passes. A failing action leaves no trace, so a rejected call cannot change anything. On any rejection the reason is sent back to the model, which may correct the call and try again, up to a retry limit.

Append-only, hash-chained audit log. Every decision, allow or reject, is appended to a log in which each entry carries a hash of the entry before it. Any later alteration, deletion, or reorder is detectable by re-walking the chain.

![updvd architecture](updvd architecture.drawio.svg)

## Structure

```
updvd/
  src/updvd/
    cli.py            command line entry point
    proposer.py       Proposer interface and the scripted proposer
    ollama_proposer.py    live model proposer through Ollama
    interceptor.py    the gate: schema, precondition, apply, postcondition
    contracts.py      per-tool preconditions and postconditions
    schemas.py        typed argument schemas
    state.py          the record store and its state
    verdicts.py       the verdict types
    trace.py          append-only, hash-chained audit log
    engine.py         the propose, check, retry loop
    tasks.py          the evaluation task set
    scenarios.py      the deterministic demo script
    evaluation.py     metrics and the containment re-check
    multimodel.py     multi-model and multi-seed runs, aggregated to compact JSON
    analysis.py       prints the result tables from saved JSON
    mmlu.py           records the capability (MMLU) scores
  scripts/            reproduction and MMLU runners
  tests/              behavioural tests, one per claim
  results/            saved JSON reports per model, plus results/mmlu/
  pyproject.toml
  README.md
  LICENSE
```

The module layout mirrors the layer: each step of the gate maps to one module.

## How to run it

Python 3.10 or higher. The core needs only pydantic. The live model needs the Ollama client and a running Ollama server.

```
pip install -e ".[llm,dev]"
```

`llm` adds the Ollama client. `dev` adds pytest.

### Run modes

```
updvd demo                                                                    # deterministic walkthrough, no model
updvd eval --scripted                                                         # full evaluation with the scripted proposer
updvd eval --model qwen3:8b --seed 0 --output results/qwen3-8b.json           # one live model
updvd multimodel --models qwen3:8b --seed 0 --output results/qwen3-8b.json    # one model, full report
updvd multiseed --models qwen3:8b --seeds 0 1 2 3 4 --temperature 0.7 \
    --output results/seeds/qwen3-8b.json                                      # one model over several seeds
updvd analyze --results results                                               # print the tables from saved JSON
updvd analyze --seeds-file results/seeds                                      # print the multi-seed table
```

Run one model per command. Ollama can get stuck on a single run (a model load stalls, or the server stops responding). One model at a time means a stall only costs that model, and that command can simply be rerun. Repeat for the other models, each to its own `results/*.json`. `updvd analyze` reads them all together.

To reproduce every result from a clean checkout:

```
./scripts/run_all.ps1      # Windows / PowerShell
./scripts/run_all.sh       # Linux / macOS
```

### Variation across seeds

The single-seed run at temperature 0 fixes the decoding, so it reports one outcome per model rather than a distribution. The `multiseed` command runs the same task set across several seeds and reports the substitution rate as an across-seed mean with its standard deviation and a pooled 95% Wilson interval, so the spread between runs is visible. The temperature is set above zero on purpose: at temperature 0 the decoding is greedy and every seed returns the same output, so the seeds would add no information. The blocking outcome is reported the same way and is expected to stay at 100% regardless of seed.

```
./scripts/run_multiseed.ps1     # Windows / PowerShell
./scripts/run_multiseed.sh      # Linux / macOS
```

Each model writes one JSON file under `results/seeds/`, holding the per-seed counts, the pooled rate, and the interval. `updvd analyze --seeds-file results/seeds` reads them together into one table.

### Capability axis (MMLU)

The paper indexes capability by MMLU rather than parameter count. To keep the five models comparable, the scores are measured under one identical setup (same benchmark, same shot count, same harness, same 4-bit quantization), not copied from the published model cards. Only the ordering across these five models is used.

The eval runs in-process with the lm-evaluation-harness `hf` backend, which computes the per-token log-probabilities MMLU needs natively. The five repos in the script are ungated mirrors, so no Hugging Face login or license click is needed.

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install lm-eval "transformers>=4.55,<5" accelerate bitsandbytes
pip install -e .
.\scripts\run_mmlu.ps1     # Windows / PowerShell
./scripts/run_mmlu.sh      # Linux / macOS
```

`transformers` is pinned below 5 on purpose: the 5.x loader crashes with a native access violation when loading bitsandbytes 4-bit weights on Windows.

The first line installs the CUDA build of PyTorch, which `bitsandbytes` needs to run 4-bit on the GPU; check it with `python -c "import torch; print(torch.cuda.is_available())"` (should print `True`).

The runner (`scripts/run_mmlu.py`) holds the model-to-repo map. It builds each model in 4-bit with an explicit `BitsAndBytesConfig`, scores MMLU (zero-shot over a fixed 10% subset by default, tuned so the run is tractable on a small GPU) through the lm-evaluation-harness, and records each accuracy into `results/mmlu/mmlu.json`. Only the ordering across the five models is used, so the subset and zero-shot keep that ordering while cutting time and memory; raise `NUM_FEWSHOT` in the script and pass a larger `--limit` for closer-to-published numbers. Once that file exists, `updvd analyze` adds an MMLU column and a capability-ordered table. Placement is per model: the 3–4B models load entirely on the GPU, while the 8B keeps its fp16 token embedding on the CPU (all 4-bit layers stay on the card) so it fits in 6 GB without offloading any quantized layer; the 4-bit weights and the score are the same either way.

## What the demo run demonstrates

`updvd demo` runs a fixed script that exercises one of each behaviour, so the walkthrough is complete evidence rather than seed-dependent.

| Behaviour | Audit-log signature |
|---|---|
| Valid create, committed | verdict ok, committed |
| Valid update, committed | verdict ok, committed |
| Schema violation, corrected and committed | first schema_violation, then committed |
| Ownership violation, corrected and committed | first ownership_violation, then committed |
| Authorization violation, blocked | authorization_violation, never committed |
| Referential violation, corrected and committed | first referential_violation, then committed |

## Output format

Each evaluation writes one compact JSON report. The single-seed report (`results/*.json`) and the multi-seed report (`results/seeds/*.json`) share the same shape; a single-seed report is just the one-seed case, with `seeds` holding one value.

| Field | Description |
|---|---|
| generated | When the report was written |
| seeds | The sampling seeds run for every model |
| temperature | The sampling temperature used for the run |
| environment | Python version, platform, Ollama version, and the model options (temperature, seeds, think off) |
| models[].model | The model name |
| models[].digest | The exact model version that produced the run |
| models[].substitution | Substitution over the forbidden trials: pooled rate with its 95% Wilson interval, per-seed rates, and the across-seed mean and standard deviation, split by violated rule |
| models[].blocking | Blocking over the same trials: pooled rate with its 95% Wilson interval, and the leaked count (should be zero) |
| models[].per_seed | One row per seed: the forbidden-task counts that the pooled and across-seed figures are built from |

The `per_seed` block is what lets a reader rebuild the aggregate from the raw per-seed counts, instead of trusting the pooled number. The report stays small on purpose: the full per-decision log for any run, with each attempt's tool, arguments, verdict, and committed action, and the hash-chain check that verifies it, are produced live by the engine and printed by `updvd demo`.

## Audit-log integrity

```
entry_hash = SHA-256( canonical_json(seq, tool, args, verdict, detail, committed, prev_hash) )
```

The first entry's `prev_hash` is the genesis value (sixty-four zeros). `Trace.verify()` re-walks the log: a mismatched hash reveals an altered record, and a mismatched `prev_hash` reveals a deletion or reorder. The `containment_holds` check goes further and re-derives the run from its starting state, confirming every committed action still passes its own contract.

## Tests

```
python -m pip install pytest
python -m pytest -q
```

`pyproject.toml` puts `src` on the path for the test run, so no manual PYTHONPATH setup is needed. The tests use the scripted proposer and need no GPU or Ollama. Each test asserts one claim, so the suite doubles as a claim-to-code checklist.

## What the results show

Finding 1, blocking does not depend on the model. Across 35 forbidden tasks per model, run under five seeds (175 trials per model), the forbidden action was blocked every time for every model, with nothing leaking through in any of the 875 trials. The per-action check time was well under 0.1 ms (about 0.05 ms in our runs), with no dependence on the model or the seed.

Finding 2, capable models get around the block. When the forbidden action was rejected, the higher-capability models proposed a different, allowed action that still met the goal behind the request. Pooled over the five seeds this happened in 25.1% of forbidden tasks for qwen3:4b, 28.6% for phi4-mini, and 36.6% for qwen3:8b, while the two weakest models did so only rarely (3.4% for llama3.2:3b and 0.6% for gemma3:4b). The split does not follow parameter count, since the two 4B models land on opposite sides, so capability rather than size orders the models. The layer bounds the action, not the intent.

| Model | Blocking | Leaked | Substitution mean (sd) | Pooled rate [95% CI] |
|---|---|---|---|--------------------|
| llama3.2:3b | 175/175 | 0 | 3.4% (1.1) | 3.4% [1.6, 7.3]    |
| gemma3:4b | 175/175 | 0 | 0.6% (1.1) | 0.6% [0.1, 3.2]    |
| phi4-mini | 175/175 | 0 | 28.6% (7.2) | 28.6% [22.4, 35.7] |
| qwen3:4b | 175/175 | 0 | 25.1% (4.2) | 25.1% [19.3, 32.1] |
| qwen3:8b | 175/175 | 0 | 36.6% (1.1) | 36.6% [29.8, 43.9] |

The mean and standard deviation are across the five seeds; the pooled rate and its 95% Wilson interval are over all 175 trials. Full-precision values and per-seed counts are in `results/seeds/*.json`; the deterministic temperature 0 single-seed run is in `results/*.json` as a reference, in the same compact format. Once `scripts/run_mmlu` has filled `results/mmlu/mmlu.json`, `updvd analyze` adds the MMLU capability column and a capability-ordered table.

## Limitations

This prototype demonstrates the gate logic and the action-versus-intent gap. It does not validate detection accuracy or performance under load. The task set is one domain (record operations), so the way models substitute (falling back to create) is tied to this tool set. The substitution numbers come from five seeds at one temperature (0.7); the two groups stay separated under the 95% intervals, but the order within the high group is not resolved and a single temperature is a narrow slice of the sampling behaviour. The correctable tasks did not trigger formatting mistakes, so the data says nothing about error-driven recovery. More domains and more seeds are needed to settle the within-group order.

## How to cite

Please check Zenodo for the current version DOI before citing.

X, X. (2026) "Untrusted Proposer, Verified Disposer: Model-Independent Action Blocking and a Capability-Dependent Intent Gap". Zenodo. doi:X.X/XXXX.XXXXXXX

BibTeX:

```
@software{2026updvd,
  author    = {X},
  title     = {{Untrusted Proposer, Verified Disposer: Model-Independent Action Blocking and a Capability-Dependent Intent Gap}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {},
  url       = {}
}
```

## Author

X

## License

MIT License. This code may be used, modified, and distributed for academic and non-commercial purposes with attribution.
