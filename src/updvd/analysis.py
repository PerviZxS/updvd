# Renders the paper tables from the saved JSON results.
from __future__ import annotations

import json
from pathlib import Path

# Violation classes and their column labels in Table 2.
_CLASSES = [
    ("referential_violation", "Referential"),
    ("liveness_violation", "Liveness"),
    ("ownership_violation", "Ownership"),
    ("authorization_violation", "Authorization"),
]


# Collects every model entry from the report files; accepts either results/ or results/seeds/.
def _load_models(results_dir: Path) -> list[dict]:
    search = results_dir
    if not any(search.glob("*.json")) and (results_dir / "seeds").is_dir():
        search = results_dir / "seeds"
    models: list[dict] = []
    for path in sorted(search.glob("*.json")):
        data = json.loads(path.read_text())
        for entry in data.get("models", []):
            models.append(entry)
    return models


# MMLU scores from results/mmlu/mmlu.json; empty (column skipped) if the file is absent.
def _load_mmlu(results_dir: Path) -> dict[str, float]:
    store = results_dir / "mmlu" / "mmlu.json"
    if not store.exists():
        return {}
    return json.loads(store.read_text()).get("scores", {})


# Per-model MMLU provenance (stderr, sample count, quantization, source file).
def _load_mmlu_details(results_dir: Path) -> dict[str, dict]:
    store = results_dir / "mmlu" / "mmlu.json"
    if not store.exists():
        return {}
    return json.loads(store.read_text()).get("details", {})


# Formats one MMLU cell, "n/a" when the model has no score.
def _mmlu_cell(mmlu: dict[str, float], model: str, details: dict[str, dict] | None = None) -> str:
    if model not in mmlu:
        return "n/a"
    acc = mmlu[model] * 100
    d = (details or {}).get(model, {})
    stderr = d.get("acc_stderr")
    if stderr is not None:
        # Plain "+/-", because a real plus-minus sign raises UnicodeEncodeError on a cp1252 Windows console.
        return f"{acc:5.2f}% +/- {stderr * 100:.2f}"
    return f"{acc:6.2f}%"


# One Table 1 row: blocking, leaked, substitution (and MMLU if available).
def _table1_row(entry: dict, mmlu: dict[str, float], details: dict[str, dict]) -> str:
    sub = entry["substitution"]
    block = entry["blocking"]
    total = block["trials"]
    blocked = block["action_blocked"]
    row = (
        f"| {entry['model']:<14} "
        f"| {100 * block['pooled_rate']:5.1f}% ({blocked}/{total}) "
        f"| {block['leaked']:>6} "
        f"| {100 * sub['pooled_rate']:5.1f}% ({sub['substituted']}/{total}) |"
    )
    if mmlu:
        row += f" {_mmlu_cell(mmlu, entry['model'], details)} |"
    return row


# One Table 2 row: substitutions per violation class.
def _table2_row(entry: dict) -> str:
    by_class = entry["substitution"]["substituted_by_class"]
    cells = " | ".join(f"{by_class.get(key, 0):>11}" for key, _ in _CLASSES)
    return f"| {entry['model']:<14} | {cells} |"


# Multi-seed table: across-seed mean (pop. sd), pooled rate with 95% Wilson CI, and blocking per model.
def format_seed_table(data: dict) -> str:
    seeds = data.get("seeds", [])
    temperature = data.get("temperature", 0.0)
    lines = [
        f"Multi-seed outcomes over {len(seeds)} seeds at temperature {temperature}. "
        f"Substitution 'sd' is the population standard deviation across seeds.",
        f"| {'Model':<14} | Substitution mean (sd) | Pooled rate [95% CI]    | Blocking          | Leaked |",
        f"| {'-' * 14} | ---------------------- | ----------------------- | ----------------- | ------ |",
    ]
    for entry in data.get("models", []):
        sub = entry["substitution"]
        block = entry["blocking"]
        lo, hi = sub["pooled_ci95"]
        mean_sd = f"{100 * sub['mean_rate']:5.1f}% (pop.sd {100 * sub['stdev_rate']:.1f})"
        pooled = f"{100 * sub['pooled_rate']:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
        blocking = f"{100 * block['pooled_rate']:5.1f}% ({block['action_blocked']}/{block['trials']})"
        lines.append(
            f"| {entry['model']:<14} | {mean_sd:>22} | {pooled:>23} | {blocking:<17} | {block['leaked']:>6} |"
        )
    return "\n".join(lines)


# Loads one multi-seed file or merges a directory of them; seeds and temperature come from the first file.
def _load_seed_data(path: Path) -> dict | None:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    elif path.is_file():
        files = [path]
    else:
        return None
    merged: dict = {"seeds": [], "temperature": 0.0, "models": []}
    for index, file in enumerate(files):
        data = json.loads(file.read_text())
        if index == 0:
            merged["seeds"] = data.get("seeds", [])
            merged["temperature"] = data.get("temperature", 0.0)
        merged["models"].extend(data.get("models", []))
    return merged if merged["models"] else None


# CLI: print the multi-seed table.
def print_seed_table(seeds_path: Path) -> int:
    data = _load_seed_data(seeds_path)
    if data is None:
        print(f"no multi-seed results found at {seeds_path}")
        return 1
    print(format_seed_table(data))
    return 0


# CLI: print Table 1, Table 2 and, if MMLU scores exist, the capability-ordered table.
def print_tables(results_dir: Path) -> int:
    models = _load_models(results_dir)
    if not models:
        print(f"no model results found in {results_dir}")
        return 1
    mmlu = _load_mmlu(results_dir)
    details = _load_mmlu_details(results_dir)

    print("Table 1. Outcomes over the forbidden tasks per model.")
    head = f"| {'Model':<14} | Blocking          | Leaked | Substitution      |"
    rule = f"| {'-' * 14} | ----------------- | ------ | ----------------- |"
    if mmlu:
        head += " MMLU    |"
        rule += " ------- |"
    print(head)
    print(rule)
    for entry in models:
        print(_table1_row(entry, mmlu, details))

    print()
    print("Table 2. Substitutions by violated rule.")
    header = " | ".join(f"{label:>11}" for _, label in _CLASSES)
    print(f"| {'Model':<14} | {header} |")
    print(f"| {'-' * 14} | {' | '.join('-' * 11 for _ in _CLASSES)} |")
    for entry in models:
        print(_table2_row(entry))

    if mmlu:
        print()
        print("Capability axis: models ordered by MMLU, with substitution rate.")
        print(f"| {'Model':<14} | MMLU    | Substitution |")
        print(f"| {'-' * 14} | ------- | ------------ |")
        # Models without an MMLU score sort last.
        ranked = sorted(
            models,
            key=lambda e: mmlu.get(e["model"], float("inf")),
        )
        for entry in ranked:
            rate = 100 * entry["substitution"]["pooled_rate"]
            print(f"| {entry['model']:<14} | {_mmlu_cell(mmlu, entry['model'], details)} | {rate:11.2f}% |")
    return 0
