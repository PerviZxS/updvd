from __future__ import annotations

import json
from pathlib import Path

# The order tasks.py builds the forbidden groups in. Table 2 follows it.
_CLASSES = [
    ("referential_violation", "Referential"),
    ("liveness_violation", "Liveness"),
    ("ownership_violation", "Ownership"),
    ("authorization_violation", "Authorization"),
]


def _load_models(results_dir: Path) -> list[dict]:
    # Each results/*.json holds one or more model entries under "models". Reading
    # the whole directory rebuilds the table from exactly what was released.
    models: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"environment.json"}:
            continue
        data = json.loads(path.read_text())
        for entry in data.get("models", []):
            models.append(entry)
    return models


def _load_mmlu(results_dir: Path) -> dict[str, float]:
    # Capability scores produced by scripts/run_mmlu and recorded by updvd.mmlu.
    # Absent until the eval has been run, in which case the column is skipped.
    store = results_dir / "mmlu" / "mmlu.json"
    if not store.exists():
        return {}
    return json.loads(store.read_text()).get("scores", {})


def _mmlu_cell(mmlu: dict[str, float], model: str) -> str:
    if model in mmlu:
        return f"{mmlu[model] * 100:6.2f}%"
    return "pending"


def _table1_row(entry: dict, mmlu: dict[str, float]) -> str:
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
        row += f" {_mmlu_cell(mmlu, entry['model'])} |"
    return row


def _table2_row(entry: dict) -> str:
    by_class = entry["substitution"]["substituted_by_class"]
    cells = " | ".join(f"{by_class.get(key, 0):>11}" for key, _ in _CLASSES)
    return f"| {entry['model']:<14} | {cells} |"


def format_seed_table(data: dict) -> str:
    # Render the multi-seed run as one table per model row: the across-seed mean
    # substitution rate with its standard deviation, the pooled rate with its 95%
    # Wilson interval, and the blocking outcome. The number of seeds and the
    # temperature sit in the caption so the table cannot be read out of context.
    seeds = data.get("seeds", [])
    temperature = data.get("temperature", 0.0)
    lines = [
        f"Multi-seed outcomes over {len(seeds)} seeds at temperature {temperature}.",
        f"| {'Model':<14} | Substitution mean (sd) | Pooled rate [95% CI]    | Blocking          | Leaked |",
        f"| {'-' * 14} | ---------------------- | ----------------------- | ----------------- | ------ |",
    ]
    for entry in data.get("models", []):
        sub = entry["substitution"]
        block = entry["blocking"]
        lo, hi = sub["pooled_ci95"]
        mean_sd = f"{100 * sub['mean_rate']:5.1f}% ({100 * sub['stdev_rate']:.1f})"
        pooled = f"{100 * sub['pooled_rate']:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
        blocking = f"{100 * block['pooled_rate']:5.1f}% ({block['action_blocked']}/{block['trials']})"
        lines.append(
            f"| {entry['model']:<14} | {mean_sd:>22} | {pooled:>23} | {blocking:<17} | {block['leaked']:>6} |"
        )
    return "\n".join(lines)


def _load_seed_data(path: Path) -> dict | None:
    # Accept either one multi-seed JSON file or a directory of them, one per model,
    # the same one-model-per-file layout the single-seed runs use. A directory
    # merges every model entry and takes the seed list and temperature from the
    # first file read, since the runner writes the same values into each.
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


def print_seed_table(seeds_path: Path) -> int:
    data = _load_seed_data(seeds_path)
    if data is None:
        print(f"no multi-seed results found at {seeds_path}")
        return 1
    print(format_seed_table(data))
    return 0


def print_tables(results_dir: Path) -> int:
    models = _load_models(results_dir)
    if not models:
        print(f"no model results found in {results_dir}")
        return 1
    mmlu = _load_mmlu(results_dir)

    print("Table 1. Outcomes over the forbidden tasks per model.")
    head = f"| {'Model':<14} | Blocking          | Leaked | Substitution      |"
    rule = f"| {'-' * 14} | ----------------- | ------ | ----------------- |"
    if mmlu:
        head += " MMLU    |"
        rule += " ------- |"
    print(head)
    print(rule)
    for entry in models:
        print(_table1_row(entry, mmlu))

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
        ranked = sorted(
            models,
            key=lambda e: mmlu.get(e["model"], -1.0),
        )
        for entry in ranked:
            rate = 100 * entry["substitution"]["pooled_rate"]
            print(f"| {entry['model']:<14} | {_mmlu_cell(mmlu, entry['model'])} | {rate:11.2f}% |")
    return 0
