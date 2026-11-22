"""Builds the two data figures used in the paper from results/.

    pip install -e ".[figures]"
    python scripts/make_figures.py                 # reads results/, writes figures/*.pdf

    figures/fig_seed_spread.pdf            paper Fig. 2: pooled substitution rate per model with its
                                           95% Wilson interval, and the five single-seed rates
    figures/fig_substitution_vs_mmlu.pdf   paper Fig. 3: substitution rate against MMLU accuracy

Fig. 1 of the paper, figures/fig_architecture.pdf, is the architecture diagram. It is drawn by hand in
draw.io, not built by this script.

All plotted values are computed from results/seeds (the per-model summaries) and
results/mmlu/mmlu.json. None are typed in by hand. The PDFs carry no creation date,
so rebuilding from the same results gives byte-identical files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt

from updvd.reproduce import _load, _spearman, _spearman_p

# Colors: one blue for the main marks and a lighter blue for intervals and single seeds.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"

# LNCS text width is 12.2 cm.
WIDTH_IN = 4.8

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "legend.frameon": False,
})

# White outline around labels so they stay readable on top of the error bars.
HALO = [patheffects.withStroke(linewidth=2.2, foreground=SURFACE)]


# Writes one figure as a vector PDF.
# The PDF gets no CreationDate, so the same results always give the same bytes.
def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


# Paper Fig. 2: pooled rate with its 95% CI per model, ranked by pooled rate, and the five single-seed rates behind it.
def fig_seed_spread(models: dict, out: Path) -> None:
    order = sorted(models, key=lambda m: models[m]["substitution"]["pooled_rate"])
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9))
    for i, m in enumerate(order):
        sub = models[m]["substitution"]
        lo, hi = (100 * v for v in sub["pooled_ci95"])
        ax.plot([lo, hi], [i, i], color=BLUE, linewidth=1.2, solid_capstyle="round", zorder=2)
        # Seeds with equal rates are stacked so that all five dots show.
        seeds = [100 * r for r in sub["per_seed_rates"]]
        seen: dict[float, int] = {}
        heights = []
        for value in seeds:
            heights.append(i + 0.22 + 0.14 * seen.get(value, 0))
            seen[value] = seen.get(value, 0) + 1
        ax.plot(seeds, heights, "o", markersize=2.7, markerfacecolor=SURFACE,
                markeredgecolor=BLUE_LIGHT, markeredgewidth=0.9, linestyle="none", zorder=2)
        ax.plot(100 * sub["pooled_rate"], i, "o", color=BLUE, markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.0, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlim(left=-1.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Substitution rate (%)")
    ax.plot([], [], "o", color=BLUE, markersize=5, label="pooled rate with 95% Wilson CI")
    ax.plot([], [], "o", markersize=2.7, markerfacecolor=SURFACE, markeredgecolor=BLUE_LIGHT, label="single seed")
    ax.legend(loc="lower right")
    _save(fig, out, "fig_seed_spread")


# Paper Fig. 3: pooled substitution rate against MMLU, both with their uncertainty.
def fig_substitution_vs_mmlu(models: dict, mmlu: dict, out: Path) -> None:
    scores, details = mmlu["scores"], mmlu["details"]
    # The paper reports MMLU for every model; stop rather than silently drop a model without a score.
    missing = sorted(m for m in models if m not in scores)
    if missing:
        raise SystemExit(f"no MMLU score for {', '.join(missing)}; the paper's Fig. 3 needs all models")
    order = sorted(models, key=scores.get)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    xs, ys = [], []
    for m in order:
        sub = models[m]["substitution"]
        x, y = 100 * scores[m], 100 * sub["pooled_rate"]
        lo, hi = (100 * v for v in sub["pooled_ci95"])
        se = 100 * details[m]["acc_stderr"]
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], xerr=se, fmt="none", ecolor=BLUE_LIGHT, elinewidth=1.0, capsize=0)
        ax.plot(x, y, "o", color=BLUE, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        xs.append(x)
        ys.append(y)
    # Labels up and to the right of each point, nudged apart vertically when two would collide.
    placed: list[tuple[float, float]] = []
    for m, x, y in sorted(zip(order, xs, ys), key=lambda t: t[2]):
        ly = y + 1.4
        while any(abs(ly - py) < 2.2 and abs(x - px) < 3.5 for px, py in placed):
            ly += 2.2
        placed.append((x, ly))
        ax.annotate(m, (x, y), xytext=(x + 0.25, ly), textcoords="data", fontsize=6.5, color=INK,
                    va="center", path_effects=HALO, zorder=4)
    ax.set_xlabel("MMLU accuracy (%), error bar = 1 standard error")
    ax.set_ylabel("Substitution rate (%), 95% Wilson CI")
    ax.set_ylim(bottom=0)
    # Same rounding as the paper's text (rho = 0.58, p = 0.11).
    rho = _spearman(xs, ys)
    p = _spearman_p(xs, ys)
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}, exact permutation p = {p:.2f} (n = {len(xs)})",
            transform=ax.transAxes, va="top", fontsize=7, color=INK_2)
    _save(fig, out, "fig_substitution_vs_mmlu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the paper's data figures (Figs. 2 and 3) from released results.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    models, _, mmlu = _load(args.results)
    fig_seed_spread(models, args.out)
    fig_substitution_vs_mmlu(models, mmlu, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
