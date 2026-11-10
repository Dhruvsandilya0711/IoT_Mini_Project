"""Figures for the Phase 2 report and slides that combine several result tables.

    python scripts/report_figures.py --results results --out report/figures

Writes:
    repro_per_class_8c.png   per-class recall: the base paper's Random Forest (its Table 8) vs ours
    cost_vs_f1_8c.png        macro-F1 against model size, baseline -> best imbalance method, per model
    confusion_rf_pair.png    Random Forest confusion matrices, baseline vs threshold tuning (8-class)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import AXIS, INK, INK_2, MUTED, SEQUENTIAL, SERIES, STRATEGY_COLORS, STRATEGY_NAMES, _save, plt  # noqa: E402
from src.train import MODEL_NAMES  # noqa: E402

# Neto et al. (2023), Table 8: Random Forest confusion matrix, 8 classes (rows = true class).
PAPER_CLASSES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
PAPER_RF_CM = np.array([
    [234929, 4, 24, 2, 4, 3192, 5159, 8],
    [1342, 169, 1, 0, 0, 844, 626, 1],
    [15, 0, 7525049, 557, 18, 339, 173, 0],
    [7, 0, 1088, 1790979, 34, 12, 47, 0],
    [5, 0, 603, 18, 582921, 100, 30, 0],
    [11565, 6, 1418, 11, 16, 60006, 5591, 17],
    [14618, 1, 18, 6, 11, 4743, 88371, 30],
    [1140, 1, 3, 1, 1, 1265, 2792, 230],
])
ORDER = ["DDoS", "DoS", "Mirai", "Benign", "Spoofing", "Recon", "Web", "BruteForce"]


def paper_recall():
    recall = np.diag(PAPER_RF_CM) / PAPER_RF_CM.sum(1)
    return dict(zip(PAPER_CLASSES, recall))


def repro_per_class(results, out):
    pc = pd.read_csv(results / "tables" / "compare_8c_rf_test_per_class.csv")
    ours = pc.pivot(index="class", columns="strategy", values="recall")
    series = [
        ("Base paper RF (its Table 8)", MUTED, pd.Series(paper_recall())),
        (STRATEGY_NAMES["none"] + ", ours", STRATEGY_COLORS["none"], ours["none"]),
        (STRATEGY_NAMES["smote"], STRATEGY_COLORS["smote"], ours["smote"]),
        (STRATEGY_NAMES["threshold"], STRATEGY_COLORS["threshold"], ours["threshold"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(ORDER))
    slot = 0.8 / len(series)
    for i, (label, color, values) in enumerate(series):
        ax.bar(x - 0.4 + slot * (i + 0.5), values.reindex(ORDER), width=slot * 0.88, color=color, label=label)
    for i, cls in enumerate(ORDER[-2:], start=len(ORDER) - 2):
        ax.text(i - 0.4 + slot * 0.5, paper_recall()[cls] + 0.02, f"{paper_recall()[cls]:.0%}", ha="center", fontsize=7,
                color=INK_2)
    ax.set_xticks(x, ORDER)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (test set)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Per-class recall, Random Forest, 8 classes: base paper vs this project", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8, handlelength=1, borderaxespad=0.2)
    _save(fig, out / "repro_per_class_8c.png")


def cost_vs_f1(results, out):
    board = pd.read_csv(results / "tables" / "leaderboard_8c.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for color, model in zip(SERIES, ["dt", "rf", "lgbm", "brf"]):
        rows = board[board["model"] == model]
        base = rows[rows["strategy"] == "none"].iloc[0]
        best = rows.sort_values("test_macro_f1", ascending=False).iloc[0]
        ax.scatter(base["model_size_mb"], base["test_macro_f1"], s=60, facecolor="#fcfcfb", edgecolor=color, linewidth=2,
                   zorder=3)
        if best["strategy"] != "none":
            ax.annotate("", xy=(best["model_size_mb"], best["test_macro_f1"]),
                        xytext=(base["model_size_mb"], base["test_macro_f1"]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=6, shrinkB=6))
        ax.scatter(best["model_size_mb"], best["test_macro_f1"], s=60, color=color, edgecolor="#fcfcfb", linewidth=1.5,
                   zorder=4, label=MODEL_NAMES[model])
        label = f"{MODEL_NAMES[model]}\n{STRATEGY_NAMES[best['strategy']]}: {best['test_macro_f1']:.3f}"
        ax.annotate(label, (best["model_size_mb"], best["test_macro_f1"]), textcoords="offset points", xytext=(8, -4),
                    fontsize=7.5, color=INK_2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (MB, log scale)")
    ax.set_ylabel("Macro-F1 (test set)")
    ax.set_ylim(0.45, 0.9)
    ax.grid(axis="both")
    ax.set_axisbelow(True)
    ax.set_title("Detection quality vs model size, 8 classes (hollow = no rebalancing, filled = best method)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out / "cost_vs_f1_8c.png")


def confusion_pair(results, out):
    runs = [("8c-rf-none-kall-s0", "Baseline (no rebalancing)"), ("8c-rf-threshold-kall-s0", "Threshold tuning")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (run, title) in zip(axes, runs):
        cm = pd.read_csv(results / "confusion" / f"{run}.csv", index_col=0)
        image = ax.imshow(cm.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
        n = len(cm)
        ax.set_xticks(range(n), cm.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), cm.index, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(n):
            for j in range(n):
                v = cm.iat[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.5 else INK)
        ax.set_xlabel("Predicted class")
        ax.set_title(f"Random Forest, {title}", fontsize=9)
    axes[0].set_ylabel("True class")
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("Share of the true class", color=INK_2)
    _save(fig, out / "confusion_rf_pair.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repro_per_class(results, out)
    cost_vs_f1(results, out)
    confusion_pair(results, out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
