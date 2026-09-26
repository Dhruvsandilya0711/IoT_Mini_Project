"""Report figures. Callers write each figure's data to a CSV next to it.

Colours follow a fixed palette: categorical slots in a fixed order (a
strategy keeps its colour in every figure), a one-hue blue ramp for
magnitudes (confusion matrices) and blue <-> red around a grey midpoint for
signed values (correlations).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

STRATEGY_COLORS = {"none": SERIES[0], "smote": SERIES[1], "hybrid": SERIES[2], "classweight": SERIES[3]}
STRATEGY_NAMES = {
    "none": "Baseline (no rebalancing)",
    "smote": "SMOTE",
    "hybrid": "SMOTE + undersampling",
    "classweight": "Class weights",
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left",
    "axes.labelcolor": INK_2, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_class_distribution(counts, path, title):
    """Horizontal bars on a log axis; counts has columns class, count, share."""
    n = len(counts)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * n + 1.1))
    y = np.arange(n)
    ax.barh(y, counts["count"], height=0.6, color=SERIES[0])
    ax.set_xscale("log")
    ax.set_xlim(left=0.8, right=counts["count"].max() * 20)
    ax.set_yticks(y, counts["class"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for yi, (count, share) in enumerate(zip(counts["count"], counts["share"])):
        ax.text(count * 1.15, yi, f"{count:,}  ({share:.2%})", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Rows in sample (log scale)")
    ax.set_title(title)
    _save(fig, path)


def plot_confusion(cm, path, title):
    """Row-normalised confusion matrix (cm is a DataFrame, true classes as rows)."""
    n = len(cm)
    size = max(4.5, 0.32 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    image = ax.imshow(cm.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(n), cm.columns, rotation=45, ha="right", fontsize=8 if n <= 12 else 6)
    ax.set_yticks(range(n), cm.index, fontsize=8 if n <= 12 else 6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                value = cm.iat[i, j]
                if value >= 0.005:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if value > 0.5 else INK)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=9)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_label("Share of the true class", color=INK_2)
    _save(fig, path)


def plot_correlation(corr, path, title):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.17 * n + 2.5, 0.17 * n + 2))
    image = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(n), corr.index, fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("Pearson correlation", color=INK_2)
    _save(fig, path)


def plot_grouped_bars(table, value, path, title, ylabel, ylim=None, classes=None, strategies=None):
    """Columns grouped by class, one colour per strategy.

    table has columns class, strategy, <value> and optionally <value>_std.
    """
    classes = classes or list(dict.fromkeys(table["class"]))
    strategies = strategies or list(dict.fromkeys(table["strategy"]))
    n_c, n_s = len(classes), len(strategies)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.22 * n_c * n_s + 1.5), 3.6))
    slot = 0.8 / n_s
    x = np.arange(n_c)
    for i, strategy in enumerate(strategies):
        rows = table[table["strategy"] == strategy].set_index("class").reindex(classes)
        offsets = x - 0.4 + slot * (i + 0.5)
        ax.bar(offsets, rows[value], width=slot * 0.88, color=STRATEGY_COLORS[strategy],
               label=STRATEGY_NAMES[strategy])
        std = rows.get(f"{value}_std")
        if std is not None and (std.fillna(0) > 0).any():
            ax.errorbar(offsets, rows[value], yerr=std, fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=0)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(x, classes, rotation=0 if n_c <= 10 else 60, ha="center" if n_c <= 10 else "right")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n_s, fontsize=8, handlelength=1, borderaxespad=0.2)
    _save(fig, path)


def plot_k_sweep(sweep, path, title):
    """Performance and cost against the number of features, as small multiples (no dual axes)."""
    panels = [
        ("predict_us_per_flow", "Prediction time (µs per flow)"),
        ("model_size_mb", "Model size (MB)"),
        ("train_s", "Training time (s)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    k = sweep["n_features"]
    marker = dict(marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax = axes[0]
    for (col, label), color in zip([("test_macro_f1", "Macro-F1"), ("test_minority_recall", "Minority-class recall")],
                                   SERIES):
        ax.plot(k, sweep[col], color=color, label=label, **marker)
        std = sweep.get(f"{col}_std")
        if std is not None and (std > 0).any():
            ax.fill_between(k, sweep[col] - std, sweep[col] + std, color=color, alpha=0.1, linewidth=0)
    ax.set_title("Detection performance (test)")
    ax.legend(loc="lower right", fontsize=8)

    for ax, (col, label) in zip(axes[1:], panels):
        ax.plot(k, sweep[col], color=SERIES[0], **marker)
        ax.set_title(label)
        ax.set_ylim(bottom=0)

    for ax in axes:
        ax.set_xticks(k)
        ax.set_xlabel("Number of features")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
