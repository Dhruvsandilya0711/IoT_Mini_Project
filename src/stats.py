"""Statistical checks on saved test predictions (results/predictions/<run_id>.npz).

    bootstrap CI   95% interval for macro-F1 and minority-class recall of one run
    paired test    difference between two runs on the same test rows: bootstrap CI,
                   one-sided bootstrap p-value, and McNemar's test on per-row correctness
"""

from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

N_BOOT = 1000


def save_predictions(results_dir, run_id, y_true, y_pred, classes):
    index = {c: i for i, c in enumerate(classes)}
    path = Path(results_dir) / "predictions" / f"{run_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=np.array([index[c] for c in y_true], dtype=np.int16),
                        y_pred=np.array([index[c] for c in y_pred], dtype=np.int16), classes=np.array(classes))


def load_predictions(results_dir, run_id):
    path = Path(results_dir) / "predictions" / f"{run_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["y_true"].astype(np.int64), data["y_pred"].astype(np.int64), list(data["classes"])


def _scores(y, p, n, minority):
    cm = np.bincount(y * n + p, minlength=n * n).reshape(n, n)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    denom = support + predicted
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
    macro_f1 = f1[denom > 0].mean()
    minority_recall = np.nanmean(recall[minority]) if len(minority) else np.nan
    return macro_f1, minority_recall


def _boot_indices(n_rows, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n_rows, n_rows)


def bootstrap_ci(y, p, n, minority, n_boot=N_BOOT, seed=0):
    draws = np.array([_scores(y[i], p[i], n, minority) for i in _boot_indices(len(y), n_boot, seed)])
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {"macro_f1_ci": (low[0], high[0]), "minority_recall_ci": (low[1], high[1])}


def mcnemar(y, p_a, p_b):
    """Exact (binomial) McNemar test on which rows each model gets right."""
    a_right, b_right = p_a == y, p_b == y
    only_a, only_b = int(np.sum(a_right & ~b_right)), int(np.sum(~a_right & b_right))
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    if only_a + only_b < 10_000:
        return only_a, only_b, binomtest(only_a, only_a + only_b, 0.5).pvalue
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
    return only_a, only_b, float(chi2.sf(stat, 1))


def paired_test(y, p_base, p_new, n, minority, n_boot=N_BOOT, seed=0):
    """How much better is p_new than p_base on the same test rows?"""
    base, new = _scores(y, p_base, n, minority), _scores(y, p_new, n, minority)
    diffs = np.array([np.subtract(_scores(y[i], p_new[i], n, minority), _scores(y[i], p_base[i], n, minority))
                      for i in _boot_indices(len(y), n_boot, seed)])
    low, high = np.nanpercentile(diffs, [2.5, 97.5], axis=0)
    only_base, only_new, p_mcnemar = mcnemar(y, p_base, p_new)
    return {
        "delta_macro_f1": new[0] - base[0], "delta_macro_f1_low": low[0], "delta_macro_f1_high": high[0],
        "p_macro_f1": float(np.mean(diffs[:, 0] <= 0)),
        "delta_minority_recall": new[1] - base[1], "delta_minority_recall_low": low[1],
        "delta_minority_recall_high": high[1], "p_minority_recall": float(np.nanmean(diffs[:, 1] <= 0)),
        "rows_only_baseline_right": only_base, "rows_only_new_right": only_new, "p_mcnemar": p_mcnemar,
    }
