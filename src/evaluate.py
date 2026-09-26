"""Metrics, cost measurements and result persistence shared by every experiment."""

import os
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# A class counts as "minority" if it is under 1% of the (original) training data.
MINORITY_SHARE = 0.01


def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def per_class_table(y_true, y_pred, classes):
    """Precision, recall, F1 and support per class, in the given display order."""
    present = set(y_true) | set(y_pred)
    labels = [c for c in classes if c in present]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s})


def confusion_table(y_true, y_pred, classes, normalize="true"):
    present = set(y_true) | set(y_pred)
    labels = [c for c in classes if c in present]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="predicted"))


def minority_classes(y_train):
    shares = y_train.value_counts(normalize=True)
    return sorted(shares[shares < MINORITY_SHARE].index)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def model_size_mb(model):
    """Size of the model serialised with joblib, i.e. what would be shipped to a gateway."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        joblib.dump(model, path)
        return os.path.getsize(path) / 1e6


def node_count(model):
    trees = getattr(model, "estimators_", [model])
    return int(sum(t.tree_.node_count for t in trees))


def append_run(results_dir, row):
    """Append one run to results/runs.csv (a re-run of the same run_id replaces the old row)."""
    path = Path(results_dir) / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != list(new.columns):
            raise ValueError(f"{path} has different columns from this run; move it aside to start a new file")
        old[old["run_id"] != row["run_id"]].to_csv(path, index=False)
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        new.to_csv(path, index=False)


def load_runs(results_dir):
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run src.train first")
    runs = pd.read_csv(path)
    runs["tag"] = runs["tag"].fillna("")
    runs["feature_ranking"] = runs["feature_ranking"].fillna("")
    return runs


def mean_std(df, by, cols):
    """Mean and standard deviation over seeds (std is 0 for a single seed)."""
    grouped = df.groupby(by, sort=False)[cols]
    return grouped.mean().join(grouped.std(ddof=0).fillna(0.0), rsuffix="_std").reset_index()
