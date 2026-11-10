"""Metrics, cost measurements and result persistence shared by every experiment."""

import pickle
import time
from pathlib import Path

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


class _ByteCounter:
    """A write-only file that only counts bytes, so large models are measured without touching disk."""

    def __init__(self):
        self.n = 0

    def write(self, data):
        size = memoryview(data).nbytes
        self.n += size
        return size


def model_size_mb(model):
    """Size of the serialised model (pickle), i.e. what would be shipped to a gateway."""
    counter = _ByteCounter()
    pickle.dump(model, counter, protocol=pickle.HIGHEST_PROTOCOL)
    return counter.n / 1e6


def node_count(model):
    model = getattr(model, "model", model)  # unwrap a ThresholdedClassifier
    if hasattr(model, "booster_"):  # LightGBM: a tree with L leaves has 2L - 1 nodes
        return int(sum(2 * t["num_leaves"] - 1 for t in model.booster_.dump_model()["tree_info"]))
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
    if "tuned_weights" not in runs:  # results from before threshold tuning existed
        runs["tuned_weights"] = ""
    return runs


def mean_std(df, by, cols):
    """Mean and standard deviation over seeds (std is 0 for a single seed)."""
    grouped = df.groupby(by, sort=False)[cols]
    return grouped.mean().join(grouped.std(ddof=0).fillna(0.0).add_suffix("_std")).reset_index()
