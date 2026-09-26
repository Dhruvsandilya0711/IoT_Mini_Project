"""Train and evaluate one configuration; append the result to results/runs.csv.

Scaling, resampling and feature selection are all fit on the training split
only. The test split is only ever predicted.

    python -m src.train --level 8 --model rf --strategy none
    python -m src.train --level 8 --model rf --strategy smote --smote-target 50000
    python -m src.train --level 8 --model rf --strategy smote --k 15 \
        --ranking results/features/ranking_mi_8class.csv

Outputs (per run_id):
    results/runs.csv                  one row per run: settings, metrics, costs
    results/per_class/<run_id>.csv    precision/recall/F1/support per class (val and test)
    results/confusion/<run_id>.csv    row-normalised test confusion matrix (+ .png)

A run whose run_id is already in runs.csv is skipped unless --force is given,
so an interrupted batch (e.g. a Kaggle session timing out) can be resumed.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import (
    append_run,
    compute_metrics,
    confusion_table,
    minority_classes,
    model_size_mb,
    node_count,
    per_class_table,
    timed,
)
from src.feature_selection import load_ranking
from src.labels import LEVELS, classes_for
from src.plots import STRATEGY_NAMES, plot_confusion
from src.preprocess import load_split
from src.resample import DEFAULT_K_NEIGHBORS, DEFAULT_SMOTE_TARGET, DEFAULT_UNDERSAMPLE_CAP, STRATEGIES, resample

MODELS = ("dt", "rf")
MODEL_NAMES = {"dt": "Decision Tree", "rf": "Random Forest"}
DEFAULT_N_ESTIMATORS = 100


def build_model(name, class_weight=None, seed=42, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None, n_jobs=-1):
    if name == "dt":
        return DecisionTreeClassifier(class_weight=class_weight, max_depth=max_depth, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight, max_depth=max_depth,
            n_jobs=n_jobs, random_state=seed,
        )
    raise ValueError(f"model must be one of {MODELS}, got {name!r}")


def config_id(args, n_features):
    """Everything that identifies a configuration except the seed."""
    strategy = args.strategy
    if args.strategy in ("smote", "hybrid"):
        strategy += str(args.smote_target)
        if args.strategy == "hybrid":
            strategy += f"-cap{args.undersample_cap}"
        if args.k_neighbors != DEFAULT_K_NEIGHBORS:
            strategy += f"-kn{args.k_neighbors}"
    model = args.model
    if args.model == "rf" and args.n_estimators != DEFAULT_N_ESTIMATORS:
        model += f"{args.n_estimators}"
    if args.max_depth:
        model += f"-d{args.max_depth}"
    features = f"k{n_features}-{Path(args.ranking).stem}" if args.k else "kall"
    parts = [f"{args.level}c", model, strategy, features] + ([args.tag] if args.tag else [])
    return "-".join(parts)


def already_done(results_dir, run_id):
    path = Path(results_dir) / "runs.csv"
    return path.exists() and run_id in set(pd.read_csv(path, usecols=["run_id"])["run_id"])


def run(args):
    data = load_split(args.data_dir, args.level)
    X_train, y_train = data["train"]
    features = list(X_train.columns)
    if args.k:
        if not args.ranking:
            raise SystemExit("--k needs --ranking (a CSV from src.feature_selection)")
        features = load_ranking(args.ranking)[: args.k]

    cfg = config_id(args, len(features))
    run_id = f"{cfg}-s{args.seed}"
    if not args.force and already_done(args.results_dir, run_id):
        print(f"{run_id} is already in runs.csv, skipping (use --force to re-run)")
        return None
    print(f"== {run_id}")

    scaler = StandardScaler().fit(X_train[features])

    def scale(X):
        return pd.DataFrame(scaler.transform(X[features]).astype("float32"), columns=features)

    X_train, y_train = scale(X_train), y_train.reset_index(drop=True)
    X_val, y_val = scale(data["val"][0]), data["val"][1].reset_index(drop=True)
    X_test, y_test = scale(data["test"][0]), data["test"][1].reset_index(drop=True)
    minority = minority_classes(y_train)

    (X_fit, y_fit), resample_s = timed(
        resample, X_train, y_train, args.strategy,
        args.smote_target, args.undersample_cap, args.k_neighbors, args.seed,
    )
    print(f"training rows: {len(y_train):,} -> {len(y_fit):,} after '{args.strategy}'")

    model = build_model(
        args.model, "balanced" if args.strategy == "classweight" else None,
        args.seed, args.n_estimators, args.max_depth, args.n_jobs,
    )
    _, train_s = timed(model.fit, X_fit, y_fit)
    y_pred_test, test_predict_s = timed(model.predict, X_test)
    y_pred_val = model.predict(X_val)

    classes = classes_for(args.level)
    test_metrics = compute_metrics(y_test, y_pred_test)
    val_metrics = compute_metrics(y_val, y_pred_val)
    test_classes = per_class_table(y_test, y_pred_test, classes)
    val_classes = per_class_table(y_val, y_pred_val, classes)
    test_minority = test_classes[test_classes["class"].isin(minority)]
    val_minority = val_classes[val_classes["class"].isin(minority)]

    results_dir = Path(args.results_dir)
    for sub in ("per_class", "confusion"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    pd.concat([val_classes.assign(split="val"), test_classes.assign(split="test")]).to_csv(
        results_dir / "per_class" / f"{run_id}.csv", index=False
    )
    cm = confusion_table(y_test, y_pred_test, classes)
    cm.to_csv(results_dir / "confusion" / f"{run_id}.csv")
    plot_confusion(
        cm, results_dir / "confusion" / f"{run_id}.png",
        f"{MODEL_NAMES[args.model]}, {args.level}-class, {STRATEGY_NAMES[args.strategy]}, "
        f"{len(features)} features (test)",
    )
    if args.save_model:
        (results_dir / "models").mkdir(exist_ok=True)
        joblib.dump({"scaler": scaler, "features": features, "model": model}, results_dir / "models" / f"{run_id}.joblib")

    row = {
        "run_id": run_id, "config_id": cfg,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "level": args.level, "model": args.model, "strategy": args.strategy,
        "smote_target": args.smote_target if args.strategy in ("smote", "hybrid") else None,
        "undersample_cap": args.undersample_cap if args.strategy == "hybrid" else None,
        "k_neighbors": args.k_neighbors if args.strategy in ("smote", "hybrid") else None,
        "n_features": len(features),
        "feature_ranking": Path(args.ranking).stem if args.k else "",
        "seed": args.seed, "tag": args.tag,
        "n_estimators": args.n_estimators if args.model == "rf" else None, "max_depth": args.max_depth,
        "train_rows": len(y_train), "train_rows_resampled": len(y_fit), "test_rows": len(y_test),
        "minority_classes": ";".join(minority),
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "test_minority_recall": test_minority["recall"].mean() if len(test_minority) else float("nan"),
        "test_minority_f1": test_minority["f1"].mean() if len(test_minority) else float("nan"),
        "test_worst_class_recall": test_classes["recall"].min(),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "val_minority_recall": val_minority["recall"].mean() if len(val_minority) else float("nan"),
        "resample_s": resample_s, "train_s": train_s,
        "test_predict_s": test_predict_s,
        "predict_us_per_flow": 1e6 * test_predict_s / len(y_test),
        "model_size_mb": model_size_mb(model), "tree_nodes": node_count(model),
    }
    append_run(results_dir, row)

    print(test_classes.to_string(index=False, float_format="%.4f"))
    print(
        f"test: accuracy {row['test_accuracy']:.4f}  macro-F1 {row['test_macro_f1']:.4f}  "
        f"minority recall {row['test_minority_recall']:.4f}  "
        f"train {train_s:.1f}s  predict {row['predict_us_per_flow']:.2f}us/flow"
    )
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--level", type=int, choices=LEVELS, default=8)
    parser.add_argument("--model", choices=MODELS, default="rf")
    parser.add_argument("--strategy", choices=STRATEGIES, default="none")
    parser.add_argument("--smote-target", type=int, default=DEFAULT_SMOTE_TARGET,
                        help="classes below this many training rows are raised to it")
    parser.add_argument("--undersample-cap", type=int, default=DEFAULT_UNDERSAMPLE_CAP,
                        help="hybrid only: classes above this many training rows are cut to it")
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS, help="SMOTE neighbours")
    parser.add_argument("--k", type=int, help="use only the top-k features of --ranking")
    parser.add_argument("--ranking", help="feature ranking CSV from src.feature_selection")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="label for runs with other non-default settings")
    parser.add_argument("--save-model", action="store_true", help="also save scaler + model to results/models/")
    parser.add_argument("--force", action="store_true", help="re-run even if run_id is already in runs.csv")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
