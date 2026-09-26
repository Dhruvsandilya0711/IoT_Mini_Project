"""Rank the features on the training split (validation and test data are never used).

By default the ranking uses real training rows only, before any resampling,
so SMOTE's synthetic points cannot bias it. Pass --strategy smote (or hybrid)
to rank on rebalanced data as well and compare the two rankings.

    mi   mutual information (mutual_info_classif)
    rf   random-forest impurity importance
    rfe  recursive feature elimination with a random forest (slowest; use a smaller --n-rows)

    python -m src.feature_selection --level 8 --method mi
    python -m src.feature_selection --level 8 --method rfe --n-rows 50000
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.labels import LEVELS
from src.preprocess import load_split
from src.resample import DEFAULT_K_NEIGHBORS, DEFAULT_SMOTE_TARGET, DEFAULT_UNDERSAMPLE_CAP, STRATEGIES, resample

METHODS = ("mi", "rf", "rfe")


def stratified_subsample(X, y, n_rows, seed):
    if len(y) <= n_rows:
        return X, y
    X, _, y, _ = train_test_split(X, y, train_size=n_rows, stratify=y, random_state=seed)
    return X, y


def rank_features(X, y, method, seed=42, rfe_step=1):
    """Return a DataFrame of feature, score, rank (rank 1 = most useful)."""
    if method == "mi":
        scores = mutual_info_classif(X, y, random_state=seed)
    elif method == "rf":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(X, y)
        scores = model.feature_importances_
    elif method == "rfe":
        estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed)
        rfe = RFE(estimator, n_features_to_select=1, step=rfe_step).fit(X, y)
        scores = rfe.ranking_.max() - rfe.ranking_ + 1  # higher is better, like the other methods
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    ranking = pd.DataFrame({"feature": X.columns, "score": scores})
    ranking = ranking.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def load_ranking(path):
    return pd.read_csv(path).sort_values("rank")["feature"].tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--level", type=int, choices=LEVELS, default=8)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--n-rows", type=int, default=200_000, help="stratified training subsample size")
    parser.add_argument("--strategy", choices=STRATEGIES, default="none", help="rank on rebalanced data instead")
    parser.add_argument("--smote-target", type=int, default=DEFAULT_SMOTE_TARGET)
    parser.add_argument("--undersample-cap", type=int, default=DEFAULT_UNDERSAMPLE_CAP)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--rfe-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="output CSV (default results/features/ranking_<method>_<level>class.csv)")
    args = parser.parse_args(argv)

    X, y = load_split(args.data_dir, args.level)["train"]
    X, y = stratified_subsample(X, y, args.n_rows, args.seed)
    X = pd.DataFrame(StandardScaler().fit_transform(X).astype("float32"), columns=X.columns)
    y = y.reset_index(drop=True)
    X, y = resample(X, y, args.strategy, args.smote_target, args.undersample_cap, args.k_neighbors, args.seed)

    ranking = rank_features(X, y, args.method, args.seed, args.rfe_step)
    suffix = "" if args.strategy in ("none", "classweight") else f"_{args.strategy}"
    out = Path(args.out or Path(args.results_dir) / "features" / f"ranking_{args.method}_{args.level}class{suffix}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)
    print(f"Ranked {len(ranking)} features on {len(y):,} rows -> {out}")
    print(ranking.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
