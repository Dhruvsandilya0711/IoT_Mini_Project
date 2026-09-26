"""Aggregate results/runs.csv into report tables and figures (mean ± std over seeds).

    python -m src.analyze compare --level 8 --model rf
        Baseline vs SMOTE / hybrid / class weights, all features (the improvement).
    python -m src.analyze sweep --level 8 --model rf --strategy smote --ranking ranking_mi_8class
        Performance and cost against the number of selected features (the extension).

Runs are matched on their settings (SMOTE target, undersampling cap, SMOTE
k_neighbors, forest size, depth, tag), so ablation runs never mix into the
main comparison. The defaults match the defaults of src.train.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import load_runs, mean_std
from src.labels import LEVELS, classes_for
from src.plots import STRATEGY_NAMES, plot_grouped_bars, plot_k_sweep
from src.resample import DEFAULT_K_NEIGHBORS, DEFAULT_SMOTE_TARGET, DEFAULT_UNDERSAMPLE_CAP, STRATEGIES
from src.train import DEFAULT_N_ESTIMATORS, MODEL_NAMES, MODELS

RANKING_NAMES = {"mi": "mutual information", "rf": "random-forest importance", "rfe": "RFE"}

SUMMARY_COLS = [
    "test_accuracy", "test_balanced_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "test_minority_recall", "test_minority_f1", "test_worst_class_recall", "val_macro_f1", "val_minority_recall",
    "train_rows_resampled", "resample_s", "train_s", "predict_us_per_flow", "model_size_mb", "tree_nodes",
]


def select_runs(runs, args, strategies):
    r = runs[(runs["level"] == args.level) & (runs["model"] == args.model)
             & runs["strategy"].isin(strategies) & (runs["tag"] == args.tag)]
    uses_smote = r["strategy"].isin(["smote", "hybrid"])
    r = r[~uses_smote | ((r["smote_target"] == args.smote_target) & (r["k_neighbors"] == args.k_neighbors))]
    r = r[(r["strategy"] != "hybrid") | (r["undersample_cap"] == args.undersample_cap)]
    if args.model == "rf":
        r = r[r["n_estimators"] == args.n_estimators]
    r = r[r["max_depth"].isna()] if args.max_depth is None else r[r["max_depth"] == args.max_depth]
    return r


def per_class_runs(results_dir, runs, split):
    frames = []
    for run in runs.itertuples():
        table = pd.read_csv(Path(results_dir) / "per_class" / f"{run.run_id}.csv")
        frames.append(table[table["split"] == split].assign(strategy=run.strategy, seed=run.seed))
    return pd.concat(frames, ignore_index=True)


def compare(args):
    runs = select_runs(load_runs(args.results_dir), args, args.strategies)
    runs = runs[runs["feature_ranking"] == ""]
    if runs.empty:
        raise SystemExit("No matching runs with all features; train some with src.train first")
    strategies = [s for s in STRATEGIES if s in set(runs["strategy"])]
    classes = classes_for(args.level)
    name = f"compare_{args.level}c_{args.model}_{args.split}"
    tables, figures = Path(args.results_dir) / "tables", Path(args.results_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    summary = mean_std(runs, ["strategy"], SUMMARY_COLS)
    summary["seeds"] = summary["strategy"].map(runs.groupby("strategy").size())
    summary = summary.set_index("strategy").loc[strategies].reset_index()
    summary.to_csv(tables / f"{name}_summary.csv", index=False)

    per_class = mean_std(per_class_runs(args.results_dir, runs, args.split),
                         ["strategy", "class"], ["precision", "recall", "f1"])
    classes = [c for c in classes if c in set(per_class["class"])]
    per_class.to_csv(tables / f"{name}_per_class.csv", index=False)

    title = f"{MODEL_NAMES[args.model]}, {args.level}-class ({args.split} set)"
    plot_grouped_bars(per_class, "recall", figures / f"{name}_recall.png", f"Per-class recall: {title}",
                      "Recall", (0, 1.05), classes, strategies)
    plot_grouped_bars(per_class, "f1", figures / f"{name}_f1.png", f"Per-class F1: {title}",
                      "F1-score", (0, 1.05), classes, strategies)

    if "none" in strategies and len(strategies) > 1:
        wide = per_class.pivot(index="class", columns="strategy", values=["recall", "f1"])
        gain = pd.concat(
            [pd.DataFrame({"class": wide.index, "strategy": s,
                           "recall_gain": wide[("recall", s)] - wide[("recall", "none")],
                           "f1_gain": wide[("f1", s)] - wide[("f1", "none")]})
             for s in strategies if s != "none"],
            ignore_index=True,
        )
        gain.to_csv(tables / f"{name}_gain.csv", index=False)
        plot_grouped_bars(gain, "recall_gain", figures / f"{name}_recall_gain.png",
                          f"Change in recall vs baseline: {title}", "Recall change",
                          classes=classes, strategies=[s for s in strategies if s != "none"])

    shown = summary[["strategy", "seeds", "test_accuracy", "test_macro_f1", "test_macro_f1_std",
                     "test_minority_recall", "test_worst_class_recall", "train_s"]]
    print(shown.replace({"strategy": STRATEGY_NAMES}).to_string(index=False, float_format="%.4f"))
    print(f"\nTables in {tables}/{name}_*.csv, figures in {figures}/{name}_*.png")


def sweep(args):
    runs = select_runs(load_runs(args.results_dir), args, [args.strategy])
    runs = runs[runs["feature_ranking"].isin([args.ranking, ""])]
    if runs["feature_ranking"].eq(args.ranking).sum() == 0:
        raise SystemExit(f"No runs with feature ranking '{args.ranking}'; train with --k and --ranking first")
    table = mean_std(runs, ["n_features"], SUMMARY_COLS).sort_values("n_features")
    name = f"sweep_{args.level}c_{args.model}_{args.strategy}_{args.ranking}"
    out = Path(args.results_dir) / "tables" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    plot_k_sweep(
        table, Path(args.results_dir) / "figures" / f"{name}.png",
        f"{MODEL_NAMES[args.model]}, {args.level}-class, {STRATEGY_NAMES[args.strategy]}, "
        f"features ranked by {RANKING_NAMES.get(args.ranking.removeprefix('ranking_').split('_')[0], args.ranking)}",
    )
    print(table[["n_features", "test_macro_f1", "test_minority_recall", "predict_us_per_flow",
                 "model_size_mb", "train_s"]].to_string(index=False, float_format="%.4f"))
    print(f"\nTable {out}, figure in {Path(args.results_dir) / 'figures'}/{name}.png")


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-dir", default="results")
    common.add_argument("--level", type=int, choices=LEVELS, default=8)
    common.add_argument("--model", choices=MODELS, default="rf")
    common.add_argument("--smote-target", type=int, default=DEFAULT_SMOTE_TARGET)
    common.add_argument("--undersample-cap", type=int, default=DEFAULT_UNDERSAMPLE_CAP)
    common.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    common.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    common.add_argument("--max-depth", type=int)
    common.add_argument("--tag", default="")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare", parents=[common], help="baseline vs rebalancing strategies")
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.set_defaults(func=compare)
    p = sub.add_parser("sweep", parents=[common], help="performance and cost vs number of features")
    p.add_argument("--strategy", choices=STRATEGIES, default="smote")
    p.add_argument("--ranking", required=True, help="ranking file stem, e.g. ranking_mi_8class")
    p.set_defaults(func=sweep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
