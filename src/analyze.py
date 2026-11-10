"""Aggregate results/runs.csv into report tables and figures (mean ± std over seeds).

    python -m src.analyze compare --level 8 --model rf
        Baseline vs every rebalancing strategy for one model, all features (the improvement),
        with bootstrap confidence intervals and McNemar tests against the baseline.
    python -m src.analyze leaderboard --level 8
        Every model x strategy in one table and chart.
    python -m src.analyze best --level 8 --model rf
        Prints the strategy with the best validation macro-F1 (used to pick the extension's strategy).
    python -m src.analyze sweep --level 8 --model rf --strategy smote --ranking ranking_mi_8class
        Performance and cost against the number of selected features (the extension).
    python -m src.analyze duplicates --level 8
        Baseline scores with and without exact duplicate rows (runs tagged 'withdups').

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
from src.resample import DEFAULT_K_NEIGHBORS, DEFAULT_SMOTE_TARGET, DEFAULT_UNDERSAMPLE_CAP, OVERSAMPLING, STRATEGIES
from src.stats import bootstrap_ci, load_predictions, paired_test
from src.train import DEFAULT_N_ESTIMATORS, ENSEMBLES, MODEL_NAMES, MODELS

RANKING_NAMES = {"mi": "mutual information", "rf": "random-forest importance", "rfe": "RFE"}

SUMMARY_COLS = [
    "test_accuracy", "test_balanced_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "test_mcc", "test_minority_recall", "test_minority_f1", "test_worst_class_recall", "val_macro_f1",
    "val_minority_recall", "train_rows_resampled", "resample_s", "train_s", "predict_us_per_flow", "model_size_mb",
    "tree_nodes",
]


def select_runs(runs, args, strategies, models=None, tag=None):
    models = models or [args.model]
    tag = args.tag if tag is None else tag
    r = runs[(runs["level"] == args.level) & runs["model"].isin(models)
             & runs["strategy"].isin(strategies) & (runs["tag"] == tag)]
    uses_smote = r["strategy"].isin(OVERSAMPLING)
    r = r[~uses_smote | ((r["smote_target"] == args.smote_target) & (r["k_neighbors"] == args.k_neighbors))]
    r = r[(r["strategy"] != "hybrid") | (r["undersample_cap"] == args.undersample_cap)]
    r = r[~r["model"].isin(ENSEMBLES) | (r["n_estimators"] == args.n_estimators)]
    r = r[r["max_depth"].isna()] if args.max_depth is None else r[r["max_depth"] == args.max_depth]
    return r


def summarise(runs, by):
    cols = [c for c in SUMMARY_COLS if c in runs]
    table = mean_std(runs, by, cols)
    table["seeds"] = runs.groupby(by, sort=False).size().to_numpy()
    return table


def per_class_runs(results_dir, runs, split):
    frames = []
    for run in runs.itertuples():
        table = pd.read_csv(Path(results_dir) / "per_class" / f"{run.run_id}.csv")
        frames.append(table[table["split"] == split].assign(strategy=run.strategy, seed=run.seed))
    return pd.concat(frames, ignore_index=True)


def minority_indices(run, classes):
    names = str(run.minority_classes).split(";") if isinstance(run.minority_classes, str) else []
    return [classes.index(c) for c in names if c in classes]


def significance(results_dir, runs):
    """Each strategy against the baseline with the same seed, on the same test rows."""
    rows = []
    base = runs[runs["strategy"] == "none"].set_index("seed")
    for run in runs[runs["strategy"] != "none"].itertuples():
        if run.seed not in base.index:
            continue
        a = load_predictions(results_dir, base.loc[run.seed, "run_id"])
        b = load_predictions(results_dir, run.run_id)
        if a is None or b is None:
            continue
        (y, p_base, classes_a), (_, p_new, classes_b) = a, b
        if classes_a != classes_b:  # a class that only one model predicts: re-code on the union
            union = sorted(set(classes_a) | set(classes_b), key=(classes_a + classes_b).index)
            recode = lambda codes, cl: pd.Series(codes).map(lambda i: union.index(cl[i])).to_numpy()
            y, p_base, p_new = recode(y, classes_a), recode(p_base, classes_a), recode(p_new, classes_b)
            classes_a = union
        result = paired_test(y, p_base, p_new, len(classes_a), minority_indices(run, classes_a), seed=int(run.seed))
        ci = bootstrap_ci(y, p_new, len(classes_a), minority_indices(run, classes_a), seed=int(run.seed))
        rows.append({"strategy": run.strategy, "seed": run.seed, **result,
                     "macro_f1_low": ci["macro_f1_ci"][0], "macro_f1_high": ci["macro_f1_ci"][1]})
    return pd.DataFrame(rows)


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

    summary = summarise(runs, ["strategy"]).set_index("strategy").loc[strategies].reset_index()
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
        if args.split == "test":
            sig = significance(args.results_dir, runs)
            if len(sig):
                sig.to_csv(tables / f"{name}_significance.csv", index=False)
                shown = sig.groupby("strategy", sort=False).agg(
                    delta_macro_f1=("delta_macro_f1", "mean"), delta_minority_recall=("delta_minority_recall", "mean"),
                    seeds=("seed", "size"), seeds_significant=("p_mcnemar", lambda p: int((p < 0.05).sum())),
                ).reset_index()
                print("\nAgainst the baseline (same seed, same test rows; significant = McNemar p < 0.05):")
                print(shown.replace({"strategy": STRATEGY_NAMES}).to_string(index=False, float_format="%+.4f"))

    shown = summary[["strategy", "seeds", "test_accuracy", "test_macro_f1", "test_macro_f1_std",
                     "test_minority_recall", "test_worst_class_recall", "train_s"]]
    print(shown.replace({"strategy": STRATEGY_NAMES}).to_string(index=False, float_format="%.4f"))
    print(f"\nTables in {tables}/{name}_*.csv, figures in {figures}/{name}_*.png")


def leaderboard(args):
    runs = select_runs(load_runs(args.results_dir), args, STRATEGIES, models=list(MODELS))
    runs = runs[runs["feature_ranking"] == ""]
    if runs.empty:
        raise SystemExit("No matching runs")
    table = summarise(runs, ["model", "strategy"]).sort_values("test_macro_f1", ascending=False)
    name = f"leaderboard_{args.level}c"
    out = Path(args.results_dir) / "tables" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    models = [m for m in MODELS if m in set(table["model"])]
    strategies = [s for s in STRATEGIES if s in set(table["strategy"])]
    chart = table.assign(**{"class": table["model"].map(MODEL_NAMES)})
    names = [MODEL_NAMES[m] for m in models]
    figures = Path(args.results_dir) / "figures"
    for metric, label in [("test_macro_f1", "Macro-F1"), ("test_minority_recall", "Minority-class recall")]:
        plot_grouped_bars(chart, metric, figures / f"{name}_{metric}.png",
                          f"{label} by model and strategy, {args.level}-class (test set)", label, (0, 1.05),
                          names, strategies)
    shown = table[["model", "strategy", "seeds", "test_macro_f1", "test_minority_recall", "test_worst_class_recall",
                   "predict_us_per_flow", "model_size_mb"]]
    print(shown.replace({"strategy": STRATEGY_NAMES, "model": MODEL_NAMES}).to_string(index=False, float_format="%.4f"))
    print(f"\nTable {out}")


def best(args):
    """Print the strategy with the best mean validation macro-F1 for one model (all features)."""
    runs = select_runs(load_runs(args.results_dir), args, STRATEGIES)
    runs = runs[runs["feature_ranking"] == ""]
    if runs.empty:
        print("smote")  # nothing to choose from yet: fall back to the default
        return
    print(runs.groupby("strategy")["val_macro_f1"].mean().idxmax())


def duplicates(args):
    """Same models without rebalancing, trained and tested with and without exact duplicate rows."""
    runs = load_runs(args.results_dir)
    frames = []
    for tag, label in [("withdups", "with duplicates"), ("", "duplicates removed")]:
        r = select_runs(runs, args, ["none"], models=list(MODELS), tag=tag)
        r = r[r["feature_ranking"] == ""]
        if len(r):
            frames.append(summarise(r, ["model"]).assign(data=label))
    if not frames:
        raise SystemExit("No runs tagged 'withdups'")
    table = pd.concat(frames, ignore_index=True)
    out = Path(args.results_dir) / "tables" / f"duplicates_{args.level}c.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table[["data", "model", "test_accuracy", "test_macro_f1", "test_minority_recall"]]
          .replace({"model": MODEL_NAMES}).to_string(index=False, float_format="%.4f"))


def sweep(args):
    runs = select_runs(load_runs(args.results_dir), args, [args.strategy])
    runs = runs[runs["feature_ranking"].isin([args.ranking, ""])]
    if runs["feature_ranking"].eq(args.ranking).sum() == 0:
        raise SystemExit(f"No runs with feature ranking '{args.ranking}'; train with --k and --ranking first")
    table = mean_std(runs, ["n_features"], [c for c in SUMMARY_COLS if c in runs]).sort_values("n_features")
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
    p = sub.add_parser("leaderboard", parents=[common], help="every model and strategy in one table")
    p.set_defaults(func=leaderboard)
    p = sub.add_parser("best", parents=[common], help="print the best strategy on the validation split")
    p.set_defaults(func=best)
    p = sub.add_parser("duplicates", parents=[common], help="effect of duplicate rows on baseline scores")
    p.set_defaults(func=duplicates)
    p = sub.add_parser("sweep", parents=[common], help="performance and cost vs number of features")
    p.add_argument("--strategy", choices=STRATEGIES, default="smote")
    p.add_argument("--ranking", required=True, help="ranking file stem, e.g. ranking_mi_8class")
    p.set_defaults(func=sweep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
