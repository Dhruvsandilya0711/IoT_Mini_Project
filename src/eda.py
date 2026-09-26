"""Exploratory data analysis of the sample: class imbalance and data quality.

    python -m src.eda --sample data/processed/sample.parquet --out-dir results/eda

Writes class counts and imbalance ratios for the 34-, 8- and 2-class tasks,
class-distribution charts (Fig. 1 of the report), a data-quality summary,
per-feature statistics and a feature correlation heatmap.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import LABEL_COL, LEVELS, classes_for, to_level
from src.plots import plot_class_distribution, plot_correlation


def class_counts(labels, level):
    counts = to_level(labels, level).value_counts()
    counts = counts.reindex([c for c in classes_for(level) if c in counts.index])
    table = pd.DataFrame({"class": counts.index, "count": counts.to_numpy()})
    table["share"] = table["count"] / table["count"].sum()
    table["imbalance_ratio"] = table["count"].max() / table["count"]
    return table.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="data/processed/sample.parquet")
    parser.add_argument("--out-dir", default="results/eda")
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.sample)
    labels = df[LABEL_COL].astype(str)
    features = df.drop(columns=LABEL_COL)

    for level in LEVELS:
        table = class_counts(labels, level)
        table.to_csv(out / f"class_counts_{level}class.csv", index=False)
        if level != 2:
            plot_class_distribution(table, out / f"class_distribution_{level}class.png",
                                    f"CICIoT2023 class distribution, {level}-class task")
        print(f"\n{level}-class:\n{table.to_string(index=False, float_format='%.4f')}")

    values = features.to_numpy()
    quality = {
        "rows": len(df),
        "features": features.shape[1],
        "rows_with_nan": int(np.isnan(values).any(axis=1).sum()),
        "rows_with_inf": int(np.isinf(values).any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_features": [c for c in features.columns if features[c].nunique(dropna=True) <= 1],
    }
    (out / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print("\n" + json.dumps(quality, indent=2))

    finite = features.replace([np.inf, -np.inf], np.nan)
    finite.describe().T.to_csv(out / "feature_summary.csv")
    corr = finite.drop(columns=quality["constant_features"]).corr()
    corr.to_csv(out / "feature_correlation.csv")
    plot_correlation(corr, out / "feature_correlation.png", "Feature correlation (Pearson)")
    print(f"\nWrote EDA outputs to {out}")


if __name__ == "__main__":
    main()
