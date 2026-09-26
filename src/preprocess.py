"""Clean the sample and make the frozen train/val/test split.

The test set is split off first, stratified on the 34 raw classes (so it is
stratified at every level), and is never resampled or used for any choice.
The validation set is carved out of the remaining data for choosing k,
SMOTE targets and other settings.

    python -m src.preprocess --sample data/processed/sample.parquet --out-dir data/processed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.labels import LABEL_COL, to_level

SPLITS = ("train", "val", "test")


def clean(df, drop_duplicates=False):
    """Drop rows containing NaN or +/-inf, and optionally exact duplicate rows."""
    values = df.drop(columns=LABEL_COL).to_numpy()
    inf_rows = np.isinf(values).any(axis=1)
    nan_rows = np.isnan(values).any(axis=1)
    report = {"rows_in": len(df), "rows_with_inf": int(inf_rows.sum()), "rows_with_nan": int(nan_rows.sum())}
    df = df[~(inf_rows | nan_rows)]
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        report["duplicates_dropped"] = before - len(df)
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def split(df, test_size=0.2, val_size=0.1, seed=42, min_class_rows=10):
    counts = df[LABEL_COL].value_counts()
    rare = counts[counts < min_class_rows]
    if len(rare):
        print(f"WARNING: dropping classes with fewer than {min_class_rows} rows: {rare.to_dict()}")
        df = df[~df[LABEL_COL].isin(rare.index)]
    if isinstance(df[LABEL_COL].dtype, pd.CategoricalDtype):
        df = df.assign(**{LABEL_COL: df[LABEL_COL].cat.remove_unused_categories()})
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[LABEL_COL], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val[LABEL_COL], random_state=seed
    )
    return {"train": train, "val": val, "test": test}, sorted(rare.index.astype(str))


def load_split(data_dir, level):
    """Return {split: (X, y)} with y mapped to the requested level (2, 8 or 34)."""
    out = {}
    for name in SPLITS:
        df = pd.read_parquet(Path(data_dir) / f"{name}.parquet")
        y = to_level(df.pop(LABEL_COL).astype(str), level)
        out[name] = (df, y.rename(LABEL_COL))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default="data/processed/sample.parquet")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of all rows (default 0.2)")
    parser.add_argument("--val-size", type=float, default=0.1, help="share of the non-test rows (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-duplicates", action="store_true", help="drop exact duplicate rows before splitting")
    parser.add_argument("--min-class-rows", type=int, default=10)
    args = parser.parse_args(argv)

    df, report = clean(pd.read_parquet(args.sample), args.drop_duplicates)
    parts, dropped = split(df, args.test_size, args.val_size, args.seed, args.min_class_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)

    rows_per_category = pd.DataFrame(
        {name: to_level(part[LABEL_COL].astype(str), 8).value_counts() for name, part in parts.items()}
    ).fillna(0).astype(int)
    info = {
        **vars(args), "cleaning": report, "dropped_classes": dropped,
        "rows": {name: len(part) for name, part in parts.items()},
        "rows_per_category": rows_per_category.to_dict(orient="index"),
    }
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(report, indent=2))
    print(rows_per_category.to_string())


if __name__ == "__main__":
    main()
