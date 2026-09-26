"""Stream the raw CICIoT2023 files and build a random sample that keeps the class ratios.

Every row is kept independently with probability --frac, so each class keeps
its true share of the data (the imbalance is what this project studies).
Row-level sampling also works for classes with only a handful of rows per
file, where per-file rounding would drop them entirely.

    python -m src.load --raw-dir data/raw --frac 0.05 --out data/processed/sample.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import LABEL_COL, canonical, category_members, to_level

# The 46 features of the public CSV release, for a sanity check only.
EXPECTED_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


def find_raw_files(raw_dir):
    files = sorted(p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in {".csv", ".parquet"})
    if not files:
        raise FileNotFoundError(f"No .csv or .parquet files under {raw_dir}")
    return files


def _label_column(columns):
    matches = [c for c in columns if c.strip().lower() == LABEL_COL]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one 'label' column, found {matches}")
    return matches[0]


def read_part(path):
    """Read one raw file with float32 features and a canonical 'label' column."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        header = pd.read_csv(path, nrows=0).columns
        label = _label_column(header)
        df = pd.read_csv(path, dtype={c: "float32" for c in header if c != label})
    df.columns = df.columns.str.strip()
    df = df.rename(columns={_label_column(df.columns): LABEL_COL})
    features = df.columns.drop(LABEL_COL)
    df[features] = df[features].astype("float32")
    df[LABEL_COL] = canonical(df[LABEL_COL])
    return df


def build_sample(raw_dir, frac, seed=42, keep_all=(), max_files=None):
    files = find_raw_files(raw_dir)[:max_files]
    keep = {raw for name in keep_all for raw in category_members(name)}
    rng = np.random.default_rng(seed)
    parts, columns = [], None
    for i, path in enumerate(files, 1):
        df = read_part(path)
        if columns is None:
            columns = list(df.columns)
        elif list(df.columns) != columns:
            raise ValueError(f"{path.name} has different columns from {files[0].name}")
        mask = rng.random(len(df)) < frac
        if keep:
            mask |= df[LABEL_COL].isin(keep).to_numpy()
        parts.append(df[mask])
        print(f"[{i}/{len(files)}] {path.name}: kept {mask.sum():,} of {len(df):,} rows", flush=True)
    sample = pd.concat(parts, ignore_index=True)
    sample[LABEL_COL] = sample[LABEL_COL].astype("category")

    features = [c for c in columns if c != LABEL_COL]
    if features != EXPECTED_FEATURES:
        missing = sorted(set(EXPECTED_FEATURES) - set(features))
        extra = sorted(set(features) - set(EXPECTED_FEATURES))
        print(f"Note: feature columns differ from the public CSV release (missing={missing}, extra={extra})")
    return sample, len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="folder with the CICIoT2023 CSV (or Parquet) parts")
    parser.add_argument("--out", default="data/processed/sample.parquet")
    parser.add_argument("--frac", type=float, default=0.05, help="fraction of rows to keep (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, help="only read the first N files (quick tests)")
    parser.add_argument(
        "--keep-all", nargs="*", default=[],
        help="raw labels or categories (e.g. Web BruteForce) to keep in full. "
             "Changes the class ratios, so say so in the report if you use it.",
    )
    args = parser.parse_args(argv)

    sample, n_files = build_sample(args.raw_dir, args.frac, args.seed, args.keep_all, args.max_files)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)

    counts = to_level(sample[LABEL_COL].astype(str), 8).value_counts()
    info = {
        "raw_dir": str(args.raw_dir), "files_read": n_files, "frac": args.frac, "seed": args.seed,
        "keep_all": args.keep_all, "rows": len(sample), "features": sample.shape[1] - 1,
        "rows_per_category": counts.to_dict(),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"\nWrote {len(sample):,} rows to {out}\n{counts.to_string()}")


if __name__ == "__main__":
    main()
