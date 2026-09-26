"""Small fake CICIoT2023-shaped CSV parts, for testing the pipeline without the real dataset.

    python -m tests.fake_data --out-dir data/fake_raw
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import CATEGORY_OF
from src.load import EXPECTED_FEATURES

# Relative class weights per category: heavily imbalanced, like the real data.
CATEGORY_WEIGHT = {"DDoS": 10, "DoS": 8, "Mirai": 5, "Benign": 6, "Spoofing": 2, "Recon": 1.5,
                   "BruteForce": 0.8, "Web": 0.7}


def write_fake_raw(out_dir, n_files=3, rows_per_file=4000, seed=0):
    rng = np.random.default_rng(seed)
    labels = sorted(CATEGORY_OF)
    weights = np.array([CATEGORY_WEIGHT[CATEGORY_OF[label]] for label in labels])
    centres = rng.normal(0, 1.5, size=(len(labels), len(EXPECTED_FEATURES)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        y = rng.choice(len(labels), size=rows_per_file, p=weights / weights.sum())
        X = centres[y] + rng.normal(0, 1.0, size=(rows_per_file, len(EXPECTED_FEATURES)))
        df = pd.DataFrame(X, columns=EXPECTED_FEATURES)
        df.loc[rng.choice(rows_per_file, 3, replace=False), "Rate"] = np.inf  # like the real data
        df["label"] = np.array(labels)[y]
        df.to_csv(out_dir / f"part-{i:05d}.csv", index=False)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fake_raw")
    parser.add_argument("--files", type=int, default=3)
    parser.add_argument("--rows", type=int, default=4000)
    args = parser.parse_args()
    print(f"Wrote fake data to {write_fake_raw(args.out_dir, args.files, args.rows)}")
