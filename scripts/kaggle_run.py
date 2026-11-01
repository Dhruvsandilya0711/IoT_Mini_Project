"""Run the whole project on Kaggle and pack the results into /kaggle/working/results.zip.

Paste into ONE code cell of a new Kaggle notebook (with the CICIoT2023 dataset added and Internet on):

    !git clone -q --depth 1 -b claude/upbeat-franklin-n50ocy https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git
    %cd IoT_Mini_Project
    !python scripts/kaggle_run.py

Options: --frac 0.03 (share of rows sampled), --seeds "0 1 2", --hours 11 (time budget),
--phases (default: baseline improvement extension duplicates), --input /kaggle/input, --work /kaggle/working.

Exact duplicate rows are removed before the main split (copies of test rows inside the training data
inflate the scores). A second split that keeps them is used by the 'duplicates' phase, which gives
numbers comparable with the base paper and measures how much the duplicates inflate them.

results.zip is rewritten after every phase, and new runs stop once the time budget is used, so a
Kaggle session (12 h limit) always ends with usable results.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


def sh(cmd, env=None):
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        sys.exit(f"Command failed ({result.returncode}): {cmd}")


def has_label(path):
    try:
        cols = pd.read_csv(path, nrows=0).columns if path.endswith(".csv") else pq.read_schema(path).names
        return any(c.strip().lower() == "label" for c in cols)
    except Exception:
        return False


def link_raw_files(input_dir, raw_dir):
    """Symlink the folder with the most labelled CICIoT2023 files, so merged copies elsewhere aren't read twice."""
    files = [p for p in glob.glob(f"{input_dir}/**/*", recursive=True)
             if p.lower().endswith((".csv", ".parquet")) and "IoT_Mini_Project" not in p]
    data_files = [p for p in files if has_label(p)]
    by_dir = Counter(os.path.dirname(p) for p in data_files)
    print("Folders with labelled CICIoT2023 files:", by_dir.most_common(5))
    if not by_dir:
        sys.exit("No CICIoT2023 files found. Add the dataset with 'Add Input' (search CICIoT2023).")
    source = by_dir.most_common(1)[0][0]
    chosen = sorted(p for p in data_files if os.path.dirname(p) == source)
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir)
    for p in chosen:
        os.symlink(p, os.path.join(raw_dir, os.path.basename(p)))
    size_gb = sum(os.path.getsize(p) for p in chosen) / 1e9
    print(f"Using {len(chosen)} files ({size_gb:.1f} GB) from {source}")


def pack(work, note):
    os.makedirs("results/data_info", exist_ok=True)
    for f in glob.glob("data/processed*/*.json"):
        name = os.path.basename(f) if "withdups" not in f else "withdups_" + os.path.basename(f)
        shutil.copy(f, os.path.join("results/data_info", name))
    shutil.make_archive(f"{work}/results", "zip", "results")
    print(f"\n>>> results.zip updated after: {note}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.03)
    parser.add_argument("--seeds", default="0 1 2")
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--phases", nargs="+", default=["baseline", "improvement", "extension", "duplicates"])
    parser.add_argument("--hours", type=float, default=11.0, help="stop starting new runs after this many hours")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    start = time.time()
    sh(f"{sys.executable} -m pip install -q 'imbalanced-learn>=0.12' 'pyarrow>=14' 'lightgbm>=4' 'scipy>=1.9'")
    raw = f"{args.work}/raw"
    link_raw_files(args.input, raw)

    py = sys.executable
    sh(f"{py} -m src.load --raw-dir {raw} --frac {args.frac}")
    sh(f"{py} -m src.preprocess --drop-duplicates")
    sh(f"{py} -m src.preprocess --out-dir data/processed_withdups")
    sh(f"{py} -m src.eda")
    pack(args.work, "EDA")

    env = {**os.environ, "SEEDS": args.seeds, "PATH": os.path.dirname(py) + os.pathsep + os.environ["PATH"],
           "RUN_DEADLINE": str(start + args.hours * 3600)}
    for phase in args.phases:
        sh(f"bash scripts/run_experiments.sh {phase}", env=env)
        pack(args.work, phase)

    tables = ["leaderboard_8c.csv", "duplicates_8c.csv"]
    for f in [f"results/tables/{t}" for t in tables if os.path.exists(f"results/tables/{t}")] \
            + sorted(glob.glob("results/tables/sweep_*.csv")):
        table = pd.read_csv(f)
        cols = [c for c in ["data", "model", "strategy", "n_features", "seeds", "test_accuracy", "test_macro_f1", "test_minority_recall",
                            "test_worst_class_recall", "predict_us_per_flow", "model_size_mb", "train_s"] if c in table]
        print(f"\n== {f}\n{table[cols].to_string(index=False, float_format='%.4f')}")
    print(f"\nDone in {(time.time() - start) / 3600:.1f} h. Download results.zip from the Output panel "
          "and upload it to the Claude chat.")


if __name__ == "__main__":
    main()
