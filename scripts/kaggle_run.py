"""Run the whole project on Kaggle and pack the results into /kaggle/working/results.zip.

Paste into ONE code cell of a new Kaggle notebook (with the CICIoT2023 dataset added and Internet on):

    import os, shutil, subprocess
    os.chdir("/kaggle/working")
    shutil.rmtree("/kaggle/working/IoT_Mini_Project", ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/Dhruvsandilya0711/IoT_Mini_Project.git", "/kaggle/working/IoT_Mini_Project"], check=True)
    os.chdir("/kaggle/working/IoT_Mini_Project")
    !python scripts/kaggle_run.py

Options: --rows 1400000 (rows to sample; about 3% of the full dataset), --frac (fixed share instead),
--seeds "0 1 2", --hours 11 (time budget),
--phases (default: baseline improvement extension duplicates), --input /kaggle/input, --work /kaggle/working.

Exact duplicate rows are removed before the main split (copies of test rows inside the training data
inflate the scores). A second split that keeps them is used by the 'duplicates' phase, which gives
numbers comparable with the base paper and measures how much the duplicates inflate them.

results.zip is rewritten after every phase, and new runs stop once the time budget is used, so a
Kaggle session (12 h limit) always ends with usable results.
"""

import argparse
import glob
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


def sh(cmd, env=None):
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        sys.exit(f"Command failed ({result.returncode}): {cmd}")


DATA_EXTENSIONS = (".csv", ".csv.gz", ".parquet")


def has_label(path):
    try:
        cols = pq.read_schema(path).names if path.endswith(".parquet") else pd.read_csv(path, nrows=0).columns
        return any(c.strip().lower() == "label" for c in cols)
    except Exception:
        return False


def list_files(input_dir):
    """Every file under input_dir, following symlinked folders (Kaggle mounts inputs that way)."""
    out = []
    for root, _, names in os.walk(input_dir, followlinks=True):
        out += [os.path.join(root, n) for n in names]
    return out


def describe_inputs(input_dir, all_files):
    print(f"Everything under {input_dir}:")
    if not os.path.exists(input_dir):
        print("  (the folder does not exist)")
        return
    by_folder = Counter(os.path.dirname(p) for p in all_files)
    if not by_folder:
        print("  (no files at all: no input is attached to this run)")
    for folder, n in by_folder.most_common(12):
        names = sorted(os.path.basename(p) for p in all_files if os.path.dirname(p) == folder)
        size = sum(os.path.getsize(p) for p in all_files if os.path.dirname(p) == folder) / 1e9
        print(f"  {folder}: {n} files, {size:.1f} GB, e.g. {names[:3]}")


def link_raw_files(input_dir, raw_dir):
    """Symlink the folder with the most labelled CICIoT2023 files, so merged copies elsewhere aren't read twice."""
    all_files = [p for p in list_files(input_dir) if "IoT_Mini_Project" not in p]
    describe_inputs(input_dir, all_files)
    data_files = [p for p in all_files if p.lower().endswith(DATA_EXTENSIONS) and has_label(p)]
    by_dir = Counter(os.path.dirname(p) for p in data_files)
    print("Folders with labelled CICIoT2023 files:", by_dir.most_common(5))
    if not by_dir:
        sys.exit("No CICIoT2023 files found (CSV or Parquet files with a 'label' column). "
                 "Check the notebook's Input panel: add the dataset with '+ Add Input' (search CICIoT2023).")
    source = by_dir.most_common(1)[0][0]
    chosen = sorted(p for p in data_files if os.path.dirname(p) == source)

    # Some re-shared copies hold the same data several times: per label granularity
    # (..._34classes / _8classes / _2classes) and per sample size (0.001percent_..., 0.01percent_...).
    names = [os.path.basename(p) for p in chosen]
    fine = [p for p, n in zip(chosen, names) if re.search(r"34[ _-]?class", n, re.I)]
    if fine and len(fine) < len(chosen):
        print("This input has copies at several label granularities: using the 34-class files only")
        chosen = fine
    if len(chosen) > 1 and all(re.search(r"percent|frac|sample", os.path.basename(p), re.I) for p in chosen):
        chosen = [max(chosen, key=os.path.getsize)]
        print(f"These are samples of different sizes of the same data: using the largest, {os.path.basename(chosen[0])}")
    if len(chosen) < 20:
        print("NOTE: this does not look like the full CICIoT2023 CSV release (169 files, ~13 GB); "
              "results will rest on less data, especially for the rarest classes.")

    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir)
    for p in chosen:
        os.symlink(p, os.path.join(raw_dir, os.path.basename(p)))
    size_gb = sum(os.path.getsize(p) for p in chosen) / 1e9
    print(f"Using {len(chosen)} files ({size_gb:.1f} GB) from {source}")
    return chosen


def count_rows(paths):
    total = 0
    for p in paths:
        if p.endswith(".parquet"):
            total += pq.ParquetFile(p).metadata.num_rows
        else:
            with (gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")) as f:
                total += sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b"")) - 1
    return total


def pack(work, note):
    os.makedirs("results/data_info", exist_ok=True)
    for f in glob.glob("data/processed*/*.json"):
        name = os.path.basename(f) if "withdups" not in f else "withdups_" + os.path.basename(f)
        shutil.copy(f, os.path.join("results/data_info", name))
    shutil.make_archive(f"{work}/results", "zip", "results")
    print(f"\n>>> results.zip updated after: {note}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_400_000, help="rows to sample (about 3%% of the full dataset)")
    parser.add_argument("--frac", type=float, help="sample this share of rows instead of --rows")
    parser.add_argument("--seeds", default="0 1 2")
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--phases", nargs="+", default=["baseline", "improvement", "extension", "duplicates"])
    parser.add_argument("--hours", type=float, default=11.0, help="stop starting new runs after this many hours")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    start = time.time()
    sh(f"{sys.executable} -m pip install -q 'imbalanced-learn>=0.12' 'pyarrow>=14' 'lightgbm>=4' 'scipy>=1.9'")
    # Links to the input files live outside the output folder, so Kaggle never copies the dataset into the output.
    raw = os.path.join(tempfile.gettempdir(), "ciciot2023_raw")
    chosen = link_raw_files(args.input, raw)
    frac = args.frac
    if frac is None:
        total = count_rows(chosen)
        frac = min(1.0, args.rows / total)
        print(f"{total:,} rows in the input: sampling {frac:.4f} of them (about {min(total, args.rows):,} rows)")

    py = sys.executable
    sh(f"{py} -m src.load --raw-dir {raw} --frac {frac:.6f}")
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
    # The sampled data is not needed any more (results.zip has everything) and would bloat Kaggle's output.
    for folder in glob.glob("data/processed*"):
        shutil.rmtree(folder, ignore_errors=True)
    print(f"\nDone in {(time.time() - start) / 3600:.1f} h. Download results.zip from the Output panel "
          "and unzip it into the repository's results/ folder.")


if __name__ == "__main__":
    main()
