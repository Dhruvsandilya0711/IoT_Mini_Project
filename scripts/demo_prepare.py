"""Prepare everything the live demo needs from a few CICIoT2023 CSV files, in one command.

On Kaggle (CICIoT2023 dataset attached), from a clone of the repository:

    !python scripts/demo_prepare.py

On a laptop, with a few part-*.csv files in data/raw:

    python scripts/demo_prepare.py --input data/raw

It samples --files CSV parts, cleans and splits them like the main project, trains the class-weighted
Decision Tree (the gateway model) and writes demo/:

    demo/gateway_dt.joblib    the trained model
    demo/demo_flows.csv       test flows to replay: every rare-class flow plus --demo-per-class of each other class

Then replay them with scripts/detect.py.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.kaggle_run import link_raw_files  # noqa: E402
from src.labels import LABEL_COL, to_level  # noqa: E402

RARE = ["Recon", "Web", "BruteForce"]


def sh(*args):
    print("\n$ " + " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def subset(df, per_class, seed=0):
    """Every rare-class row plus up to per_class rows of each other 8-class category."""
    cat = to_level(df[LABEL_COL].astype(str), 8)
    parts = [df[cat == c] if c in RARE else df[cat == c].sample(min(per_class, int((cat == c).sum())), random_state=seed)
             for c in sorted(cat.unique())]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input", help="folder holding the CICIoT2023 CSV parts")
    parser.add_argument("--files", type=int, default=5, help="how many CSV parts to use (about 230,000 flows each)")
    parser.add_argument("--demo-per-class", type=int, default=1000)
    args = parser.parse_args(argv)

    raw = Path(tempfile.gettempdir()) / "ciciot2023_demo_raw"
    chosen = link_raw_files(args.input, str(raw))
    for link in sorted(raw.iterdir())[args.files:]:
        link.unlink()
    print(f"Using {min(args.files, len(chosen))} of {len(chosen)} files")

    processed = ROOT / "data" / "processed_demo"
    out = ROOT / "demo"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    sh("-m", "src.load", "--raw-dir", raw, "--frac", "1.0", "--out", processed / "sample.parquet")
    sh("-m", "src.preprocess", "--sample", processed / "sample.parquet", "--out-dir", processed, "--drop-duplicates")
    sh("-m", "src.train", "--data-dir", processed, "--results-dir", out / "results", "--level", "8",
       "--model", "dt", "--strategy", "classweight", "--save-model", "--force")
    shutil.copy(out / "results" / "models" / "8c-dt-classweight-kall-s0.joblib", out / "gateway_dt.joblib")

    subset(pd.read_parquet(processed / "test.parquet"), args.demo_per_class).to_csv(out / "demo_flows.csv", index=False)
    print(f"\nWrote {out}/gateway_dt.joblib and {out}/demo_flows.csv")
    print("Run the demo:  python scripts/detect.py --model demo/gateway_dt.joblib --flows demo/demo_flows.csv")


if __name__ == "__main__":
    main()
