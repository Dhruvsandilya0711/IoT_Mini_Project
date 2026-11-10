"""Run a trained detector as an edge IDS, e.g. on a Raspberry Pi acting as the IoT gateway.

Replays recorded flows through a model saved with `python -m src.train ... --save-model` and reports
detection quality, prediction time per flow, throughput, memory and CPU temperature.

    # whole test split at once: quality + throughput
    python scripts/pi_detect.py --model pi_results/models/8c-dt-classweight-kall-s0.joblib

    # one flow at a time, like a gateway receiving traffic; print an alert for every attack
    python scripts/pi_detect.py --model pi_results/models/8c-dt-classweight-kall-s0.joblib \\
        --stream 300 --classes Benign Web BruteForce Recon

--flows takes a processed split (data/processed/test.parquet, the default) or a raw CICIoT2023 CSV part.
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import LABEL_COL, canonical, to_level  # noqa: E402
from src.load import read_part  # noqa: E402

BENIGN = {"Benign", "BenignTraffic"}


def cpu_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB


def level_of(classes):
    names = set(map(str, classes))
    if "BruteForce" in names:
        return 8
    if "Attack" in names:
        return 2
    return 34


def load_flows(path, level):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else read_part(path)
    y = to_level(canonical(df.pop(LABEL_COL).astype(str)), level) if LABEL_COL in df else None
    return df, y


def single_threaded(model):
    """Predicting one flow with joblib threads costs more than the prediction itself."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "n_jobs"):
        inner.n_jobs = 1


def report_quality(y, pred):
    table = pd.DataFrame({"true": y.to_numpy(), "pred": pred})
    rows = []
    for cls, grp in table.groupby("true"):
        flagged = table["pred"] == cls
        rows.append({
            "class": cls,
            "flows": len(grp),
            "recall": (grp["pred"] == cls).mean(),
            "precision": (table.loc[flagged, "true"] == cls).mean() if flagged.any() else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values("flows", ascending=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"accuracy {np.mean(table['true'] == table['pred']):.4f}   macro recall {out['recall'].mean():.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=".joblib file written by src.train --save-model")
    parser.add_argument("--flows", default="data/processed/test.parquet")
    parser.add_argument("--classes", nargs="+", help="only replay flows of these classes (names at the model's level)")
    parser.add_argument("--stream", type=int, metavar="N", help="replay N random flows one at a time instead of one batch")
    parser.add_argument("--rate", type=float, default=0, help="with --stream: flows per second (0 = as fast as possible)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    bundle = joblib.load(args.model)
    scaler, features, model = bundle["scaler"], bundle["features"], bundle["model"]
    level = level_of(model.classes_)
    print(f"model {Path(args.model).name}: {Path(args.model).stat().st_size / 2**20:.1f} MB on disk, "
          f"{len(features)} features, {level} classes, loaded in {time.perf_counter() - t0:.2f} s")

    X, y = load_flows(args.flows, level)
    if args.classes:
        if y is None:
            raise SystemExit("--classes needs flows with a label column")
        keep = y.isin(args.classes).to_numpy()
        X, y = X[keep], y[keep]
    if len(X) == 0:
        raise SystemExit("no flows to replay")
    print(f"{len(X):,} flows from {args.flows}")

    def scale(frame):
        return pd.DataFrame(scaler.transform(frame[features]).astype("float32"), columns=features)

    if not args.stream:
        t0 = time.perf_counter()
        pred = model.predict(scale(X))
        elapsed = time.perf_counter() - t0
        print(f"batch: {elapsed:.2f} s, {elapsed / len(X) * 1e6:.2f} µs per flow, {len(X) / elapsed:,.0f} flows/s")
        if y is not None:
            report_quality(y, pred)
    else:
        single_threaded(model)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X), size=min(args.stream, len(X)), replace=False)
        times, alerts, hits = [], 0, 0
        for n, i in enumerate(idx, 1):
            flow = X.iloc[[i]]
            t0 = time.perf_counter()
            label = model.predict(scale(flow))[0]
            times.append(time.perf_counter() - t0)
            truth = None if y is None else y.iloc[i]
            hits += truth == label
            if label not in BENIGN:
                alerts += 1
                mark = "" if truth is None else ("  ok" if truth == label else f"  (actually {truth})")
                print(f"{time.strftime('%H:%M:%S')}  ALERT  flow {n:>5}: {label}{mark}")
            elif truth is not None and truth not in BENIGN:
                print(f"{time.strftime('%H:%M:%S')}  missed flow {n:>5}: {truth} passed as {label}")
            if args.rate:
                time.sleep(max(0.0, 1 / args.rate - times[-1]))
        ms = np.array(times) * 1000
        print(f"\n{len(idx)} flows, {alerts} alerts" + (f", {hits / len(idx):.1%} labelled correctly" if y is not None else ""))
        print(f"time per flow, scaling + prediction: median {np.median(ms):.2f} ms, 99th percentile {np.percentile(ms, 99):.2f} ms")

    temp = cpu_temp()
    print(f"peak memory {peak_memory_mb():.0f} MB" + (f", CPU {temp:.1f} °C" if temp is not None else ""))


if __name__ == "__main__":
    main()
