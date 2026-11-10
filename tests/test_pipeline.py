"""End-to-end smoke test on fake data: load -> preprocess -> EDA -> train -> rank -> sweep -> analyze."""

import numpy as np
import pandas as pd
import pytest

from src import analyze, eda, feature_selection, load, preprocess, train
from src.labels import CATEGORY_OF, CLASSES_34, canonical, to_level
from src.resample import resample
from src.threshold import macro_f1_codes, tune_weights
from tests.fake_data import write_fake_raw


def test_label_mapping_covers_all_classes():
    assert len(CATEGORY_OF) == 34
    assert sorted(CLASSES_34) == sorted(CATEGORY_OF)
    raw = pd.Series([" ddos-syn_flood ", "XSS", "BenignTraffic"])
    assert canonical(raw).tolist() == ["DDoS-SYN_Flood", "XSS", "BenignTraffic"]
    assert to_level(canonical(raw), 8).tolist() == ["DDoS", "Web", "Benign"]
    assert to_level(canonical(raw), 2).tolist() == ["Attack", "Attack", "Benign"]
    with pytest.raises(ValueError):
        canonical(pd.Series(["NotAnAttack"]))


def test_loader_cleans_messy_copies(tmp_path):
    raw = write_fake_raw(tmp_path / "raw", n_files=1, rows_per_file=300)
    path = raw / "part-00000.csv"
    df = pd.read_csv(path)
    df.insert(0, "Unnamed: 0", range(len(df)))  # index column written by to_csv
    df["note"] = "x"  # an extra text column
    df["Rate"] = df["Rate"].astype(object)
    df.loc[5, "Rate"] = "oops"  # a stray text value in a numeric column
    header = pd.DataFrame([df.columns], columns=df.columns)  # header line repeated mid-file
    pd.concat([df.iloc[:100], header, df.iloc[100:]]).to_csv(path, index=False)
    part = load.read_part(path)
    assert list(part.columns) == load.EXPECTED_FEATURES + ["label"]
    assert len(part) == 300 and part["Rate"].isna().sum() == 1
    assert all(str(t) == "float32" for t in part.dtypes.iloc[:-1])


def test_resample_strategies():
    y = pd.Series(["big"] * 300 + ["mid"] * 60 + ["small"] * 12)
    X = pd.DataFrame({"a": range(len(y)), "b": range(len(y))}, dtype="float32")
    _, y_smote = resample(X, y, "smote", smote_target=100)
    assert y_smote.value_counts().to_dict() == {"big": 300, "mid": 100, "small": 100}
    _, y_hybrid = resample(X, y, "hybrid", smote_target=100, undersample_cap=200)
    assert y_hybrid.value_counts().to_dict() == {"big": 200, "mid": 100, "small": 100}
    _, y_none = resample(X, y, "classweight")
    assert y_none.equals(y)
    _, y_border = resample(X, y, "borderline", smote_target=100)
    assert set(y_border.value_counts().to_dict()) <= {"big", "mid", "small"}


def test_threshold_tuning_never_hurts_validation_f1():
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=2000, p=[0.9, 0.08, 0.02])
    proba = rng.dirichlet([1, 1, 1], size=2000)
    proba[np.arange(2000), y] += 0.4  # informative, but biased towards nothing in particular
    proba[:, 0] += 0.5  # a model that over-predicts the majority class
    before = macro_f1_codes(y, proba.argmax(1), 3)
    weights, after = tune_weights(proba, y)
    assert after >= before and weights[0] <= 1


def test_pipeline_end_to_end(tmp_path):
    raw, data, results = tmp_path / "raw", tmp_path / "processed", tmp_path / "results"
    write_fake_raw(raw)
    sample = data / "sample.parquet"
    load.main(["--raw-dir", str(raw), "--out", str(sample), "--frac", "1.0"])
    preprocess.main(["--sample", str(sample), "--out-dir", str(data), "--drop-duplicates"])
    preprocess.main(["--sample", str(sample), "--out-dir", str(tmp_path / "withdups")])
    eda.main(["--sample", str(sample), "--out-dir", str(results / "eda")])
    assert (results / "eda" / "class_distribution_8class.png").exists()

    test_before = pd.read_parquet(data / "test.parquet")
    common = ["--data-dir", str(data), "--results-dir", str(results), "--n-estimators", "10"]
    for strategy in ["none", "smote", "hybrid", "classweight", "borderline", "threshold"]:
        train.main(common + ["--model", "rf", "--strategy", strategy, "--smote-target", "300",
                             "--undersample-cap", "1000"])
    train.main(common + ["--model", "rf", "--strategy", "smote", "--smote-target", "300", "--seed", "1"])
    train.main(common + ["--model", "rf", "--strategy", "none", "--seed", "1"])
    for model in ["lgbm", "brf"]:
        train.main(common + ["--model", model, "--strategy", "threshold"])
    train.main(common + ["--model", "lgbm", "--strategy", "classweight"])
    with pytest.raises(SystemExit):
        train.main(common + ["--model", "brf", "--strategy", "smote"])
    train.main(common + ["--model", "dt", "--level", "34"])
    train.main(["--data-dir", str(tmp_path / "withdups"), "--results-dir", str(results), "--n-estimators", "10",
                "--model", "rf", "--tag", "withdups"])
    assert pd.read_parquet(data / "test.parquet").equals(test_before)

    for method in ["mi", "rf", "rfe"]:
        feature_selection.main(["--data-dir", str(data), "--results-dir", str(results), "--method", method,
                                "--n-rows", "2000", "--rfe-step", "5"])
    ranking = results / "features" / "ranking_mi_8class.csv"
    for k in [5, 10]:
        train.main(common + ["--model", "rf", "--strategy", "smote", "--smote-target", "300",
                             "--k", str(k), "--ranking", str(ranking)])

    runs = pd.read_csv(results / "runs.csv")
    assert len(runs) == 15
    assert runs["test_accuracy"].between(0, 1).all()
    assert set(runs["n_features"]) == {46, 5, 10}
    assert runs.loc[runs["strategy"] == "threshold", "tuned_weights"].str.startswith("{").all()

    # A second identical call is skipped rather than duplicated.
    train.main(common + ["--model", "rf", "--strategy", "none"])
    assert len(pd.read_csv(results / "runs.csv")) == 15

    analyze.main(["compare", "--results-dir", str(results), "--smote-target", "300",
                  "--undersample-cap", "1000", "--n-estimators", "10"])
    sig = pd.read_csv(results / "tables" / "compare_8c_rf_test_significance.csv")
    assert set(sig["strategy"]) == {"smote", "hybrid", "classweight", "borderline", "threshold"}
    assert sig["p_mcnemar"].between(0, 1).all()
    analyze.main(["leaderboard", "--results-dir", str(results), "--smote-target", "300",
                  "--undersample-cap", "1000", "--n-estimators", "10"])
    board = pd.read_csv(results / "tables" / "leaderboard_8c.csv")
    assert {"rf", "lgbm", "brf"} <= set(board["model"])
    analyze.main(["duplicates", "--results-dir", str(results), "--n-estimators", "10"])
    assert (results / "tables" / "duplicates_8c.csv").exists()
    analyze.main(["sweep", "--results-dir", str(results), "--smote-target", "300", "--n-estimators", "10",
                  "--ranking", "ranking_mi_8class"])
    summary = pd.read_csv(results / "tables" / "compare_8c_rf_test_summary.csv")
    assert summary["strategy"].tolist() == ["none", "smote", "hybrid", "classweight", "borderline", "threshold"]
    assert (results / "figures" / "compare_8c_rf_test_recall_gain.png").exists()
    sweep = pd.read_csv(results / "tables" / "sweep_8c_rf_smote_ranking_mi_8class.csv")
    assert sweep["n_features"].tolist() == [5, 10, 46]
