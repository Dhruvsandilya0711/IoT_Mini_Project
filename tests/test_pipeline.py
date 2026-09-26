"""End-to-end smoke test on fake data: load -> preprocess -> EDA -> train -> rank -> sweep -> analyze."""

import pandas as pd
import pytest

from src import analyze, eda, feature_selection, load, preprocess, train
from src.labels import CATEGORY_OF, CLASSES_34, canonical, to_level
from src.resample import resample
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


def test_resample_strategies():
    y = pd.Series(["big"] * 300 + ["mid"] * 60 + ["small"] * 12)
    X = pd.DataFrame({"a": range(len(y)), "b": range(len(y))}, dtype="float32")
    _, y_smote = resample(X, y, "smote", smote_target=100)
    assert y_smote.value_counts().to_dict() == {"big": 300, "mid": 100, "small": 100}
    _, y_hybrid = resample(X, y, "hybrid", smote_target=100, undersample_cap=200)
    assert y_hybrid.value_counts().to_dict() == {"big": 200, "mid": 100, "small": 100}
    _, y_none = resample(X, y, "classweight")
    assert y_none.equals(y)


def test_pipeline_end_to_end(tmp_path):
    raw, data, results = tmp_path / "raw", tmp_path / "processed", tmp_path / "results"
    write_fake_raw(raw)
    sample = data / "sample.parquet"
    load.main(["--raw-dir", str(raw), "--out", str(sample), "--frac", "1.0"])
    preprocess.main(["--sample", str(sample), "--out-dir", str(data)])
    eda.main(["--sample", str(sample), "--out-dir", str(results / "eda")])
    assert (results / "eda" / "class_distribution_8class.png").exists()

    test_before = pd.read_parquet(data / "test.parquet")
    common = ["--data-dir", str(data), "--results-dir", str(results), "--n-estimators", "10"]
    for strategy in ["none", "smote", "hybrid", "classweight"]:
        train.main(common + ["--model", "rf", "--strategy", strategy, "--smote-target", "300",
                             "--undersample-cap", "1000"])
    train.main(common + ["--model", "dt", "--level", "34"])
    assert pd.read_parquet(data / "test.parquet").equals(test_before)

    for method in ["mi", "rf", "rfe"]:
        feature_selection.main(["--data-dir", str(data), "--results-dir", str(results), "--method", method,
                                "--n-rows", "2000", "--rfe-step", "5"])
    ranking = results / "features" / "ranking_mi_8class.csv"
    for k in [5, 10]:
        train.main(common + ["--model", "rf", "--strategy", "smote", "--smote-target", "300",
                             "--k", str(k), "--ranking", str(ranking)])

    runs = pd.read_csv(results / "runs.csv")
    assert len(runs) == 7
    assert runs["test_accuracy"].between(0, 1).all()
    assert set(runs["n_features"]) == {46, 5, 10}

    # A second identical call is skipped rather than duplicated.
    train.main(common + ["--model", "rf", "--strategy", "none"])
    assert len(pd.read_csv(results / "runs.csv")) == 7

    analyze.main(["compare", "--results-dir", str(results), "--smote-target", "300",
                  "--undersample-cap", "1000", "--n-estimators", "10"])
    analyze.main(["sweep", "--results-dir", str(results), "--smote-target", "300", "--n-estimators", "10",
                  "--ranking", "ranking_mi_8class"])
    summary = pd.read_csv(results / "tables" / "compare_8c_rf_test_summary.csv")
    assert summary["strategy"].tolist() == ["none", "smote", "hybrid", "classweight"]
    assert (results / "figures" / "compare_8c_rf_test_recall_gain.png").exists()
    sweep = pd.read_csv(results / "tables" / "sweep_8c_rf_smote_ranking_mi_8class.csv")
    assert sweep["n_features"].tolist() == [5, 10, 46]
