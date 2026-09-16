from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "pl_rade",
    "pl_orbper",
    "pl_orbsmax",
    "pl_eqt",
    "pl_orbeccen",
    "st_teff",
    "st_rad",
    "st_mass",
    "sy_snum",
    "sy_pnum",
    "discoverymethod",
]


def test_processed_datasets_are_clean_and_disjoint() -> None:
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    assert len(train) > 100
    assert len(test) > 20
    assert not train[FEATURES + ["pl_bmasse"]].isna().any().any()
    assert set(train["pl_name"]).isdisjoint(set(test["pl_name"]))
    assert train["pl_bmasse"].between(0, 13 * 317.82838).all()
    assert test["pl_bmasse"].between(0, 13 * 317.82838).all()


def test_preparation_summary_records_real_cleaning() -> None:
    summary = json.loads(
        (PROJECT_ROOT / "data" / "processed" / "preparation_summary.json").read_text()
    )
    assert summary["initial_rows"] > summary["rows_after_validation"]
    assert summary["missing_values_after_imputation"] == 0
    assert sum(summary["missing_values_before_imputation"].values()) > 0


def test_packaged_model_predicts_a_positive_mass() -> None:
    package = joblib.load(PROJECT_ROOT / "models" / "model.joblib")
    sample = pd.DataFrame(
        [
            {
                "pl_rade": 1.0,
                "pl_orbper": 365.25,
                "pl_orbsmax": 1.0,
                "pl_eqt": 255.0,
                "pl_orbeccen": 0.0167,
                "st_teff": 5772.0,
                "st_rad": 1.0,
                "st_mass": 1.0,
                "sy_snum": 1,
                "sy_pnum": 1,
                "discoverymethod": "Transit",
            }
        ]
    )
    prediction = package["model"].predict(sample)

    assert package["metadata"]["target"] == "pl_bmasse"
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0


def test_saved_metrics_meet_smoke_thresholds() -> None:
    metrics = json.loads((PROJECT_ROOT / "metrics" / "metrics.json").read_text())
    assert metrics["r2"] > 0.5
    assert metrics["rmsle"] < 1.5
    assert metrics["within_factor_2_fraction"] > 0.5

