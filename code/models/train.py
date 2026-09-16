"""Train, evaluate, and package the exoplanet mass regression model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_log_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "test.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "metrics"

TARGET = "pl_bmasse"
LOG_FEATURES = ["pl_rade", "pl_orbper", "pl_orbsmax", "pl_eqt"]
LINEAR_FEATURES = [
    "pl_orbeccen",
    "st_teff",
    "st_rad",
    "st_mass",
    "sy_snum",
    "sy_pnum",
]
CATEGORICAL_FEATURES = ["discoverymethod"]
FEATURES = LOG_FEATURES + LINEAR_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_frame(frame: pd.DataFrame, name: str) -> None:
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {missing_columns}")
    if frame.empty:
        raise ValueError(f"{name} contains no rows")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError(f"{name} contains an invalid target value")


def build_estimator() -> TransformedTargetRegressor:
    """Create one serializable preprocessing-and-regression estimator."""
    log_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "log_transform",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    linear_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("log", log_pipeline, LOG_FEATURES),
            ("linear", linear_pipeline, LINEAR_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
    predictions = np.clip(predictions, a_min=0, a_max=None)
    ratio = np.maximum(y_true.to_numpy(), predictions) / np.maximum(
        np.minimum(y_true.to_numpy(), predictions), 1e-9
    )
    return {
        "mae_earth_masses": float(mean_absolute_error(y_true, predictions)),
        "median_ae_earth_masses": float(median_absolute_error(y_true, predictions)),
        "rmse_earth_masses": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "rmsle": float(root_mean_squared_log_error(y_true, predictions)),
        "within_factor_2_fraction": float(np.mean(ratio <= 2.0)),
    }


def extract_feature_importance(estimator: TransformedTargetRegressor) -> pd.DataFrame:
    fitted_pipeline = estimator.regressor_
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    regressor = fitted_pipeline.named_steps["regressor"]
    names: List[str] = list(preprocessor.get_feature_names_out())
    return (
        pd.DataFrame(
            {"feature": names, "importance": regressor.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_package(
    train_path: Path,
    test_path: Path,
    model_path: Path,
    metrics_dir: Path,
) -> Dict[str, Any]:
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets are missing. Run `make prepare` first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_training_frame(train, "training dataset")
    validate_training_frame(test, "testing dataset")

    estimator = build_estimator()
    estimator.fit(train[FEATURES], train[TARGET])
    predictions = np.clip(estimator.predict(test[FEATURES]), a_min=0, a_max=None)
    metrics = calculate_metrics(test[TARGET], predictions)

    discovery_methods = sorted(train["discoverymethod"].dropna().unique().tolist())
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        "model_name": "exoplanet-mass-random-forest",
        "model_version": "1.0.0",
        "trained_at_utc": trained_at,
        "target": TARGET,
        "target_unit": "Earth masses",
        "features": FEATURES,
        "log_transformed_features": LOG_FEATURES,
        "train_rows": len(train),
        "test_rows": len(test),
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "training_data_sha256": sha256(train_path),
        "testing_data_sha256": sha256(test_path),
        "discovery_methods": discovery_methods,
        "metrics": metrics,
    }
    package = {"model": estimator, "metadata": metadata}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_path, compress=3)
    (model_path.parent / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    extract_feature_importance(estimator).to_csv(
        metrics_dir / "feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "pl_name": test.get("pl_name", pd.Series(index=test.index, dtype=str)),
            "actual_mass_earth": test[TARGET],
            "predicted_mass_earth": predictions,
        }
    ).to_csv(metrics_dir / "test_predictions.csv", index=False)

    print(f"Saved trained model to {model_path}")
    print(json.dumps(metrics, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_and_package(
        train_path=arguments.train,
        test_path=arguments.test,
        model_path=arguments.model,
        metrics_dir=arguments.metrics_dir,
    )
