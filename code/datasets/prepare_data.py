"""Validate, clean, impute, and split the NASA exoplanet snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "exoplanets.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "pl_bmasse"
ID_COLUMNS = ["pl_name", "hostname", "disc_year"]
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
REQUIRED_COLUMNS = ID_COLUMNS + FEATURES + [TARGET]

EARTH_MASSES_PER_JUPITER = 317.82838
MAX_PLANET_MASS_EARTH = 13 * EARTH_MASSES_PER_JUPITER


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar values to regular JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `make download` first."
        )
    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Raw dataset is missing columns: {missing_columns}")
    return frame[REQUIRED_COLUMNS].copy()


def _drop_by_rule(
    frame: pd.DataFrame,
    mask: pd.Series,
    name: str,
    removal_counts: Dict[str, int],
) -> pd.DataFrame:
    count = int(mask.fillna(False).sum())
    removal_counts[name] = count
    return frame.loc[~mask.fillna(False)].copy()


def clean_data(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove invalid rows and physically implausible extreme observations."""
    initial_rows = len(frame)
    numeric_columns: Iterable[str] = ID_COLUMNS[2:] + LOG_FEATURES + LINEAR_FEATURES + [TARGET]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_count = int(frame.duplicated(subset=["pl_name"], keep="last").sum())
    frame = frame.drop_duplicates(subset=["pl_name"], keep="last").copy()

    removal_counts: Dict[str, int] = {}
    frame = _drop_by_rule(
        frame,
        frame[TARGET].isna() | frame["pl_rade"].isna(),
        "missing_target_or_radius",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        (frame[TARGET] <= 0) | (frame[TARGET] > MAX_PLANET_MASS_EARTH),
        "mass_outside_planetary_range",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        (frame["pl_rade"] <= 0) | (frame["pl_rade"] > 30),
        "radius_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["pl_orbper"].notna()
        & ((frame["pl_orbper"] <= 0) | (frame["pl_orbper"] > 1_000_000)),
        "orbital_period_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["pl_orbsmax"].notna()
        & ((frame["pl_orbsmax"] <= 0) | (frame["pl_orbsmax"] > 1_000)),
        "semi_major_axis_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["pl_orbeccen"].notna()
        & ~frame["pl_orbeccen"].between(0, 1, inclusive="both"),
        "invalid_eccentricity",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["pl_eqt"].notna()
        & ((frame["pl_eqt"] <= 0) | (frame["pl_eqt"] > 10_000)),
        "equilibrium_temperature_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["st_teff"].notna()
        & ((frame["st_teff"] < 1_000) | (frame["st_teff"] > 50_000)),
        "stellar_temperature_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["st_rad"].notna()
        & ((frame["st_rad"] <= 0) | (frame["st_rad"] > 100)),
        "stellar_radius_outlier",
        removal_counts,
    )
    frame = _drop_by_rule(
        frame,
        frame["st_mass"].notna()
        & ((frame["st_mass"] <= 0) | (frame["st_mass"] > 100)),
        "stellar_mass_outlier",
        removal_counts,
    )

    frame["discoverymethod"] = frame["discoverymethod"].replace("", np.nan)
    frame = frame.reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError(f"Only {len(frame)} valid rows remain; refusing to train")

    summary = {
        "initial_rows": initial_rows,
        "duplicate_planet_names_removed": duplicate_count,
        "rule_removals": removal_counts,
        "rows_after_validation": len(frame),
        "missing_values_before_imputation": {
            column: int(frame[column].isna().sum()) for column in FEATURES
        },
        "planetary_mass_upper_limit_earth_masses": MAX_PLANET_MASS_EARTH,
    }
    return frame, summary


def split_and_impute(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split first, then learn imputation values only from the training fold."""
    mass_bins = pd.qcut(np.log1p(frame[TARGET]), q=10, duplicates="drop")
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=mass_bins,
    )
    train = train.copy()
    test = test.copy()

    imputation: Dict[str, Any] = {"numeric": {}, "categorical": {}}
    for column in LOG_FEATURES + LINEAR_FEATURES:
        value = _json_value(train[column].median())
        if pd.isna(value):
            raise ValueError(f"Cannot impute {column}: the training column is empty")
        train[column] = train[column].fillna(value)
        test[column] = test[column].fillna(value)
        imputation["numeric"][column] = value

    for column in CATEGORICAL_FEATURES:
        modes = train[column].mode(dropna=True)
        value = str(modes.iloc[0]) if not modes.empty else "Unknown"
        train[column] = train[column].fillna(value)
        test[column] = test[column].fillna(value)
        imputation["categorical"][column] = value

    train = train.sort_values("pl_name").reset_index(drop=True)
    test = test.sort_values("pl_name").reset_index(drop=True)
    return train, test, imputation


def prepare(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    raw = load_raw_data(input_path)
    clean, summary = clean_data(raw)
    train, test, imputation = split_and_impute(clean, test_size, random_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "preparation_summary.json"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    summary.update(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_size": test_size,
            "random_state": random_state,
            "features": FEATURES,
            "target": TARGET,
            "imputation_values_learned_from_train": imputation,
            "missing_values_after_imputation": int(
                train[FEATURES].isna().sum().sum() + test[FEATURES].isna().sum().sum()
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Prepared {len(train):,} training rows and {len(test):,} testing rows "
        f"in {output_dir}"
    )
    print(f"Removed {summary['initial_rows'] - summary['rows_after_validation']:,} rows")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not 0 < arguments.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1")
    prepare(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        test_size=arguments.test_size,
        random_state=arguments.random_state,
    )
