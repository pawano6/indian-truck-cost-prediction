"""
data_utils.py — Data loading, validation, and preprocessing utilities.

This module provides reusable functions for the Indian Truck Trip Cost
Prediction project. It handles CSV loading, column validation,
feature/target separation, and scikit-learn preprocessing pipeline
construction.

IMPORTANT: The dataset is synthetic (educational) and does not represent
actual company or government logistics data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "record_type", "trip_id", "origin_city", "origin_state",
    "destination_city", "destination_state", "distance_km",
    "truck_type", "truck_capacity_tons", "load_weight_tons",
    "fuel_efficiency_km_per_litre", "diesel_price_inr_per_litre",
    "toll_cost_inr", "driver_allowance_inr", "maintenance_cost_inr",
    "loading_unloading_cost_inr", "other_cost_inr", "route_factor",
    "traffic_factor", "month", "actual_trip_cost_inr",
    "freight_revenue_inr", "trip_profit_inr",
]

TARGET_COLUMN = "actual_trip_cost_inr"

# Columns excluded from model inputs (leakage / identifier / label)
DROP_COLUMNS = [
    "actual_trip_cost_inr",   # target
    "trip_id",                # identifier
    "record_type",            # provenance label
    "freight_revenue_inr",    # downstream outcome
    "trip_profit_inr",        # downstream outcome
]

# Feature groupings (derived after dropping excluded columns)
CATEGORICAL_FEATURES = [
    "origin_city", "origin_state",
    "destination_city", "destination_state",
    "truck_type",
]

NUMERICAL_FEATURES = [
    "distance_km", "truck_capacity_tons", "load_weight_tons",
    "fuel_efficiency_km_per_litre", "diesel_price_inr_per_litre",
    "toll_cost_inr", "driver_allowance_inr", "maintenance_cost_inr",
    "loading_unloading_cost_inr", "other_cost_inr",
    "route_factor", "traffic_factor", "month",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> bool:
    """Check that all required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
    required : list[str], optional
        Defaults to REQUIRED_COLUMNS.

    Returns
    -------
    bool
        True if all required columns are present.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    required = required or REQUIRED_COLUMNS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return a dictionary summarising data quality metrics.

    Returns
    -------
    dict
        Keys: shape, dtypes, missing_counts, duplicate_rows, target_stats.
    """
    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_stats": df[TARGET_COLUMN].describe().to_dict()
        if TARGET_COLUMN in df.columns else {},
    }
    return report


# ---------------------------------------------------------------------------
# Feature / Target separation
# ---------------------------------------------------------------------------

def separate_features_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    drop_cols : list[str], optional
        Columns to drop. Defaults to DROP_COLUMNS.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """
    drop_cols = drop_cols or DROP_COLUMNS
    y = df[target].copy()
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Construct a ColumnTransformer for scaling + encoding.

    Numerical features → StandardScaler
    Categorical features → OneHotEncoder (handle_unknown='ignore')

    Parameters
    ----------
    numerical_features : list[str], optional
    categorical_features : list[str], optional

    Returns
    -------
    ColumnTransformer
    """
    numerical_features = numerical_features or NUMERICAL_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Must be fitted.

    Returns
    -------
    list[str]
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def build_baseline(strategy: str = "mean") -> DummyRegressor:
    """Return a DummyRegressor baseline.

    Parameters
    ----------
    strategy : str
        One of 'mean', 'median', 'constant', 'quantile'.

    Returns
    -------
    DummyRegressor
    """
    return DummyRegressor(strategy=strategy)
