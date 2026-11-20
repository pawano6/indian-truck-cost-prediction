"""
feature_engineering.py — Feature engineering for Indian Truck Logistics Cost Prediction.

Separates PRE-TRIP features (available before dispatch) from POST-TRIP
metrics (available only after trip completion).

IMPORTANT: This project uses synthetic educational data.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Feature classification documentation
# ---------------------------------------------------------------------------
#
# PRE-TRIP FEATURES (known before dispatch — safe for prediction inputs):
#   - origin_city, origin_state, destination_city, destination_state
#   - distance_km (estimated route distance)
#   - truck_type, truck_capacity_tons
#   - load_weight_tons (known at loading)
#   - fuel_efficiency_km_per_litre (truck specification)
#   - diesel_price_inr_per_litre (current market rate)
#   - toll_cost_inr (estimated tolls on planned route)
#   - driver_allowance_inr (contractual / estimated)
#   - maintenance_cost_inr (estimated/budgeted)
#   - loading_unloading_cost_inr (estimated)
#   - other_cost_inr (estimated miscellaneous)
#   - route_factor (route difficulty estimate)
#   - traffic_factor (traffic condition estimate)
#   - month (planned dispatch month)
#
# POST-TRIP METRICS (only available after trip completion — NEVER for prediction):
#   - actual_trip_cost_inr (TARGET — the value we predict)
#   - freight_revenue_inr (revenue realized after trip)
#   - trip_profit_inr (revenue minus cost — post-trip)
#   - Any derived metric using actual_trip_cost_inr
#
# ENGINEERED PRE-TRIP FEATURES:
#   - load_utilization_pct: load_weight_tons / truck_capacity_tons * 100
#   - estimated_fuel_litres: distance_km / fuel_efficiency_km_per_litre
#   - estimated_fuel_cost_inr: estimated_fuel_litres * diesel_price_inr_per_litre
#   - estimated_direct_expenses_inr: sum of all estimated expense components
#   - cost_per_ton_km_estimate: estimated_direct_expenses / (load_weight * distance)
#
# ENGINEERED POST-TRIP METRICS (for analysis only, not prediction):
#   - actual_cost_per_km: actual_trip_cost_inr / distance_km
#   - profit_margin_pct: trip_profit_inr / freight_revenue_inr * 100


def add_load_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Add load utilization percentage.

    load_utilization_pct = (load_weight_tons / truck_capacity_tons) * 100

    Only computed where both values are positive and load <= capacity.
    """
    df = df.copy()
    mask = (df["truck_capacity_tons"] > 0) & (df["load_weight_tons"] >= 0)
    df["load_utilization_pct"] = np.where(
        mask,
        (df["load_weight_tons"] / df["truck_capacity_tons"]) * 100,
        np.nan,
    )
    return df


def add_estimated_fuel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated fuel consumption and fuel cost (pre-trip estimates).

    Only computed when fuel_efficiency > 0 and diesel_price > 0.
    """
    df = df.copy()
    has_fuel_eff = (
        "fuel_efficiency_km_per_litre" in df.columns
        and "distance_km" in df.columns
    )
    has_diesel = "diesel_price_inr_per_litre" in df.columns

    if has_fuel_eff:
        valid = df["fuel_efficiency_km_per_litre"] > 0
        df["estimated_fuel_litres"] = np.where(
            valid,
            df["distance_km"] / df["fuel_efficiency_km_per_litre"],
            np.nan,
        )
        if has_diesel:
            df["estimated_fuel_cost_inr"] = np.where(
                valid & (df["diesel_price_inr_per_litre"] > 0),
                df["estimated_fuel_litres"] * df["diesel_price_inr_per_litre"],
                np.nan,
            )
    return df


def add_operational_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add operational efficiency features (pre-trip estimates).

    - estimated_direct_expenses_inr: sum of estimated cost components
    - cost_per_ton_km_estimate: expense estimate normalized by load and distance
    """
    df = df.copy()
    expense_cols = [
        "toll_cost_inr", "driver_allowance_inr", "maintenance_cost_inr",
        "loading_unloading_cost_inr", "other_cost_inr",
    ]
    present = [c for c in expense_cols if c in df.columns]

    # Add estimated fuel cost if present
    fuel_col = "estimated_fuel_cost_inr"
    if fuel_col in df.columns:
        df["estimated_direct_expenses_inr"] = df[present].sum(axis=1) + df[fuel_col].fillna(0)
    elif present:
        df["estimated_direct_expenses_inr"] = df[present].sum(axis=1)

    # Cost per ton-km estimate (pre-trip, using estimated expenses)
    if "estimated_direct_expenses_inr" in df.columns:
        denom = df["load_weight_tons"] * df["distance_km"]
        valid = denom > 0
        df["cost_per_ton_km_estimate"] = np.where(
            valid,
            df["estimated_direct_expenses_inr"] / denom,
            np.nan,
        )
    return df


def add_post_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add post-trip analysis metrics. NEVER use these as prediction inputs.

    - actual_cost_per_km
    - profit_margin_pct
    """
    df = df.copy()
    if "actual_trip_cost_inr" in df.columns and "distance_km" in df.columns:
        valid = df["distance_km"] > 0
        df["actual_cost_per_km"] = np.where(
            valid, df["actual_trip_cost_inr"] / df["distance_km"], np.nan
        )

    if "trip_profit_inr" in df.columns and "freight_revenue_inr" in df.columns:
        valid = df["freight_revenue_inr"] > 0
        df["profit_margin_pct"] = np.where(
            valid,
            (df["trip_profit_inr"] / df["freight_revenue_inr"]) * 100,
            np.nan,
        )
    return df


def engineer_features(df: pd.DataFrame, include_post_trip: bool = False) -> pd.DataFrame:
    """Apply all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    include_post_trip : bool
        If True, include post-trip metrics for analysis.
        MUST be False when preparing data for prediction.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """
    df = add_load_utilization(df)
    df = add_estimated_fuel_metrics(df)
    df = add_operational_efficiency_metrics(df)
    if include_post_trip:
        df = add_post_trip_metrics(df)
    return df


# ---------------------------------------------------------------------------
# Data validation functions
# ---------------------------------------------------------------------------

def validate_numerical_ranges(df: pd.DataFrame) -> dict:
    """Validate numerical columns for business-logic constraints.

    Returns a dict of {column: {issue: count}} for any violations found.
    """
    issues = {}

    checks = {
        "distance_km": ("positive", lambda s: s <= 0),
        "truck_capacity_tons": ("positive", lambda s: s <= 0),
        "load_weight_tons": ("non_negative", lambda s: s < 0),
        "fuel_efficiency_km_per_litre": ("positive", lambda s: s <= 0),
        "diesel_price_inr_per_litre": ("positive", lambda s: s <= 0),
        "toll_cost_inr": ("non_negative", lambda s: s < 0),
        "driver_allowance_inr": ("non_negative", lambda s: s < 0),
        "maintenance_cost_inr": ("non_negative", lambda s: s < 0),
        "loading_unloading_cost_inr": ("non_negative", lambda s: s < 0),
        "other_cost_inr": ("non_negative", lambda s: s < 0),
        "actual_trip_cost_inr": ("positive", lambda s: s <= 0),
    }

    for col, (label, check_fn) in checks.items():
        if col in df.columns:
            violations = check_fn(df[col]).sum()
            if violations > 0:
                issues[col] = {f"not_{label}": int(violations)}

    # Load should not exceed capacity
    if "load_weight_tons" in df.columns and "truck_capacity_tons" in df.columns:
        overloaded = (df["load_weight_tons"] > df["truck_capacity_tons"]).sum()
        if overloaded > 0:
            issues["load_exceeds_capacity"] = {"count": int(overloaded)}

    # Month validation
    if "month" in df.columns:
        invalid_month = (~df["month"].between(1, 12)).sum()
        if invalid_month > 0:
            issues["month"] = {"invalid_range": int(invalid_month)}

    return issues


def validate_categorical_values(df: pd.DataFrame) -> dict:
    """Validate categorical columns have expected values."""
    issues = {}
    expected_truck_types = {"Heavy", "Medium", "LCV", "Trailer"}

    if "truck_type" in df.columns:
        actual = set(df["truck_type"].unique())
        unexpected = actual - expected_truck_types
        if unexpected:
            issues["truck_type"] = {"unexpected_values": list(unexpected)}

    return issues


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None,
                        multiplier: float = 1.5) -> dict:
    """Detect outliers using the IQR method.

    Returns dict of {column: {count, lower_bound, upper_bound}}.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = {}
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if count > 0:
            outliers[col] = {
                "count": count,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
    return outliers


def comprehensive_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data quality report.

    Returns
    -------
    dict with keys: shape, missing, duplicates, numerical_issues,
    categorical_issues, outliers, target_stats.
    """
    report = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "missing": {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_issues": validate_numerical_ranges(df),
        "categorical_issues": validate_categorical_values(df),
        "outliers": detect_outliers_iqr(df, [
            "distance_km", "actual_trip_cost_inr", "load_weight_tons",
            "fuel_efficiency_km_per_litre",
        ]),
    }
    if "actual_trip_cost_inr" in df.columns:
        report["target_stats"] = df["actual_trip_cost_inr"].describe().to_dict()
    return report
