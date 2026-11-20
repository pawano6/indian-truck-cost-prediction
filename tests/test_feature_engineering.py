"""
test_feature_engineering.py — Unit tests for src/feature_engineering.py

Tests data quality validation, feature engineering, and data leakage prevention.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import load_dataset
from src.feature_engineering import (
    add_load_utilization,
    add_estimated_fuel_metrics,
    add_operational_efficiency_metrics,
    add_post_trip_metrics,
    engineer_features,
    validate_numerical_ranges,
    validate_categorical_values,
    detect_outliers_iqr,
    comprehensive_data_quality_report,
)

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")


@pytest.fixture(scope="module")
def sample_df():
    """Create a controlled sample DataFrame for unit testing."""
    return pd.DataFrame({
        "trip_id": ["T1", "T2"],
        "origin_city": ["Mumbai", "Delhi"],
        "origin_state": ["Maharashtra", "Delhi"],
        "destination_city": ["Ahmedabad", "Jaipur"],
        "destination_state": ["Gujarat", "Rajasthan"],
        "distance_km": [350, 280],
        "truck_type": ["Heavy", "Medium"],
        "truck_capacity_tons": [16.0, 9.0],
        "load_weight_tons": [12.0, 6.0],
        "fuel_efficiency_km_per_litre": [4.0, 5.0],
        "diesel_price_inr_per_litre": [95.0, 90.0],
        "toll_cost_inr": [200.0, 150.0],
        "driver_allowance_inr": [500.0, 400.0],
        "maintenance_cost_inr": [1000.0, 800.0],
        "loading_unloading_cost_inr": [1200.0, 900.0],
        "other_cost_inr": [300.0, 200.0],
        "route_factor": [1.05, 1.02],
        "traffic_factor": [1.10, 1.08],
        "month": [5, 8],
        "actual_trip_cost_inr": [15000.0, 10000.0],
        "freight_revenue_inr": [18000.0, 13000.0],
        "trip_profit_inr": [3000.0, 3000.0],
        "record_type": ["SYNTHETIC_FOR_EDUCATIONAL_USE", "SYNTHETIC_FOR_EDUCATIONAL_USE"],
    })


class TestFeatureEngineering:
    def test_load_utilization_calculation(self, sample_df):
        res = add_load_utilization(sample_df)
        assert "load_utilization_pct" in res.columns
        # 12 / 16 * 100 = 75.0%
        assert np.isclose(res.loc[0, "load_utilization_pct"], 75.0)
        # 6 / 9 * 100 = 66.67%
        assert np.isclose(res.loc[1, "load_utilization_pct"], 66.666667)

    def test_fuel_metrics_calculation(self, sample_df):
        res = add_estimated_fuel_metrics(sample_df)
        assert "estimated_fuel_litres" in res.columns
        assert "estimated_fuel_cost_inr" in res.columns
        # 350 / 4.0 = 87.5 L
        assert np.isclose(res.loc[0, "estimated_fuel_litres"], 87.5)
        # 87.5 * 95.0 = 8312.5 INR
        assert np.isclose(res.loc[0, "estimated_fuel_cost_inr"], 8312.5)

    def test_operational_efficiency_metrics(self, sample_df):
        df_with_fuel = add_estimated_fuel_metrics(sample_df)
        res = add_operational_efficiency_metrics(df_with_fuel)
        assert "estimated_direct_expenses_inr" in res.columns
        assert "cost_per_ton_km_estimate" in res.columns
        assert (res["estimated_direct_expenses_inr"] > 0).all()
        assert (res["cost_per_ton_km_estimate"] > 0).all()

    def test_leakage_prevention_default(self, sample_df):
        """Verify pre-trip feature engineering does NOT generate post-trip fields."""
        res = engineer_features(sample_df, include_post_trip=False)
        assert "actual_cost_per_km" not in res.columns
        assert "profit_margin_pct" not in res.columns

    def test_post_trip_metrics_when_requested(self, sample_df):
        res = add_post_trip_metrics(sample_df)
        assert "actual_cost_per_km" in res.columns
        assert "profit_margin_pct" in res.columns
        # 15000 / 350 = 42.857
        assert np.isclose(res.loc[0, "actual_cost_per_km"], 15000.0 / 350.0)


class TestDataValidation:
    def test_validate_numerical_ranges_clean(self, sample_df):
        issues = validate_numerical_ranges(sample_df)
        assert len(issues) == 0

    def test_validate_numerical_ranges_catches_negative(self, sample_df):
        bad_df = sample_df.copy()
        bad_df.loc[0, "distance_km"] = -50
        bad_df.loc[0, "toll_cost_inr"] = -10
        issues = validate_numerical_ranges(bad_df)
        assert "distance_km" in issues
        assert "toll_cost_inr" in issues

    def test_validate_load_exceeding_capacity(self, sample_df):
        overloaded_df = sample_df.copy()
        overloaded_df.loc[0, "load_weight_tons"] = 30.0  # capacity is 16.0
        issues = validate_numerical_ranges(overloaded_df)
        assert "load_exceeds_capacity" in issues

    def test_validate_categorical_values(self, sample_df):
        bad_cat_df = sample_df.copy()
        bad_cat_df.loc[0, "truck_type"] = "ElectricHovercraft"
        issues = validate_categorical_values(bad_cat_df)
        assert "truck_type" in issues

    def test_detect_outliers_iqr(self, sample_df):
        outliers = detect_outliers_iqr(sample_df, ["distance_km"])
        assert isinstance(outliers, dict)

    def test_comprehensive_quality_report(self, sample_df):
        report = comprehensive_data_quality_report(sample_df)
        assert "shape" in report
        assert "missing" in report
        assert "duplicate_rows" in report
        assert "numerical_issues" in report
        assert "categorical_issues" in report
