"""
test_prediction.py — Unit tests for src/prediction.py

Tests single and batch inference, input validation, and model loading.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import load_dataset, ALL_FEATURE_COLUMNS
from src.prediction import (
    validate_prediction_input,
    load_prediction_model,
    predict_single,
    predict_batch,
)

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")


@pytest.fixture(scope="module")
def valid_input():
    return {
        "origin_city": "Mumbai",
        "origin_state": "Maharashtra",
        "destination_city": "Delhi",
        "destination_state": "Delhi",
        "distance_km": 1400,
        "truck_type": "Heavy",
        "truck_capacity_tons": 16.0,
        "load_weight_tons": 12.0,
        "fuel_efficiency_km_per_litre": 4.2,
        "diesel_price_inr_per_litre": 94.5,
        "toll_cost_inr": 1200.0,
        "driver_allowance_inr": 1800.0,
        "maintenance_cost_inr": 2500.0,
        "loading_unloading_cost_inr": 2000.0,
        "other_cost_inr": 800.0,
        "route_factor": 1.05,
        "traffic_factor": 1.10,
        "month": 6,
    }


class TestInputValidation:
    def test_valid_input_passes(self, valid_input):
        errors = validate_prediction_input(valid_input)
        assert len(errors) == 0

    def test_negative_distance_rejected(self, valid_input):
        bad = valid_input.copy()
        bad["distance_km"] = -100
        errors = validate_prediction_input(bad)
        assert any("Distance" in e for e in errors)

    def test_zero_capacity_rejected(self, valid_input):
        bad = valid_input.copy()
        bad["truck_capacity_tons"] = 0
        errors = validate_prediction_input(bad)
        assert any("capacity" in e for e in errors)

    def test_overload_rejected(self, valid_input):
        bad = valid_input.copy()
        bad["load_weight_tons"] = 25.0  # capacity is 16.0
        errors = validate_prediction_input(bad)
        assert any("exceeds" in e for e in errors)

    def test_invalid_month_rejected(self, valid_input):
        bad = valid_input.copy()
        bad["month"] = 15
        errors = validate_prediction_input(bad)
        assert any("Month" in e for e in errors)

    def test_negative_allowance_rejected(self, valid_input):
        bad = valid_input.copy()
        bad["driver_allowance_inr"] = -500
        errors = validate_prediction_input(bad)
        assert any("Driver Allowance" in e for e in errors)


class TestInferencePipeline:
    def test_model_loading(self):
        model, preprocessor, schema, model_type = load_prediction_model()
        assert model is not None
        assert preprocessor is not None
        assert "all_feature_columns" in schema
        assert model_type in ["ann", "sklearn"]

    def test_predict_single_returns_positive_float(self, valid_input):
        model, preprocessor, schema, model_type = load_prediction_model()
        pred = predict_single(model, preprocessor, schema, valid_input, model_type)
        assert isinstance(pred, float)
        assert pred > 0, "Predicted cost must be positive"
        # Sanity check: 1400 km heavy truck should cost reasonable INR (> 10,000)
        assert pred > 5000

    def test_predict_batch_returns_dataframe(self):
        model, preprocessor, schema, model_type = load_prediction_model()
        df = load_dataset(DATA_PATH)
        sample = df.head(10).copy()
        res = predict_batch(model, preprocessor, schema, sample, model_type)
        assert "predicted_cost_inr" in res.columns
        assert len(res) == 10
        assert (res["predicted_cost_inr"] > 0).all()
