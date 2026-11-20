"""
test_explainability.py — Unit tests for src/explainability.py

Tests SHAP explainability, feature contributions, and disclaimers.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import load_dataset, separate_features_target, build_preprocessor
from src.explainability import (
    compute_shap_values,
    global_feature_importance,
    explain_single_prediction,
    _load_best_tree_model,
)

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")


@pytest.fixture(scope="module")
def tree_model():
    model, name = _load_best_tree_model()
    return model, name


@pytest.fixture(scope="module")
def processed_sample():
    df = load_dataset(DATA_PATH)
    X, _ = separate_features_target(df)
    sample = X.iloc[:15]
    import joblib
    prep_path = os.path.join("models", "preprocessor.pkl")
    if os.path.exists(prep_path):
        preprocessor = joblib.load(prep_path)
        X_proc = preprocessor.transform(sample)
    else:
        preprocessor = build_preprocessor()
        X_proc = preprocessor.fit_transform(X)
    return X_proc


class TestExplainability:
    def test_tree_model_exists(self, tree_model):
        model, name = tree_model
        if model is None:
            pytest.skip("No tree model trained yet")
        assert model is not None
        assert name in ["HistGradientBoosting", "RandomForest"]

    def test_compute_shap_values(self, tree_model, processed_sample):
        model, name = tree_model
        if model is None:
            pytest.skip("No tree model trained yet")
        result = compute_shap_values(processed_sample[:5], model=model, model_name=name)
        assert "shap_values" in result
        assert "feature_names" in result
        assert "expected_value" in result
        assert result["shap_values"].shape[0] == 5

    def test_global_feature_importance(self, tree_model, processed_sample):
        model, name = tree_model
        if model is None:
            pytest.skip("No tree model trained yet")
        result = compute_shap_values(processed_sample[:5], model=model, model_name=name)
        importance_df = global_feature_importance(result, top_n=5)
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 5
        assert "mean_abs_shap" in importance_df.columns
        assert "feature" in importance_df.columns

    def test_single_prediction_explanation(self, tree_model, processed_sample):
        model, name = tree_model
        if model is None:
            pytest.skip("No tree model trained yet")
        explanation = explain_single_prediction(
            processed_sample[0:1], model=model, model_name=name
        )
        assert "prediction" in explanation
        assert "feature_contributions" in explanation
        assert "disclaimer" in explanation
        assert "not causal" in explanation["disclaimer"].lower()
        assert len(explanation["feature_contributions"]) > 0
