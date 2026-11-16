"""
test_data_utils.py — Unit tests for src/data_utils.py

Run with:
    python -m pytest tests/test_data_utils.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import (
    load_dataset,
    validate_columns,
    data_quality_report,
    separate_features_target,
    build_preprocessor,
    get_feature_names,
    build_baseline,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
)

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Load the dataset once for the test module."""
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="module")
def features_and_target(raw_df):
    """Return (X, y) after separation."""
    return separate_features_target(raw_df)


# ---------------------------------------------------------------------------
# Tests — Loading & Validation
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_file_exists(self):
        assert os.path.isfile(DATA_PATH), f"Dataset not found at {DATA_PATH}"

    def test_load_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_shape_rows(self, raw_df):
        assert raw_df.shape[0] > 10_000, "Expected > 10,000 rows"

    def test_shape_columns(self, raw_df):
        assert raw_df.shape[1] == 23, f"Expected 23 columns, got {raw_df.shape[1]}"

    def test_required_columns_present(self, raw_df):
        assert validate_columns(raw_df) is True

    def test_missing_column_raises(self):
        dummy = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            validate_columns(dummy)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent/path.csv")


class TestDataQuality:
    def test_no_missing_values(self, raw_df):
        missing = raw_df.isnull().sum().sum()
        assert missing == 0, f"Found {missing} missing values"

    def test_no_duplicate_rows(self, raw_df):
        dupes = raw_df.duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate rows"

    def test_target_positive(self, raw_df):
        assert (raw_df[TARGET_COLUMN] > 0).all(), "Target has non-positive values"

    def test_data_quality_report(self, raw_df):
        report = data_quality_report(raw_df)
        assert "shape" in report
        assert "missing_counts" in report
        assert report["duplicate_rows"] == 0


# ---------------------------------------------------------------------------
# Tests — Feature / Target Separation
# ---------------------------------------------------------------------------

class TestFeatureSeparation:
    def test_target_not_in_features(self, features_and_target):
        X, y = features_and_target
        assert TARGET_COLUMN not in X.columns

    def test_excluded_cols_not_in_features(self, features_and_target):
        X, _ = features_and_target
        for col in DROP_COLUMNS:
            assert col not in X.columns, f"{col} should be dropped"

    def test_target_is_series(self, features_and_target):
        _, y = features_and_target
        assert isinstance(y, pd.Series)

    def test_feature_count(self, features_and_target):
        X, _ = features_and_target
        assert X.shape[1] == len(ALL_FEATURE_COLUMNS)

    def test_all_feature_columns_present(self, features_and_target):
        X, _ = features_and_target
        for col in ALL_FEATURE_COLUMNS:
            assert col in X.columns, f"Feature {col} missing"


# ---------------------------------------------------------------------------
# Tests — Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_preprocessor_fits(self, features_and_target):
        X, _ = features_and_target
        preprocessor = build_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        assert X_processed.shape[0] == X.shape[0]
        assert X_processed.shape[1] > 0

    def test_feature_names_extracted(self, features_and_target):
        X, _ = features_and_target
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        names = get_feature_names(preprocessor)
        assert len(names) > 0
        assert len(names) == preprocessor.transform(X[:1]).shape[1]

    def test_no_nans_after_processing(self, features_and_target):
        X, _ = features_and_target
        preprocessor = build_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        assert not np.isnan(X_processed).any(), "NaNs found in processed features"

    def test_unknown_category_handled(self, features_and_target):
        X, _ = features_and_target
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        # Create a row with an unseen city
        test_row = X.iloc[:1].copy()
        test_row["origin_city"] = "UnseenCityXYZ"
        result = preprocessor.transform(test_row)
        assert result.shape[0] == 1
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# Tests — Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_dummy_regressor_trains(self, features_and_target):
        X, y = features_and_target
        preprocessor = build_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        baseline = build_baseline()
        baseline.fit(X_processed, y)
        preds = baseline.predict(X_processed)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# Tests — Inference (post-training artefacts)
# ---------------------------------------------------------------------------

class TestInference:
    """Tests that require trained model artefacts to exist."""

    @pytest.fixture(autouse=True)
    def check_artefacts(self):
        self.model_path = os.path.join("models", "truck_cost_ann.keras")
        self.preprocessor_path = os.path.join("models", "preprocessor.pkl")
        self.schema_path = os.path.join("models", "feature_schema.json")
        self.metrics_path = os.path.join("models", "evaluation_metrics.json")

    def test_model_file_exists(self):
        if not os.path.exists(self.model_path):
            pytest.skip("Model not yet trained")
        assert os.path.isfile(self.model_path)

    def test_preprocessor_file_exists(self):
        if not os.path.exists(self.preprocessor_path):
            pytest.skip("Preprocessor not yet saved")
        assert os.path.isfile(self.preprocessor_path)

    def test_schema_file_exists(self):
        if not os.path.exists(self.schema_path):
            pytest.skip("Schema not yet saved")
        import json
        with open(self.schema_path) as f:
            schema = json.load(f)
        assert "input_dim" in schema
        assert "numerical_features" in schema

    def test_metrics_file_exists(self):
        if not os.path.exists(self.metrics_path):
            pytest.skip("Metrics not yet saved")
        import json
        with open(self.metrics_path) as f:
            metrics = json.load(f)
        assert "ann" in metrics
        assert "mae" in metrics["ann"]

    def test_end_to_end_prediction(self, raw_df):
        """Test that a single row can be predicted end-to-end."""
        import joblib

        if not os.path.exists(self.model_path):
            pytest.skip("Model not yet trained")

        from tensorflow import keras
        model = keras.models.load_model(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)

        # Prepare a single-row input
        X, _ = separate_features_target(raw_df)
        sample = X.iloc[:1]
        sample_processed = preprocessor.transform(sample)
        prediction = model.predict(sample_processed, verbose=0)
        assert prediction.shape == (1, 1)
        assert prediction[0][0] > 0, "Prediction should be positive"
