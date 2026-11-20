"""
prediction.py — Prediction utilities for model loading and inference.

Handles loading the best model and preprocessor, single predictions,
and batch predictions from CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from src.data_utils import ALL_FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_prediction_model():
    """Load the best model and preprocessor for inference.

    Returns
    -------
    tuple (model, preprocessor, schema, model_type)
        model_type is one of: 'ann', 'sklearn'
    """
    schema_path = os.path.join(MODEL_DIR, "feature_schema.json")
    prep_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")

    if not os.path.isfile(schema_path) or not os.path.isfile(prep_path):
        raise FileNotFoundError("Model artefacts not found. Run training first.")

    with open(schema_path) as f:
        schema = json.load(f)
    preprocessor = joblib.load(prep_path)

    # Determine best model from metrics
    best_name = "ann"  # default fallback
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_info = metrics.get("best_model", {})
        best_name = best_info.get("best_model_name", "ann")

    # Load appropriate model
    model_map = {
        "ann": ("truck_cost_ann.keras", "ann"),
        "hist_gradient_boosting": ("hgbr_model.pkl", "sklearn"),
        "random_forest": ("rf_model.pkl", "sklearn"),
        "linear_regression": ("baseline_lr.pkl", "sklearn"),
    }

    filename, model_type = model_map.get(best_name, ("truck_cost_ann.keras", "ann"))
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.isfile(model_path):
        # Fallback: try ANN
        model_path = os.path.join(MODEL_DIR, "truck_cost_ann.keras")
        model_type = "ann"
        best_name = "ann"

    if model_type == "ann":
        os.environ.setdefault("KERAS_BACKEND", "torch")
        import keras
        model = keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    return model, preprocessor, schema, model_type


def predict_single(model, preprocessor, schema, input_dict: dict,
                   model_type: str = "ann") -> float:
    """Make a single prediction from a dictionary of inputs.

    Parameters
    ----------
    model : trained model
    preprocessor : fitted ColumnTransformer
    schema : dict with feature schema
    input_dict : dict of feature values
    model_type : 'ann' or 'sklearn'

    Returns
    -------
    float — predicted trip cost in INR
    """
    input_df = pd.DataFrame([input_dict])
    feature_cols = schema.get("all_feature_columns", ALL_FEATURE_COLUMNS)
    input_df = input_df[feature_cols]
    processed = preprocessor.transform(input_df)

    if model_type == "ann":
        prediction = model.predict(processed, verbose=0).flatten()[0]
    else:
        prediction = model.predict(processed)[0]

    return float(prediction)


def predict_batch(model, preprocessor, schema, df: pd.DataFrame,
                  model_type: str = "ann") -> pd.DataFrame:
    """Make batch predictions on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame with required feature columns

    Returns
    -------
    pd.DataFrame — original data with 'predicted_cost_inr' column added
    """
    feature_cols = schema.get("all_feature_columns", ALL_FEATURE_COLUMNS)
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in uploaded data: {missing}")

    input_df = df[feature_cols].copy()
    processed = preprocessor.transform(input_df)

    if model_type == "ann":
        predictions = model.predict(processed, verbose=0).flatten()
    else:
        predictions = model.predict(processed)

    result = df.copy()
    result["predicted_cost_inr"] = predictions
    return result


def validate_prediction_input(input_dict: dict) -> list[str]:
    """Validate a prediction input dictionary.

    Returns list of error messages (empty if valid).
    """
    errors = []

    if input_dict.get("distance_km", 0) <= 0:
        errors.append("Distance must be positive.")
    if input_dict.get("truck_capacity_tons", 0) <= 0:
        errors.append("Truck capacity must be positive.")
    if input_dict.get("load_weight_tons", 0) < 0:
        errors.append("Load weight cannot be negative.")
    if input_dict.get("fuel_efficiency_km_per_litre", 0) <= 0:
        errors.append("Fuel efficiency must be positive.")
    if input_dict.get("diesel_price_inr_per_litre", 0) <= 0:
        errors.append("Diesel price must be positive.")

    load = input_dict.get("load_weight_tons", 0)
    cap = input_dict.get("truck_capacity_tons", 0)
    if cap > 0 and load > cap:
        errors.append(
            f"Load ({load}t) exceeds truck capacity ({cap}t)."
        )

    for field in ["toll_cost_inr", "driver_allowance_inr", "maintenance_cost_inr",
                   "loading_unloading_cost_inr", "other_cost_inr"]:
        if input_dict.get(field, 0) < 0:
            errors.append(f"{field.replace('_', ' ').title()} cannot be negative.")

    month = input_dict.get("month", 0)
    if not (1 <= month <= 12):
        errors.append("Month must be between 1 and 12.")

    return errors
