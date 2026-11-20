"""
explainability.py — SHAP-based model explainability for cost predictions.

Provides global feature importance and individual prediction explanations.

IMPORTANT: SHAP feature contributions indicate correlations in the model,
NOT causal relationships. High SHAP values mean the model relies on a feature,
not that changing that feature will necessarily change the actual cost.

This module explains the best tree-based model (HistGradientBoosting or
RandomForest) for computational efficiency. The ANN model is documented
separately but not explained via SHAP due to computational cost.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


_CACHED_MODEL = None
_CACHED_MODEL_NAME = None


def _load_best_tree_model():
    """Load the best tree-based model for SHAP explanation (cached).

    Returns (model, model_name) or (None, None) if no tree model found.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_MODEL_NAME

    # Try HistGradientBoosting first (usually best), then RandomForest
    for name, filename in [
        ("HistGradientBoosting", "hgbr_model.pkl"),
        ("RandomForest", "rf_model.pkl"),
    ]:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.isfile(path):
            _CACHED_MODEL = joblib.load(path)
            _CACHED_MODEL_NAME = name
            return _CACHED_MODEL, _CACHED_MODEL_NAME
    return None, None


def _load_preprocessor():
    """Load the fitted preprocessor."""
    path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    if os.path.isfile(path):
        return joblib.load(path)
    return None


def _get_feature_names():
    """Load feature names from schema."""
    path = os.path.join(MODEL_DIR, "feature_schema.json")
    if os.path.isfile(path):
        with open(path) as f:
            schema = json.load(f)
        return schema.get("processed_feature_names", [])
    return []


def compute_shap_values(X_processed: np.ndarray, model=None,
                        model_name: str | None = None,
                        max_samples: int = 500) -> dict:
    """Compute SHAP values for the given processed feature matrix.

    Parameters
    ----------
    X_processed : np.ndarray
        Preprocessed feature matrix (after ColumnTransformer).
    model : sklearn model, optional
        If None, loads the best tree-based model from disk.
    model_name : str, optional
        Name of the model being explained.
    max_samples : int
        Maximum samples to explain.

    Returns
    -------
    dict with keys: shap_values, feature_names, model_name, expected_value
    """
    if model is None:
        model, model_name = _load_best_tree_model()
    if model is None:
        raise FileNotFoundError(
            "No tree-based model found in models/. Train models first."
        )

    feature_names = _get_feature_names()

    if X_processed.shape[0] > max_samples:
        X_eval = X_processed[:max_samples]
    else:
        X_eval = X_processed

    # TreeExplainer calculates tree-path SHAP values directly
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_eval)
    expected_val = float(np.atleast_1d(explainer.expected_value)[0])

    return {
        "shap_values": np.array(shap_vals),
        "feature_names": feature_names if feature_names else [
            f"feature_{i}" for i in range(X_processed.shape[1])
        ],
        "model_name": model_name or "Unknown",
        "expected_value": expected_val,
    }


def global_feature_importance(shap_result: dict, top_n: int = 20) -> pd.DataFrame:
    """Compute global feature importance from SHAP values.

    Returns a DataFrame sorted by mean |SHAP value|.
    """
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    mean_abs = np.abs(shap_values).mean(axis=0)

    # Clean feature names for display
    clean_names = []
    for name in feature_names:
        name = name.replace("num__", "").replace("cat__", "")
        clean_names.append(name)

    importance_df = pd.DataFrame({
        "feature": clean_names[:len(mean_abs)],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    importance_df = importance_df.reset_index(drop=True)
    return importance_df


def explain_single_prediction(
    input_row: np.ndarray,
    model=None,
    model_name: str | None = None,
    background: np.ndarray | None = None,
    max_background: int = 100,
) -> dict:
    """Explain a single prediction with SHAP values.

    Parameters
    ----------
    input_row : np.ndarray
        Single preprocessed row (1, n_features).
    model : optional
    model_name : optional
    background : np.ndarray, optional (kept for compatibility)
    max_background : int, optional

    Returns
    -------
    dict with: prediction, expected_value, feature_contributions (list of
    dicts with feature, shap_value, direction).
    """
    if model is None:
        model, model_name = _load_best_tree_model()
    if model is None:
        raise FileNotFoundError("No tree-based model found.")

    feature_names = _get_feature_names()

    if input_row.ndim == 1:
        input_row = input_row.reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(input_row)
    if isinstance(sv, list):
        sv_arr = np.array(sv[0])
    else:
        sv_arr = np.array(sv)
    if sv_arr.ndim > 1:
        sv_row = sv_arr[0]
    else:
        sv_row = sv_arr

    expected_val = float(np.atleast_1d(explainer.expected_value)[0])
    prediction = float(model.predict(input_row)[0])

    # Build contribution list
    contributions = []
    for i, val in enumerate(sv_row):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        name = name.replace("num__", "").replace("cat__", "")
        contributions.append({
            "feature": name,
            "shap_value": float(val),
            "direction": "increases cost" if val > 0 else "decreases cost",
        })

    # Sort by absolute value
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "prediction": prediction,
        "expected_value": expected_val,
        "model_name": model_name or "Unknown",
        "feature_contributions": contributions,
        "disclaimer": (
            "SHAP values show feature correlations in the model, "
            "NOT causal relationships. This is a synthetic dataset."
        ),
    }


def plot_global_importance(shap_result: dict, save_path: str | None = None,
                           top_n: int = 15) -> plt.Figure:
    """Create a bar plot of global feature importance."""
    importance_df = global_feature_importance(shap_result, top_n=top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(len(importance_df)),
        importance_df["mean_abs_shap"].values,
        color="#ff6b35", alpha=0.8,
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (INR)", fontsize=11)
    ax.set_title(
        f"Global Feature Importance — {shap_result['model_name']}\n"
        "(SHAP values indicate model reliance, not causation)",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_single_explanation(explanation: dict, top_n: int = 10,
                            save_path: str | None = None) -> plt.Figure:
    """Create a waterfall-style bar chart for a single prediction explanation."""
    contribs = explanation["feature_contributions"][:top_n]

    features = [c["feature"] for c in contribs]
    values = [c["shap_value"] for c in contribs]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(features)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (INR impact on prediction)", fontsize=11)
    ax.set_title(
        f"Prediction Explanation — ₹{explanation['prediction']:,.0f}\n"
        "Red = increases cost · Green = decreases cost\n"
        "(Correlations, not causal effects)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
