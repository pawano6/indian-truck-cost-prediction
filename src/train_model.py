"""
train_model.py — Multi-model training pipeline with comprehensive evaluation.

Compares DummyRegressor, Linear Regression, RandomForest,
HistGradientBoosting, and ANN (TensorFlow/Keras) on the Indian
truck logistics cost prediction task.

IMPORTANT: This dataset is synthetic (formula-generated). High R² values
reflect the deterministic formula, not real-world prediction accuracy.
Random train/test splitting may overestimate generalization because records
sharing the same synthetic generation formula can appear in both sets.
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import sys
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor

import keras
from keras import layers, callbacks

from src.data_utils import (
    load_dataset,
    validate_columns,
    separate_features_target,
    build_preprocessor,
    get_feature_names,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.feature_engineering import (
    engineer_features,
    comprehensive_data_quality_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.15
EPOCHS = 150
BATCH_SIZE = 64
PATIENCE = 15

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")
MODEL_DIR = "models"

np.random.seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# ANN architecture (preserved from original)
# ---------------------------------------------------------------------------

def build_ann(input_dim: int) -> keras.Model:
    """Build ANN: Input → Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(1)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", name="hidden_1"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", name="hidden_2"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu", name="hidden_3"),
        layers.Dense(1, activation="linear", name="output"),
    ], name="TruckCostANN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R²."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def cross_validate_model(estimator, X, y, cv=5):
    """
    Run cross-validation with preprocessing fitted separately
    inside each fold to prevent data leakage.
    """
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    return -scores

def error_analysis_by_group(y_true, y_pred, groups: pd.Series,
                            group_name: str) -> dict:
    """Compute error metrics for each group value."""
    results = {}
    for val in sorted(groups.unique()):
        mask = groups == val
        if mask.sum() < 5:
            continue
        m = evaluate_model(y_true[mask], y_pred[mask])
        m["count"] = int(mask.sum())
        results[str(val)] = m
    return results


def distance_band(distance: pd.Series) -> pd.Series:
    """Categorize distance into bands."""
    return pd.cut(
        distance,
        bins=[0, 300, 800, 1200, float("inf")],
        labels=["Short (<300km)", "Medium (300-800km)",
                "Long (800-1200km)", "Very Long (>1200km)"],
    )


def plot_training_history(history, save_path: str):
    """Save ANN training/validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("Training & Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (INR)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, save_path: str, model_name: str = ""):
    """Save actual vs predicted scatter + residual plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    axes[0].set_xlabel("Actual Trip Cost (INR)")
    axes[0].set_ylabel("Predicted Trip Cost (INR)")
    axes[0].set_title(f"Actual vs Predicted — {model_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color="coral")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Predicted Trip Cost (INR)")
    axes[1].set_ylabel("Residual (INR)")
    axes[1].set_title(f"Residual Analysis — {model_name}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(all_metrics: dict, save_path: str):
    """Bar chart comparing all models on MAE, RMSE, R²."""
    models = []
    mae_vals, rmse_vals, r2_vals = [], [], []
    for name, m in all_metrics.items():
        if isinstance(m, dict) and "mae" in m:
            models.append(name.replace("_", " ").title())
            mae_vals.append(m["mae"])
            rmse_vals.append(m["rmse"])
            r2_vals.append(m["r2"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(len(models))

    bars = axes[0].bar(x, mae_vals, color="#ff6b35", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("MAE (INR)")
    axes[0].set_title("Mean Absolute Error")
    axes[0].bar_label(bars, fmt="₹{:,.0f}", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(x, rmse_vals, color="#4ecdc4", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("RMSE (INR)")
    axes[1].set_title("Root Mean Squared Error")
    axes[1].bar_label(bars, fmt="₹{:,.0f}", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    bars = axes[2].bar(x, r2_vals, color="#a855f7", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("R² Score")
    axes[2].set_title("R-Squared")
    axes[2].bar_label(bars, fmt="{:.4f}", fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].set_ylim(min(0, min(r2_vals) - 0.05), 1.05)

    plt.suptitle("Model Comparison — Synthetic Indian Truck Logistics Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  Indian Truck Trip Cost Prediction — Training Pipeline")
    print("=" * 60)

    # 1. Load & validate ---------------------------------------------------
    print("\n[1/9] Loading dataset...")
    df = load_dataset(DATA_PATH)
    validate_columns(df)
    print(f"  → Shape: {df.shape}")

    # 2. Data quality report -----------------------------------------------
    print("\n[2/9] Data quality check...")
    quality = comprehensive_data_quality_report(df)
    print(f"  → Missing values: {len(quality['missing'])} columns with nulls")
    print(f"  → Duplicates: {quality['duplicate_rows']}")
    print(f"  → Numerical issues: {len(quality['numerical_issues'])}")

    # 3. Feature engineering -----------------------------------------------
    print("\n[3/9] Feature engineering...")
    df = engineer_features(df, include_post_trip=False)
    print(f"  → Columns after engineering: {df.shape[1]}")

    # 4. Separate features & target ----------------------------------------
    print("\n[4/9] Separating features and target...")
    X, y = separate_features_target(df)
    print(f"  → Features: {X.shape[1]} columns")
    print(f"  → Target range: ₹{y.min():,.0f} – ₹{y.max():,.0f}")

    # 5. Train/test split --------------------------------------------------
    print("\n[5/9] Train/test split (seed=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Save truck_type and distance for error analysis before preprocessing
    test_truck_types = X_test["truck_type"].copy()
    test_distances = X_test["distance_km"].copy()
    test_distance_bands = distance_band(test_distances)

    print(f"  → Train: {X_train.shape[0]} · Test: {X_test.shape[0]}")

    # 6. Preprocessing (fit on train only) ---------------------------------
    print("\n[6/9] Preprocessing (fit on training data only)...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"  → Processed features: {X_train_proc.shape[1]}")

    # Save preprocessor
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    # Save feature schema
    schema = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "processed_feature_names": feature_names,
        "input_dim": X_train_proc.shape[1],
    }
    with open(os.path.join(MODEL_DIR, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    # 7. Train all models --------------------------------------------------
    print("\n[7/9] Training and evaluating models...")
    all_metrics = {}

    # 7a. DummyRegressor baseline
    print("  Training DummyRegressor (mean)...")
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_proc, y_train)
    dummy_pred = dummy.predict(X_test_proc)
    all_metrics["dummy_baseline"] = evaluate_model(y_test.values, dummy_pred)
    print(f"    MAE=₹{all_metrics['dummy_baseline']['mae']:,.0f}  "
          f"R²={all_metrics['dummy_baseline']['r2']:.4f}")

    # 7b. Linear Regression
    print("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_proc, y_train)
    lr_pred = lr.predict(X_test_proc)
    all_metrics["linear_regression"] = evaluate_model(y_test.values, lr_pred)
    joblib.dump(lr, os.path.join(MODEL_DIR, "baseline_lr.pkl"))
    # Cross-validation
    lr_cv = cross_validate_model(lr, X_train, y_train, cv=5)
    all_metrics["linear_regression"]["cv_mae_mean"] = float(lr_cv.mean())
    all_metrics["linear_regression"]["cv_mae_std"] = float(lr_cv.std())
    print(f"    MAE=₹{all_metrics['linear_regression']['mae']:,.0f}  "
        f"R²={all_metrics['linear_regression']['r2']:.4f}  "
        f"CV-MAE=₹{lr_cv.mean():,.0f} ± {lr_cv.std():,.0f}")

    # 7c. RandomForest
    print("  Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train_proc, y_train)
    rf_pred = rf.predict(X_test_proc)
    all_metrics["random_forest"] = evaluate_model(y_test.values, rf_pred)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
    rf_cv = cross_validate_model(rf, X_train, y_train, cv=3)
    all_metrics["random_forest"]["cv_mae_mean"] = float(rf_cv.mean())
    all_metrics["random_forest"]["cv_mae_std"] = float(rf_cv.std())
    print(f"    MAE=₹{all_metrics['random_forest']['mae']:,.0f}  "
        f"R²={all_metrics['random_forest']['r2']:.4f}  "
        f"CV-MAE=₹{rf_cv.mean():,.0f} ± {rf_cv.std():,.0f}")

    # 7d. HistGradientBoosting
    print("  Training HistGradientBoostingRegressor...")
    hgbr = HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED,
    )
    hgbr.fit(X_train_proc, y_train)
    hgbr_pred = hgbr.predict(X_test_proc)
    all_metrics["hist_gradient_boosting"] = evaluate_model(y_test.values, hgbr_pred)
    joblib.dump(hgbr, os.path.join(MODEL_DIR, "hgbr_model.pkl"))
    hgbr_cv = cross_validate_model(hgbr, X_train, y_train, cv=3)
    all_metrics["hist_gradient_boosting"]["cv_mae_mean"] = float(hgbr_cv.mean())
    all_metrics["hist_gradient_boosting"]["cv_mae_std"] = float(hgbr_cv.std())

    print(f"    MAE=₹{all_metrics['hist_gradient_boosting']['mae']:,.0f}  "
      f"R²={all_metrics['hist_gradient_boosting']['r2']:.4f}  "
      f"CV-MAE=₹{hgbr_cv.mean():,.0f} ± {hgbr_cv.std():,.0f}")

    # 7e. ANN
    print("  Training ANN (Keras with PyTorch backend)...")
    input_dim = X_train_proc.shape[1]
    ann = build_ann(input_dim)
    ann.summary(print_fn=lambda x: print(f"    {x}"))

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1,
    )
    history = ann.fit(
        X_train_proc, y_train.values,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=1,
    )
    plot_training_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    ann_pred = ann.predict(X_test_proc, verbose=0).flatten()
    all_metrics["ann"] = evaluate_model(y_test.values, ann_pred)
    ann.save(os.path.join(MODEL_DIR, "truck_cost_ann.keras"))
    print(f"    MAE=₹{all_metrics['ann']['mae']:,.0f}  "
          f"R²={all_metrics['ann']['r2']:.4f}")

    # 8. Determine best model & error analysis -----------------------------
    print("\n[8/9] Model comparison and error analysis...")

    # Select using training-only validation results, not test-set MAE.
    model_selection_maes = {
    "linear_regression": all_metrics["linear_regression"]["cv_mae_mean"],
    "random_forest": all_metrics["random_forest"]["cv_mae_mean"],
    "hist_gradient_boosting": all_metrics["hist_gradient_boosting"]["cv_mae_mean"],
    "ann": float(min(history.history["val_mae"])),
}

    best_name = min(model_selection_maes, key=model_selection_maes.get)
    best_preds = {
        "linear_regression": lr_pred,
        "random_forest": rf_pred,
        "hist_gradient_boosting": hgbr_pred,
        "ann": ann_pred,
    }

    print(f"\n  Best model (lowest MAE): {best_name}")
    print(f"  MAE=₹{all_metrics[best_name]['mae']:,.0f}  "
          f"RMSE=₹{all_metrics[best_name]['rmse']:,.0f}  "
          f"R²={all_metrics[best_name]['r2']:.4f}")

    # Error analysis for best model
    bp = best_preds[best_name]
    error_by_truck = error_analysis_by_group(
        y_test.values, bp, test_truck_types, "truck_type"
    )
    error_by_distance = error_analysis_by_group(
        y_test.values, bp, test_distance_bands, "distance_band"
    )

    # Save actual vs predicted for best model
    plot_actual_vs_predicted(
        y_test.values, bp,
        os.path.join(MODEL_DIR, "actual_vs_predicted.png"),
        model_name=best_name.replace("_", " ").title(),
    )

    # Model comparison chart
    plot_model_comparison(all_metrics, os.path.join(MODEL_DIR, "model_comparison.png"))

    # Save best model reference
    best_model_info = {
        "best_model_name": best_name,
        "selection_criterion": "lowest validation MAE; CV for classical models and ANN validation split",
    }

    # 9. Save all artefacts ------------------------------------------------
    print("\n[9/9] Saving artefacts...")

    final_metrics = {
        **all_metrics,
        "best_model": best_model_info,
        "error_analysis": {
            "by_truck_type": error_by_truck,
            "by_distance_band": error_by_distance,
        },
        "data_quality": {
            "total_rows": int(df.shape[0]),
            "missing_value_columns": len(quality["missing"]),
            "duplicate_rows": quality["duplicate_rows"],
        },
        "validation_note": (
            "This dataset is synthetic (formula-generated). High R² values "
            "reflect the deterministic generation formula, not real-world "
            "prediction accuracy. Random train/test splitting may overestimate "
            "generalization because records sharing the same synthetic formula "
            "can appear in both sets. A grouped or scenario-based split would "
            "provide a more conservative estimate but is not applicable here "
            "since all records share the same formula."
        ),
    }

    with open(os.path.join(MODEL_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Export data with features for Streamlit/analytics
    export_df = df.copy()
    export_path = os.path.join("data", "processed_logistics_data.csv")
    export_df.to_csv(export_path, index=False)
    print(f"  → Processed data exported to {export_path}")

    print("\n" + "=" * 60)
    print("  Training complete! All artefacts saved in models/")
    print(f"  Best model: {best_name} (MAE=₹{all_metrics[best_name]['mae']:,.0f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
