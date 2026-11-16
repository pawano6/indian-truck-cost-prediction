"""
train_model.py — ANN model building, training, evaluation, and artifact saving.

Uses TensorFlow/Keras for the regression ANN and scikit-learn for
preprocessing. All artefacts (model, preprocessor, feature schema,
metrics) are saved to the models/ directory.

IMPORTANT: This project uses a synthetic educational dataset.
High evaluation metrics may reflect the formula-generated nature of
the target variable and should NOT be interpreted as production-grade
accuracy on real logistics data.
"""

import os
import sys
import json

# Ensure console output handles encodings cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.data_utils import (
    load_dataset,
    validate_columns,
    separate_features_target,
    build_preprocessor,
    get_feature_names,
    build_baseline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.15  # of remaining training data
EPOCHS = 150
BATCH_SIZE = 64
PATIENCE = 15  # early-stopping patience

DATA_PATH = os.path.join("data", "indian_truck_logistics_synthetic_15000.csv")
MODEL_DIR = "models"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# ANN architecture
# ---------------------------------------------------------------------------

def build_ann(input_dim: int) -> keras.Model:
    """Build an ANN regression model.

    Architecture:
        Input -> Dense(128, ReLU) -> Dropout(0.2) ->
        Dense(64, ReLU) -> Dropout(0.2) ->
        Dense(32, ReLU) ->
        Dense(1, linear)

    Parameters
    ----------
    input_dim : int
        Number of input features after preprocessing.

    Returns
    -------
    keras.Model
    """
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
    """Compute regression metrics.

    Returns
    -------
    dict
        Keys: mae, rmse, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def plot_training_history(history, save_path: str):
    """Save training/validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
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
    print(f"  -> Training history plot saved to {save_path}")


def plot_actual_vs_predicted(y_true, y_pred, save_path: str):
    """Save actual vs. predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    axes[0].set_xlabel("Actual Trip Cost (INR)")
    axes[0].set_ylabel("Predicted Trip Cost (INR)")
    axes[0].set_title("Actual vs. Predicted Trip Cost")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=8, color="coral")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Predicted Trip Cost (INR)")
    axes[1].set_ylabel("Residual (INR)")
    axes[1].set_title("Residual Analysis")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Actual vs. Predicted plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    """End-to-end training pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 60)
    print("  ANN-Based Indian Truck Trip Cost Prediction")
    print("  Training Pipeline")
    print("=" * 60)

    # 1. Load & validate -------------------------------------------------
    print("\n[1/8] Loading dataset...")
    df = load_dataset(DATA_PATH)
    validate_columns(df)
    print(f"  -> Shape: {df.shape}")
    print(f"  -> Missing values: {df.isnull().sum().sum()}")
    print(f"  -> Duplicates: {df.duplicated().sum()}")

    # 2. Separate features & target -------------------------------------
    print("\n[2/8] Separating features and target...")
    X, y = separate_features_target(df)
    print(f"  -> Features: {X.shape[1]} columns")
    print(f"  -> Target: {TARGET_COLUMN}")
    print(f"  -> Target range: INR {y.min():,.2f} - INR {y.max():,.2f}")

    # 3. Train/test split -----------------------------------------------
    print("\n[3/8] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"  -> Train: {X_train.shape[0]} samples")
    print(f"  -> Test:  {X_test.shape[0]} samples")

    # 4. Preprocessing (fit on train only) ------------------------------
    print("\n[4/8] Building & fitting preprocessor (train only)...")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    print(f"  -> Processed feature count: {X_train_processed.shape[1]}")

    # Save preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  -> Preprocessor saved to {preprocessor_path}")

    # Save feature schema
    schema = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "processed_feature_names": feature_names,
        "input_dim": X_train_processed.shape[1],
    }
    schema_path = os.path.join(MODEL_DIR, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  -> Feature schema saved to {schema_path}")

    # 5. Baseline -------------------------------------------------------
    print("\n[5/8] Training baseline models...")
    baseline = build_baseline(strategy="mean")
    baseline.fit(X_train_processed, y_train)
    baseline_pred = baseline.predict(X_test_processed)
    baseline_metrics = evaluate_model(y_test.values, baseline_pred)
    print(f"  -> DummyRegressor (mean): MAE=INR {baseline_metrics['mae']:,.2f}  "
          f"RMSE=INR {baseline_metrics['rmse']:,.2f}  R2={baseline_metrics['r2']:.4f}")

    # Optional: Linear Regression baseline
    lr = LinearRegression()
    lr.fit(X_train_processed, y_train)
    lr_pred = lr.predict(X_test_processed)
    lr_metrics = evaluate_model(y_test.values, lr_pred)
    print(f"  -> LinearRegression:      MAE=INR {lr_metrics['mae']:,.2f}  "
          f"RMSE=INR {lr_metrics['rmse']:,.2f}  R2={lr_metrics['r2']:.4f}")

    # 6. Build & train ANN ----------------------------------------------
    print("\n[6/8] Building and training ANN...")
    input_dim = X_train_processed.shape[1]
    model = build_ann(input_dim)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_processed, y_train.values,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    # Save training history plot
    plot_training_history(
        history,
        os.path.join(MODEL_DIR, "training_history.png"),
    )

    # 7. Evaluate on test set -------------------------------------------
    print("\n[7/8] Evaluating on held-out test set...")
    y_pred = model.predict(X_test_processed, verbose=0).flatten()
    ann_metrics = evaluate_model(y_test.values, y_pred)
    print(f"  -> ANN:  MAE=INR {ann_metrics['mae']:,.2f}  "
          f"RMSE=INR {ann_metrics['rmse']:,.2f}  R2={ann_metrics['r2']:.4f}")

    # Save actual vs predicted plot
    plot_actual_vs_predicted(
        y_test.values, y_pred,
        os.path.join(MODEL_DIR, "actual_vs_predicted.png"),
    )

    # 8. Save artefacts -------------------------------------------------
    print("\n[8/8] Saving artefacts...")

    # Keras model
    model_path = os.path.join(MODEL_DIR, "truck_cost_ann.keras")
    model.save(model_path)
    print(f"  -> ANN model saved to {model_path}")

    # Metrics
    all_metrics = {
        "baseline_dummy_mean": baseline_metrics,
        "baseline_linear_regression": lr_metrics,
        "ann": ann_metrics,
        "note": (
            "This dataset is synthetic (formula-generated). High R2 values "
            "reflect the deterministic formula, not real-world prediction "
            "accuracy. Do not quote these metrics as production performance."
        ),
    }
    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  -> Evaluation metrics saved to {metrics_path}")

    # Save baseline model
    baseline_lr_path = os.path.join(MODEL_DIR, "baseline_lr.pkl")
    joblib.dump(lr, baseline_lr_path)
    print(f"  -> Linear Regression baseline saved to {baseline_lr_path}")

    print("\n" + "=" * 60)
    print("  Training complete! All artefacts saved in models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
