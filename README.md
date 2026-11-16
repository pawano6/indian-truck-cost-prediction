# 🚛 ANN-Based Indian Truck Trip Cost Prediction & Fleet Expense Analytics

An end-to-end **Artificial Neural Network (ANN)** regression project that predicts the operating cost of truck trips across Indian routes. Built with TensorFlow/Keras, scikit-learn, and Streamlit.

> ⚠️ **Synthetic Data Disclosure:** This project uses a **synthetic educational dataset** (~15,000 simulated truck trips). It does **not** represent actual company data, Delhivery data, or government logistics statistics. High evaluation metrics reflect the formula-generated nature of the target variable.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Architecture](#-project-architecture)
- [Dataset Overview](#-dataset-overview)
- [Model Architecture](#-model-architecture)
- [Evaluation Metrics](#-evaluation-metrics)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 Problem Statement

A transport operator needs to estimate trip operating costs before dispatch to:
- Plan budgets and set freight rates
- Compare cost estimates across routes and vehicle types
- Analyse fleet expense breakdowns

This project builds an ANN regression model that predicts `actual_trip_cost_inr` (trip operating cost in Indian Rupees) using route, vehicle, fuel, and cost parameters.

---

## 🏗️ Project Architecture

```
indian-truck-cost-prediction/
│
├── data/
│   └── indian_truck_logistics_synthetic_15000.csv
│
├── notebooks/
│   └── truck_cost_prediction.ipynb      # Full EDA + training notebook
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                    # Data loading, validation, preprocessing
│   └── train_model.py                   # Training pipeline (script mode)
│
├── models/                              # Saved artefacts (generated after training)
│   ├── truck_cost_ann.keras             # Trained ANN model
│   ├── preprocessor.pkl                 # Fitted preprocessing pipeline
│   ├── feature_schema.json              # Feature names and input dimensions
│   ├── evaluation_metrics.json          # MAE, RMSE, R² for all models
│   ├── baseline_lr.pkl                  # Baseline Linear Regression
│   ├── training_history.png             # Loss curves
│   └── actual_vs_predicted.png          # Prediction diagnostic plots
│
├── tests/
│   └── test_data_utils.py               # Unit tests
│
├── app.py                               # Streamlit dashboard
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| Rows | ~15,000 |
| Columns | 23 |
| Type | Synthetic / Educational |
| Target | `actual_trip_cost_inr` |
| Cities | 16 Indian cities |
| States | 10 Indian states |
| Truck Types | Heavy, Medium, LCV, Trailer |

### Features Used (18 columns)

**Numerical (13):** distance_km, truck_capacity_tons, load_weight_tons, fuel_efficiency_km_per_litre, diesel_price_inr_per_litre, toll_cost_inr, driver_allowance_inr, maintenance_cost_inr, loading_unloading_cost_inr, other_cost_inr, route_factor, traffic_factor, month

**Categorical (5):** origin_city, origin_state, destination_city, destination_state, truck_type

### Excluded Columns
- `actual_trip_cost_inr` — target variable
- `trip_id` — identifier
- `record_type` — provenance label
- `freight_revenue_inr` — downstream outcome (leakage)
- `trip_profit_inr` — downstream outcome (leakage)

---

## 🧠 Model Architecture

```
Input (N features)
    │
    ├── Dense(128, ReLU)
    ├── Dropout(0.2)
    ├── Dense(64, ReLU)
    ├── Dropout(0.2)
    ├── Dense(32, ReLU)
    └── Dense(1, Linear)  →  Trip Cost (₹)
```

- **Loss:** MSE (Mean Squared Error)
- **Optimizer:** Adam (lr=0.001)
- **Early Stopping:** patience=15, restore_best_weights=True
- **Preprocessing:** StandardScaler (numerical) + OneHotEncoder (categorical)

---

## 📈 Evaluation Metrics

Metrics are computed on a **held-out 20% test set** (fitted preprocessing on train only):

| Model | MAE (₹) | RMSE (₹) | R² |
|-------|---------|----------|-----|
| DummyRegressor (Mean) | ₹16,170.01 | ₹19,688.21 | -0.0040 |
| Linear Regression | ₹3,635.26 | ₹4,992.56 | 0.9354 |
| **ANN (Ours)** | **₹823.35** | **₹1,083.82** | **0.9970** |

> ⚠️ **Evaluation Context:** As noted throughout this documentation, this dataset is synthetic and formula-generated. The high R² (0.9970) reflects the deterministic mathematical formulation used to simulate the dataset and should not be quoted as real-world freight prediction performance.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+ (tested with 3.11.9)
- Windows 10/11 with PowerShell
- Git

### Step 1: Clone / Navigate to project

```powershell
cd "path\to\indian-truck-cost-prediction"
```

### Step 2: Create virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> **If blocked by execution policy**, run this first (current session only):
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### Step 4: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note:** If TensorFlow installation fails due to Windows long path limitations, enable long paths:
> ```powershell
> # Run PowerShell as Administrator:
> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
> ```
> Then restart your terminal and retry the install.

### Step 5: Train the model

**Option A — Run the training script:**
```powershell
python -m src.train_model
```

**Option B — Run the Jupyter notebook:**
```powershell
jupyter notebook notebooks/truck_cost_prediction.ipynb
```

### Step 6: Run the Streamlit app

```powershell
python -m streamlit run app.py
```

---

## 🖥️ Usage

### Streamlit Dashboard
1. Open the app in your browser (default: http://localhost:8501)
2. Configure trip parameters in the sidebar (origin, destination, truck type, costs, etc.)
3. Click **Predict Trip Cost**
4. View the predicted operating cost, expense breakdown, and fuel estimate
5. Explore the dataset and model performance in the dashboard tabs

### Training Script
```powershell
python -m src.train_model
```
This runs the full pipeline: load → validate → split → preprocess → train → evaluate → save.

### Tests
```powershell
python -m pytest tests/ -v
```

---

## 📸 Screenshots

<!-- Add screenshots of your Streamlit app here -->
*Screenshots will be added after deployment.*

---

## ⚠️ Limitations

1. **Synthetic Data:** The target was formula-generated. Metrics do not represent real-world accuracy.
2. **No Real Validation:** Not validated against actual truck operating cost data.
3. **Static Inputs:** Diesel prices and other costs are static inputs; real values fluctuate.
4. **Limited Feature Engineering:** No derived features (cost-per-km, load utilisation ratio).
5. **No Temporal Encoding:** Month is treated as numerical; cyclical encoding could help.

---

## 🚀 Future Improvements

1. Collect and validate against real-world truck trip cost data
2. Feature engineering (cost ratios, cyclical month encoding, load utilisation)
3. Hyperparameter tuning (Keras Tuner / Optuna)
4. Compare with tree-based models (XGBoost, LightGBM)
5. SHAP / LIME for model interpretability
6. REST API endpoint for batch predictions
7. CI/CD pipeline for automated retraining
8. Docker containerisation for deployment

---

## 📜 License

This project is for educational purposes. The dataset is synthetic.

---

## 🙏 Acknowledgements

- Built with [TensorFlow](https://www.tensorflow.org/), [scikit-learn](https://scikit-learn.org/), and [Streamlit](https://streamlit.io/)
- Dataset is synthetic, generated for ANN regression training exercises
