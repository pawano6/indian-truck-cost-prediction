"""
app.py — Streamlit Dashboard for Indian Truck Trip Cost Prediction

A polished, professional Streamlit application that loads the trained
ANN model and preprocessing pipeline, accepts user inputs for trip
parameters, and predicts the operating cost in INR.

DISCLAIMER: This application uses a SYNTHETIC educational dataset.
Predictions do not reflect actual freight industry costs.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🚛 Indian Truck Trip Cost Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a professional logistics theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a40 50%, #24243e 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #ff6b35 0%, #f7c948 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        padding: 10px 0;
    }

    .sub-header {
        text-align: center;
        color: #a0a0c0;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 107, 53, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 107, 53, 0.6);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff6b35;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        margin-top: 5px;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(247,201,72,0.1));
        border: 2px solid #ff6b35;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    .prediction-amount {
        font-size: 3rem;
        font-weight: 800;
        color: #f7c948;
    }

    .prediction-label {
        font-size: 1.2rem;
        color: #ff6b35;
        margin-bottom: 10px;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #e0d090;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a40 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 107, 53, 0.2);
    }

    /* Expense table */
    .expense-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .expense-label {
        color: #c0c0e0;
    }

    .expense-value {
        color: #ff6b35;
        font-weight: 600;
    }

    /* Divider */
    .custom-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,107,53,0.4), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants / paths (anchored to script directory for robust resolution)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "truck_cost_ann.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.json")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "indian_truck_logistics_synthetic_15000.csv")

# City → State mapping from the dataset

CITY_STATE_MAP = {
    "Ahmedabad": "Gujarat",
    "Bengaluru": "Karnataka",
    "Bhopal": "Madhya Pradesh",
    "Chennai": "Tamil Nadu",
    "Delhi": "Delhi",
    "Hyderabad": "Telangana",
    "Indore": "Madhya Pradesh",
    "Jaipur": "Rajasthan",
    "Kolkata": "West Bengal",
    "Lucknow": "Uttar Pradesh",
    "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra",
    "Pune": "Maharashtra",
    "Surat": "Gujarat",
    "Vadodara": "Gujarat",
}


TRUCK_TYPES = ["Heavy", "Medium", "LCV", "Trailer"]

TRUCK_CAPACITY_DEFAULTS = {
    "Heavy": 16.0,
    "Medium": 9.0,
    "LCV": 3.5,
    "Trailer": 25.0,
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    from tensorflow import keras
    if not os.path.isfile(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    """Load the fitted preprocessing pipeline."""
    if not os.path.isfile(PREPROCESSOR_PATH):
        return None
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_schema():
    """Load the feature schema."""
    if not os.path.isfile(SCHEMA_PATH):
        return None
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    """Load evaluation metrics."""
    if not os.path.isfile(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_data():
    """Load the dataset for exploration."""
    if not os.path.isfile(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-header">🚛 Indian Truck Trip Cost Predictor</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">ANN-Powered Fleet Expense Analytics Dashboard</p>',
            unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Synthetic Data Disclosure:</strong> This application uses a
    synthetic educational dataset (≈15,000 simulated truck trips). Predictions
    are for learning and demonstration purposes only. They do not represent
    actual freight costs, real company data, or government statistics.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

model = load_model()
preprocessor = load_preprocessor()
schema = load_schema()
metrics = load_metrics()
dataset = load_data()

# Check if model artefacts exist
artefacts_ok = all([model is not None, preprocessor is not None, schema is not None])

if not artefacts_ok:
    st.error(
        "⚠️ **Model artefacts not found!** Please train the model first by running:\n\n"
        "```powershell\n"
        "python -m src.train_model\n"
        "```\n\n"
        "Expected files in `models/`:\n"
        "- `truck_cost_ann.keras`\n"
        "- `preprocessor.pkl`\n"
        "- `feature_schema.json`"
    )

# ---------------------------------------------------------------------------
# Sidebar — Trip Input Form
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🗺️ Trip Parameters")
    st.markdown("Configure the trip details below:")

    cities = sorted(CITY_STATE_MAP.keys())

    origin_city = st.selectbox("🏙️ Origin City", cities, index=cities.index("Mumbai"))
    origin_state = CITY_STATE_MAP.get(origin_city, "Unknown")
    st.caption(f"State: {origin_state}")

    destination_city = st.selectbox(
        "🏙️ Destination City", cities, index=cities.index("Delhi")
    )
    destination_state = CITY_STATE_MAP.get(destination_city, "Unknown")
    st.caption(f"State: {destination_state}")

    st.markdown("---")
    st.markdown("### 🚚 Vehicle & Load")

    truck_type = st.selectbox("Truck Type", TRUCK_TYPES, index=0)
    default_capacity = TRUCK_CAPACITY_DEFAULTS.get(truck_type, 10.0)

    distance_km = st.number_input(
        "Distance (km)", min_value=50, max_value=4000, value=800, step=50
    )
    truck_capacity_tons = st.number_input(
        "Truck Capacity (tons)", min_value=1.0, max_value=40.0,
        value=default_capacity, step=0.5
    )
    load_weight_tons = st.number_input(
        "Load Weight (tons)", min_value=0.5, max_value=40.0,
        value=min(default_capacity * 0.7, 10.0), step=0.5
    )

    st.markdown("---")
    st.markdown("### ⛽ Fuel & Costs")

    fuel_efficiency = st.number_input(
        "Fuel Efficiency (km/litre)", min_value=1.0, max_value=15.0,
        value=4.5, step=0.1
    )
    diesel_price = st.number_input(
        "Diesel Price (₹/litre)", min_value=60.0, max_value=150.0,
        value=92.0, step=0.5
    )
    toll_cost = st.number_input(
        "Toll Cost (₹)", min_value=0.0, max_value=10000.0,
        value=500.0, step=50.0
    )
    driver_allowance = st.number_input(
        "Driver Allowance (₹)", min_value=0.0, max_value=10000.0,
        value=800.0, step=50.0
    )
    maintenance_cost = st.number_input(
        "Maintenance Cost (₹)", min_value=0.0, max_value=15000.0,
        value=2000.0, step=100.0
    )
    loading_unloading_cost = st.number_input(
        "Loading/Unloading Cost (₹)", min_value=0.0, max_value=15000.0,
        value=2000.0, step=100.0
    )
    other_cost = st.number_input(
        "Other Costs (₹)", min_value=0.0, max_value=10000.0,
        value=500.0, step=50.0
    )

    st.markdown("---")
    st.markdown("### 📊 Route & Traffic")

    route_factor = st.slider(
        "Route Factor", min_value=0.8, max_value=1.5, value=1.1, step=0.01,
        help="Multiplier for route difficulty (1.0 = standard)"
    )
    traffic_factor = st.slider(
        "Traffic Factor", min_value=0.8, max_value=1.5, value=1.1, step=0.01,
        help="Multiplier for traffic conditions (1.0 = low traffic)"
    )
    month = st.selectbox(
        "Month", list(range(1, 13)),
        format_func=lambda m: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ][m - 1],
        index=0,
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Trip Cost", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

def make_prediction(
    model, preprocessor, schema,
    origin_city, origin_state, destination_city, destination_state,
    distance_km, truck_type, truck_capacity_tons, load_weight_tons,
    fuel_efficiency, diesel_price, toll_cost, driver_allowance,
    maintenance_cost, loading_unloading_cost, other_cost,
    route_factor, traffic_factor, month,
):
    """Construct input DataFrame, preprocess, and predict."""
    input_data = pd.DataFrame([{
        "origin_city": origin_city,
        "origin_state": origin_state,
        "destination_city": destination_city,
        "destination_state": destination_state,
        "distance_km": distance_km,
        "truck_type": truck_type,
        "truck_capacity_tons": truck_capacity_tons,
        "load_weight_tons": load_weight_tons,
        "fuel_efficiency_km_per_litre": fuel_efficiency,
        "diesel_price_inr_per_litre": diesel_price,
        "toll_cost_inr": toll_cost,
        "driver_allowance_inr": driver_allowance,
        "maintenance_cost_inr": maintenance_cost,
        "loading_unloading_cost_inr": loading_unloading_cost,
        "other_cost_inr": other_cost,
        "route_factor": route_factor,
        "traffic_factor": traffic_factor,
        "month": month,
    }])

    # Ensure column order matches training
    feature_cols = schema["all_feature_columns"]
    input_data = input_data[feature_cols]

    # Transform
    processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed, verbose=0).flatten()[0]
    return float(prediction), input_data


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 2])

with col1:
    if predict_btn:
        if not artefacts_ok:
            st.error("Cannot predict — model artefacts are missing. Train the model first.")
        elif load_weight_tons > truck_capacity_tons:
            st.warning("⚠️ Load weight exceeds truck capacity! Please adjust inputs.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    predicted_cost, input_df = make_prediction(
                        model, preprocessor, schema,
                        origin_city, origin_state, destination_city, destination_state,
                        distance_km, truck_type, truck_capacity_tons, load_weight_tons,
                        fuel_efficiency, diesel_price, toll_cost, driver_allowance,
                        maintenance_cost, loading_unloading_cost, other_cost,
                        route_factor, traffic_factor, month,
                    )

                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-label">Estimated Trip Operating Cost</div>
                        <div class="prediction-amount">₹ {predicted_cost:,.2f}</div>
                        <div style="color: #a0a0c0; font-size: 0.85rem; margin-top: 8px;">
                            {origin_city} → {destination_city} · {distance_km} km · {truck_type}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Fuel estimate
                    est_fuel_litres = distance_km / fuel_efficiency
                    est_fuel_cost = est_fuel_litres * diesel_price

                    # Expense breakdown
                    st.markdown("### 📋 Expense Assumptions Breakdown")

                    expenses = {
                        "⛽ Estimated Fuel Cost": est_fuel_cost,
                        "🛣️ Toll Cost": toll_cost,
                        "👨‍✈️ Driver Allowance": driver_allowance,
                        "🔧 Maintenance": maintenance_cost,
                        "📦 Loading/Unloading": loading_unloading_cost,
                        "📝 Other Costs": other_cost,
                    }

                    expense_df = pd.DataFrame(
                        list(expenses.items()), columns=["Component", "Amount (₹)"]
                    )
                    expense_df["Amount (₹)"] = expense_df["Amount (₹)"].apply(
                        lambda x: f"₹ {x:,.2f}"
                    )

                    st.table(expense_df.set_index("Component"))

                    total_direct = sum(expenses.values())
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.03);
                                border-radius: 8px; margin-top: 10px;">
                        <span style="color: #a0a0c0;">Sum of entered expenses: </span>
                        <span style="color: #f7c948; font-weight: 700; font-size: 1.2rem;">
                            ₹ {total_direct:,.2f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Fuel calculation detail
                    st.markdown("### ⛽ Fuel Estimate Calculation")
                    st.markdown(f"""
                    | Parameter | Value |
                    |-----------|-------|
                    | Distance | {distance_km} km |
                    | Fuel Efficiency | {fuel_efficiency} km/litre |
                    | Diesel Price | ₹{diesel_price}/litre |
                    | **Estimated Litres** | **{est_fuel_litres:,.1f} L** |
                    | **Estimated Fuel Cost** | **₹ {est_fuel_cost:,.2f}** |
                    """)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
    else:
        st.info("👈 Configure trip parameters in the sidebar and click **Predict Trip Cost**.")

        # Quick overview
        if dataset is not None:
            st.markdown("### 📊 Dataset Quick Overview")
            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
            with overview_col1:
                st.metric("Total Trips", f"{len(dataset):,}")
            with overview_col2:
                st.metric("Avg Cost", f"₹{dataset['actual_trip_cost_inr'].mean():,.0f}")
            with overview_col3:
                st.metric("Cities", f"{dataset['origin_city'].nunique()}")
            with overview_col4:
                st.metric("Truck Types", f"{dataset['truck_type'].nunique()}")


with col2:
    # Model performance metrics
    st.markdown("### 🎯 Model Performance")

    if metrics:
        ann = metrics.get("ann", {})
        baseline = metrics.get("baseline_dummy_mean", {})
        lr = metrics.get("baseline_linear_regression", {})

        tab1, tab2, tab3 = st.tabs(["ANN", "Linear Regression", "Baseline (Mean)"])

        with tab1:
            if ann:
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"₹{ann['mae']:,.0f}")
                m2.metric("RMSE", f"₹{ann['rmse']:,.0f}")
                m3.metric("R²", f"{ann['r2']:.4f}")

        with tab2:
            if lr:
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"₹{lr['mae']:,.0f}")
                m2.metric("RMSE", f"₹{lr['rmse']:,.0f}")
                m3.metric("R²", f"{lr['r2']:.4f}")

        with tab3:
            if baseline:
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"₹{baseline['mae']:,.0f}")
                m2.metric("RMSE", f"₹{baseline['rmse']:,.0f}")
                m3.metric("R²", f"{baseline['r2']:.4f}")

        note = metrics.get("note", "")
        if note:
            st.caption(f"ℹ️ {note}")
    else:
        st.info("Train the model to see evaluation metrics.")

    # Training plots
    st.markdown("### 📈 Training Visualisations")
    history_plot = os.path.join(MODEL_DIR, "training_history.png")
    avp_plot = os.path.join(MODEL_DIR, "actual_vs_predicted.png")

    if os.path.isfile(history_plot):
        st.image(history_plot, caption="Training & Validation Loss", use_container_width=True)
    if os.path.isfile(avp_plot):
        st.image(avp_plot, caption="Actual vs. Predicted", use_container_width=True)

    if not os.path.isfile(history_plot) and not os.path.isfile(avp_plot):
        st.info("Training plots will appear after model training.")


# ---------------------------------------------------------------------------
# Data exploration section
# ---------------------------------------------------------------------------

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("### 🔍 Data Exploration")

if dataset is not None:
    exp_tab1, exp_tab2, exp_tab3 = st.tabs([
        "📋 Sample Data", "📊 Cost Distribution", "🗺️ Routes Analysis"
    ])

    with exp_tab1:
        st.markdown("**First 10 rows of the dataset** (synthetic educational data)")
        display_cols = [
            "origin_city", "destination_city", "distance_km", "truck_type",
            "load_weight_tons", "actual_trip_cost_inr"
        ]
        st.dataframe(
            dataset[display_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

    with exp_tab2:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#1a1a40')

        sns.histplot(
            dataset["actual_trip_cost_inr"], bins=50, kde=True,
            color="#ff6b35", ax=ax, edgecolor="none", alpha=0.7
        )
        ax.set_xlabel("Trip Cost (₹)", color="white", fontsize=12)
        ax.set_ylabel("Frequency", color="white", fontsize=12)
        ax.set_title("Distribution of Actual Trip Cost", color="white", fontsize=14)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        st.pyplot(fig)

    with exp_tab3:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor('#0f0c29')
        ax2.set_facecolor('#1a1a40')

        avg_cost = dataset.groupby("truck_type")["actual_trip_cost_inr"].mean().sort_values()
        bars = ax2.barh(avg_cost.index, avg_cost.values, color=["#ff6b35", "#f7c948", "#4ecdc4", "#a855f7"])
        ax2.set_xlabel("Average Trip Cost (₹)", color="white", fontsize=12)
        ax2.set_title("Average Cost by Truck Type", color="white", fontsize=14)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_color("#333")
        ax2.bar_label(bars, fmt="₹{:,.0f}", color="white", fontsize=10, padding=5)
        st.pyplot(fig2)
        plt.close("all")
else:
    st.warning("Dataset not found. Place `indian_truck_logistics_synthetic_15000.csv` in `data/`.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #505070; font-size: 0.8rem; padding: 20px 0;">
    🚛 Indian Truck Trip Cost Predictor · ANN Regression Project<br>
    Built with TensorFlow/Keras · Streamlit · scikit-learn<br>
    <strong>Synthetic data for educational purposes only</strong>
</div>
""", unsafe_allow_html=True)
