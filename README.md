
# Indian Logistics Cost Prediction & Fleet Operations Intelligence

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Keras%20%7C%20PyTorch-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![Database](https://img.shields.io/badge/Database-MySQL-4479A1?logo=mysql)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end Machine Learning and Business Intelligence project for predicting truck transportation costs, analyzing logistics operations, and supporting fleet management decisions across Indian freight corridors.

Built using **Python, Pandas, Scikit-Learn, Artificial Neural Networks, Keras, PyTorch, SHAP, MySQL, Power BI, and Streamlit**.

> **Important: Synthetic Dataset Disclosure**
>
> This project uses approximately 15,000 synthetically generated truck transportation records across 16 Indian cities. The target variable is generated using a mathematical cost formula with simulated noise. Model performance on this dataset is for educational and technical demonstration purposes and does not represent validated real-world logistics performance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Project Objectives](#3-project-objectives)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture](#5-system-architecture)
6. [Dataset Description](#6-dataset-description)
7. [Data Preprocessing and Feature Engineering](#7-data-preprocessing-and-feature-engineering)
8. [Machine Learning Models](#8-machine-learning-models)
9. [Model Evaluation](#9-model-evaluation)
10. [Explainable AI using SHAP](#10-explainable-ai-using-shap)
11. [MySQL Data Warehouse](#11-mysql-data-warehouse)
12. [Power BI Dashboard](#12-power-bi-dashboard)
13. [Streamlit Application](#13-streamlit-application)
14. [Project Structure](#14-project-structure)
15. [Installation and Setup](#15-installation-and-setup)
16. [Running the Project](#16-running-the-project)
17. [Running Tests](#17-running-tests)
18. [Model Artifacts](#18-model-artifacts)
19. [Business Applications](#19-business-applications)
20. [Limitations and Future Improvements](#20-limitations-and-future-improvements)
21. [License](#21-license)

---

## 1. Project Overview

Indian logistics and transportation companies manage multiple vehicles, routes, fuel expenses, maintenance costs, and delivery operations.

Estimating transportation expenses before dispatch is important for controlling operational costs and making better fleet management decisions.

This project develops a Logistics Cost Prediction and Fleet Operations Intelligence platform that combines Machine Learning, Deep Learning, Explainable AI, SQL analytics, and Business Intelligence.

The application allows users to:

- Predict transportation costs for individual truck trips.
- Generate predictions for multiple trips using CSV uploads.
- Compare different Machine Learning and Deep Learning models.
- Understand important factors influencing predicted transportation costs.
- Analyze route-wise and vehicle-wise transportation expenses.
- Explore fleet operational data through interactive dashboards.
- Perform relational data analysis using MySQL.
- Visualize business KPIs through Power BI.

---

## 2. Business Problem

Transportation companies need to estimate trip costs before assigning trucks to delivery routes.

However, transportation expenses depend on several factors:

- Distance travelled.
- Truck type and loading capacity.
- Weight of transported goods.
- Fuel efficiency.
- Diesel prices.
- Toll charges.
- Driver allowances.
- Maintenance expenses.
- Route conditions.
- Traffic conditions.

An inaccurate cost estimate can affect freight quotations, operational budgets, and fleet profitability.

**Problem Statement:**

Develop a Machine Learning system that predicts estimated truck transportation costs using pre-dispatch trip information and provides analytical insights into fleet operations.

Target variable:

`actual_trip_cost_inr`

The model uses information available before dispatch rather than realized post-trip information.

---

## 3. Project Objectives

### Machine Learning
- Develop a regression-based transportation cost prediction system.
- Compare Linear Regression, Random Forest, HistGradientBoosting, and Artificial Neural Networks.
- Evaluate models using MAE, RMSE, and R².
- Save trained models for application-level inference.

### Data Science
- Perform data validation and exploratory analysis.
- Handle numerical and categorical features.
- Engineer meaningful logistics-related variables.
- Prevent target leakage from post-trip information.

### Explainable AI
- Use SHAP to understand model predictions.
- Identify important cost-driving features.
- Explain individual transportation cost predictions.

### Data Engineering and SQL
- Design a MySQL relational warehouse.
- Implement fact and dimension tables.
- Write analytical queries using joins, CTEs, aggregations, and window functions.

### Business Intelligence
- Develop an interactive Streamlit dashboard.
- Build Power BI reports for operational and management insights.

---

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Altair |
| Machine Learning | Scikit-Learn |
| Deep Learning | Keras, PyTorch |
| Explainable AI | SHAP |
| Model Serialization | Joblib |
| Database | MySQL |
| Database Integration | SQLAlchemy, MySQL Connector |
| Dashboard | Streamlit |
| Business Intelligence | Power BI |
| Testing | Pytest |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

## 5. System Architecture

```text
                 INDIAN LOGISTICS COST INTELLIGENCE
                              |
                              v
                     Synthetic Trip Dataset
                         (15,000 Rows)
                              |
                              v
                   Data Validation & Cleaning
                              |
                              v
                    Feature Engineering
                              |
                              v
                    Data Preprocessing
                 (ColumnTransformer)
                              |
                              v
                   Train / Validation / Test
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
       Linear Regression  Random Forest   HistGradientBoosting
             |                |                |
             +----------------+----------------+
                              |
                              v
                    Artificial Neural Network
                              |
                              v
                     Model Evaluation
                    MAE / RMSE / R2
                              |
                              v
                    Best Model Selection
                              |
               +--------------+--------------+
               |                             |
               v                             v
          SHAP Explainability          Model Artifacts
               |                             |
               +--------------+--------------+
                              |
                              v
                    Streamlit Application
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         Cost Prediction  Batch Prediction  Route Analytics
                              |
                              v
                       MySQL Warehouse
                              |
                              v
                      Power BI Dashboard
```

---

## 6. Dataset Description

The project uses a synthetic dataset containing approximately **15,000 truck transportation records**.

### Geographic Coverage

The dataset represents transportation routes involving 16 Indian cities:

- Ahmedabad
- Bengaluru
- Bhopal
- Chennai
- Delhi
- Hyderabad
- Indore
- Jaipur
- Kolkata
- Lucknow
- Mumbai
- Nagpur
- Nashik
- Pune
- Surat
- Vadodara

### Vehicle Categories

| Vehicle Type | Typical Capacity |
|---|---:|
| LCV | 3.5 Tons |
| Medium | 9 Tons |
| Heavy | 16 Tons |
| Trailer | 25 Tons |

### Important Dataset Features

| Feature | Description |
|---|---|
| distance_km | Transportation distance |
| truck_type | Vehicle category |
| truck_capacity_tons | Maximum vehicle capacity |
| load_weight_tons | Weight of transported goods |
| fuel_efficiency_km_per_litre | Vehicle fuel efficiency |
| diesel_price_inr_per_litre | Diesel price |
| toll_cost_inr | Toll expenses |
| driver_allowance_inr | Driver allowance |
| maintenance_cost_inr | Estimated maintenance expense |
| loading_unloading_cost_inr | Loading and unloading expense |
| other_cost_inr | Other operational expenses |
| route_factor | Route-related cost factor |
| traffic_factor | Traffic-related cost factor |
| month | Dispatch month |
| actual_trip_cost_inr | Target variable |

### Data Quality

The project includes checks for:

- Missing values.
- Duplicate records.
- Invalid distances.
- Invalid truck capacities.
- Negative numerical values.
- Vehicle overloading.
- Invalid vehicle categories.
- Unusual numerical observations.

---

## 7. Data Preprocessing and Feature Engineering

Data preprocessing is implemented in:

`src/data_utils.py`

Feature engineering is implemented in:

`src/feature_engineering.py`

### Data Preprocessing

The preprocessing pipeline handles numerical and categorical variables separately.

**Numerical Features**
- StandardScaler

**Categorical Features**
- OneHotEncoder

A Scikit-Learn ColumnTransformer combines these transformations into a single preprocessing workflow.

### Feature Engineering

The project creates additional variables to represent logistics operations more effectively.

**1. Load Utilization Percentage**

```python
load_utilization_pct = (
    load_weight_tons / truck_capacity_tons
) * 100
```

**2. Estimated Fuel Consumption**

```python
estimated_fuel_litres = (
    distance_km / fuel_efficiency_km_per_litre
)
```

**3. Estimated Fuel Cost**

```python
estimated_fuel_cost_inr = (
    estimated_fuel_litres * diesel_price_inr_per_litre
)
```

**4. Estimated Direct Expenses**

Combines estimated fuel cost, tolls, driver allowance, maintenance, handling, and other expenses.

### Target Leakage Prevention

The model is designed for pre-dispatch predictions.

Therefore, realized post-trip information is excluded from model inputs.

| Feature Type | Model Input |
|---|---|
| Distance and vehicle information | Included |
| Fuel and diesel information | Included |
| Route and traffic information | Included |
| Estimated pre-trip expenses | Included |
| Actual trip cost | Target only |
| Freight revenue | Excluded |
| Trip profit | Excluded |
| Actual cost per kilometre | Excluded |
| Profit margin | Excluded |

This separation is important because information generated after completing a trip should not be used to predict its cost before dispatch.

---

## 8. Machine Learning Models

The project compares multiple regression algorithms.

### 1. Linear Regression

Used as a baseline regression model to understand relationships between input features and transportation costs.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to capture nonlinear relationships and interactions between features.

### 3. HistGradientBoosting Regressor

A gradient boosting algorithm that builds decision trees sequentially to reduce prediction errors.

### 4. Artificial Neural Network

A feed-forward Artificial Neural Network built using Keras with a PyTorch backend.

The ANN learns complex nonlinear relationships between transportation parameters and trip costs.

The network uses Dense layers, Dropout regularization, an Adam optimizer, and validation-based training.

### Model Selection

The project compares models using validation or cross-validation performance rather than choosing a model based on the final held-out test score.

The final test set is reserved for model evaluation.

---

## 9. Model Evaluation

The project evaluates regression models using three standard metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted transportation costs.

```text
MAE = Average(|Actual Cost - Predicted Cost|)
```

Lower MAE indicates smaller average prediction errors.

### Root Mean Squared Error (RMSE)

Measures prediction errors while penalizing larger errors more heavily.

Lower RMSE indicates better performance.

### R-squared (R²)

Measures how much variation in the target variable is explained by the regression model.

A higher R² indicates a better fit to the evaluated dataset.

### Existing Saved Benchmark Results

The following values are from the benchmark artifact included in the project ZIP. They represent the previous training run, before the training-pipeline corrections described in this documentation.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Dummy Baseline | ₹16,170 | ₹19,688 | -0.0040 |
| Linear Regression | ₹3,635 | ₹4,993 | 0.9354 |
| Random Forest | ₹1,927 | ₹2,620 | 0.9822 |
| ANN | ₹1,298 | ₹1,722 | 0.9923 |
| HistGradientBoosting | ₹746 | ₹1,028 | 0.9973 |

**Important:** These are historical results, not a claim about the corrected training pipeline. Rerun training and replace this table with the newly generated results before presenting the updated metrics as final.

Because the dataset is synthetic and formula-generated, these metrics should not be interpreted as evidence of real-world transportation prediction accuracy.

---

## 10. Explainable AI using SHAP

The project integrates SHAP (SHapley Additive exPlanations) to make model predictions easier to interpret.

Implementation:

`src/explainability.py`

### Global Feature Importance

SHAP helps identify which features contribute most to model predictions across the dataset.

Examples include:
- Distance.
- Fuel efficiency.
- Estimated fuel expenses.
- Maintenance expenses.
- Traffic conditions.
- Route-related factors.

### Individual Prediction Explanation

For individual predictions, SHAP helps explain which input features increased or decreased the predicted transportation cost relative to the model's baseline output.

### Business Value

Instead of presenting only a predicted cost, the application can help users understand the factors influencing the estimate.

**Note:** SHAP explains model behaviour and feature contributions. It does not establish physical causality.

---

## 11. MySQL Data Warehouse

The project includes a MySQL warehouse for structured logistics analysis.

SQL scripts are available inside the `sql/` directory.

### Database Schema

The warehouse follows a star-schema approach.

**Dimension Tables**

`dim_truck_type`
- Truck type.
- Vehicle capacity.
- Vehicle category information.

`dim_route`
- Origin city.
- Destination city.
- Route information.

**Fact Table**

`fact_trip_cost`
- Trip-related measures.
- Route identifiers.
- Truck type identifiers.
- Trip month.
- Transportation cost information.

### SQL Concepts Implemented

The analytical SQL scripts demonstrate:

- INNER JOIN and LEFT JOIN.
- GROUP BY and HAVING.
- Aggregate functions.
- Common Table Expressions (CTEs).
- Window functions.
- RANK() and DENSE_RANK().
- CASE WHEN statements.
- Moving averages.
- Route-wise cost comparisons.
- Vehicle efficiency analysis.

### SQL Files

| File | Purpose |
|---|---|
| `01_create_database.sql` | Database creation |
| `02_create_tables.sql` | Warehouse table creation |
| `03_logistics_analysis.sql` | Analytical SQL queries |
| `04_load_data.sql` | Data loading instructions |

**Database safety:** Review the SQL scripts before running them. Table reset operations can remove existing data. The CSV loading script also requires the correct local project path and MySQL local-file loading permissions.

---

## 12. Power BI Dashboard

The project includes a Power BI dashboard:

`powerbi/dashboard.pbix`

The dashboard is intended to support management-level analysis of transportation operations.

Potential analytical views include:

- Total transportation expenses.
- Average cost per kilometre.
- Vehicle category comparisons.
- Route-wise cost analysis.
- Monthly transportation trends.
- Fleet expense breakdowns.
- Load utilization analysis.

The Power BI file can be opened using Microsoft Power BI Desktop.

---

## 13. Streamlit Application

The interactive application is built using Streamlit.

Main application:

`app.py`

The dashboard integrates the trained prediction models, data-processing functions, analytics, and explainability modules.

### Main Functional Areas

**1. Operations Overview**

Provides an overview of logistics and transportation metrics.

**2. AI Cost Prediction**

Allows users to enter truck and trip information and receive an estimated transportation cost.

**3. Batch Prediction**

Supports CSV-based batch prediction for multiple transportation records.

**4. Route Intelligence**

Provides route-related transportation analysis.

**5. Expense and Load Analysis**

Supports analysis of operating expenses and vehicle load utilization.

**6. Model Benchmarks and Explainability**

Provides model comparison and SHAP-related insights.

---

## 14. Project Structure

```text
indian-truck-cost-prediction/
│
├── data/
│   ├── indian_truck_logistics_synthetic_15000.csv
│   └── processed_logistics_data.csv
│
├── models/
│   ├── baseline_lr.pkl
│   ├── rf_model.pkl
│   ├── hgbr_model.pkl
│   ├── truck_cost_ann.keras
│   ├── preprocessor.pkl
│   ├── feature_schema.json
│   ├── evaluation_metrics.json
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── training_history.png
│
├── notebooks/
│   └── truck_cost_prediction.ipynb
│
├── powerbi/
│   └── dashboard.pbix
│
├── sql/
│   ├── 01_create_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_logistics_analysis.sql
│   └── 04_load_data.sql
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── explainability.py
│   ├── database.py
│   └── analytics.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_feature_engineering.py
│   ├── test_prediction.py
│   └── test_explainability.py
│
├── .env.example
├── .gitignore
├── .gitattributes
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
```

**Note:** The `models/` directory contains generated model artifacts. They are included in the repository when available so that the application can load trained models without requiring users to train them first.

---

## 15. Installation and Setup

### Prerequisites

- Python 3.11 recommended.
- Git.
- MySQL 8.0+ (optional, for database integration).
- Power BI Desktop (optional, for dashboard viewing).

### Step 1: Clone the Repository

```bash
git clone https://github.com/pawano6/indian-truck-cost-prediction.git

cd indian-truck-cost-prediction
```

### Step 2: Create a Virtual Environment

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a local `.env` file using the example template.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Update the MySQL configuration if database integration is required.

**Security:** Never upload `.env` or real database passwords to GitHub.

---

## 16. Running the Project

### Run Model Training

```bash
python -m src.train_model
```

This executes the model training and evaluation pipeline and generates model artifacts inside the `models/` directory.

### Launch Streamlit

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### MySQL Setup

If using the database integration:

1. Open MySQL.
2. Execute `sql/01_create_database.sql`.
3. Execute `sql/02_create_tables.sql`.
4. Configure the `.env` database credentials.
5. Review and execute `sql/04_load_data.sql` using your local dataset path.
6. Execute analytical queries from `sql/03_logistics_analysis.sql`.

Review destructive SQL statements before running database setup scripts.

---

## 17. Running Tests

The project uses Pytest for automated testing.

Run all tests:

```bash
python -m pytest tests/ -v
```

Check Python syntax:

```bash
python -m compileall src app.py
```

The test modules cover:

| Test File | Testing Area |
|---|---|
| `test_data_utils.py` | Data loading, validation, preprocessing |
| `test_feature_engineering.py` | Feature calculations and leakage prevention |
| `test_prediction.py` | Single and batch prediction |
| `test_explainability.py` | SHAP explanation functionality |

Run the tests after installing dependencies and before uploading code changes.

---

## 18. Model Artifacts

The trained models and supporting artifacts are stored in `models/`.

| File | Purpose |
|---|---|
| `baseline_lr.pkl` | Linear Regression model |
| `rf_model.pkl` | Random Forest model |
| `hgbr_model.pkl` | HistGradientBoosting model |
| `truck_cost_ann.keras` | Artificial Neural Network |
| `preprocessor.pkl` | Fitted preprocessing object |
| `feature_schema.json` | Feature and schema information |
| `evaluation_metrics.json` | Model evaluation results |
| `model_comparison.png` | Model comparison visualization |
| `actual_vs_predicted.png` | Prediction diagnostic visualization |
| `training_history.png` | ANN training history |

The model artifacts must remain compatible with the application code.

If you retrain the models, test prediction and explainability functionality before committing the newly generated artifacts.

---

## 19. Business Applications

Although this project uses synthetic data, its workflow demonstrates concepts relevant to logistics and supply chain analytics.

### Transportation Cost Estimation
Estimate trip expenses before dispatch using vehicle, route, fuel, and load information.

### Fleet Cost Analysis
Compare expenses across vehicle categories and transportation routes.

### Operational Budgeting
Analyze expected transportation expenses and identify high-cost operating scenarios.

### Route Intelligence
Compare cost patterns across origin-destination corridors.

### Explainable Decision Support
Understand which input factors contribute to model predictions.

### Management Reporting
Present operational metrics through Power BI and interactive dashboards.

---

## 20. Limitations and Future Improvements

### Current Limitations

1. **Synthetic Dataset:** The dataset is simulated and does not represent actual commercial fleet transactions.
2. **Formula-Generated Target:** The target variable is generated using a mathematical formula with simulated noise.
3. **Generalization:** Random train/test splitting on synthetic records may not reflect performance on genuinely unseen routes, fleets, or operating conditions.
4. **Limited Time Information:** The dataset uses month-level information rather than complete trip timestamps.
5. **External Data:** Live fuel prices, GPS information, weather, toll changes, and actual traffic APIs are not integrated.

### Future Improvements

- Integrate real-world transportation datasets.
- Introduce scenario-based and group-based model validation.
- Incorporate live diesel prices.
- Integrate GPS and route-distance APIs.
- Include weather and traffic information.
- Improve model monitoring and retraining workflows.
- Deploy the application using a production hosting platform.
- Extend the warehouse with additional fleet and driver dimensions.
- Develop cost anomaly detection for operational expenses.

---

## 21. License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## Author

**Pawan Singh**

GitHub: [github.com/pawano6](https://github.com/pawano6)

Project Repository:

[Indian Logistics Cost Prediction & Fleet Operations Intelligence](https://github.com/pawano6/indian-truck-cost-prediction)
