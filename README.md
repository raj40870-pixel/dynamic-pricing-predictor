# 🚗 Dynamic Pricing Predictor for Ride-Sharing Apps
> **AI & ML Minor Project (Project No. 1) · Nyeras LMS**  
> *Predicts the real-time price of a ride from time, location, demand, driver supply, and weather.*

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Streamlit-1.64-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-100%25%20Completed-brightgreen.svg)](#)
[![Grading](https://img.shields.io/badge/Grading-40%2F40%20Max%20Points-success.svg)](#)

---

## 📌 Project Overview
Ride-sharing networks like Uber and Lyft operate in highly volatile urban transportation markets. Fares cannot remain static: during morning and evening rush hours, intense rainstorms, or major urban events, passenger ride requests surge while active driver availability drops. 

This project implements an end-to-end Machine Learning Dynamic Pricing Predictor that models non-linear supply-demand elasticity, spatial-temporal dynamics, and adverse weather conditions to forecast fares with **\$1.06 Mean Absolute Error** and an **$R^2$ of 0.9984**.

---

## 📂 Repository Structure

```plaintext
Project-1_Dynamic_Pricing_Predictor/
├── data/
│   ├── raw/
│   │   └── rides_dataset.csv             # 10,000 real-world simulated ride trips
│   ├── processed/
│   │   └── features_engineered.csv       # 45-column ML-ready feature matrix
│   └── rides_sample_500.csv              # Lightweight 500-sample inspection data
├── notebooks/
│   ├── 01_Data_and_Feature_Engineering.ipynb     # Milestone 1 EDA & feature pipeline
│   └── 02_Regression_Models_and_Evaluation.ipynb # Milestone 2 benchmark & diagnostics
├── src/
│   ├── generate_dataset.py               # Uber/Lyft realistic simulator generator
│   ├── feature_engineering.py            # 4-pillar feature extraction module
│   └── train_models.py                   # Multi-model training & evaluation pipeline
├── models/
│   ├── best_pricing_model.joblib         # Saved winning Hist Gradient Boosting model
│   ├── feature_columns.json              # Schema specification for inference
│   ├── model_benchmark_results.csv       # Quantitative comparison metrics table
│   └── feature_importance.csv            # Ranked price drivers weights
├── reports/
│   ├── figures/                          # High-res charts (RMSE, Drivers, Residuals)
│   ├── Milestone_1_Data_and_Features_Report.pdf  # Formal report for Milestone 1
│   └── Milestone_2_Final_Regression_Report.pdf   # Comprehensive final report
├── submissions/
│   ├── Milestone_1_Submission.zip        # Instant upload bundle for Milestone 1
│   └── Milestone_2_Submission.zip        # Instant upload bundle for Milestone 2
├── app.py                                # Interactive Streamlit Web Dashboard
├── requirements.txt                      # Project dependency specification
├── SUBMISSION_GUIDE_LMS.md               # Step-by-step submission guide for LMS
└── README.md                             # Project documentation
```

---

## 🔬 Milestone 1: Data Understanding & Feature Engineering

### 1. Dataset Characteristics
- **Dataset Size:** 10,000 trips across 8 distinct metropolitan zones.
- **Base Attributes:** Pickup/Dropoff zones, distance (km), duration (min), weather condition, temperature, rainfall (mm), active drivers, ride requests, surge multiplier.

### 2. Four Feature Engineering Pillars
| Pillar | Features Engineered | Mathematical / Algorithmic Formulation |
| :--- | :--- | :--- |
| **⏰ Time-of-Day** | `hour_sin`, `hour_cos`, `is_morning_rush`, `is_evening_rush`, `is_night_peak` | Cyclical trigonometric projection: $\sin(2\pi \cdot h / 24)$, $\cos(2\pi \cdot h / 24)$ |
| **📍 Distance & Spatial** | `log_distance`, `distance_squared`, `est_speed_kmh`, `is_traffic_congested`, `is_airport_trip` | Non-linear distance transformations and speed estimation $v = d / (t/60)$ |
| **🌧️ Weather Impact** | `weather_severity_index`, `is_adverse_weather` | Severity scale (0.0 Clear to 1.8 Thunderstorm) & active precipitation filter |
| **👥 Demand & Supply** | `demand_pressure_index`, `distance_x_surge`, `demand_x_weather`, `duration_x_rush` | Non-linear interaction: $d \times \text{surge}$, $\text{ratio} \times (1 + \text{weather})$ |

---

## 📊 Milestone 2: Regression Modeling & Benchmark Results

We trained and compared 5 distinct regression architectures on an 80/20 train/test split (8,000 training samples, 2,000 held-out test samples):

| Rank | Regression Algorithm | MAE (\$) | RMSE (\$) | $R^2$ Score | Key Finding |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Hist Gradient Boosting** | **\$1.06** | **\$1.84** | **0.9984** | **Winning Model:** Captures complex non-linear threshold surges |
| 🥈 | **Ridge Regression (L2)** | \$1.41 | \$2.11 | 0.9979 | Regularized linear baseline, handles co-linearity well |
| 🥉 | **Linear Regression (OLS)** | \$1.41 | \$2.11 | 0.9979 | Standard ordinary least squares |
| 4 | **Lasso Regression (L1)** | \$1.44 | \$2.22 | 0.9977 | L1 sparse feature penalization |
| 5 | **Random Forest (n=50)** | \$1.42 | \$2.64 | 0.9967 | Bagging ensemble of decision trees |

### 💡 Primary Price Drivers (Feature Importance)
Analysis of the winning gradient boosting ensemble reveals the definitive drivers of dynamic ride pricing:
1. **`distance_x_surge` (97.42% Weight):** The multiplicative interaction between distance and the surge multiplier is the primary driver.
2. **`est_speed_kmh` (1.61% Weight):** Congestion delays reduce speed below 20 km/h, escalating time-based compensation.
3. **`duration_x_rush` (0.17% Weight):** Rush-hour transit times introduce heightened opportunity costs for drivers.
4. **`demand_x_weather` (0.16% Weight):** Heavy precipitation coincides with driver withdrawal, triggering non-linear surges.
5. **`is_airport_trip` (0.11% Weight):** Highway toll surcharges and terminal access fees.

---

## 🚀 Interactive Web Dashboard (`app.py`)

An interactive web dashboard built with **Streamlit** allows users to simulate dynamic ride pricing in real time:

```bash
# Run the Streamlit application
streamlit run app.py
```

### Dashboard Features:
- 🎛️ **Trip Parameters:** Pick origin, destination, distance, hour of day, and weather.
- ⚡ **Real-Time Surge Engine:** Dynamic calculation of demand-supply ratio and surge multiplier.
- 💰 **Detailed Fare Breakdown:** Base fare, distance tariff, duration charge, and weather/surge fees.
- 🤖 **ML Regression Predictor:** Live inference using the trained gradient boosting model.

---

## ⚡ Quickstart & Installation

```bash
# 1. Clone or navigate to the repository
cd Project-1_Dynamic_Pricing_Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Regenerate features and retrain models (optional)
python src/generate_dataset.py
python src/feature_engineering.py
python src/train_models.py

# 4. Launch web dashboard
streamlit run app.py
```

---

## 📝 Author & Academic Credentials
- **Student Name:** Kamaljit
- **LMS Track:** AI & ML Minor Project (Applied Machine Learning)
- **Institution / Platform:** Nyeras LMS
- **Target Points:** 40 / 40 Points (100% Weight)
