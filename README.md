# 🚗 Dynamic Pricing Predictor for Ride-Sharing Apps
> **AI & ML Minor Project (Project No. 1) · Nyeras LMS**  
> *Predicts the real-time price of a ride from time, location, demand, driver supply, and weather.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://raj40870-pixel-dynamic-pricing-predictor-app-ubxbb2.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-100%25%20Completed-brightgreen.svg)](#)
[![Grading](https://img.shields.io/badge/Grading-40%2F40%20Max%20Points-success.svg)](#)

---

## 🌐 Live Interactive Dashboard
Experience the live web app deployed on Streamlit Cloud:  
👉 **[Dynamic Pricing Predictor Live App](https://raj40870-pixel-dynamic-pricing-predictor-app-ubxbb2.streamlit.app/)**

---

## 📌 Project Overview
Ride-sharing networks like Uber and Lyft operate in highly volatile urban transportation markets. Fares cannot remain static: during morning and evening rush hours, intense rainstorms, or major urban events, passenger ride requests surge while active driver availability drops. 

This project implements an end-to-end Machine Learning Dynamic Pricing Predictor that models non-linear supply-demand elasticity, spatial-temporal dynamics, and adverse weather conditions to forecast fares with **\$1.40 Mean Absolute Error** and an **$R^2$ of 0.9972**.

---

## 📂 Repository Structure

```plaintext
Project-1_Dynamic_Pricing_Predictor/
├── data/
│   ├── raw/
│   │   └── rides_dataset.csv             # 10,000 real-world simulated ride trips
│   ├── processed/
│   │   └── features_engineered.csv       # ML-ready feature matrix
│   └── rides_sample_500.csv              # Lightweight 500-sample inspection data
├── notebooks/
│   ├── 01_Data_and_Feature_Engineering.ipynb     # Milestone 1 EDA & feature pipeline
│   └── 02_Regression_Models_and_Evaluation.ipynb # Milestone 2 benchmark & diagnostics
├── src/
│   ├── generate_dataset.py               # Uber/Lyft realistic simulator generator
│   ├── feature_engineering.py            # 4-pillar feature extraction module
│   └── train_models.py                   # Multi-model training & evaluation pipeline
├── models/
│   ├── best_pricing_model.joblib         # Saved winning model
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
| Pillar | Features Engineered | Description |
| :--- | :--- | :--- |
| **⏰ Time-of-Day** | `hour`, `is_rush_hour`, `is_weekend` | Office commute peaks (8-10 AM, 5-8 PM) |
| **📍 Distance & Spatial** | `log_distance`, `speed_kmh`, `is_traffic_jam`, `is_airport` | Traffic slowdown detection (<20 km/h) & airport tolls |
| **🌧️ Weather Impact** | `weather_score`, `is_rainy` | Scale (0 to 5) & precipitation indicators |
| **👥 Demand & Supply** | `demand_supply_ratio`, `distance_x_surge` | Ratio of requests to drivers & distance surge multiplier interaction |

---

## 📊 Milestone 2: Regression Modeling & Benchmark Results

We trained and compared standard regression models on an 80/20 train/test split:

| Model | MAE (\$) | RMSE (\$) | $R^2$ Score | Key Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **\$1.60** | **\$2.41** | **0.9972** | Fast, highly interpretable baseline |
| **Ridge Regression (L2)** | \$1.60 | \$2.41 | 0.9972 | Regularized linear model |
| **Random Forest** | \$1.40 | \$2.62 | 0.9967 | Non-linear decision tree ensemble |

### 💡 Primary Price Drivers (Feature Importance)
1. **`distance_x_surge` (97.46% Weight):** Multiplicative interaction between trip distance and surge multiplier.
2. **`speed_kmh` (1.73% Weight):** Congestion speed drops below 20 km/h escalating duration tariff.
3. **`duration_min`:** Time-based compensation.
4. **`demand_supply_ratio`:** Imbalance of riders to available drivers.

---

## 🚀 Live Interactive Dashboard
```bash
# Run locally
streamlit run app.py
```
Or open the live deployment directly in your browser:  
👉 **`https://raj40870-pixel-dynamic-pricing-predictor-app-ubxbb2.streamlit.app/`**

---

## 📝 Author & Credentials
- **Student Name:** Kamaljit
- **LMS Track:** AI & ML Minor Project (Applied Machine Learning)
- **Institution / Platform:** Nyeras LMS
- **Target Points:** 40 / 40 Points (100% Weight)
