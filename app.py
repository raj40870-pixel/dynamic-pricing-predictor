"""
Streamlit Web Dashboard: Dynamic Pricing Predictor for Ride-Sharing Apps
Author: Kamaljit
Project: AI & ML Minor Project - Dynamic Pricing Predictor
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json

st.set_page_config(
    page_title="Dynamic Pricing Predictor | Ride-Sharing IDE",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3799;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .surge-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Locations list
LOCATIONS = [
    "Downtown Financial District",
    "Midtown Center",
    "International Airport",
    "Tech Park / University",
    "Westside Residential",
    "East Riverfront",
    "North Suburbs",
    "South Bay District"
]

LOC_COORDS = {
    "Downtown Financial District": (0, 0),
    "Midtown Center": (2, 3),
    "International Airport": (18, 12),
    "Tech Park / University": (8, -4),
    "Westside Residential": (-5, 6),
    "East Riverfront": (4, 1),
    "North Suburbs": (1, 15),
    "South Bay District": (-3, -12)
}

# Header Section
st.markdown('<div class="main-header">🚗 Dynamic Pricing Predictor for Ride-Sharing Apps</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Machine Learning Fare Prediction based on Real-Time Demand, Supply, Location, and Weather Conditions</div>', unsafe_allow_html=True)

# Sidebar - Ride Trip Parameters
st.sidebar.header("🛠️ Ride Trip Configuration")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    pickup = st.selectbox("Pickup Zone", LOCATIONS, index=0)
with col_sb2:
    dropoff = st.selectbox("Dropoff Zone", LOCATIONS, index=2)

# Distance calculation based on locations
p1 = LOC_COORDS[pickup]
p2 = LOC_COORDS[dropoff]
default_dist = round(max(2.5, np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) * 1.15), 1)

distance_km = st.sidebar.slider("Trip Distance (km)", min_value=1.0, max_value=45.0, value=float(default_dist), step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⏰ Temporal Factors")
hour = st.sidebar.slider("Hour of Day (24h)", min_value=0, max_value=23, value=18, step=1)
is_weekend = st.sidebar.checkbox("Is Weekend (Saturday/Sunday)", value=False)

is_morning_rush = (7 <= hour <= 9 and not is_weekend)
is_evening_rush = (17 <= hour <= 20 and not is_weekend)
is_rush_hour = is_morning_rush or is_evening_rush

st.sidebar.markdown("---")
st.sidebar.header("🌧️ Weather Conditions")
weather = st.sidebar.selectbox("Current Weather", ["Clear", "Cloudy", "Fog", "Rain", "Heavy Rain", "Thunderstorm"], index=0)

temp_map = {"Clear": 26.0, "Cloudy": 22.0, "Fog": 16.0, "Rain": 19.0, "Heavy Rain": 18.0, "Thunderstorm": 17.0}
precip_map = {"Clear": 0.0, "Cloudy": 0.0, "Fog": 0.0, "Rain": 2.5, "Heavy Rain": 12.0, "Thunderstorm": 22.0}

temperature_c = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=42.0, value=temp_map[weather])
precipitation_mm = st.sidebar.slider("Precipitation (mm)", min_value=0.0, max_value=35.0, value=precip_map[weather])

st.sidebar.markdown("---")
st.sidebar.header("👥 Market Demand & Driver Supply")
active_drivers = st.sidebar.slider("Active Drivers in Area", min_value=10, max_value=120, value=35)
ride_requests = st.sidebar.slider("Ride Requests in Area", min_value=10, max_value=160, value=65)

# Calculate dynamic surge
demand_supply_ratio = round(ride_requests / max(1, active_drivers), 2)

if demand_supply_ratio < 0.9:
    surge_multiplier = 1.0
elif demand_supply_ratio < 1.3:
    surge_multiplier = round(1.0 + (demand_supply_ratio - 0.9) * 0.5, 2)
elif demand_supply_ratio < 2.0:
    surge_multiplier = round(1.2 + (demand_supply_ratio - 1.3) * 0.8, 2)
else:
    surge_multiplier = round(min(3.5, 1.76 + (demand_supply_ratio - 2.0) * 0.9), 2)

# Estimate duration based on speed
speed_kmh = 32.0
if is_rush_hour:
    speed_kmh *= 0.65
if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
    speed_kmh *= 0.78
if weather == "Fog":
    speed_kmh *= 0.85

duration_min = round(max(5.0, (distance_km / speed_kmh) * 60 + 2.0), 1)

# Base fare formula
base_fare = 3.50
per_km_rate = 1.65
per_minute_rate = 0.35
booking_fee = 2.00
airport_surcharge = 5.00 if ("Airport" in pickup or "Airport" in dropoff) else 0.0

standard_subtotal = base_fare + (distance_km * per_km_rate) + (duration_min * per_minute_rate) + booking_fee + airport_surcharge

weather_fee = 0.0
if weather == "Heavy Rain":
    weather_fee = 2.50
elif weather == "Thunderstorm":
    weather_fee = 4.00
elif weather == "Rain":
    weather_fee = 1.50

computed_fare = round(standard_subtotal * surge_multiplier + weather_fee, 2)
standard_base_fare = round(standard_subtotal, 2)

# Load trained ML model if available
model_path = os.path.join(os.path.dirname(__file__), "models", "best_pricing_model.joblib")
cols_path = os.path.join(os.path.dirname(__file__), "models", "feature_columns.json")

ml_predicted_fare = None
if os.path.exists(model_path) and os.path.exists(cols_path):
    try:
        model = joblib.load(model_path)
        with open(cols_path, "r") as f:
            feat_cols = json.load(f)

        # Build feature vector
        sample_dict = {col: 0.0 for col in feat_cols}
        sample_dict['hour'] = hour
        sample_dict['is_weekend'] = 1 if is_weekend else 0
        sample_dict['distance_km'] = distance_km
        sample_dict['duration_min'] = duration_min
        sample_dict['temperature_c'] = temperature_c
        sample_dict['precipitation_mm'] = precipitation_mm
        sample_dict['wind_speed_kmh'] = 15.0
        sample_dict['active_drivers'] = active_drivers
        sample_dict['ride_requests'] = ride_requests
        sample_dict['demand_supply_ratio'] = demand_supply_ratio
        sample_dict['surge_multiplier'] = surge_multiplier
        
        # Engineered features
        sample_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        sample_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        sample_dict['is_morning_rush'] = 1 if is_morning_rush else 0
        sample_dict['is_evening_rush'] = 1 if is_evening_rush else 0
        sample_dict['is_rush_hour'] = 1 if is_rush_hour else 0
        sample_dict['log_distance'] = np.log1p(distance_km)
        sample_dict['distance_squared'] = distance_km ** 2
        sample_dict['est_speed_kmh'] = speed_kmh
        sample_dict['is_traffic_congested'] = 1 if speed_kmh < 20 else 0
        sample_dict['is_airport_trip'] = 1 if ("Airport" in pickup or "Airport" in dropoff) else 0
        
        severity_map = {"Clear": 0.0, "Cloudy": 0.2, "Fog": 0.5, "Rain": 0.7, "Heavy Rain": 1.2, "Thunderstorm": 1.8}
        sample_dict['weather_severity_index'] = severity_map.get(weather, 0.0)
        sample_dict['is_adverse_weather'] = 1 if precipitation_mm > 0.5 else 0
        sample_dict['demand_pressure_index'] = min(3.5, max(0.5, demand_supply_ratio))
        sample_dict['is_surge_active'] = 1 if surge_multiplier > 1.0 else 0
        sample_dict['distance_x_surge'] = distance_km * surge_multiplier
        sample_dict['demand_x_weather'] = sample_dict['demand_pressure_index'] * (1.0 + sample_dict['weather_severity_index'])
        sample_dict['duration_x_rush'] = duration_min * (1.0 + (0.5 if is_rush_hour else 0.0))

        # Categorical dummies
        weather_col = f"weather_condition_{weather}"
        if weather_col in sample_dict:
            sample_dict[weather_col] = 1.0
        
        pickup_col = f"pickup_location_{pickup}"
        if pickup_col in sample_dict:
            sample_dict[pickup_col] = 1.0

        sample_df = pd.DataFrame([sample_dict])[feat_cols]
        ml_predicted_fare = round(float(model.predict(sample_df)[0]), 2)
    except Exception as e:
        ml_predicted_fare = None

# Main Dashboard Content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Standard Base Fare (1.0x)", value=f"${standard_base_fare:.2f}")

with col2:
    st.metric(
        label="Dynamic Surge Multiplier", 
        value=f"{surge_multiplier:.2f}x",
        delta=f"Ratio: {demand_supply_ratio:.2f}" if surge_multiplier > 1.0 else "Normal Demand"
    )

with col3:
    st.metric(
        label="Rule-Based Dynamic Fare", 
        value=f"${computed_fare:.2f}",
        delta=f"+${computed_fare - standard_base_fare:.2f}" if computed_fare > standard_base_fare else "$0.00"
    )

with col4:
    if ml_predicted_fare is not None:
        st.metric(
            label="🤖 ML Regression Prediction", 
            value=f"${ml_predicted_fare:.2f}",
            delta=f"Diff: ${ml_predicted_fare - computed_fare:.2f}"
        )
    else:
        st.metric(label="🤖 ML Model Fare", value=f"${computed_fare:.2f}")

st.markdown("---")

# Middle Section: Market Insights and Price Drivers
c_left, c_right = st.columns([3, 2])

with c_left:
    st.subheader("📊 Dynamic Fare Breakdown")
    fare_breakdown = pd.DataFrame({
        "Component": [
            "Base Pickup Fare",
            f"Distance Fare ({distance_km} km @ $1.65/km)",
            f"Duration Time ({duration_min} min @ $0.35/min)",
            "Mandatory Booking Fee",
            "Airport Surcharge" if airport_surcharge > 0 else "Airport Surcharge (N/A)",
            f"Surge Multiplier Extra ({surge_multiplier:.2f}x)",
            f"Weather Penalty ({weather})"
        ],
        "Amount ($)": [
            base_fare,
            round(distance_km * per_km_rate, 2),
            round(duration_min * per_minute_rate, 2),
            booking_fee,
            airport_surcharge,
            round((standard_subtotal * (surge_multiplier - 1.0)), 2),
            weather_fee
        ]
    })
    st.dataframe(fare_breakdown, use_container_width=True)

with c_right:
    st.subheader("💡 Real-Time Market Drivers")
    if surge_multiplier > 1.5:
        st.error(f"🔥 **High Surge Active ({surge_multiplier:.2f}x):** High demand pressure! {ride_requests} ride requests with only {active_drivers} active drivers.")
    elif surge_multiplier > 1.0:
        st.warning(f"⚡ **Moderate Surge ({surge_multiplier:.2f}x):** Demand slightly exceeding normal driver supply.")
    else:
        st.success("✅ **Balanced Market:** Driver supply meets passenger demand. Standard rates apply.")

    if is_rush_hour:
        st.info("🚦 **Peak Rush Hour:** Slower travel speeds (expected 21 km/h) increase trip duration.")
    if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
        st.warning(f"🌧️ **Adverse Weather ({weather}):** Precipitation reduces driving speed and triggers bad-weather safety allowance.")

st.markdown("---")

# Bottom Section: Model Metrics Reference
st.subheader("📈 Model Performance & Comparison Benchmark")
metrics_csv_path = os.path.join(os.path.dirname(__file__), "models", "model_benchmark_results.csv")
if os.path.exists(metrics_csv_path):
    bm_df = pd.read_csv(metrics_csv_path)
    st.table(bm_df)
else:
    st.info("Model benchmark table will appear once model training finishes.")

st.caption("Nyeras LMS AI & ML Minor Project · Dynamic Pricing Predictor · Developer: Kamaljit")
