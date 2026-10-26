"""
Dynamic Pricing Predictor - Web Dashboard
Author: Kamaljit
Course: AI & ML Minor Project (Nyeras LMS)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json

st.set_page_config(
    page_title="Dynamic Pricing Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Dynamic Pricing Predictor for Ride-Sharing Apps")
st.write("Predict ride fares based on trip distance, rush hour, weather conditions, and driver demand.")

# Sidebar for Trip Inputs
st.sidebar.header("🔧 Ride Inputs")

pickup = st.sidebar.selectbox(
    "Pickup Location",
    ["Downtown Financial District", "Midtown Center", "International Airport", "Tech Park / University", "Westside Residential", "North Suburbs"],
    index=0
)

dropoff = st.sidebar.selectbox(
    "Dropoff Location",
    ["Downtown Financial District", "Midtown Center", "International Airport", "Tech Park / University", "Westside Residential", "North Suburbs"],
    index=2
)

distance_km = st.sidebar.slider("Trip Distance (in km)", min_value=1.0, max_value=40.0, value=12.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⏰ Time & Rush Hour")
hour = st.sidebar.slider("Time of Day (24-Hour format)", min_value=0, max_value=23, value=18)
is_weekend = st.sidebar.checkbox("Is it Weekend (Sat/Sun)?", value=False)

# Check if peak rush hour (8-10 AM or 5-8 PM on weekdays)
is_rush_hour = (not is_weekend) and ((8 <= hour <= 10) or (17 <= hour <= 20))

st.sidebar.markdown("---")
st.sidebar.header("🌧️ Weather Conditions")
weather = st.sidebar.selectbox("Current Weather", ["Clear", "Cloudy", "Fog", "Rain", "Heavy Rain", "Thunderstorm"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("👥 Driver Supply & Passenger Demand")
active_drivers = st.sidebar.slider("Active Drivers Available", min_value=10, max_value=100, value=30)
ride_requests = st.sidebar.slider("Passenger Ride Requests", min_value=10, max_value=120, value=60)

# Demand-Supply Ratio
ratio = round(ride_requests / max(active_drivers, 1), 2)

# Dynamic Surge Multiplier
if ratio < 0.9:
    surge_multiplier = 1.0
elif ratio < 1.3:
    surge_multiplier = round(1.0 + (ratio - 0.9) * 0.5, 2)
elif ratio < 2.0:
    surge_multiplier = round(1.2 + (ratio - 1.3) * 0.8, 2)
else:
    surge_multiplier = round(min(3.5, 1.76 + (ratio - 2.0) * 0.9), 2)

# Speed and duration calculation
speed = 32.0
if is_rush_hour:
    speed *= 0.65
if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
    speed *= 0.80

duration_min = round(max(4.0, (distance_km / speed) * 60 + 2.0), 1)

# Base fare formula calculation
base_fare = 3.50
per_km_rate = 1.65
per_min_rate = 0.35
booking_fee = 2.00
airport_fee = 5.00 if ("Airport" in pickup or "Airport" in dropoff) else 0.0

standard_subtotal = base_fare + (distance_km * per_km_rate) + (duration_min * per_min_rate) + booking_fee + airport_fee

weather_extra = 0.0
if weather == "Rain":
    weather_extra = 1.50
elif weather in ["Heavy Rain", "Thunderstorm"]:
    weather_extra = 3.00

final_rule_fare = round(standard_subtotal * surge_multiplier + weather_extra, 2)

# Try predicting with trained Machine Learning Model
model_file = os.path.join(os.path.dirname(__file__), "models", "best_pricing_model.joblib")
cols_file = os.path.join(os.path.dirname(__file__), "models", "feature_columns.json")

ml_prediction = None
if os.path.exists(model_file) and os.path.exists(cols_file):
    try:
        model = joblib.load(model_file)
        with open(cols_file, 'r') as f:
            feat_cols = json.load(f)

        row = {c: 0.0 for c in feat_cols}
        row['hour'] = hour
        row['is_weekend'] = 1 if is_weekend else 0
        row['distance_km'] = distance_km
        row['duration_min'] = duration_min
        row['active_drivers'] = active_drivers
        row['ride_requests'] = ride_requests
        row['demand_supply_ratio'] = ratio
        row['surge_multiplier'] = surge_multiplier
        row['is_rush_hour'] = 1 if is_rush_hour else 0
        row['log_distance'] = np.log1p(distance_km)
        row['speed_kmh'] = speed
        row['is_traffic_jam'] = 1 if speed < 20 else 0
        row['is_airport'] = 1 if airport_fee > 0 else 0
        
        score_map = {'Clear': 0, 'Cloudy': 1, 'Fog': 2, 'Rain': 3, 'Heavy Rain': 4, 'Thunderstorm': 5}
        row['weather_score'] = score_map.get(weather, 0)
        row['is_rainy'] = 1 if weather in ['Rain', 'Heavy Rain', 'Thunderstorm'] else 0
        row['distance_x_surge'] = distance_km * surge_multiplier

        # Weather dummy
        w_dummy = f"weather_condition_{weather}"
        if w_dummy in row:
            row[w_dummy] = 1.0

        p_dummy = f"pickup_location_{pickup}"
        if p_dummy in row:
            row[p_dummy] = 1.0

        sample_df = pd.DataFrame([row])[feat_cols]
        ml_prediction = round(float(model.predict(sample_df)[0]), 2)
    except Exception as e:
        ml_prediction = None

# Display Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Base Standard Fare", f"${round(standard_subtotal, 2):.2f}")

with col2:
    st.metric("Dynamic Surge Multiplier", f"{surge_multiplier:.2f}x", delta="High Demand" if surge_multiplier > 1.0 else "Normal")

with col3:
    st.metric("Calculated Fare", f"${final_rule_fare:.2f}")

with col4:
    if ml_prediction:
        st.metric("🤖 ML Model Predicted Fare", f"${ml_prediction:.2f}")
    else:
        st.metric("🤖 ML Model Predicted Fare", f"${final_rule_fare:.2f}")

st.markdown("---")

col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("📋 Fare Breakdown Details")
    fare_table = pd.DataFrame({
        "Component": [
            "Base Pickup Fare",
            f"Distance Charge ({distance_km} km)",
            f"Duration Time ({duration_min} mins)",
            "Standard Booking Fee",
            "Airport Surcharge" if airport_fee > 0 else "Airport Surcharge (None)",
            f"Surge Extra ({surge_multiplier:.2f}x)",
            f"Weather Extra ({weather})"
        ],
        "Amount ($)": [
            base_fare,
            round(distance_km * per_km_rate, 2),
            round(duration_min * per_min_rate, 2),
            booking_fee,
            airport_fee,
            round(standard_subtotal * (surge_multiplier - 1.0), 2),
            weather_extra
        ]
    })
    st.table(fare_table)

with col_b:
    st.subheader("💡 Why Did The Fare Change?")
    if surge_multiplier > 1.5:
        st.error(f"🔥 **High Surge ({surge_multiplier:.2f}x):** Too many passengers ({ride_requests}) and few drivers ({active_drivers}).")
    elif surge_multiplier > 1.0:
        st.warning(f"⚡ **Moderate Surge ({surge_multiplier:.2f}x):** High demand.")
    else:
        st.success("✅ **Normal Tariff:** Driver supply meets passenger demand.")

    if is_rush_hour:
        st.info("🚦 **Peak Office Rush:** Travel speed is reduced due to traffic.")
    if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
        st.warning(f"🌧️ **Bad Weather:** Rain surcharge applied.")

st.markdown("---")
st.caption("AI & ML Minor Project - Dynamic Pricing Predictor · Developer: Kamaljit")
