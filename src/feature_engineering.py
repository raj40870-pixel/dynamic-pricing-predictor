"""
Feature Engineering Module for Dynamic Pricing Predictor
Processes raw ride-sharing data into high-value ML features.
Author: Kamaljit
"""

import numpy as np
import pandas as pd
import os

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw dataframe into feature-rich dataset for regression modeling.
    Extracts time-of-day, distance, weather, and demand features.
    """
    df = df.copy()

    # 1. TIME-OF-DAY FEATURES
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week_num'] = df['datetime'].dt.dayofweek
    else:
        df['day_of_week_num'] = df['hour'] % 7  # fallback

    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

    # Rush hour flags
    is_wknd = df['is_weekend'] if 'is_weekend' in df.columns else (df['day_of_week_num'] >= 5).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (is_wknd == 0)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (is_wknd == 0)).astype(int)
    df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']
    df['is_night_peak'] = (((df['hour'] >= 22) | (df['hour'] <= 2)) & (is_wknd == 1)).astype(int)

    # Time of day categories
    def get_time_bucket(hour):
        if 5 <= hour < 11:
            return "Morning"
        elif 11 <= hour < 16:
            return "Afternoon"
        elif 16 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    df['time_bucket'] = df['hour'].apply(get_time_bucket)

    # 2. DISTANCE & DURATION FEATURES
    df['log_distance'] = np.log1p(df['distance_km'])
    df['distance_squared'] = df['distance_km'] ** 2
    df['est_speed_kmh'] = np.where(df['duration_min'] > 0, (df['distance_km'] / (df['duration_min'] / 60.0)), 30.0)
    df['is_traffic_congested'] = (df['est_speed_kmh'] < 20.0).astype(int)
    
    # Location features
    airport_name = "International Airport"
    df['is_airport_trip'] = ((df['pickup_location'] == airport_name) | (df['dropoff_location'] == airport_name)).astype(int)

    # 3. WEATHER FEATURES
    weather_severity_map = {
        "Clear": 0.0,
        "Cloudy": 0.2,
        "Fog": 0.5,
        "Rain": 0.7,
        "Heavy Rain": 1.2,
        "Thunderstorm": 1.8
    }
    df['weather_severity_index'] = df['weather_condition'].map(weather_severity_map).fillna(0.0)
    df['is_adverse_weather'] = (df['precipitation_mm'] > 0.5) | (df['weather_condition'].isin(["Heavy Rain", "Thunderstorm"]))
    df['is_adverse_weather'] = df['is_adverse_weather'].astype(int)

    # 4. DEMAND & SUPPLY FEATURES
    if 'demand_supply_ratio' not in df.columns:
        df['demand_supply_ratio'] = df['ride_requests'] / np.maximum(df['active_drivers'], 1)

    df['demand_pressure_index'] = np.clip(df['demand_supply_ratio'], 0.5, 3.5)
    df['is_surge_active'] = (df['surge_multiplier'] > 1.0).astype(int)

    # 5. INTERACTION FEATURES (Non-linear market dynamics)
    df['distance_x_surge'] = df['distance_km'] * df['surge_multiplier']
    df['demand_x_weather'] = df['demand_pressure_index'] * (1.0 + df['weather_severity_index'])
    df['duration_x_rush'] = df['duration_min'] * (1.0 + df['is_rush_hour'] * 0.5)

    return df

def get_feature_matrix(df: pd.DataFrame):
    """
    Splits into numerical feature matrix X and target y.
    One-hot encodes categorical columns like weather_condition and time_bucket.
    """
    df_feat = create_features(df)
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_feat, columns=['weather_condition', 'time_bucket', 'pickup_location'], drop_first=True)
    
    # Exclude identifiers, raw timestamps, and target
    drop_cols = ['ride_id', 'timestamp', 'datetime', 'dropoff_location', 'day_of_week', 'price', 'base_fare']
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    
    X = df_encoded[feature_cols].copy()
    y = df_encoded['price'].copy()
    
    return X, y, feature_cols

if __name__ == "__main__":
    base_dir = r"C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor"
    raw_path = os.path.join(base_dir, "data", "raw", "rides_dataset.csv")
    out_path = os.path.join(base_dir, "data", "processed", "features_engineered.csv")

    df_raw = pd.read_csv(raw_path)
    df_feat = create_features(df_raw)
    df_feat.to_csv(out_path, index=False)
    print(f"Feature engineering completed. Processed rows: {len(df_feat)}, columns: {df_feat.shape[1]}")
    print("New engineered features:")
    new_cols = [c for c in df_feat.columns if c not in df_raw.columns]
    print(new_cols)
