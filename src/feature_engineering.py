"""
Feature Engineering for Ride-Sharing Dynamic Pricing
Created by: Kamaljit
Course: AI & ML Minor Project
"""

import pandas as pd
import numpy as np

def add_features(df):
    """
    Function to extract time, distance, weather, and demand features
    from raw ride data.
    """
    df = df.copy()

    # 1. TIME FEATURES
    # Convert timestamp string to datetime object
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['is_weekend'] = df['datetime'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

    # Rush hour flag: Morning (8 AM - 10 AM) and Evening (5 PM - 8 PM) on weekdays
    df['is_rush_hour'] = df.apply(
        lambda row: 1 if (row['is_weekend'] == 0 and (8 <= row['hour'] <= 10 or 17 <= row['hour'] <= 20)) else 0,
        axis=1
    )

    # 2. DISTANCE FEATURES
    # Log distance to handle long-distance rides smoothly
    df['log_distance'] = np.log1p(df['distance_km'])

    # Travel speed (km/h) to check if there is traffic congestion
    df['speed_kmh'] = df['distance_km'] / (df['duration_min'] / 60.0)
    df['is_traffic_jam'] = df['speed_kmh'].apply(lambda s: 1 if s < 20.0 else 0)

    # Airport trip indicator (often has airport toll fee)
    df['is_airport'] = df.apply(
        lambda r: 1 if (r['pickup_location'] == 'International Airport' or r['dropoff_location'] == 'International Airport') else 0,
        axis=1
    )

    # 3. WEATHER FEATURES
    # Map weather conditions to a simple numeric severity score
    weather_score_map = {
        'Clear': 0,
        'Cloudy': 1,
        'Fog': 2,
        'Rain': 3,
        'Heavy Rain': 4,
        'Thunderstorm': 5
    }
    df['weather_score'] = df['weather_condition'].map(weather_score_map).fillna(0)
    df['is_rainy'] = df['weather_condition'].apply(lambda w: 1 if w in ['Rain', 'Heavy Rain', 'Thunderstorm'] else 0)

    # 4. DEMAND & SUPPLY FEATURES
    # Demand-Supply Ratio: requests divided by available drivers
    df['demand_supply_ratio'] = df['ride_requests'] / df['active_drivers']

    # Interaction feature: Distance multiplied by Surge Multiplier
    df['distance_x_surge'] = df['distance_km'] * df['surge_multiplier']

    return df

def prepare_training_data(df):
    """
    One-hot encodes categorical columns and returns X (features) and y (target price).
    """
    df_feat = add_features(df)

    # Convert categorical text columns to dummy variables (0/1)
    df_encoded = pd.get_dummies(df_feat, columns=['weather_condition', 'pickup_location'], drop_first=True)

    # Remove non-numeric or raw identifier columns
    drop_columns = ['ride_id', 'timestamp', 'datetime', 'dropoff_location', 'day_of_week', 'price', 'base_fare']
    feature_cols = [col for col in df_encoded.columns if col not in drop_columns]

    X = df_encoded[feature_cols]
    y = df_encoded['price']

    return X, y, feature_cols

if __name__ == '__main__':
    # Test script locally
    raw_file = r'C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor\data\raw\rides_dataset.csv'
    df_raw = pd.read_csv(raw_file)
    print("Original data shape:", df_raw.shape)

    X, y, feature_names = prepare_training_data(df_raw)
    print("Processed feature matrix shape:", X.shape)
    print("Total features:", len(feature_names))
    print("Sample features:", feature_names[:8])
