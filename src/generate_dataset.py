"""
Dynamic Pricing Predictor for Ride-Sharing Apps
Dataset Generation Script: Realistic Uber/Lyft style ride dataset.
Author: Kamaljit
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_ride_dataset(num_records=10000, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2026, 8, 1, 0, 0, 0)
    
    locations = [
        "Downtown Financial District",
        "Midtown Center",
        "International Airport",
        "Tech Park / University",
        "Westside Residential",
        "East Riverfront",
        "North Suburbs",
        "South Bay District"
    ]
    
    # Distance matrix between locations (approx km)
    loc_coords = {
        "Downtown Financial District": (0, 0),
        "Midtown Center": (2, 3),
        "International Airport": (18, 12),
        "Tech Park / University": (8, -4),
        "Westside Residential": (-5, 6),
        "East Riverfront": (4, 1),
        "North Suburbs": (1, 15),
        "South Bay District": (-3, -12)
    }

    weather_types = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Thunderstorm"]
    weather_probs = [0.45, 0.25, 0.15, 0.07, 0.05, 0.03]

    records = []

    for i in range(num_records):
        ride_id = f"RIDE_{100000 + i}"
        
        # Random time over 60 days
        seconds_offset = random.randint(0, 60 * 24 * 3600)
        dt = start_date + timedelta(seconds=seconds_offset)
        hour = dt.hour
        day_of_week = dt.strftime("%A")
        is_weekend = 1 if dt.weekday() >= 5 else 0

        # Location pickup & dropoff
        pickup = random.choice(locations)
        dropoff = random.choice([loc for loc in locations if loc != pickup])
        
        # Calculate base geometric distance + variance
        p1 = loc_coords[pickup]
        p2 = loc_coords[dropoff]
        base_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        distance_km = round(max(1.2, base_dist * np.random.uniform(0.9, 1.25)), 2)

        # Weather condition
        weather = np.random.choice(weather_types, p=weather_probs)
        if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
            precipitation_mm = round(np.random.uniform(2.0, 25.0) if weather != "Rain" else np.random.uniform(0.5, 4.0), 1)
            temperature_c = round(np.random.uniform(14.0, 22.0), 1)
            wind_speed_kmh = round(np.random.uniform(15.0, 45.0), 1)
        elif weather == "Fog":
            precipitation_mm = 0.0
            temperature_c = round(np.random.uniform(10.0, 18.0), 1)
            wind_speed_kmh = round(np.random.uniform(5.0, 15.0), 1)
        else:
            precipitation_mm = 0.0
            temperature_c = round(np.random.uniform(18.0, 32.0), 1)
            wind_speed_kmh = round(np.random.uniform(3.0, 20.0), 1)

        # Time of day rush dynamics
        is_morning_rush = 1 if (7 <= hour <= 9 and not is_weekend) else 0
        is_evening_rush = 1 if (17 <= hour <= 20 and not is_weekend) else 0
        is_night_party = 1 if ((hour >= 22 or hour <= 2) and is_weekend) else 0

        # Driver Supply and Passenger Demand
        base_supply = np.random.randint(40, 95)
        base_demand = np.random.randint(30, 85)

        if is_morning_rush:
            base_demand = int(base_demand * np.random.uniform(1.6, 2.2))
            base_supply = int(base_supply * np.random.uniform(0.8, 1.1))
        elif is_evening_rush:
            base_demand = int(base_demand * np.random.uniform(1.8, 2.5))
            base_supply = int(base_supply * np.random.uniform(0.7, 1.0))
        elif is_night_party:
            base_demand = int(base_demand * np.random.uniform(1.4, 2.0))
            base_supply = int(base_supply * np.random.uniform(0.6, 0.9))

        # Bad weather reduces supply and increases demand
        if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
            base_demand = int(base_demand * np.random.uniform(1.2, 1.5))
            base_supply = int(base_supply * np.random.uniform(0.7, 0.85))

        active_drivers = max(10, base_supply)
        ride_requests = max(10, base_demand)
        
        demand_supply_ratio = round(ride_requests / active_drivers, 3)

        # Surge Multiplier Calculation (Dynamic Pricing algorithm)
        if demand_supply_ratio < 0.9:
            surge_multiplier = 1.0
        elif demand_supply_ratio < 1.3:
            surge_multiplier = round(1.0 + (demand_supply_ratio - 0.9) * 0.5, 2)
        elif demand_supply_ratio < 2.0:
            surge_multiplier = round(1.2 + (demand_supply_ratio - 1.3) * 0.8, 2)
        else:
            surge_multiplier = round(min(3.5, 1.76 + (demand_supply_ratio - 2.0) * 0.9), 2)

        # Ride duration based on distance, traffic congestion & weather
        speed_kmh = 32.0  # standard urban speed
        if is_morning_rush or is_evening_rush:
            speed_kmh *= 0.65
        if weather in ["Rain", "Heavy Rain", "Thunderstorm"]:
            speed_kmh *= 0.78
        if weather == "Fog":
            speed_kmh *= 0.85

        duration_min = round(max(4.0, (distance_km / speed_kmh) * 60 + np.random.normal(2.0, 1.5)), 1)

        # Base pricing formula
        base_fare = 3.50
        per_km_rate = 1.65
        per_minute_rate = 0.35
        booking_fee = 2.00
        
        # Airport surcharge
        airport_surcharge = 5.00 if (pickup == "International Airport" or dropoff == "International Airport") else 0.0

        # Subtotal
        standard_fare = base_fare + (distance_km * per_km_rate) + (duration_min * per_minute_rate) + booking_fee + airport_surcharge

        # Weather surcharge adjustment
        weather_fee = 0.0
        if weather == "Heavy Rain":
            weather_fee = 2.50
        elif weather == "Thunderstorm":
            weather_fee = 4.00
        elif weather == "Rain":
            weather_fee = 1.50

        # Final dynamic price = standard_fare * surge_multiplier + weather_fee + noise
        noise = np.random.normal(0, 0.75)
        raw_price = standard_fare * surge_multiplier + weather_fee + noise
        final_price = round(max(6.50, raw_price), 2)

        records.append({
            "ride_id": ride_id,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "pickup_location": pickup,
            "dropoff_location": dropoff,
            "distance_km": distance_km,
            "duration_min": duration_min,
            "weather_condition": weather,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "wind_speed_kmh": wind_speed_kmh,
            "active_drivers": active_drivers,
            "ride_requests": ride_requests,
            "demand_supply_ratio": demand_supply_ratio,
            "surge_multiplier": surge_multiplier,
            "base_fare": base_fare,
            "price": final_price
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    import os
    base_dir = r"C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor"
    raw_path = os.path.join(base_dir, "data", "raw", "rides_dataset.csv")
    df = generate_ride_dataset(num_records=10000, random_seed=42)
    df.to_csv(raw_path, index=False)
    print(f"Generated {len(df)} records and saved to {raw_path}")
    print(df.head())
    print("\nSummary Statistics:")
    print(df[['distance_km', 'duration_min', 'surge_multiplier', 'price']].describe())
