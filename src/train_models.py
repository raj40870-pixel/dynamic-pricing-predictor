"""
Model Training and Evaluation Script
Created by: Kamaljit
Course: AI & ML Minor Project
Project: Dynamic Pricing Predictor for Ride-Sharing Apps
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our feature function
from feature_engineering import prepare_training_data

def train_and_evaluate():
    base_dir = r"C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor"
    data_file = os.path.join(base_dir, "data", "raw", "rides_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    fig_dir = os.path.join(base_dir, "reports", "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Step 1: Loading raw dataset...")
    df_raw = pd.read_csv(data_file)
    print(f"Loaded {len(df_raw)} ride rows.")

    print("\nStep 2: Preparing features and target...")
    X, y, feature_names = prepare_training_data(df_raw)
    print(f"Features: {X.shape[1]} columns, Target: {y.name}")

    print("\nStep 3: Splitting into 80% Train and 20% Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Define standard regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=1)
    }

    results = []
    trained_models = {}

    print("\nStep 4: Training models and calculating evaluation metrics...")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        # Predict on unseen test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": model_name,
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })

        trained_models[model_name] = (model, y_pred)
        print(f"  Result -> MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2 Score: {r2:.4f}")

    # Summary table
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    print("\n=== Model Benchmark Comparison Table ===")
    print(results_df.to_string(index=False))

    # Save benchmark table to CSV
    results_df.to_csv(os.path.join(models_dir, "model_benchmark_results.csv"), index=False)

    # Save the best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model, best_pred = trained_models[best_model_name]
    print(f"\nBest Model selected: {best_model_name}")

    joblib.dump(best_model, os.path.join(models_dir, "best_pricing_model.joblib"))
    with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Plot 1: Model Error Comparison
    plt.figure(figsize=(8, 4.5))
    x_pos = np.arange(len(results_df))
    plt.bar(x_pos - 0.15, results_df["RMSE ($)"], width=0.3, label="RMSE ($)", color="#e74c3c")
    plt.bar(x_pos + 0.15, results_df["MAE ($)"], width=0.3, label="MAE ($)", color="#3498db")
    plt.xticks(x_pos, results_df["Model"])
    plt.ylabel("Error in Dollars ($)")
    plt.title("Model Error Comparison (RMSE vs MAE)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison_metrics.png"), dpi=200)
    plt.close()

    # Plot 2: Top Price Drivers (Feature Importance)
    rf_model = trained_models["Random Forest"][0]
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    feat_imp.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    plt.figure(figsize=(9, 5))
    top_drivers = feat_imp.head(8).iloc[::-1]
    plt.barh(top_drivers["Feature"], top_drivers["Importance"], color="#27ae60")
    plt.xlabel("Importance Score")
    plt.title("Top Price Drivers in Dynamic Ride Pricing", fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance_drivers.png"), dpi=200)
    plt.close()

    # Plot 3: Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_pred, alpha=0.3, color="#2980b9", s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Fare ($)")
    plt.ylabel("Predicted Fare ($)")
    plt.title(f"Actual vs Predicted Ride Fares ({best_model_name})", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()

    print("\nTraining and evaluation finished successfully!")

if __name__ == "__main__":
    train_and_evaluate()
