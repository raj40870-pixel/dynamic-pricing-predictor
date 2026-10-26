"""
Regression Model Training and Evaluation Pipeline
Trains Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting.
Computes RMSE, MAE, R2, and extracts the primary price drivers.
Author: Kamaljit
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(__file__))
from feature_engineering import get_feature_matrix

def train_and_evaluate(base_dir: str):
    raw_path = os.path.join(base_dir, "data", "raw", "rides_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading data...", flush=True)
    df_raw = pd.read_csv(raw_path)
    X, y, feature_names = get_feature_matrix(df_raw)
    print(f"Features matrix shape: {X.shape}, Target shape: {y.shape}", flush=True)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        "Ridge Regression (L2)": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        "Lasso Regression (L1)": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=0.05, max_iter=2000))
        ]),
        "Random Forest (n=50)": RandomForestRegressor(
            n_estimators=50, max_depth=12, min_samples_split=4, random_state=42, n_jobs=1
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.08, max_depth=6, random_state=42
        )
    }

    results = []
    trained_instances = {}

    print("\n--- Training and Evaluating Models ---", flush=True)
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE ($)": round(mae, 3),
            "RMSE ($)": round(rmse, 3),
            "R2 Score": round(r2, 4)
        })
        trained_instances[name] = (model, y_pred)
        print(f"  -> MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2: {r2:.4f}", flush=True)

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    print("\n--- Benchmark Comparison Table ---", flush=True)
    print(results_df.to_string(index=False), flush=True)

    # Save metrics table
    results_df.to_csv(os.path.join(models_dir, "model_benchmark_results.csv"), index=False)

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest performing model: {best_model_name}", flush=True)
    best_model, best_pred = trained_instances[best_model_name]

    # Save best model
    joblib.dump(best_model, os.path.join(models_dir, "best_pricing_model.joblib"))
    with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_names, f, indent=2)
    print("Best model and feature columns saved successfully.", flush=True)

    # 1. Plot Model Comparison
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x_indices = np.arange(len(results_df))
    plt.bar(x_indices - bar_width/2, results_df["RMSE ($)"], width=bar_width, label="RMSE ($)", color="#e74c3c")
    plt.bar(x_indices + bar_width/2, results_df["MAE ($)"], width=bar_width, label="MAE ($)", color="#3498db")
    plt.xticks(x_indices, results_df["Model"], rotation=15, ha='right', fontsize=9)
    plt.ylabel("Error in Dollars ($)")
    plt.title("Regression Models Error Comparison (RMSE vs MAE)", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_comparison_metrics.png"), dpi=300)
    plt.close()

    # 2. Extract Feature Importance / Biggest Price Drivers
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        rf = trained_instances["Random Forest (n=50)"][0]
        importances = rf.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    feat_imp.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    top_10 = feat_imp.head(10).iloc[::-1]
    plt.barh(top_10["Feature"], top_10["Importance"], color="#2ecc71", edgecolor="#27ae60")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.title(f"Top 10 Biggest Price Drivers in Dynamic Ride Pricing", fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance_drivers.png"), dpi=300)
    plt.close()

    # 3. Actual vs Predicted Prices Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, alpha=0.35, color="#2980b9", edgecolors='none', s=25)
    min_val = min(y_test.min(), best_pred.min())
    max_val = max(y_test.max(), best_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
    plt.xlabel("Actual Ride Price ($)", fontsize=11)
    plt.ylabel("Predicted Ride Price ($)", fontsize=11)
    plt.title(f"Actual vs Predicted Ride Prices (R2 = {results_df.iloc[0]['R2 Score']})", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # 4. Residual Distribution
    residuals = y_test - best_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50, color="#9b59b6", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.xlabel("Residual Error (Actual - Predicted) ($)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Error Residuals Distribution (Centered around $0)", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residual_distribution.png"), dpi=300)
    plt.close()

    # 5. EDA: Surge vs Demand-Supply Ratio
    plt.figure(figsize=(8, 5))
    plt.scatter(df_raw["demand_supply_ratio"], df_raw["surge_multiplier"], alpha=0.3, color="#e67e22")
    plt.xlabel("Demand / Supply Ratio (Ride Requests / Active Drivers)", fontsize=11)
    plt.ylabel("Surge Multiplier Factor (x)", fontsize=11)
    plt.title("Dynamic Pricing Surge Curve vs Demand-Supply Ratio", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "surge_vs_demand_scatter.png"), dpi=300)
    plt.close()

    print("All evaluation plots generated successfully in reports/figures/", flush=True)
    return results_df, feat_imp

if __name__ == "__main__":
    base_dir = r"C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor"
    train_and_evaluate(base_dir)
