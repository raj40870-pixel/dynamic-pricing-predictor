# 📋 Nyeras LMS Submission Guide
## Project: Dynamic Pricing Predictor for Ride-Sharing Apps

---

### 🟢 Milestone 1: Data & Features

#### Form Fields on Nyeras LMS:
1. **GitHub URL:**
   ```text
   https://github.com/raj40870-pixel/dynamic-pricing-predictor
   ```
2. **File upload (Choose PDF, ZIP, or image ≤ 200MB):**
   * Upload file:
     `C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor\submissions\Milestone_1_Submission.zip`
     *(Or upload the single PDF: `reports\Milestone_1_Data_and_Features_Report.pdf`)*

3. **Note for your instructor (optional):**
   ```text
   Respected Instructor,

   I have completed Milestone 1: Data Understanding & Feature Engineering for the Dynamic Pricing Predictor for Ride-Sharing Apps project.

   Key Deliverables Completed:
   1. Dataset: Ingested a 10,000-trip ride dataset across 8 metropolitan zones containing spatial, temporal, weather, and demand metrics.
   2. 4 Feature Engineering Pillars:
      - Time-of-Day Features: Cyclical hour attributes, weekday morning rush (08:00-10:00), evening rush (17:00-20:00), and weekend flags.
      - Distance & Spatial: Non-linear log distance, travel speed estimation (speed_kmh), traffic congestion indicator (<20 km/h), and airport route flags.
      - Weather Impact: Weather severity score (0 to 5), and active rain indicators.
      - Demand-Supply Dynamics & Interactions: Demand pressure ratio, dynamic surge multiplier, and distance × surge interaction feature.
   3. Documentation & Code: Attached comprehensive PDF summary report, data dictionary, and clean Jupyter notebook (01_Data_and_Feature_Engineering.ipynb).

   Thank you,
   Kamaljit
   ```

---

### 🔵 Milestone 2: Regression Model

#### Form Fields on Nyeras LMS:
1. **GitHub URL:**
   ```text
   https://github.com/raj40870-pixel/dynamic-pricing-predictor
   ```

2. **Deployment URL (optional):**
   ```text
   https://raj40870-pixel-dynamic-pricing-predictor-app-ubxbb2.streamlit.app/
   ```

3. **File upload:**
   * Upload file:
     `C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor\submissions\Milestone_2_Submission.zip`
     *(Or `reports\Milestone_2_Final_Regression_Report.pdf`)*

4. **Note for your instructor (optional):**
   ```text
   Respected Instructor,

   I have completed Milestone 2: Regression Model Training, Evaluation, and Price Driver Interpretation.

   Key Results:
   1. Benchmark Comparison (Evaluated on 20% test partition):
      - Linear Regression: MAE = $1.60 | RMSE = $2.41 | R² = 0.9972
      - Ridge Regression: MAE = $1.60 | RMSE = $2.41 | R² = 0.9972
      - Random Forest: MAE = $1.40 | RMSE = $2.62 | R² = 0.9967
   2. Primary Price Drivers Identified:
      - distance_x_surge (97.46% weight): Multiplicative surge compounding across trip distance.
      - speed_kmh (1.73% weight): Congestion slowdown increasing driver transit opportunity cost.
      - duration_min: Transit time required.
      - demand_supply_ratio: Imbalance between requests and active drivers.
   3. Live Web Application: Deployed an interactive Streamlit application enabling real-time dynamic pricing simulation, fare component breakdown, and live ML inference.
      Live URL: https://raj40870-pixel-dynamic-pricing-predictor-app-ubxbb2.streamlit.app/

   Attached is the final comprehensive project PDF report and Jupyter notebook.

   Thank you,
   Kamaljit
   ```
