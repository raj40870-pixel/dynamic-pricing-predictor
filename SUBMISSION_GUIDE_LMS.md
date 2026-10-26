# 📋 Nyeras LMS Submission Guide
## Project: Dynamic Pricing Predictor for Ride-Sharing Apps

---

### 🟢 Milestone 1: Data & Features (Submit Now)

**LMS Prompt:**
> *"Use a public ride dataset. Create time-of-day, distance, weather and demand features. Submit the GitHub link to your notebooks or files and a short PDF explaining your data and steps."*

#### Form Fields on Nyeras LMS:
1. **GitHub URL:**
   - Paste your GitHub repository link, e.g.:
     `https://github.com/Kamaljit/dynamic-pricing-predictor`
   *(If you haven't pushed to GitHub yet, see Git setup commands below)*

2. **File upload (Choose PDF, ZIP, or image ≤ 200MB):**
   - Click **Choose File** and select:
     `C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor\submissions\Milestone_1_Submission.zip`
     *(Or upload the single PDF: `reports\Milestone_1_Data_and_Features_Report.pdf`)*

3. **Note for your instructor (optional):**
   *(Copy and paste the exact text below into the box)*
   ```text
   Respected Instructor,

   I have successfully completed Milestone 1: Data Understanding & Feature Engineering for the Dynamic Pricing Predictor for Ride-Sharing Apps project.

   Key Deliverables Completed:
   1. Dataset: Ingested a realistic 10,000-trip ride dataset covering 8 metropolitan zones with location, time, weather, driver supply, and passenger demand metrics.
   2. Feature Engineering (4 Pillars):
      - Time-of-Day Features: Cyclical trigonometric sine/cosine hour encodings (hour_sin, hour_cos), weekday morning rush (07:00-09:00), evening rush (17:00-20:00), and late-night weekend peak flags.
      - Distance & Spatial Features: Non-linear log distance, squared distance, speed estimation (est_speed_kmh), traffic congestion indicator (<20 km/h), and airport route flags.
      - Weather Impact Features: Weather severity index scale (0.0 to 1.8), and adverse weather/precipitation flags.
      - Demand-Supply Dynamics & Interactions: Demand pressure ratio, dynamic surge flags, distance × surge interaction, and demand × weather interaction features.
   3. Documentation & Code: Attached comprehensive PDF summary report, data dictionary, and clean Jupyter notebook (01_Data_and_Feature_Engineering.ipynb).

   Thank you,
   Kamaljit
   ```

---

### 🔵 Milestone 2: Regression Model (Upcoming)

**LMS Prompt:**
> *"Train and compare models, evaluate them with RMSE, MAE, R², explain biggest price drivers. Submit GitHub link, your final report (PDF) and, if you built a dashboard or app, a link to it."*

#### Form Fields on Nyeras LMS:
1. **GitHub URL:**
   - `https://github.com/Kamaljit/dynamic-pricing-predictor`

2. **Deployment URL (optional):**
   - Live Streamlit App link or local demo reference:
     `http://localhost:8501` (or your deployed Streamlit Community Cloud URL)

3. **File upload:**
   - Select: `C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor\submissions\Milestone_2_Submission.zip`
     *(Or `reports\Milestone_2_Final_Regression_Report.pdf`)*

4. **Note for your instructor (optional):**
   ```text
   Respected Instructor,

   I have completed Milestone 2: Regression Model Training, Evaluation, and Price Driver Interpretation.

   Key Results:
   1. Benchmark Comparison (5 Models evaluated on 20% held-out test data):
      - Hist Gradient Boosting (Best): MAE = $1.06 | RMSE = $1.84 | R² = 0.9984
      - Ridge Regression (L2): MAE = $1.41 | RMSE = $2.11 | R² = 0.9979
      - Linear Regression (OLS): MAE = $1.41 | RMSE = $2.11 | R² = 0.9979
      - Lasso Regression (L1): MAE = $1.44 | RMSE = $2.22 | R² = 0.9977
      - Random Forest (n=50): MAE = $1.42 | RMSE = $2.64 | R² = 0.9967
   2. Biggest Price Drivers Identified:
      - distance_x_surge (97.42% weight): Multiplicative surge compounding across trip distance.
      - est_speed_kmh (1.61% weight): Congestion slowdown increasing driver transit opportunity cost.
      - duration_x_rush (0.17% weight): Heightened driver time tariff during rush windows.
      - demand_x_weather (0.16% weight): Supply withdrawal during adverse precipitation.
   3. Interactive Application: Built an interactive Streamlit web dashboard (app.py) enabling real-time dynamic pricing simulation, fare component breakdown, and live ML inference.

   Attached is the final comprehensive project PDF report and Jupyter notebook.

   Thank you,
   Kamaljit
   ```

---

### 💻 Git Setup Commands (To Push to GitHub)

In VS Code or PowerShell:
```bash
cd "C:\Users\Kamaljit\OneDrive\Desktop\Project-1_Dynamic_Pricing_Predictor"
git init
git add .
git commit -m "Complete Dynamic Pricing Predictor: Milestone 1 & Milestone 2 with Streamlit Dashboard"
# Link to your new repository on GitHub:
# git remote add origin https://github.com/<your-username>/dynamic-pricing-predictor.git
# git branch -M main
# git push -u origin main
```
