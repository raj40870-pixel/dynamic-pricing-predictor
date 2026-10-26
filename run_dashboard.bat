@echo off
echo =======================================================
echo  🚗 Starting Dynamic Pricing Predictor Dashboard...
echo =======================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
