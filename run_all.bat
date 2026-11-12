@echo off
echo ====================================================================
echo   360 Auto Vision - Accident Damage Detection & Cost Estimation
echo ====================================================================
echo Starting Flask Backend API on http://127.0.0.1:5000 ...
start "360 Auto Vision Backend" cmd /k "call .venv\Scripts\activate.bat && python -m backend.app"

timeout /t 3 /nobreak > nul

echo Starting Streamlit Frontend Dashboard on http://localhost:8501 ...
start "360 Auto Vision Frontend" cmd /k "call .venv\Scripts\activate.bat && streamlit run frontend\streamlit_app.py"

echo System launched! Check your browser windows.
pause
