@echo off
echo Starting 360 Auto Vision Streamlit Frontend...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run frontend\streamlit_app.py
pause
