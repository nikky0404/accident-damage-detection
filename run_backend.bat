@echo off
echo Starting 360 Auto Vision Flask Backend Server...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m backend.app
pause
