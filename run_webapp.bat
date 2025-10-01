@echo off
echo Starting Baby Cry Classification Web App...
echo.

REM Check if model exists
if not exist "models\hybrid_model.pth" (
    echo Model not found! Please train the model first by running run_training.bat
    pause
    exit /b 1
)

REM Change to web_app directory
cd web_app

REM Start the Flask app
echo Starting Flask application on http://localhost:5000
python app.py

pause