@echo off
echo Starting Baby Cry Classification Training...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Run setup to check requirements
echo Running setup check...
python setup.py
if %errorlevel% neq 0 (
    echo Setup check failed
    pause
    exit /b 1
)

echo.
echo Starting model training...
python train_model.py
if %errorlevel% neq 0 (
    echo Training failed
    pause
    exit /b 1
)

echo.
echo Training completed successfully!
echo You can now run the web application with:
echo cd web_app
echo python app.py
echo.
pause