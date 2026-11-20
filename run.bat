@echo off
setlocal enabledelayedexpansion

:: Window Title
title SignSense - Run Application

:: Navigate to project directory
cd /d "%~dp0"

echo ============================================================
echo               SignSense: Run Application
echo ============================================================
echo.

:: Detect Python from virtual environment
if exist "venv\Scripts\python.exe" (
    echo [OK] Using virtual environment Python: venv\Scripts\python.exe
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        echo [INFO] Virtual environment not found, using system Python.
        set "PYTHON_EXE=python"
    ) else (
        echo [ERROR] Python was not found on your system!
        echo Please ensure Python is installed and added to PATH, or setup venv.
        pause
        exit /b 1
    )
)

:: Check if trained checkpoint exists
if not exist "checkpoints\best_static.pt" (
    echo.
    echo [WARNING] No trained model found at checkpoints\best_static.pt!
    echo Would you like to train the model first? (Y/N)
    set /p "CHOICE=Enter choice [Y/N, default=Y]: "
    if /i "!CHOICE!"=="N" (
        echo [INFO] Starting in landmark-only mode...
    ) else (
        echo.
        echo [INFO] Redirecting to training first...
        call "%~dp0train_and_run.bat"
        exit /b 0
    )
)

echo.
echo Launching SignSense Web Application...
echo Server will open http://localhost:5000/camera in your default browser.
echo Press Ctrl+C in this terminal anytime to stop the server.
echo ============================================================
echo.

"%PYTHON_EXE%" src/run_app.py --config configs/static.yaml %*

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application exited with code %errorlevel%.
)

pause
