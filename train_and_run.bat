@echo off
setlocal enabledelayedexpansion

:: Window Title
title SignSense - Train and Run

:: Navigate to project directory
cd /d "%~dp0"

echo ============================================================
echo          SignSense: Train Model and Run Application
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

echo.
echo [1/2] Starting Model Training...
echo       Config: configs\static.yaml (Classes: A-Z, 0-9)
echo ------------------------------------------------------------
"%PYTHON_EXE%" src/train.py --config configs/static.yaml --model static
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo [ERROR] Training failed with exit code %errorlevel%!
    echo Please review the error messages above.
    echo ============================================================
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo [SUCCESS] Training completed successfully!
echo.
echo [2/2] Launching SignSense Application...
echo ------------------------------------------------------------
echo Server will open http://localhost:5000/camera in your browser.
echo Press Ctrl+C in this terminal anytime to stop the server.
echo ============================================================
echo.

"%PYTHON_EXE%" src/run_app.py --config configs/static.yaml %*

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Application exited with code %errorlevel%.
)

pause
