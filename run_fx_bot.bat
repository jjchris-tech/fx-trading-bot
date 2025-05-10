@echo off
echo =====================================
echo   EUR/USD FX Trading Bot - Launcher
echo =====================================
echo.

REM Set up color output
color 0A

REM Create necessary directories if they don't exist
if not exist data_files mkdir data_files
if not exist data_files\market_data_cache mkdir data_files\market_data_cache
if not exist data_files\sentiment_cache mkdir data_files\sentiment_cache
if not exist data_files\dashboard mkdir data_files\dashboard
if not exist data_files\optimization mkdir data_files\optimization
if not exist data_files\walk_forward mkdir data_files\walk_forward
if not exist logs mkdir logs
if not exist reports mkdir reports
if not exist reports\metrics mkdir reports\metrics

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8+ is required
    echo Your current Python version is:
    python --version
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist venv (
    echo [INFO] Using existing virtual environment
    call venv\Scripts\activate
) else (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        echo Please make sure Python 3.8+ is installed with venv module
        pause
        exit /b 1
    )
    call venv\Scripts\activate
    
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt --no-cache-dir
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if required modules exist
python -c "import os, sys; sys.path.insert(0, os.getcwd()); all_good = True; required_modules = ['numpy', 'pandas', 'matplotlib', 'optuna', 'scipy', 'requests', 'python_dotenv', 'ta', 'fpdf2', 'discord_webhook', 'streamlit', 'joblib', 'sklearn', 'tabulate', 'coloredlogs']; missing = [m for m in required_modules if not any(True for loader, name, is_pkg in pkgutil.iter_modules() if name == m or name == m.replace('_', '-'))]; all_good = len(missing) == 0; print('Missing modules: ' + ', '.join(missing) if missing else 'All modules installed'); sys.exit(0 if all_good else 1)" >nul 2>&1

if %errorlevel% neq 0 (
    echo [WARNING] Some modules may be missing, installing again...
    pip install -r requirements.txt --no-cache-dir
)

echo.
echo [INFO] Environment is ready!
echo.

REM Create module structure if it doesn't exist
python -c "import os; dirs = ['config', 'data', 'execution', 'strategies', 'optimization', 'reporting', 'alerts', 'utils']; [os.makedirs(d, exist_ok=True) for d in dirs]; [open(os.path.join(d, '__init__.py'), 'a').close() for d in dirs]"

REM Run the fix and run script
echo [INFO] Running the Trading Bot setup and launcher...
python fix_and_run.py

pause