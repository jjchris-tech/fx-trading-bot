"""
Setup Script
Helps set up the FX Trading Bot by checking dependencies and
creating necessary directories.
"""
import os
import sys
import subprocess
import codecs
import importlib.util
import shutil
from pathlib import Path

# Force UTF-8 encoding for console output
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def print_header(text):
    """Print a header with decoration."""
    print("\n" + "=" * 60)
    print(f" {text} ".center(60, "="))
    print("=" * 60)

def print_step(step_number, step_text):
    """Print a step in the setup process."""
    print(f"\n[{step_number}] {step_text}")

def check_python_version():
    """Check if Python version is 3.8+."""
    print_step(1, "Checking Python version...")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"[FAIL] Python 3.8+ is required. You have Python {major}.{minor}")
        return False
    
    print(f"[OK] Python {major}.{minor} detected")
    return True

def check_pip():
    """Check if pip is installed."""
    print_step(2, "Checking pip installation...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("[OK] pip is installed")
        return True
    except subprocess.CalledProcessError:
        print("[FAIL] pip is not installed")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print_step(3, "Checking dependencies...")
    
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "plotly", "dash", "optuna",
        "scipy", "requests", "python-dotenv", "ta", "fpdf2", "discord-webhook",
        "pytest", "streamlit", "joblib", "scikit-learn", "tabulate", "coloredlogs"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"[FAIL] The following packages are missing: {', '.join(missing_packages)}")
        return False, missing_packages
    
    print("[OK] All required packages are installed")
    return True, []

def install_dependencies(missing_packages):
    """Install missing dependencies."""
    print_step(4, "Installing missing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing_packages], check=True)
        print("[OK] All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print_step(5, "Creating necessary directories...")
    
    directories = [
        "data_files",
        "data_files/market_data_cache",
        "data_files/sentiment_cache",
        "data_files/dashboard",
        "data_files/optimization",
        "data_files/walk_forward",
        "logs",
        "reports",
        "reports/metrics"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}")
    
    print("[OK] All directories created")
    return True

def check_project_structure():
    """Check if the project structure is correct."""
    print_step(6, "Checking project structure...")
    
    required_files = [
        "main.py",
        "backtest.py",
        "config/api_keys.py",
        "config/config.py",
        "config/parameters.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"[FAIL] The following files are missing: {', '.join(missing_files)}")
        return False
    
    print("[OK] Project structure looks good")
    return True

def create_empty_init_files():
    """Create empty __init__.py files in all directories."""
    print_step(7, "Creating __init__.py files...")
    
    module_dirs = [
        "config",
        "data",
        "execution",
        "strategies",
        "optimization",
        "reporting",
        "alerts",
        "utils",
        "tests"
    ]
    
    for directory in module_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    pass  # Create an empty file
                print(f"  Created file: {init_file}")
    
    print("[OK] All __init__.py files created")
    return True

def run_test_import():
    """Test importing the main modules."""
    print_step(8, "Testing imports...")
    
    try:
        import config
        import data
        import strategies
        import execution
        import optimization
        import reporting
        import alerts
        import utils
        
        print("[OK] All modules imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Error importing modules: {e}")
        return False

def setup():
    """Run the setup process."""
    print_header("FX Trading Bot Setup")
    
    # Check Python version
    if not check_python_version():
        print("\n[FAIL] Setup failed: Python version requirement not met")
        return False
    
    # Check pip
    if not check_pip():
        print("\n[FAIL] Setup failed: pip is not installed")
        return False
    
    # Check dependencies
    deps_ok, missing_packages = check_dependencies()
    if not deps_ok:
        # Install missing dependencies
        if not install_dependencies(missing_packages):
            print("\n[FAIL] Setup failed: could not install dependencies")
            return False
    
    # Create directories
    if not create_directories():
        print("\n[FAIL] Setup failed: could not create directories")
        return False
    
    # Check project structure
    if not check_project_structure():
        print("\n[FAIL] Setup failed: project structure is incorrect")
        return False
    
    # Create empty __init__.py files
    if not create_empty_init_files():
        print("\n[FAIL] Setup failed: could not create __init__.py files")
        return False
    
    # Run test import
    if not run_test_import():
        print("\n[FAIL] Setup failed: could not import modules")
        return False
    
    print_header("Setup Completed Successfully")
    print("\nYou can now run the bot with:")
    print("  python main.py")
    print("\nOr run a backtest with:")
    print("  python backtest.py --start-date 2023-01-01 --end-date 2023-12-31")
    
    return True

if __name__ == "__main__":
    setup()