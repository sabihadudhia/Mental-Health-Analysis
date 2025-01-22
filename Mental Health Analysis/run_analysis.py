import subprocess
import os
from datetime import datetime

def create_base_directories():
    """Create base directories needed for all scripts"""
    directories = [
        'plots',
        'results',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def run_script(script_name):
    """Run a script with proper error handling"""
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"Successfully ran {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

if __name__ == "__main__":
    # Create all necessary directories
    create_base_directories()
    
    # List of scripts to run in order
    scripts = [
        'data_exploration.py',
        'data_processing.py',
        'dimensionality_reduction.py',
        'clustering.py'
    ]

    # Run each script
    for script in scripts:
        run_script(script)