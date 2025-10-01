#!/usr/bin/env python3
"""
Setup script for Baby Cry Classification project
This script helps set up the environment and check requirements
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_installed_packages():
    """Check if required packages are installed"""
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print("❌ requirements.txt not found")
        return False
    
    print("🔍 Checking installed packages...")
    
    with open(requirements_file, 'r') as f:
        requirements = f.read().splitlines()
    
    missing_packages = []
    for requirement in requirements:
        if requirement.strip() and not requirement.startswith('#'):
            package_name = requirement.split('>=')[0].split('==')[0].split('[')[0]
            try:
                pkg_resources.get_distribution(package_name)
                print(f"✅ {package_name}")
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package_name)
                print(f"❌ {package_name} - Not installed")
    
    return len(missing_packages) == 0, missing_packages

def check_data_structure():
    """Check if data directory structure is correct"""
    print("📁 Checking data structure...")
    
    data_dir = Path("Data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("Please ensure you have the dataset in the 'Data' folder")
        return False
    
    expected_classes = ["belly pain", "cold_hot", "discomfort", "hungry", "tired"]
    missing_classes = []
    
    for class_name in expected_classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            print(f"❌ {class_name} - Directory not found")
        else:
            audio_files = list(class_dir.glob("*.wav"))
            print(f"✅ {class_name} - {len(audio_files)} audio files")
    
    if missing_classes:
        print(f"❌ Missing class directories: {missing_classes}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    
    directories = ["models", "web_app/uploads", "web_app/static"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {directory}")

def check_gpu_availability():
    """Check if GPU is available for training"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("ℹ️  No GPU found, will use CPU (training will be slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def check_disk_space():
    """Check available disk space"""
    print("💾 Checking disk space...")
    
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (1024**3)
    
    if free_gb < 5:
        print(f"⚠️  Low disk space: {free_gb}GB available")
        print("Recommended: At least 5GB for models and processed data")
        return False
    else:
        print(f"✅ Sufficient disk space: {free_gb}GB available")
        return True

def main():
    """Main setup function"""
    print("🚀 Baby Cry Classification - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check disk space
    check_disk_space()
    
    # Create directories
    create_directories()
    
    # Check/install packages
    packages_ok, missing = check_installed_packages()
    if not packages_ok:
        print(f"\n📦 Missing packages: {missing}")
        install_choice = input("Do you want to install missing packages? (y/n): ")
        if install_choice.lower() == 'y':
            if not install_requirements():
                print("❌ Failed to install packages")
                sys.exit(1)
        else:
            print("❌ Cannot proceed without required packages")
            sys.exit(1)
    
    # Check GPU
    check_gpu_availability()
    
    # Check data structure
    if not check_data_structure():
        print("\n⚠️  Data structure check failed")
        print("Please ensure you have the dataset properly organized in the 'Data' folder")
        print("Expected structure:")
        print("Data/")
        print("├── belly pain/")
        print("├── cold_hot/")
        print("├── discomfort/")
        print("├── hungry/")
        print("└── tired/")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Ensure your dataset is in the 'Data' folder")
    print("2. Run: python train_model.py (to train the model)")
    print("3. Run: cd web_app && python app.py (to start the web interface)")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()