# AI Image Classifier - Complete Installation Guide

## Project Overview
A custom AI image classifier built with TensorFlow that can recognize and categorize images based on your own dataset.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Environment Setup](#environment-setup)
4. [Package Installation](#package-installation)
5. [Verification](#verification)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.8 or higher** (We used Python 3.14.0)
  - Download from: https://www.python.org/downloads/
  - During installation, make sure to check "Add Python to PATH"

- **Code Editor** (Recommended)
  - VS Code: https://code.visualstudio.com/
  - Or any text editor of your choice

- **Git** (Optional, for version control)
  - Download from: https://git-scm.com/

### Check Python Installation
Open Command Prompt or PowerShell and run:
```cmd
python --version
```
You should see: `Python 3.x.x`

---

## Project Setup

### Step 1: Create Project Structure

Navigate to where you want your project and run these commands:

```bash
# Create main project directory
mkdir Ai-Image-Classifier
cd Ai-Image-Classifier

# Create subdirectories
mkdir data\train data\test data\raw
mkdir models
mkdir notebooks
mkdir results\plots results\predictions
mkdir utils

# Create Python files
type nul > config.py
type nul > model.py
type nul > train.py
type nul > predict.py
type nul > utils\__init__.py
type nul > utils\data_loader.py
type nul > utils\preprocessor.py
```

### Step 2: Create requirements.txt

Create a file named `requirements.txt` in the root directory with this content:

```
tensorflow==2.15.0
numpy==1.24.3
matplotlib==3.7.1
pillow==10.0.0
scikit-learn==1.3.0
pandas==2.0.3
```

### Step 3: Create .gitignore

Create a `.gitignore` file to exclude unnecessary files from version control:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# Data files
data/raw/*
data/train/*
data/test/*

# Model files
models/*.h5
models/*.keras
*.pkl

# Jupyter Notebooks
.ipynb_checkpoints
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Results
results/plots/*.png
results/predictions/*
```

### Final Project Structure
```
Ai-Image-Classifier/
│   .gitignore
│   config.py
│   model.py
│   predict.py
│   README.md
│   requirements.txt
│   train.py
│
├───data/
│   ├───raw/
│   ├───test/
│   └───train/
│
├───models/
├───notebooks/
├───results/
│   ├───plots/
│   └───predictions/
└───utils/
    │   __init__.py
    │   data_loader.py
    └───preprocessor.py
```

---

## Environment Setup

### Option 1: Using Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated from other Python projects.

#### Create Virtual Environment

**In Command Prompt:**
```cmd
python -m venv venv
```

**In PowerShell:**
```powershell
python -m venv venv
```

**In VS Code:**
- VS Code may prompt you to create a virtual environment automatically
- Click "Create" when prompted
- Select your Python installation (Python 3.14.0 or higher)

#### Activate Virtual Environment

**Windows Command Prompt:**
```cmd
venv\Scripts\activate
```

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**If PowerShell gives an error, enable scripts:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

#### Verify Activation
You should see `(venv)` or `(.venv)` at the beginning of your command line:
```
(venv) D:\Ai-Image-Classifier>
```

### Option 2: Global Installation (Simpler, but not recommended for multiple projects)

Skip the virtual environment and install packages directly to your system Python.

---

## Package Installation

### Step 1: Install All Dependencies

Make sure you're in the `Ai-Image-Classifier` directory.

**If using virtual environment (recommended):**
```cmd
# Make sure venv is activated first (you should see (venv) in terminal)
pip install -r requirements.txt
```

**If installing globally:**
```cmd
pip install tensorflow numpy matplotlib pillow scikit-learn pandas
```

### Step 2: Wait for Installation

- TensorFlow is a large package (~500MB)
- Installation may take 5-10 minutes depending on your internet speed
- You may see warnings about PATH - these are safe to ignore
- The installation will show progress bars for each package

### What Gets Installed:

| Package | Version | Purpose |
|---------|---------|---------|
| **tensorflow** | 2.15.0 | Deep learning framework for building neural networks |
| **numpy** | 1.24.3 | Numerical computing and array operations |
| **matplotlib** | 3.7.1 | Data visualization and plotting |
| **pillow** | 10.0.0 | Image processing and manipulation |
| **scikit-learn** | 1.3.0 | Machine learning utilities and metrics |
| **pandas** | 2.0.3 | Data manipulation and analysis |

---

## Verification

### Step 1: Verify TensorFlow Installation

Run this command to check if TensorFlow is installed correctly:

```cmd
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Expected Output:**
```
TensorFlow version: 2.15.0
```

You may also see an informational message about oneDNN optimizations - this is normal and indicates TensorFlow is working correctly.

### Step 2: Verify All Packages

Create a test script to verify all packages:

```cmd
python -c "import tensorflow, numpy, matplotlib, PIL, sklearn, pandas; print('All packages installed successfully!')"
```

**Expected Output:**
```
All packages installed successfully!
```

### Step 3: Check Python Environment

Verify which Python you're using:

```cmd
python -c "import sys; print(sys.executable)"
```

- If using venv: Should show path to `venv\Scripts\python.exe`
- If using global: Should show path to your system Python

---

## Troubleshooting

### Issue 1: "python is not recognized"

**Solution:** Python is not in your PATH.
- Reinstall Python and check "Add Python to PATH" during installation
- Or manually add Python to your system PATH

### Issue 2: Virtual Environment Won't Activate

**For PowerShell users:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Alternative - Use Command Prompt instead:**
- Close PowerShell
- Open Command Prompt (cmd)
- Try activation again

### Issue 3: pip install fails with "permission denied"

**Solution:** Run terminal as Administrator or use:
```cmd
pip install --user tensorflow numpy matplotlib pillow scikit-learn pandas
```

### Issue 4: TensorFlow import shows warnings

**These are normal informational messages:**
- oneDNN optimization warnings
- GPU not found warnings (if you don't have a compatible GPU)
- These don't affect functionality

### Issue 5: Import errors after installation

**Solution:** Make sure virtual environment is activated
```cmd
# Deactivate and reactivate
deactivate
venv\Scripts\activate

# Reinstall packages
pip install -r requirements.txt
```

---

## Next Steps

Now that installation is complete, you're ready to:

1. **Prepare your dataset** - Organize images into category folders
2. **Configure the model** - Edit `config.py` with your settings
3. **Train the model** - Run `train.py` to train on your images
4. **Make predictions** - Use `predict.py` to classify new images

Refer to the main README.md for detailed usage instructions.

---

## Additional Resources

- **TensorFlow Documentation:** https://www.tensorflow.org/tutorials
- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Image Classification Guide:** https://www.tensorflow.org/tutorials/images/classification
- **Transfer Learning Tutorial:** https://www.tensorflow.org/tutorials/images/transfer_learning

---

## Installation Summary Checklist

- [ ] Python 3.8+ installed
- [ ] Project directory structure created
- [ ] requirements.txt file created
- [ ] Virtual environment created (optional but recommended)
- [ ] Virtual environment activated
- [ ] All packages installed successfully
- [ ] TensorFlow import verified
- [ ] Ready to start coding!

---

**Installation Date:** December 6, 2025  
**Python Version Used:** 3.14.0  
**TensorFlow Version:** 2.15.0  
**Installation Method:** Global installation (without virtual environment)

---
