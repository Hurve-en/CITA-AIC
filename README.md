# Ai-Image-Classifier
A simple AI image classifier that uses machine learning to recognize and categorize images. This project trains a model to predict what an image shows and can be adapted for any dataset, such as animals, objects, or everyday items.

# POWER SHELL SCRIPT

# Create main project directory
mkdir Ai-Image-Classifier
cd Ai-Image-Classifier

# Folders
mkdir data
mkdir data/train
mkdir data/test
mkdir data/raw

mkdir models
mkdir notebooks
mkdir utils

mkdir results
mkdir results/plots
mkdir results/predictions

# Files
New-Item train.py -ItemType File
New-Item predict.py -ItemType File
New-Item model.py -ItemType File
New-Item config.py -ItemType File

New-Item utils/__init__.py -ItemType File
New-Item utils/data_loader.py -ItemType File
New-Item utils/preprocessor.py -ItemType File

# .gitkeep files
New-Item data/train/.gitkeep -ItemType File
New-Item data/test/.gitkeep -ItemType File
New-Item data/raw/.gitkeep -ItemType File
New-Item models/.gitkeep -ItemType File
New-Item notebooks/.gitkeep -ItemType File

## Project Structure

Ai-Image-Classifier/
├── data/              # Dataset storage
│   ├── train/        # Training images
│   ├── test/         # Testing images
│   └── raw/          # Raw unprocessed images
├── models/           # Saved trained models
├── notebooks/        # Jupyter notebooks for experiments
├── utils/            # Helper functions
├── results/          # Output results and plots
├── train.py          # Training script
├── predict.py        # Prediction script
├── model.py          # Model architecture
└── config.py         # Configuration settings


## Dataset
Add your images to \`data/train/\` organized in subdirectories by category:

data/train/
├── category1/
│   ├── img1.jpg
│   └── img2.jpg
└── category2/
    ├── img1.jpg
    └── img2.jpg
