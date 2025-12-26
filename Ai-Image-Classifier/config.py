"""
Configuration file for the AI Image Classifier
===============================================
This file contains all the settings and hyperparameters for training and prediction.
By keeping all settings in one place, we can easily adjust our model without 
changing the main code.
"""

import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# These are the paths where our data, models, and results will be stored

# Base directory (current project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories - where our images are stored
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')  # Training images folder
TEST_DIR = os.path.join(DATA_DIR, 'test')     # Testing images folder (optional)
RAW_DIR = os.path.join(DATA_DIR, 'raw')       # Raw unprocessed images

# Model directory - where trained models are saved
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'image_classifier.keras')  # Saved model file

# Results directory - where outputs and visualizations are saved
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')


# ==============================================================================
# IMAGE PREPROCESSING SETTINGS
# ==============================================================================
# These settings control how images are processed before feeding to the model

# Image dimensions - all images will be resized to this size
# Smaller = faster training but less detail
# Larger = slower training but more detail
# 224x224 is a good balance and standard for many pre-trained models
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Number of color channels
# 3 = RGB (color images)
# 1 = Grayscale (black and white)
IMG_CHANNELS = 3


# ==============================================================================
# MODEL ARCHITECTURE SETTINGS
# ==============================================================================
# These settings define the structure of our neural network

# Choose model type:
# 'custom_cnn' = Build a CNN from scratch (good for learning)
# 'transfer_learning' = Use a pre-trained model (better accuracy, faster)
MODEL_TYPE = 'transfer_learning'  # Change to 'custom_cnn' to build from scratch

# For transfer learning, which pre-trained model to use:
# Options: 'MobileNetV2', 'ResNet50', 'EfficientNetB0'
# MobileNetV2 is lightweight and fast - good for beginners
PRETRAINED_MODEL = 'MobileNetV2'

# Whether to freeze the pre-trained layers initially
# True = Only train the final layers (faster, good for small datasets)
# False = Train all layers (slower, better if you have lots of data)
FREEZE_BASE_MODEL = True

# Dropout rate (0.0 to 1.0)
# Helps prevent overfitting by randomly turning off neurons during training
# 0.3-0.5 is typical (30-50% of neurons dropped)
DROPOUT_RATE = 0.4


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
# These control how the model learns

# Batch size - number of images processed together in one step
# Smaller = uses less memory but slower training
# Larger = faster but needs more memory
# 16 or 32 is good for most computers
BATCH_SIZE = 32

# Number of epochs - how many times to go through the entire dataset
# More epochs = more learning but risk of overfitting
# Start with 10-20, increase if model isn't learning enough
EPOCHS = 15

# Learning rate - how big the steps are when adjusting the model
# Smaller = more careful learning but slower
# Larger = faster but might miss the optimal solution
# 0.001 is a safe default, 0.0001 for fine-tuning
LEARNING_RATE = 0.001

# Validation split - what percentage of training data to use for validation
# 0.2 = 20% for validation, 80% for training
# Validation data helps us see if the model is overfitting
VALIDATION_SPLIT = 0.2


# ==============================================================================
# DATA AUGMENTATION SETTINGS
# ==============================================================================
# Data augmentation creates variations of images to help the model generalize better
# This is like showing the model the same image from different angles/lighting

# Enable data augmentation (True/False)
USE_DATA_AUGMENTATION = True

# Augmentation parameters (only used if USE_DATA_AUGMENTATION = True)
AUGMENTATION_CONFIG = {
    'rotation_range': 20,        # Randomly rotate images by up to 20 degrees
    'width_shift_range': 0.2,    # Randomly shift images horizontally by 20%
    'height_shift_range': 0.2,   # Randomly shift images vertically by 20%
    'horizontal_flip': True,     # Randomly flip images horizontally
    'zoom_range': 0.2,           # Randomly zoom in/out by 20%
    'shear_range': 0.1,          # Randomly apply shear transformation
    'fill_mode': 'nearest'       # How to fill in newly created pixels
}


# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================
# Callbacks are functions that run during training to help optimize the process

# Early stopping - stop training if model stops improving
# This prevents wasting time and overfitting
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5  # Stop after 5 epochs without improvement

# Model checkpoint - save the best model during training
# This ensures we keep the best version even if training gets worse later
SAVE_BEST_MODEL = True

# Reduce learning rate on plateau
# If model stops improving, reduce learning rate to make smaller adjustments
USE_REDUCE_LR = True
REDUCE_LR_PATIENCE = 3  # Reduce learning rate after 3 epochs without improvement
REDUCE_LR_FACTOR = 0.5  # Multiply learning rate by 0.5 when reducing


# ==============================================================================
# PREDICTION SETTINGS
# ==============================================================================
# Settings for making predictions on new images

# Confidence threshold - minimum probability to consider a prediction valid
# 0.5 = 50% confidence, 0.8 = 80% confidence
CONFIDENCE_THRESHOLD = 0.6

# Whether to show prediction probabilities for all classes
SHOW_ALL_PREDICTIONS = True


# ==============================================================================
# VISUALIZATION SETTINGS
# ==============================================================================
# Settings for plotting training results and predictions

# Whether to display plots during training
SHOW_PLOTS = True

# Whether to save plots to files
SAVE_PLOTS = True

# Plot figure size (width, height in inches)
PLOT_FIGSIZE = (12, 5)

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'  # Makes plots look nice


# ==============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ==============================================================================
# Setting a random seed ensures we get the same results each time we run the code
# This is useful for debugging and comparing different models

RANDOM_SEED = 42

# Note: Even with a seed, some operations (especially on GPU) may not be 
# perfectly reproducible due to hardware-level randomness


# ==============================================================================
# SYSTEM SETTINGS
# ==============================================================================

# Number of CPU cores to use for data loading
# -1 = use all available cores
# Set to a specific number if you want to limit CPU usage
NUM_WORKERS = 4

# Verbosity level for training output
# 0 = silent, 1 = progress bar, 2 = one line per epoch
VERBOSE = 1


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def create_directories():
    """
    Create all necessary directories if they don't exist.
    This function should be called at the start of training.
    """
    directories = [
        DATA_DIR, TRAIN_DIR, TEST_DIR, RAW_DIR,
        MODEL_DIR, RESULTS_DIR, PLOTS_DIR, PREDICTIONS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ All directories created/verified")


def print_config():
    """
    Print the current configuration settings.
    Useful for debugging and keeping track of experiments.
    """
    print("\n" + "="*70)
    print("IMAGE CLASSIFIER CONFIGURATION")
    print("="*70)
    print(f"\nModel Type: {MODEL_TYPE}")
    if MODEL_TYPE == 'transfer_learning':
        print(f"Pre-trained Model: {PRETRAINED_MODEL}")
        print(f"Freeze Base Model: {FREEZE_BASE_MODEL}")
    print(f"\nImage Size: {IMG_WIDTH}x{IMG_HEIGHT}x{IMG_CHANNELS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Validation Split: {VALIDATION_SPLIT * 100}%")
    print(f"Data Augmentation: {'Enabled' if USE_DATA_AUGMENTATION else 'Disabled'}")
    print(f"Early Stopping: {'Enabled' if USE_EARLY_STOPPING else 'Disabled'}")
    print("="*70 + "\n")


# ==============================================================================
# USAGE NOTES
# ==============================================================================
"""
How to use this config file:
-----------------------------

1. In other Python files, import settings like this:
   from config import IMG_SIZE, BATCH_SIZE, EPOCHS
   
2. Before training, call create_directories() to set up folders:
   from config import create_directories
   create_directories()
   
3. To see your current settings:
   from config import print_config
   print_config()
   
4. To change settings, just edit the values above and save this file.

Common adjustments:
-------------------
- Start overfitting? → Increase DROPOUT_RATE or enable USE_DATA_AUGMENTATION
- Training too slow? → Decrease IMG_SIZE or BATCH_SIZE
- Model not learning? → Increase EPOCHS or LEARNING_RATE
- Not enough data? → Enable USE_DATA_AUGMENTATION
- Want better accuracy? → Switch MODEL_TYPE to 'transfer_learning'
"""