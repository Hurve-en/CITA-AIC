"""
Data Loader for AI Image Classifier
====================================
This module handles loading images from folders, preprocessing them,
and preparing them for training or prediction.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import config


# ==============================================================================
# DATA LOADING FROM DIRECTORIES
# ==============================================================================

def load_data_from_directory(data_dir, validation_split=None):
    """
    Load images from a directory structure where each subfolder is a class.
    
    Expected Directory Structure:
    -----------------------------
    data/train/
    ├── category1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── category2/
    │   ├── image1.jpg
    │   └── ...
    └── category3/
        └── ...
    
    How it works:
    -------------
    1. Scans the directory for subfolders (each subfolder = one class)
    2. Loads all images from each subfolder
    3. Resizes images to the size specified in config
    4. Normalizes pixel values to 0-1 range
    5. Creates labels based on folder names
    
    Args:
        data_dir (str): Path to the directory containing class subfolders
        validation_split (float): Fraction of data to use for validation (e.g., 0.2 = 20%)
        
    Returns:
        tuple: (train_dataset, val_dataset, class_names) if validation_split is provided
               (dataset, class_names) otherwise
    """
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Check if directory has subdirectories (classes)
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) == 0:
        raise ValueError(f"No subdirectories found in {data_dir}. "
                        "Please organize images into class folders.")
    
    print(f"\n{'='*70}")
    print(f"LOADING DATA FROM: {data_dir}")
    print(f"{'='*70}")
    print(f"Found {len(subdirs)} categories: {subdirs}")
    
    # Count images per class
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        num_images = len([f for f in os.listdir(subdir_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {subdir}: {num_images} images")
    
    # -------------------------------------------------------------------------
    # Create Image Data Generator
    # -------------------------------------------------------------------------
    # ImageDataGenerator handles loading, preprocessing, and augmentation
    if validation_split and validation_split > 0:
        # Create generator with validation split
        # The data will be automatically split into training and validation
        datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values from 0-255 to 0-1
            validation_split=validation_split  # Split data into train/val
        )
        
        # Training dataset
        train_dataset = datagen.flow_from_directory(
            data_dir,
            target_size=config.IMG_SIZE,  # Resize all images to this size
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',     # For multi-class classification
            subset='training',            # This is the training split
            shuffle=True,                 # Shuffle data for better learning
            seed=config.RANDOM_SEED
        )
        
        # Validation dataset
        val_dataset = datagen.flow_from_directory(
            data_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',          # This is the validation split
            shuffle=False,                # Don't shuffle validation data
            seed=config.RANDOM_SEED
        )
        
        print(f"\n✓ Training samples: {train_dataset.samples}")
        print(f"✓ Validation samples: {val_dataset.samples}")
        print(f"{'='*70}\n")
        
        return train_dataset, val_dataset, train_dataset.class_indices
    
    else:
        # No validation split - load all data as training
        datagen = ImageDataGenerator(rescale=1./255)
        
        dataset = datagen.flow_from_directory(
            data_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        
        print(f"\n✓ Total samples: {dataset.samples}")
        print(f"{'='*70}\n")
        
        return dataset, dataset.class_indices


# ==============================================================================
# DATA AUGMENTATION
# ==============================================================================

def create_augmented_generator(data_dir, validation_split=None):
    """
    Create an image data generator with augmentation enabled.
    
    What is Data Augmentation?
    ---------------------------
    Data augmentation creates variations of your training images by:
    - Rotating them slightly
    - Flipping them horizontally
    - Zooming in/out
    - Shifting them left/right/up/down
    
    Why use it?
    -----------
    - Helps the model generalize better (works on new, unseen images)
    - Prevents overfitting (memorizing training data)
    - Effectively increases the size of your dataset
    - Makes your model more robust to variations in input
    
    Args:
        data_dir (str): Path to directory containing class subfolders
        validation_split (float): Fraction of data for validation
        
    Returns:
        tuple: (train_dataset, val_dataset, class_names)
    """
    
    print(f"\n{'='*70}")
    print("DATA AUGMENTATION ENABLED")
    print(f"{'='*70}")
    print("Augmentation techniques being applied:")
    for key, value in config.AUGMENTATION_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # -------------------------------------------------------------------------
    # Training Data Generator (with augmentation)
    # -------------------------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Always normalize
        validation_split=validation_split if validation_split else 0,
        **config.AUGMENTATION_CONFIG  # Apply all augmentation settings from config
    )
    
    train_dataset = train_datagen.flow_from_directory(
        data_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training' if validation_split else None,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # -------------------------------------------------------------------------
    # Validation Data Generator (NO augmentation)
    # -------------------------------------------------------------------------
    # Important: We don't augment validation data!
    # We want to test on real, unmodified images
    if validation_split and validation_split > 0:
        val_datagen = ImageDataGenerator(
            rescale=1./255,  # Only normalize, no augmentation
            validation_split=validation_split
        )
        
        val_dataset = val_datagen.flow_from_directory(
            data_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,  # Don't shuffle validation
            seed=config.RANDOM_SEED
        )
        
        print(f"✓ Training samples (augmented): {train_dataset.samples}")
        print(f"✓ Validation samples (original): {val_dataset.samples}")
        
        return train_dataset, val_dataset, train_dataset.class_indices
    
    else:
        print(f"✓ Training samples (augmented): {train_dataset.samples}")
        return train_dataset, train_dataset.class_indices


# ==============================================================================
# SINGLE IMAGE LOADING (for prediction)
# ==============================================================================

def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.
    
    Steps:
    ------
    1. Load image from file
    2. Resize to model input size
    3. Convert to array
    4. Normalize pixel values (0-1)
    5. Add batch dimension
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.array: Preprocessed image ready for prediction (shape: 1, height, width, channels)
    """
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image using TensorFlow
    img = keras.preprocessing.image.load_img(
        image_path,
        target_size=config.IMG_SIZE  # Resize to match model input
    )
    
    # Convert image to numpy array
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Normalize pixel values from 0-255 to 0-1
    # This matches the normalization used during training
    img_array = img_array / 255.0
    
    # Add batch dimension (models expect batches, even if batch size = 1)
    # Shape changes from (height, width, channels) to (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_image_for_display(image_path):
    """
    Load an image for display (without preprocessing).
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Image object that can be displayed with matplotlib
    """
    return keras.preprocessing.image.load_img(image_path)


# ==============================================================================
# DATA VISUALIZATION
# ==============================================================================

def visualize_augmentation(data_dir, class_name=None, num_images=9):
    """
    Visualize data augmentation by showing original and augmented versions.
    
    This helps you understand what augmentation is doing to your images
    and verify that the transformations look reasonable.
    
    Args:
        data_dir (str): Path to data directory
        class_name (str): Specific class to visualize (optional)
        num_images (int): Number of augmented versions to show
    """
    
    # Create augmented generator
    datagen = ImageDataGenerator(**config.AUGMENTATION_CONFIG)
    
    # Get one image to augment
    if class_name:
        # Load from specific class
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"No images found in {class_dir}")
            return
        img_path = os.path.join(class_dir, images[0])
    else:
        # Load from first available class
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        img_path = os.path.join(data_dir, classes[0], 
                               os.listdir(os.path.join(data_dir, classes[0]))[0])
    
    # Load and prepare image
    img = keras.preprocessing.image.load_img(img_path, target_size=config.IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Generate augmented images
    print(f"Showing {num_images} augmented versions of: {os.path.basename(img_path)}")
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    
    # Show original
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    i = 0
    for batch in datagen.flow(img_array, batch_size=1):
        if i >= num_images - 1:
            break
        axes[i + 1].imshow(batch[0].astype('uint8'))
        axes[i + 1].set_title(f'Augmented {i+1}')
        axes[i + 1].axis('off')
        i += 1
    
    plt.tight_layout()
    plt.show()


def plot_sample_images(dataset, class_names, num_images=9):
    """
    Display a grid of sample images from the dataset.
    
    Useful for:
    -----------
    - Verifying your data loaded correctly
    - Checking image quality
    - Understanding what your model will see
    
    Args:
        dataset: TensorFlow dataset or data generator
        class_names (dict): Dictionary mapping class indices to names
        num_images (int): Number of images to display
    """
    
    plt.figure(figsize=(12, 12))
    
    # Get one batch of images
    images, labels = next(dataset)
    
    # Create reverse mapping (index -> name)
    class_names_list = list(class_names.keys())
    
    # Plot images
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        # Get class name from label
        label_idx = np.argmax(labels[i])
        class_name = class_names_list[label_idx]
        
        plt.title(f'Class: {class_name}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

def validate_dataset(data_dir):
    """
    Validate the dataset structure and contents.
    
    Checks:
    -------
    - Directory exists
    - Contains class subdirectories
    - Each class has enough images
    - Image files are valid
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    
    print(f"\n{'='*70}")
    print("VALIDATING DATASET")
    print(f"{'='*70}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory not found: {data_dir}")
        return False
    
    # Get class directories
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(classes) == 0:
        print(f"❌ Error: No class directories found in {data_dir}")
        print("   Please create subdirectories for each class")
        return False
    
    print(f"✓ Found {len(classes)} classes: {classes}")
    
    # Validate each class
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    min_images_per_class = 10  # Minimum recommended
    
    all_valid = True
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        
        # Count images
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(valid_extensions)]
        num_images = len(images)
        total_images += num_images
        
        # Check minimum images
        if num_images < min_images_per_class:
            print(f"⚠️  Warning: Class '{class_name}' has only {num_images} images")
            print(f"   Recommended minimum: {min_images_per_class} images")
            all_valid = False
        else:
            print(f"✓ Class '{class_name}': {num_images} images")
    
    print(f"\nTotal images: {total_images}")
    
    if all_valid:
        print(f"✓ Dataset validation passed!")
    else:
        print(f"⚠️  Dataset has warnings (see above)")
    
    print(f"{'='*70}\n")
    
    return all_valid


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
"""
How to use this module:
-----------------------

1. Load data for training:
   from utils.data_loader import load_data_from_directory
   train_ds, val_ds, classes = load_data_from_directory(
       'data/train', 
       validation_split=0.2
   )

2. Load data with augmentation:
   from utils.data_loader import create_augmented_generator
   train_ds, val_ds, classes = create_augmented_generator(
       'data/train',
       validation_split=0.2
   )

3. Load a single image for prediction:
   from utils.data_loader import load_and_preprocess_image
   img = load_and_preprocess_image('path/to/image.jpg')
   prediction = model.predict(img)

4. Validate your dataset:
   from utils.data_loader import validate_dataset
   is_valid = validate_dataset('data/train')

5. Visualize augmentation:
   from utils.data_loader import visualize_augmentation
   visualize_augmentation('data/train')
"""