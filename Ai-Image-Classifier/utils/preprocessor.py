"""
Image Preprocessing Utilities
==============================
This module contains utility functions for preparing and cleaning image data
before training or prediction.
"""

import os
import shutil
import numpy as np
from PIL import Image
import config


# ==============================================================================
# IMAGE VALIDATION
# ==============================================================================

def is_valid_image(image_path):
    """
    Check if a file is a valid image that can be opened.
    
    Why validate?
    -------------
    - Corrupted images can crash training
    - Non-image files might sneak into your dataset
    - Better to catch problems early
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    
    try:
        # Try to open and verify the image
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's actually an image
        return True
    
    except (IOError, OSError):
        # File is corrupted or not an image
        return False


def check_image_dimensions(image_path):
    """
    Get the dimensions of an image.
    
    Useful for:
    -----------
    - Finding images that are too small
    - Checking aspect ratios
    - Understanding your dataset
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        tuple: (width, height, channels) or None if invalid
    """
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Get number of channels (RGB=3, RGBA=4, Grayscale=1)
            channels = len(img.getbands())
            return (width, height, channels)
    
    except Exception:
        return None


# ==============================================================================
# DATASET CLEANING
# ==============================================================================

def clean_dataset(data_dir, min_size=(50, 50), remove_invalid=False):
    """
    Clean a dataset by finding and optionally removing problematic images.
    
    Problems checked:
    -----------------
    - Corrupted/invalid images
    - Images that are too small
    - Images with wrong number of channels (if not RGB)
    
    Args:
        data_dir (str): Path to dataset directory
        min_size (tuple): Minimum (width, height) for images
        remove_invalid (bool): If True, delete invalid images. If False, just report them.
        
    Returns:
        dict: Statistics about cleaning process
    """
    
    print(f"\n{'='*70}")
    print(f"CLEANING DATASET: {data_dir}")
    print(f"{'='*70}")
    print(f"Minimum size: {min_size[0]}x{min_size[1]}")
    print(f"Remove invalid: {remove_invalid}")
    print(f"{'='*70}\n")
    
    stats = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'too_small': 0,
        'wrong_channels': 0,
        'removed': 0
    }
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        print(f"Checking class: {class_name}")
        
        # Get all files in this class
        files = os.listdir(class_path)
        
        for filename in files:
            file_path = os.path.join(class_path, filename)
            
            # Skip non-files
            if not os.path.isfile(file_path):
                continue
            
            stats['total_images'] += 1
            
            # Check if valid image
            if not is_valid_image(file_path):
                stats['invalid_images'] += 1
                print(f"  ❌ Invalid: {filename}")
                
                if remove_invalid:
                    os.remove(file_path)
                    stats['removed'] += 1
                continue
            
            # Check dimensions
            dims = check_image_dimensions(file_path)
            if dims is None:
                stats['invalid_images'] += 1
                print(f"  ❌ Can't read: {filename}")
                
                if remove_invalid:
                    os.remove(file_path)
                    stats['removed'] += 1
                continue
            
            width, height, channels = dims
            
            # Check if too small
            if width < min_size[0] or height < min_size[1]:
                stats['too_small'] += 1
                print(f"  ⚠️  Too small: {filename} ({width}x{height})")
                
                if remove_invalid:
                    os.remove(file_path)
                    stats['removed'] += 1
                continue
            
            # Check channels (we want RGB = 3 channels)
            if channels != 3:
                stats['wrong_channels'] += 1
                print(f"  ⚠️  Wrong channels: {filename} ({channels} channels)")
                # Note: We might want to convert these instead of removing
            
            # Image passed all checks
            stats['valid_images'] += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("CLEANING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images checked: {stats['total_images']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Invalid images: {stats['invalid_images']}")
    print(f"Too small: {stats['too_small']}")
    print(f"Wrong channels: {stats['wrong_channels']}")
    if remove_invalid:
        print(f"Removed: {stats['removed']}")
    print(f"{'='*70}\n")
    
    return stats


# ==============================================================================
# IMAGE CONVERSION
# ==============================================================================

def convert_to_rgb(image_path, save_path=None):
    """
    Convert an image to RGB format.
    
    Why convert?
    ------------
    - Grayscale images have 1 channel, we need 3
    - RGBA images have 4 channels (with transparency)
    - Models expect RGB (3 channels)
    
    Args:
        image_path (str): Path to source image
        save_path (str): Path to save converted image (overwrites if None)
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB
        rgb_img = img.convert('RGB')
        
        # Save
        if save_path is None:
            save_path = image_path  # Overwrite original
        
        rgb_img.save(save_path)
        return True
    
    except Exception as e:
        print(f"Error converting {image_path}: {str(e)}")
        return False


def batch_convert_to_rgb(data_dir):
    """
    Convert all images in a dataset to RGB format.
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        int: Number of images converted
    """
    
    print(f"\n{'='*70}")
    print(f"CONVERTING IMAGES TO RGB: {data_dir}")
    print(f"{'='*70}\n")
    
    converted_count = 0
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        print(f"Processing class: {class_name}")
        
        # Get all image files
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            # Check if needs conversion
            dims = check_image_dimensions(file_path)
            if dims and dims[2] != 3:  # Not 3 channels
                if convert_to_rgb(file_path):
                    converted_count += 1
                    print(f"  ✓ Converted: {filename}")
    
    print(f"\n✓ Converted {converted_count} images to RGB")
    print(f"{'='*70}\n")
    
    return converted_count


# ==============================================================================
# IMAGE RESIZING
# ==============================================================================

def resize_image(image_path, target_size, save_path=None):
    """
    Resize an image to target size.
    
    Args:
        image_path (str): Path to source image
        target_size (tuple): Target (width, height)
        save_path (str): Path to save resized image (overwrites if None)
        
    Returns:
        bool: True if successful
    """
    
    try:
        img = Image.open(image_path)
        resized_img = img.resize(target_size, Image.LANCZOS)  # High-quality resize
        
        if save_path is None:
            save_path = image_path
        
        resized_img.save(save_path)
        return True
    
    except Exception as e:
        print(f"Error resizing {image_path}: {str(e)}")
        return False


# ==============================================================================
# DATASET SPLITTING
# ==============================================================================

def split_dataset(source_dir, train_dir, test_dir, test_split=0.2, seed=42):
    """
    Split a dataset into training and testing sets.
    
    Use this when:
    --------------
    - You have all images in one folder per class
    - You want to create separate train/test splits
    - You want reproducible splits (same split each time with same seed)
    
    Args:
        source_dir (str): Directory with class subfolders
        train_dir (str): Where to put training images
        test_dir (str): Where to put testing images
        test_split (float): Fraction of data for testing (0.2 = 20%)
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Statistics about the split
    """
    
    import random
    random.seed(seed)
    
    print(f"\n{'='*70}")
    print("SPLITTING DATASET")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Train: {train_dir}")
    print(f"Test: {test_dir}")
    print(f"Test split: {test_split*100}%")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    stats = {
        'classes': [],
        'total_train': 0,
        'total_test': 0,
        'per_class': {}
    }
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in class_dirs:
        print(f"Processing class: {class_name}")
        
        source_class_path = os.path.join(source_dir, class_name)
        train_class_path = os.path.join(train_dir, class_name)
        test_class_path = os.path.join(test_dir, class_name)
        
        # Create class directories
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)
        
        # Get all images in this class
        images = [f for f in os.listdir(source_class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split point
        test_count = int(len(images) * test_split)
        train_count = len(images) - test_count
        
        # Split images
        test_images = images[:test_count]
        train_images = images[test_count:]
        
        # Copy images to train directory
        for img in train_images:
            src = os.path.join(source_class_path, img)
            dst = os.path.join(train_class_path, img)
            shutil.copy2(src, dst)
        
        # Copy images to test directory
        for img in test_images:
            src = os.path.join(source_class_path, img)
            dst = os.path.join(test_class_path, img)
            shutil.copy2(src, dst)
        
        # Update statistics
        stats['classes'].append(class_name)
        stats['per_class'][class_name] = {
            'total': len(images),
            'train': train_count,
            'test': test_count
        }
        stats['total_train'] += train_count
        stats['total_test'] += test_count
        
        print(f"  ✓ {class_name}: {train_count} train, {test_count} test")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total classes: {len(stats['classes'])}")
    print(f"Total training images: {stats['total_train']}")
    print(f"Total testing images: {stats['total_test']}")
    print(f"{'='*70}\n")
    
    return stats


# ==============================================================================
# DATASET ANALYSIS
# ==============================================================================

def analyze_dataset(data_dir):
    """
    Analyze a dataset and provide detailed statistics.
    
    Provides:
    ---------
    - Number of classes
    - Images per class
    - Image size distribution
    - Channel distribution
    - Potential problems
    
    Args:
        data_dir (str): Path to dataset directory
    """
    
    print(f"\n{'='*70}")
    print(f"DATASET ANALYSIS: {data_dir}")
    print(f"{'='*70}\n")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    if not class_dirs:
        print("❌ No class directories found!")
        return
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}\n")
    
    all_sizes = []
    all_channels = []
    total_images = 0
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        
        # Get all image files
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        num_images = len(images)
        total_images += num_images
        
        print(f"{class_name}:")
        print(f"  Images: {num_images}")
        
        # Analyze a sample of images
        sample_size = min(10, num_images)
        for i in range(sample_size):
            img_path = os.path.join(class_path, images[i])
            dims = check_image_dimensions(img_path)
            if dims:
                all_sizes.append((dims[0], dims[1]))
                all_channels.append(dims[2])
        
        print()
    
    # Overall statistics
    print(f"{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total images: {total_images}")
    
    if all_sizes:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        
        print(f"\nImage dimensions:")
        print(f"  Min size: {min(widths)}x{min(heights)}")
        print(f"  Max size: {max(widths)}x{max(heights)}")
        print(f"  Avg size: {int(np.mean(widths))}x{int(np.mean(heights))}")
    
    if all_channels:
        unique_channels = set(all_channels)
        print(f"\nChannel distribution:")
        for ch in unique_channels:
            count = all_channels.count(ch)
            percentage = (count / len(all_channels)) * 100
            print(f"  {ch} channels: {count} images ({percentage:.1f}%)")
    
    print(f"{'='*70}\n")


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
"""
How to use this module:
-----------------------

1. Clean your dataset:
   from utils.preprocessor import clean_dataset
   stats = clean_dataset('data/train', min_size=(50, 50), remove_invalid=True)

2. Convert all images to RGB:
   from utils.preprocessor import batch_convert_to_rgb
   batch_convert_to_rgb('data/train')

3. Split dataset into train/test:
   from utils.preprocessor import split_dataset
   split_dataset('data/raw', 'data/train', 'data/test', test_split=0.2)

4. Analyze your dataset:
   from utils.preprocessor import analyze_dataset
   analyze_dataset('data/train')

5. Check a single image:
   from utils.preprocessor import is_valid_image, check_image_dimensions
   if is_valid_image('image.jpg'):
       width, height, channels = check_image_dimensions('image.jpg')
       print(f"Image size: {width}x{height}, Channels: {channels}")
"""