"""
Prediction Script for AI Image Classifier
==========================================
This script uses your trained model to classify new images.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder/  (to classify all images in folder)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Import our custom modules
import config
from utils.data_loader import load_and_preprocess_image, load_image_for_display


# ==============================================================================
# LOAD MODEL AND CLASS NAMES
# ==============================================================================

def load_trained_model():
    """
    Load the trained model from disk.
    
    Returns:
        keras.Model: Loaded model ready for prediction
    """
    
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}\n"
            f"Please train the model first by running: python train.py"
        )
    
    print(f"Loading model from: {config.MODEL_PATH}")
    model = keras.models.load_model(config.MODEL_PATH)
    print("✓ Model loaded successfully")
    
    return model


def load_class_names():
    """
    Load class names from the saved text file.
    
    Returns:
        list: List of class names in order
    """
    
    class_names_path = os.path.join(config.MODEL_DIR, 'class_names.txt')
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Class names file not found at {class_names_path}\n"
            f"Please train the model first by running: python train.py"
        )
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded {len(class_names)} classes: {class_names}")
    
    return class_names


# ==============================================================================
# PREDICTION FUNCTIONS
# ==============================================================================

def predict_single_image(model, image_path, class_names, show_plot=True):
    """
    Predict the class of a single image.
    
    Process:
    --------
    1. Load and preprocess the image
    2. Feed it through the model
    3. Get probability for each class
    4. Return the class with highest probability
    
    Args:
        model: Trained Keras model
        image_path (str): Path to the image file
        class_names (list): List of class names
        show_plot (bool): Whether to display the image with prediction
        
    Returns:
        dict: Prediction results containing:
            - predicted_class: Name of predicted class
            - confidence: Confidence percentage
            - all_probabilities: Probabilities for all classes
    """
    
    print(f"\nPredicting: {image_path}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load and Preprocess Image
    # -------------------------------------------------------------------------
    # Convert image to the format expected by the model
    preprocessed_img = load_and_preprocess_image(image_path)
    
    # -------------------------------------------------------------------------
    # STEP 2: Make Prediction
    # -------------------------------------------------------------------------
    # model.predict() returns probabilities for each class
    # Shape: (1, num_classes) - one row of probabilities
    predictions = model.predict(preprocessed_img, verbose=0)
    
    # Get probabilities for all classes (flatten from 2D to 1D)
    probabilities = predictions[0]
    
    # Find the class with highest probability
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    # -------------------------------------------------------------------------
    # STEP 3: Display Results
    # -------------------------------------------------------------------------
    print(f"✓ Prediction complete!")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    
    # Show all class probabilities if configured
    if config.SHOW_ALL_PREDICTIONS:
        print(f"\n  All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            print(f"    {class_name}: {prob*100:.2f}%")
    
    # Check if confidence is above threshold
    if confidence < config.CONFIDENCE_THRESHOLD * 100:
        print(f"\n  ⚠️  Warning: Confidence is below threshold "
              f"({config.CONFIDENCE_THRESHOLD*100:.2f}%)")
        print(f"     The model is uncertain about this prediction.")
    
    # -------------------------------------------------------------------------
    # STEP 4: Visualize (if requested)
    # -------------------------------------------------------------------------
    if show_plot:
        visualize_prediction(
            image_path, 
            predicted_class, 
            confidence, 
            class_names, 
            probabilities
        )
    
    # Return results as dictionary
    return {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': dict(zip(class_names, probabilities))
    }


def predict_batch(model, image_paths, class_names):
    """
    Predict classes for multiple images at once.
    
    More efficient than predicting one by one when you have many images.
    
    Args:
        model: Trained Keras model
        image_paths (list): List of paths to image files
        class_names (list): List of class names
        
    Returns:
        list: List of prediction dictionaries
    """
    
    print(f"\nPredicting {len(image_paths)} images...")
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] ", end="")
        
        try:
            result = predict_single_image(
                model, 
                image_path, 
                class_names, 
                show_plot=False  # Don't show plot for batch
            )
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error predicting {image_path}: {str(e)}")
            continue
    
    return results


def predict_folder(model, folder_path, class_names):
    """
    Predict all images in a folder.
    
    Args:
        model: Trained Keras model
        folder_path (str): Path to folder containing images
        class_names (list): List of class names
        
    Returns:
        list: List of prediction dictionaries
    """
    
    # Get all image files in folder
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Predict all images
    results = predict_batch(model, image_files, class_names)
    
    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_prediction(image_path, predicted_class, confidence, 
                        class_names, probabilities):
    """
    Display the image with prediction results.
    
    Shows:
    - The original image
    - Predicted class and confidence
    - Bar chart of probabilities for all classes
    
    Args:
        image_path (str): Path to image
        predicted_class (str): Predicted class name
        confidence (float): Confidence percentage
        class_names (list): All class names
        probabilities (np.array): Probabilities for all classes
    """
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------------------------------------------------------------------------
    # LEFT: Display Image
    # -------------------------------------------------------------------------
    img = load_image_for_display(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    
    # Add title with prediction
    title_color = 'green' if confidence >= config.CONFIDENCE_THRESHOLD * 100 else 'orange'
    ax1.set_title(
        f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%',
        fontsize=14,
        fontweight='bold',
        color=title_color
    )
    
    # -------------------------------------------------------------------------
    # RIGHT: Probability Bar Chart
    # -------------------------------------------------------------------------
    # Sort classes by probability (highest first)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices] * 100
    
    # Create horizontal bar chart
    colors = ['green' if i == sorted_indices[0] else 'skyblue' 
              for i in range(len(sorted_indices))]
    
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, sorted_probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_classes)
    ax2.set_xlabel('Probability (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels on bars
    for i, prob in enumerate(sorted_probs):
        ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save if configured
    if config.SAVE_PLOTS:
        filename = os.path.basename(image_path).split('.')[0] + '_prediction.png'
        save_path = os.path.join(config.PREDICTIONS_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction visualization saved to: {save_path}")
    
    plt.show()


def create_summary_report(results, save_path=None):
    """
    Create a summary report of batch predictions.
    
    Args:
        results (list): List of prediction dictionaries
        save_path (str): Path to save report (optional)
    """
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    
    # Count predictions per class
    class_counts = {}
    for result in results:
        pred_class = result['predicted_class']
        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
    
    # Calculate average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    # Print summary
    print(f"Total images predicted: {len(results)}")
    print(f"Average confidence: {avg_confidence:.2f}%")
    print(f"\nPredictions per class:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Low confidence warnings
    low_conf_count = sum(1 for r in results 
                        if r['confidence'] < config.CONFIDENCE_THRESHOLD * 100)
    if low_conf_count > 0:
        print(f"\n⚠️  {low_conf_count} predictions below confidence threshold")
    
    print("="*70 + "\n")
    
    # Save report if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("PREDICTION SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total images: {len(results)}\n")
            f.write(f"Average confidence: {avg_confidence:.2f}%\n\n")
            f.write("Detailed Results:\n")
            f.write("-"*70 + "\n")
            for result in results:
                f.write(f"\nImage: {os.path.basename(result['image_path'])}\n")
                f.write(f"Predicted: {result['predicted_class']}\n")
                f.write(f"Confidence: {result['confidence']:.2f}%\n")
        
        print(f"✓ Report saved to: {save_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function to handle command-line predictions.
    
    Usage:
        python predict.py image.jpg           # Single image
        python predict.py folder/             # All images in folder
        python predict.py img1.jpg img2.jpg   # Multiple images
    """
    
    print("\n" + "="*70)
    print("AI IMAGE CLASSIFIER - PREDICTION")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # Check command-line arguments
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <image_path>         # Predict single image")
        print("  python predict.py <folder_path>        # Predict all images in folder")
        print("  python predict.py <img1> <img2> ...    # Predict multiple images")
        print("\nExample:")
        print("  python predict.py data/test/sample.jpg")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Load model and class names
    # -------------------------------------------------------------------------
    try:
        model = load_trained_model()
        class_names = load_class_names()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # Handle different input types
    # -------------------------------------------------------------------------
    paths = sys.argv[1:]
    
    # Case 1: Single path provided
    if len(paths) == 1:
        path = paths[0]
        
        # Check if it's a folder
        if os.path.isdir(path):
            print(f"Processing all images in folder: {path}")
            results = predict_folder(model, path, class_names)
            
            if results:
                # Create summary report
                report_path = os.path.join(
                    config.PREDICTIONS_DIR,
                    'batch_prediction_report.txt'
                )
                create_summary_report(results, report_path)
        
        # It's a single file
        elif os.path.isfile(path):
            predict_single_image(model, path, class_names, show_plot=True)
        
        else:
            print(f"❌ Error: Path not found: {path}")
            sys.exit(1)
    
    # Case 2: Multiple paths provided
    else:
        print(f"Processing {len(paths)} images...")
        
        # Filter valid file paths
        valid_paths = [p for p in paths if os.path.isfile(p)]
        
        if not valid_paths:
            print("❌ Error: No valid image files found")
            sys.exit(1)
        
        # Predict all images
        results = predict_batch(model, valid_paths, class_names)
        
        # Create summary
        create_summary_report(results)
    
    print("\n✓ Prediction complete!")


# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================

def interactive_mode():
    """
    Run predictions in interactive mode.
    
    Allows user to predict multiple images without restarting the script.
    Useful for testing and experimentation.
    """
    
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("Type 'quit' or 'exit' to stop\n")
    
    # Load model once
    model = load_trained_model()
    class_names = load_class_names()
    
    while True:
        # Get input from user
        path = input("\nEnter image path (or folder): ").strip()
        
        # Check for exit commands
        if path.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        # Skip empty input
        if not path:
            continue
        
        # Predict
        try:
            if os.path.isfile(path):
                predict_single_image(model, path, class_names, show_plot=True)
            elif os.path.isdir(path):
                results = predict_folder(model, path, class_names)
                create_summary_report(results)
            else:
                print(f"❌ Path not found: {path}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point for the prediction script.
    
    Runs when you execute: python predict.py
    """
    
    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        main()


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
"""
Command-line Usage:
-------------------

1. Predict single image:
   python predict.py path/to/image.jpg

2. Predict all images in a folder:
   python predict.py path/to/folder/

3. Predict multiple specific images:
   python predict.py img1.jpg img2.jpg img3.jpg

4. Interactive mode:
   python predict.py --interactive

Programmatic Usage:
-------------------

from predict import load_trained_model, load_class_names, predict_single_image

# Load model
model = load_trained_model()
class_names = load_class_names()

# Predict
result = predict_single_image(model, 'image.jpg', class_names)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")
"""