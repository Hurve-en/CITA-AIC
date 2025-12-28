"""
Training Script for AI Image Classifier
========================================
This script trains the image classification model on your custom dataset.

Usage:
    python train.py

The script will:
1. Load your images from data/train/ directory
2. Build the model architecture
3. Train the model
4. Save the trained model
5. Generate training visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Import our custom modules
import config
from model import build_model
from utils.data_loader import (
    load_data_from_directory,
    create_augmented_generator,
    validate_dataset,
    plot_sample_images
)


# ==============================================================================
# TRAINING CALLBACKS
# ==============================================================================

def create_callbacks():
    """
    Create training callbacks for better training control.
    
    What are Callbacks?
    -------------------
    Callbacks are functions that run at certain points during training:
    - After each epoch
    - When validation loss stops improving
    - When learning rate needs adjustment
    
    They help:
    - Save the best model automatically
    - Stop training early if not improving (saves time)
    - Adjust learning rate dynamically
    - Log training progress
    
    Returns:
        list: List of callback objects
    """
    
    callbacks = []
    
    # -------------------------------------------------------------------------
    # 1. MODEL CHECKPOINT
    # -------------------------------------------------------------------------
    # Saves the best model during training (based on validation accuracy)
    # Even if training gets worse later, we keep the best version
    if config.SAVE_BEST_MODEL:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor='val_accuracy',  # Watch validation accuracy
            save_best_only=True,     # Only save when it's the best so far
            mode='max',              # Higher accuracy is better
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        print("✓ Model Checkpoint enabled - will save best model")
    
    # -------------------------------------------------------------------------
    # 2. EARLY STOPPING
    # -------------------------------------------------------------------------
    # Stops training if validation accuracy doesn't improve for N epochs
    # Prevents wasting time and overfitting
    if config.USE_EARLY_STOPPING:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOPPING_PATIENCE,  # Wait N epochs before stopping
            mode='max',
            restore_best_weights=True,  # Restore the best model when stopping
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        print(f"✓ Early Stopping enabled - patience: {config.EARLY_STOPPING_PATIENCE} epochs")
    
    # -------------------------------------------------------------------------
    # 3. REDUCE LEARNING RATE ON PLATEAU
    # -------------------------------------------------------------------------
    # Reduces learning rate when validation accuracy plateaus
    # Smaller learning rate = more careful adjustments = better fine-tuning
    if config.USE_REDUCE_LR:
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=config.REDUCE_LR_FACTOR,     # Multiply LR by this factor
            patience=config.REDUCE_LR_PATIENCE,  # Wait N epochs before reducing
            mode='max',
            min_lr=1e-7,  # Don't go below this learning rate
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        print(f"✓ Reduce LR on Plateau enabled - patience: {config.REDUCE_LR_PATIENCE} epochs")
    
    # -------------------------------------------------------------------------
    # 4. TENSORBOARD (Optional - for advanced monitoring)
    # -------------------------------------------------------------------------
    # Creates logs that can be visualized with TensorBoard
    # Run: tensorboard --logdir=logs
    log_dir = os.path.join(BASE_DIR, 'logs', 'fit', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1  # Log weight histograms every epoch
    )
    callbacks.append(tensorboard_callback)
    print(f"✓ TensorBoard enabled - logs saved to: {log_dir}")
    
    return callbacks


# ==============================================================================
# TRAINING VISUALIZATION
# ==============================================================================

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.
    
    This helps you understand:
    - Is the model learning? (accuracy should increase)
    - Is it overfitting? (gap between train and val accuracy)
    - When did it stop improving? (for early stopping analysis)
    
    Args:
        history: Training history object returned by model.fit()
    """
    
    # Extract metrics from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.PLOT_FIGSIZE)
    
    # -------------------------------------------------------------------------
    # Plot 1: Accuracy
    # -------------------------------------------------------------------------
    ax1.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Loss
    # -------------------------------------------------------------------------
    ax2.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if config.SAVE_PLOTS:
        plot_path = os.path.join(config.PLOTS_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training plot saved to: {plot_path}")
    
    # Show plot
    if config.SHOW_PLOTS:
        plt.show()


def plot_confusion_matrix(model, validation_dataset, class_names):
    """
    Create a confusion matrix showing which classes the model confuses.
    
    What is a Confusion Matrix?
    ----------------------------
    A table showing:
    - Rows: Actual classes
    - Columns: Predicted classes
    - Diagonal: Correct predictions
    - Off-diagonal: Mistakes
    
    Helps identify which classes are hardest to distinguish.
    
    Args:
        model: Trained model
        validation_dataset: Validation data generator
        class_names (dict): Class name mapping
    """
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Get predictions
    print("Generating confusion matrix...")
    y_true = validation_dataset.classes
    y_pred = model.predict(validation_dataset)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,  # Show numbers in cells
        fmt='d',     # Format as integers
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save
    if config.SAVE_PLOTS:
        plot_path = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {plot_path}")
    
    if config.SHOW_PLOTS:
        plt.show()


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_model():
    """
    Main training function that orchestrates the entire training process.
    
    Steps:
    ------
    1. Create necessary directories
    2. Validate dataset
    3. Load data
    4. Build model
    5. Train model
    6. Evaluate model
    7. Save model and results
    """
    
    print("\n" + "="*70)
    print("AI IMAGE CLASSIFIER - TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: Setup
    # -------------------------------------------------------------------------
    print("STEP 1: Setting up directories...")
    config.create_directories()
    config.print_config()
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # -------------------------------------------------------------------------
    # STEP 2: Validate Dataset
    # -------------------------------------------------------------------------
    print("STEP 2: Validating dataset...")
    is_valid = validate_dataset(config.TRAIN_DIR)
    
    if not is_valid:
        print("\n⚠️  Dataset validation found issues.")
        print("Please fix the issues above before training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # -------------------------------------------------------------------------
    # STEP 3: Load Data
    # -------------------------------------------------------------------------
    print("\nSTEP 3: Loading data...")
    
    if config.USE_DATA_AUGMENTATION:
        # Load with augmentation
        train_dataset, val_dataset, class_indices = create_augmented_generator(
            config.TRAIN_DIR,
            validation_split=config.VALIDATION_SPLIT
        )
    else:
        # Load without augmentation
        train_dataset, val_dataset, class_indices = load_data_from_directory(
            config.TRAIN_DIR,
            validation_split=config.VALIDATION_SPLIT
        )
    
    # Get class names (convert dict to list)
    class_names = list(class_indices.keys())
    num_classes = len(class_names)
    
    print(f"\n✓ Classes: {class_names}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Visualize some sample images
    print("\nDisplaying sample images...")
    plot_sample_images(train_dataset, class_indices, num_images=9)
    
    # -------------------------------------------------------------------------
    # STEP 4: Build Model
    # -------------------------------------------------------------------------
    print("\nSTEP 4: Building model...")
    model = build_model(num_classes)
    
    # -------------------------------------------------------------------------
    # STEP 5: Setup Callbacks
    # -------------------------------------------------------------------------
    print("\nSTEP 5: Setting up training callbacks...")
    callbacks = create_callbacks()
    
    # -------------------------------------------------------------------------
    # STEP 6: Train Model
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 6: TRAINING MODEL")
    print("="*70)
    print(f"This may take a while depending on your dataset size and hardware...")
    print(f"Training for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}")
    print("="*70 + "\n")
    
    # Calculate steps per epoch
    steps_per_epoch = train_dataset.samples // config.BATCH_SIZE
    validation_steps = val_dataset.samples // config.BATCH_SIZE
    
    # Start training!
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )
    
    # -------------------------------------------------------------------------
    # STEP 7: Evaluate Model
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 7: EVALUATING MODEL")
    print("="*70)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy, val_precision, val_recall, val_auc = model.evaluate(
        val_dataset,
        verbose=0
    )
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  AUC: {val_auc:.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 8: Save Results
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 8: SAVING RESULTS")
    print("="*70)
    
    # Save model
    model.save(config.MODEL_PATH)
    print(f"✓ Model saved to: {config.MODEL_PATH}")
    
    # Save class names
    class_names_path = os.path.join(config.MODEL_DIR, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"✓ Class names saved to: {class_names_path}")
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(model, val_dataset, class_names)
    
    # -------------------------------------------------------------------------
    # STEP 9: Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Model saved to: {config.MODEL_PATH}")
    print("\nNext steps:")
    print("1. Check the training plots in results/plots/")
    print("2. Use predict.py to test your model on new images")
    print("3. If accuracy is low, try:")
    print("   - Collecting more training data")
    print("   - Enabling data augmentation in config.py")
    print("   - Training for more epochs")
    print("   - Using transfer learning (if not already)")
    print("="*70 + "\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point for the training script.
    
    This runs when you execute: python train.py
    """
    
    try:
        # Run training
        train_model()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\nTraining interrupted by user.")
        print("Partial progress may have been saved.")
        sys.exit(0)
        
    except Exception as e:
        # Handle any errors
        print(f"\n❌ Error during training: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==============================================================================
# USAGE NOTES
# ==============================================================================
"""
How to use this script:
-----------------------

1. Prepare your data:
   - Put images in data/train/ folder
   - Organize into subfolders by class:
     data/train/
     ├── category1/
     ├── category2/
     └── category3/

2. Adjust settings in config.py:
   - Number of epochs
   - Batch size
   - Model type (custom_cnn or transfer_learning)
   - Enable/disable data augmentation

3. Run training:
   python train.py

4. Monitor training:
   - Watch the progress in terminal
   - Training plots will be generated
   - Best model automatically saved

5. After training:
   - Check results/plots/ for visualizations
   - Use predict.py to test your model
   - Model saved as models/image_classifier.keras

Common Issues:
--------------
- Out of memory? → Reduce BATCH_SIZE in config.py
- Training too slow? → Reduce IMG_SIZE or use transfer learning
- Not learning? → Increase EPOCHS or check your data
- Overfitting? → Enable data augmentation or increase DROPOUT_RATE
"""