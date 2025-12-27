"""
Model Architecture for AI Image Classifier
==========================================
This file defines the neural network architectures we can use for image classification.
It includes both a custom CNN built from scratch and transfer learning options.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
import config


# ==============================================================================
# CUSTOM CNN ARCHITECTURE (Built from Scratch)
# ==============================================================================

def build_custom_cnn(num_classes):
    """
    Build a Convolutional Neural Network (CNN) from scratch.
    
    What is a CNN?
    --------------
    A CNN is a type of neural network designed specifically for images.
    It learns to detect features like edges, shapes, and patterns.
    
    Architecture Layers Explained:
    -------------------------------
    1. Conv2D layers: Extract features (edges, textures, patterns)
    2. MaxPooling: Reduce size while keeping important information
    3. Dropout: Prevent overfitting by randomly disabling neurons
    4. Dense layers: Make final classification decision
    
    Args:
        num_classes (int): Number of categories to classify
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    
    model = models.Sequential(name='Custom_CNN')
    
    # -------------------------------------------------------------------------
    # BLOCK 1: Initial Feature Extraction
    # -------------------------------------------------------------------------
    # First convolutional layer - detects basic features like edges
    # 32 filters = learns 32 different feature patterns
    # (3,3) = each filter looks at 3x3 pixel areas
    # activation='relu' = adds non-linearity (helps model learn complex patterns)
    model.add(layers.Conv2D(
        32, (3, 3), 
        activation='relu', 
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
        padding='same',  # 'same' keeps the image size the same
        name='conv1'
    ))
    
    # MaxPooling reduces image size by taking the maximum value in each 2x2 area
    # This makes training faster and helps the model focus on important features
    model.add(layers.MaxPooling2D((2, 2), name='pool1'))
    
    # -------------------------------------------------------------------------
    # BLOCK 2: Deeper Feature Extraction
    # -------------------------------------------------------------------------
    # Second conv layer - learns more complex patterns from the basic features
    # 64 filters = learns more patterns than the first layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))
    
    # -------------------------------------------------------------------------
    # BLOCK 3: High-Level Feature Extraction
    # -------------------------------------------------------------------------
    # Third conv layer - learns even more abstract features
    # 128 filters = learns the most complex patterns
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'))
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))
    
    # Optional fourth layer for larger datasets
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4'))
    model.add(layers.MaxPooling2D((2, 2), name='pool4'))
    
    # -------------------------------------------------------------------------
    # FLATTENING
    # -------------------------------------------------------------------------
    # Convert the 2D feature maps into a 1D vector
    # This is necessary before we can use regular Dense layers
    model.add(layers.Flatten(name='flatten'))
    
    # -------------------------------------------------------------------------
    # FULLY CONNECTED LAYERS (Classification Head)
    # -------------------------------------------------------------------------
    # Dense layer - combines all features to make decisions
    # 256 neurons = good balance between complexity and speed
    model.add(layers.Dense(256, activation='relu', name='dense1'))
    
    # Dropout - randomly turns off 40% of neurons during training
    # This prevents the model from memorizing the training data (overfitting)
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout1'))
    
    # Another Dense layer for more learning capacity
    model.add(layers.Dense(128, activation='relu', name='dense2'))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout2'))
    
    # -------------------------------------------------------------------------
    # OUTPUT LAYER
    # -------------------------------------------------------------------------
    # Final layer - outputs probability for each class
    # num_classes neurons = one for each category
    # softmax = converts outputs to probabilities that sum to 1
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


# ==============================================================================
# TRANSFER LEARNING ARCHITECTURE (Using Pre-trained Models)
# ==============================================================================

def build_transfer_learning_model(num_classes):
    """
    Build a model using transfer learning with a pre-trained base.
    
    What is Transfer Learning?
    --------------------------
    Instead of training from scratch, we use a model that's already been trained
    on millions of images (ImageNet dataset). This model already knows how to
    detect edges, shapes, and patterns. We just add our own layers on top to
    classify our specific categories.
    
    Benefits:
    ---------
    - Faster training (we only train the top layers)
    - Better accuracy with less data
    - Works well even with small datasets
    - Industry standard approach
    
    Args:
        num_classes (int): Number of categories to classify
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Pre-trained Base Model
    # -------------------------------------------------------------------------
    # Get the pre-trained model specified in config
    base_model = get_pretrained_base(config.PRETRAINED_MODEL)
    
    # Freeze or unfreeze the base model layers
    # Frozen = don't train these layers (faster, good for small datasets)
    # Unfrozen = train all layers (slower, better for large datasets)
    base_model.trainable = not config.FREEZE_BASE_MODEL
    
    if config.FREEZE_BASE_MODEL:
        print(f"✓ Base model ({config.PRETRAINED_MODEL}) layers are FROZEN")
        print("  Only the top layers will be trained (faster, less data needed)")
    else:
        print(f"✓ Base model ({config.PRETRAINED_MODEL}) layers are UNFROZEN")
        print("  All layers will be trained (slower, needs more data)")
    
    # -------------------------------------------------------------------------
    # STEP 2: Build the Complete Model
    # -------------------------------------------------------------------------
    # Create a Sequential model
    model = models.Sequential(name=f'Transfer_Learning_{config.PRETRAINED_MODEL}')
    
    # Add the pre-trained base model
    model.add(base_model)
    
    # -------------------------------------------------------------------------
    # STEP 3: Add Custom Classification Head
    # -------------------------------------------------------------------------
    # These layers are specific to our classification task
    
    # Global Average Pooling - reduces the spatial dimensions
    # Instead of flattening, this averages each feature map to a single number
    # This reduces parameters and helps prevent overfitting
    model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
    
    # Dense layer for learning task-specific features
    model.add(layers.Dense(512, activation='relu', name='dense1'))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout1'))
    
    # Another Dense layer (optional, but often helpful)
    model.add(layers.Dense(256, activation='relu', name='dense2'))
    model.add(layers.Dropout(config.DROPOUT_RATE, name='dropout2'))
    
    # Output layer - one neuron per class
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def get_pretrained_base(model_name):
    """
    Get a pre-trained base model.
    
    Available Models:
    -----------------
    - MobileNetV2: Lightweight, fast, good for most cases (recommended)
    - ResNet50: More accurate but slower and larger
    - EfficientNetB0: Good balance of accuracy and speed
    
    All models are pre-trained on ImageNet (1.4 million images, 1000 categories)
    
    Args:
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        keras.Model: Pre-trained base model
    """
    
    # All models expect images of size (224, 224, 3)
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    
    # Dictionary of available models
    models_dict = {
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50,
        'EfficientNetB0': EfficientNetB0
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models_dict.keys())}")
    
    # Load the model without the top classification layer (include_top=False)
    # We'll add our own classification layer instead
    # weights='imagenet' loads the pre-trained weights
    base_model = models_dict[model_name](
        include_top=False,      # Don't include the original classification layer
        weights='imagenet',     # Use pre-trained ImageNet weights
        input_shape=input_shape
    )
    
    print(f"✓ Loaded pre-trained {model_name} base model")
    
    return base_model


# ==============================================================================
# MODEL COMPILATION
# ==============================================================================

def compile_model(model, num_classes):
    """
    Compile the model with optimizer, loss function, and metrics.
    
    What is Model Compilation?
    --------------------------
    Compilation configures the model for training by specifying:
    1. Optimizer: How the model learns (adjusts weights)
    2. Loss function: What we're trying to minimize
    3. Metrics: What we measure to track progress
    
    Args:
        model (keras.Model): The model to compile
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    
    # -------------------------------------------------------------------------
    # OPTIMIZER: Adam
    # -------------------------------------------------------------------------
    # Adam is an adaptive optimizer that adjusts the learning rate automatically
    # It's the most popular optimizer because it works well in most cases
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    # -------------------------------------------------------------------------
    # LOSS FUNCTION
    # -------------------------------------------------------------------------
    # For multi-class classification, we use categorical crossentropy
    # This measures how different our predictions are from the true labels
    if num_classes == 2:
        # Binary classification (2 classes) - simpler loss function
        loss = 'binary_crossentropy'
        output_activation = 'sigmoid'
    else:
        # Multi-class classification (3+ classes)
        loss = 'categorical_crossentropy'
        output_activation = 'softmax'
    
    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    # Metrics help us track model performance during training
    metrics = [
        'accuracy',  # Percentage of correct predictions
        keras.metrics.Precision(name='precision'),  # Of predicted positives, how many are correct?
        keras.metrics.Recall(name='recall'),        # Of actual positives, how many did we find?
        keras.metrics.AUC(name='auc')               # Area Under the ROC Curve (overall quality)
    ]
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print("✓ Model compiled successfully")
    print(f"  Optimizer: Adam (lr={config.LEARNING_RATE})")
    print(f"  Loss: {loss}")
    print(f"  Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
    
    return model


# ==============================================================================
# MODEL BUILDER (Main Function)
# ==============================================================================

def build_model(num_classes):
    """
    Build and compile the complete model based on config settings.
    
    This is the main function you'll call to create your model.
    It automatically chooses between custom CNN or transfer learning
    based on the MODEL_TYPE setting in config.py
    
    Args:
        num_classes (int): Number of categories to classify
        
    Returns:
        keras.Model: Compiled model ready for training
    """
    
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    
    # Build the model architecture based on config
    if config.MODEL_TYPE == 'custom_cnn':
        print("Building Custom CNN from scratch...")
        model = build_custom_cnn(num_classes)
    elif config.MODEL_TYPE == 'transfer_learning':
        print("Building Transfer Learning model...")
        model = build_transfer_learning_model(num_classes)
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {config.MODEL_TYPE}. Choose 'custom_cnn' or 'transfer_learning'")
    
    # Compile the model
    model = compile_model(model, num_classes)
    
    # Print model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    print("="*70 + "\n")
    
    return model


# ==============================================================================
# MODEL UTILITIES
# ==============================================================================

def count_parameters(model):
    """
    Count the total, trainable, and non-trainable parameters in the model.
    
    Parameters = the weights that the model learns
    More parameters = more complex model (but slower and needs more data)
    
    Args:
        model (keras.Model): The model to analyze
    """
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total = trainable + non_trainable
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Non-trainable: {non_trainable:,}")


def unfreeze_base_model(model, layers_to_unfreeze='all'):
    """
    Unfreeze layers in the base model for fine-tuning.
    
    Fine-tuning: After initial training with frozen base, we can unfreeze
    some layers and train them with a lower learning rate to improve accuracy.
    
    Args:
        model (keras.Model): The model to unfreeze
        layers_to_unfreeze (str or int): 'all' or number of layers to unfreeze from the end
        
    Returns:
        keras.Model: Model with unfrozen layers
    """
    
    if config.MODEL_TYPE != 'transfer_learning':
        print("Unfreezing only applies to transfer learning models")
        return model
    
    base_model = model.layers[0]  # First layer is the base model
    
    if layers_to_unfreeze == 'all':
        base_model.trainable = True
        print(f"✓ All base model layers unfrozen")
    else:
        # Unfreeze only the last N layers
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        for layer in base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True
        print(f"✓ Last {layers_to_unfreeze} base model layers unfrozen")
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
"""
How to use this module:
-----------------------

1. Build a model:
   from model import build_model
   model = build_model(num_classes=5)  # For 5 categories

2. See model architecture:
   model.summary()

3. Count parameters:
   from model import count_parameters
   count_parameters(model)

4. For fine-tuning (after initial training):
   from model import unfreeze_base_model
   model = unfreeze_base_model(model, layers_to_unfreeze=20)
   # Then continue training with a lower learning rate
"""