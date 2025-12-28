# AI Image Classifier - Complete Usage Guide

## 📚 Table of Contents
1. [Quick Start](#quick-start)
2. [Preparing Your Dataset](#preparing-your-dataset)
3. [Training Your Model](#training-your-model)
4. [Making Predictions](#making-predictions)
5. [Understanding the Code](#understanding-the-code)
6. [Customization Guide](#customization-guide)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

---

## 🚀 Quick Start

### Step 1: Prepare Your Images

Organize your images into folders by category:

```
data/train/
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── dog3.jpg
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── cat3.jpg
└── birds/
    ├── bird1.jpg
    └── bird2.jpg
```

**Minimum Requirements:**
- At least 2 categories (can have more)
- Minimum 50 images per category (100-200 recommended)
- Images in JPG, PNG, or JPEG format

### Step 2: Train Your Model

```bash
python train.py
```

This will:
- Load your images
- Build the neural network
- Train the model (10-30 minutes depending on dataset size)
- Save the trained model to `models/image_classifier.keras`

### Step 3: Make Predictions

```bash
# Predict a single image
python predict.py path/to/image.jpg

# Predict all images in a folder
python predict.py path/to/folder/

# Interactive mode (predict multiple images)
python predict.py --interactive
```

---

## 📁 Preparing Your Dataset

### Option 1: Collect Your Own Images

1. **Choose your categories** (e.g., different dog breeds, types of flowers, emotions)

2. **Gather images:**
   - Download from Google Images (use [google-images-download](https://github.com/hardikvasa/google-images-download))
   - Take your own photos
   - Use public datasets (Kaggle, ImageNet, etc.)

3. **How many images do you need?**
   - **Minimum:** 50 images per category
   - **Good:** 100-200 images per category
   - **Great:** 500+ images per category
   - More images = better accuracy!

4. **Image quality tips:**
   - Use clear, well-lit images
   - Variety is good (different angles, backgrounds, lighting)
   - Avoid blurry or heavily filtered images
   - Similar image sizes work best

### Option 2: Use Example Datasets

Download pre-made datasets to practice:

**Kaggle Datasets (requires Kaggle account):**
- [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- [Flowers Recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)
- [Fruits 360](https://www.kaggle.com/moltean/fruits)

**Steps:**
1. Download and extract dataset
2. Move images to `data/train/` with proper folder structure
3. Run `python train.py`

### Dataset Organization Script

If your images are all mixed together, use this helper:

```python
# Split dataset into train and test
from utils.preprocessor import split_dataset

split_dataset(
    source_dir='data/raw',      # Where your images are now
    train_dir='data/train',     # Training folder (80%)
    test_dir='data/test',       # Testing folder (20%)
    test_split=0.2              # 20% for testing
)
```

### Dataset Validation

Before training, validate your dataset:

```python
from utils.preprocessor import analyze_dataset, clean_dataset

# Analyze your dataset
analyze_dataset('data/train')

# Clean dataset (remove corrupted/invalid images)
clean_dataset('data/train', min_size=(50, 50), remove_invalid=True)
```

---

## 🎯 Training Your Model

### Basic Training

Simply run:
```bash
python train.py
```

The script will:
1. ✓ Validate your dataset
2. ✓ Load and preprocess images
3. ✓ Build the model architecture
4. ✓ Train for specified epochs
5. ✓ Save the best model
6. ✓ Generate training plots

### Training Configuration

Edit `config.py` to customize training:

```python
# Key settings to adjust:

EPOCHS = 15                    # Number of training cycles
BATCH_SIZE = 32               # Images processed at once
LEARNING_RATE = 0.001         # How fast the model learns

MODEL_TYPE = 'transfer_learning'  # or 'custom_cnn'
USE_DATA_AUGMENTATION = True      # Recommended!

# For small datasets:
EPOCHS = 20
USE_DATA_AUGMENTATION = True

# For large datasets:
EPOCHS = 10
BATCH_SIZE = 64
```

### Training Modes

**1. Transfer Learning (Recommended)**
```python
# In config.py
MODEL_TYPE = 'transfer_learning'
PRETRAINED_MODEL = 'MobileNetV2'
FREEZE_BASE_MODEL = True
```
- **Best for:** Small datasets (100-500 images per class)
- **Pros:** Fast training, high accuracy, less data needed
- **Training time:** 10-15 minutes

**2. Custom CNN (From Scratch)**
```python
# In config.py
MODEL_TYPE = 'custom_cnn'
```
- **Best for:** Large datasets (500+ images per class), learning
- **Pros:** Full control, good for understanding neural networks
- **Training time:** 30-60 minutes

### Monitoring Training

Watch the training progress:
- **Accuracy:** Should increase each epoch (aim for 90%+)
- **Loss:** Should decrease each epoch
- **Gap between train/val:** Small gap = good, large gap = overfitting

**Example good training:**
```
Epoch 10/15
accuracy: 0.92, val_accuracy: 0.89  ← Good!
loss: 0.25, val_loss: 0.30
```

**Example overfitting:**
```
Epoch 10/15
accuracy: 0.98, val_accuracy: 0.75  ← Bad! Big gap
loss: 0.10, val_loss: 0.65
```

**If overfitting:**
1. Enable data augmentation: `USE_DATA_AUGMENTATION = True`
2. Increase dropout: `DROPOUT_RATE = 0.5`
3. Collect more training data

### After Training

Check your results:
- Training plots: `results/plots/training_history.png`
- Confusion matrix: `results/plots/confusion_matrix.png`
- Saved model: `models/image_classifier.keras`

---

## 🔮 Making Predictions

### Single Image Prediction

```bash
python predict.py path/to/image.jpg
```

Output:
```
Predicted class: dog
Confidence: 95.43%

All class probabilities:
  dog: 95.43%
  cat: 3.21%
  bird: 1.36%
```

### Batch Predictions

Predict all images in a folder:
```bash
python predict.py path/to/folder/
```

This creates a summary report with statistics.

### Interactive Mode

Predict multiple images without restarting:
```bash
python predict.py --interactive
```

### Programmatic Usage

Use predictions in your own Python code:

```python
from predict import load_trained_model, load_class_names, predict_single_image

# Load once
model = load_trained_model()
class_names = load_class_names()

# Predict many times
result = predict_single_image(model, 'image1.jpg', class_names, show_plot=False)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Get all probabilities
for class_name, prob in result['all_probabilities'].items():
    print(f"{class_name}: {prob*100:.2f}%")
```

### Confidence Thresholds

Adjust confidence threshold in `config.py`:
```python
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence required
```

Low confidence predictions get a warning.

---

## 💡 Understanding the Code

### File Structure Explained

```
Ai-Image-Classifier/
│
├── config.py              # All settings in one place
│                          # Change: epochs, batch size, model type
│
├── model.py               # Neural network architectures
│                          # Two options: custom CNN or transfer learning
│
├── train.py               # Training script
│                          # Run this to train your model
│
├── predict.py             # Prediction script
│                          # Run this to classify new images
│
├── utils/
│   ├── data_loader.py    # Load and preprocess images
│   │                      # Handles: resizing, normalization, augmentation
│   │
│   └── preprocessor.py   # Dataset utilities
│                          # Clean, split, analyze datasets
│
├── data/
│   ├── train/            # Training images (80%)
│   └── test/             # Testing images (20%)
│
├── models/               # Saved trained models
│   └── image_classifier.keras
│
└── results/
    ├── plots/            # Training visualizations
    └── predictions/      # Prediction results
```

### Key Concepts

#### 1. **Convolutional Neural Network (CNN)**
- Specialized for images
- Learns features automatically (edges → shapes → objects)
- Multiple layers process the image

#### 2. **Transfer Learning**
- Uses a pre-trained model (trained on millions of images)
- We add our own final layers
- Much faster and more accurate than training from scratch

#### 3. **Data Augmentation**
- Creates variations of images (rotate, flip, zoom)
- Helps model generalize better
- Effectively increases dataset size

#### 4. **Epochs**
- One complete pass through all training data
- More epochs = more learning (but risk overfitting)

#### 5. **Batch Size**
- Number of images processed together
- Larger = faster but needs more memory

#### 6. **Learning Rate**
- How big the steps are when adjusting the model
- Too large = unstable learning
- Too small = very slow learning

---

## ⚙️ Customization Guide

### For Different Datasets

**2 Classes (Binary Classification):**
```python
# In config.py - everything else stays the same
# The code automatically detects binary vs multi-class
```

**3-5 Classes:**
```python
# Default settings work great
EPOCHS = 15
MODEL_TYPE = 'transfer_learning'
```

**6-10 Classes:**
```python
EPOCHS = 20
DROPOUT_RATE = 0.5
USE_DATA_AUGMENTATION = True
```

**10+ Classes:**
```python
EPOCHS = 25
LEARNING_RATE = 0.0001  # Lower learning rate
USE_DATA_AUGMENTATION = True
```

### For Different Dataset Sizes

**Small Dataset (< 500 images total):**
```python
MODEL_TYPE = 'transfer_learning'
FREEZE_BASE_MODEL = True
USE_DATA_AUGMENTATION = True
EPOCHS = 20
DROPOUT_RATE = 0.5
```

**Medium Dataset (500-2000 images):**
```python
MODEL_TYPE = 'transfer_learning'
FREEZE_BASE_MODEL = True
USE_DATA_AUGMENTATION = True
EPOCHS = 15
```

**Large Dataset (2000+ images):**
```python
MODEL_TYPE = 'transfer_learning'
FREEZE_BASE_MODEL = False  # Train all layers
USE_DATA_AUGMENTATION = False
EPOCHS = 10
BATCH_SIZE = 64
```

### Changing Pre-trained Models

```python
# In config.py

# Fastest, lightest (recommended for beginners)
PRETRAINED_MODEL = 'MobileNetV2'

# More accurate, slower
PRETRAINED_MODEL = 'ResNet50'

# Good balance
PRETRAINED_MODEL = 'EfficientNetB0'
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Out of Memory Error**

**Problem:** GPU/RAM runs out of memory during training

**Solutions:**
```python
# In config.py
BATCH_SIZE = 16  # Reduce from 32
IMG_HEIGHT = 128  # Reduce from 224
IMG_WIDTH = 128
```

#### 2. **Model Not Learning (Accuracy Stuck)**

**Problem:** Accuracy doesn't improve, stays around 30-50%

**Solutions:**
1. Check your data is organized correctly
2. Make sure you have enough images per class
3. Try transfer learning:
   ```python
   MODEL_TYPE = 'transfer_learning'
   ```
4. Increase epochs:
   ```python
   EPOCHS = 25
   ```

#### 3. **Overfitting (Train Good, Val Bad)**

**Problem:** Training accuracy 95%, validation accuracy 60%

**Solutions:**
```python
USE_DATA_AUGMENTATION = True
DROPOUT_RATE = 0.5  # Increase dropout
EPOCHS = 10  # Reduce epochs
```

#### 4. **Training Very Slow**

**Problem:** Takes hours to train

**Solutions:**
```python
BATCH_SIZE = 64  # Increase batch size
IMG_HEIGHT = 128  # Reduce image size
IMG_WIDTH = 128
MODEL_TYPE = 'transfer_learning'  # Faster than custom CNN
PRETRAINED_MODEL = 'MobileNetV2'  # Fastest model
```

#### 5. **Low Prediction Confidence**

**Problem:** Model always predicts with low confidence (< 60%)

**Solutions:**
1. Collect more training data
2. Enable data augmentation
3. Train for more epochs
4. Check if your classes are too similar
5. Clean your dataset (remove ambiguous images)

### Error Messages

**"No module named 'tensorflow'"**
```bash
# Solution: Install TensorFlow
pip install tensorflow
```

**"Directory not found: data/train"**
```bash
# Solution: Create the directory and add images
mkdir data/train
# Then organize images into class subfolders
```

**"Found 0 images"**
```bash
# Solution: Check your folder structure
# Should be: data/train/class1/image1.jpg
#            data/train/class2/image1.jpg
```

---

## 🚀 Advanced Features

### Fine-Tuning

After initial training, fine-tune for better accuracy:

```python
# After running train.py once, run this:
from model import unfreeze_base_model
from tensorflow import keras

# Load your trained model
model = keras.models.load_model('models/image_classifier.keras')

# Unfreeze last 20 layers for fine-tuning
model = unfreeze_base_model(model, layers_to_unfreeze=20)

# Train for a few more epochs with lower learning rate
# (learning rate automatically reduced by unfreeze_base_model)
```

### Custom Data Augmentation

Customize augmentation in `config.py`:

```python
AUGMENTATION_CONFIG = {
    'rotation_range': 30,        # Rotate up to 30 degrees
    'width_shift_range': 0.3,    # Shift horizontally 30%
    'height_shift_range': 0.3,   # Shift vertically 30%
    'horizontal_flip': True,     # Flip horizontally
    'vertical_flip': False,      # Don't flip vertically (usually not natural)
    'zoom_range': 0.3,           # Zoom 30%
    'brightness_range': [0.8, 1.2],  # Adjust brightness
    'fill_mode': 'nearest'
}
```

### Model Ensemble

Combine multiple models for better predictions:

```python
# Train 3 different models
# 1. MobileNetV2
# 2. ResNet50
# 3. EfficientNetB0

# Then average their predictions
predictions = []
for model_path in ['model1.keras', 'model2.keras', 'model3.keras']:
    model = keras.models.load_model(model_path)
    pred = model.predict(image)
    predictions.append(pred)

# Average predictions
final_prediction = np.mean(predictions, axis=0)
```

### Grad-CAM Visualization

See what parts of the image the model focuses on:

```python
# Add this to predict.py for visualization
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def grad_cam(model, image, class_idx, layer_name='conv5_block3_out'):
    """Generate Grad-CAM heatmap"""
    grad_model = Model([model.inputs], 
                      [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
```

---

## 📊 Performance Benchmarks

### Expected Training Times (on modern CPU)

| Dataset Size | Model Type | Epochs | Time |
|-------------|-----------|---------|------|
| 500 images | Transfer Learning | 15 | 10-15 min |
| 500 images | Custom CNN | 15 | 30-40 min |
| 2000 images | Transfer Learning | 15 | 30-45 min |
| 2000 images | Custom CNN | 15 | 1-2 hours |

*With GPU: 5-10x faster*

### Expected Accuracy

| Scenario | Expected Accuracy |
|----------|------------------|
| Well-separated classes, 100+ images each | 90-98% |
| Similar classes, 100+ images each | 75-85% |
| Small dataset (< 50 per class) | 60-75% |
| Very similar classes (dog breeds) | 70-85% |

---

## 🎓 Learning Resources

### Concepts to Learn

1. **Neural Networks Basics**
   - [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

2. **Convolutional Neural Networks**
   - [CNN Explainer (Interactive)](https://poloclub.github.io/cnn-explainer/)

3. **Transfer Learning**
   - [TensorFlow Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)

4. **Image Classification**
   - [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Practice Projects

1. **Beginner:** Dogs vs Cats classifier
2. **Intermediate:** Multi-class flower recognition
3. **Advanced:** Fine-grained classification (bird species, car models)

---

## 📝 Checklist for Success

- [ ] Collect at least 100 images per category
- [ ] Organize images into proper folder structure
- [ ] Validate dataset with `analyze_dataset()`
- [ ] Enable data augmentation for small datasets
- [ ] Start with transfer learning
- [ ] Monitor training (watch for overfitting)
- [ ] Test on new images after training
- [ ] Adjust config and retrain if accuracy is low

---

## 🤝 Getting Help

If you're stuck:

1. **Check this guide** for common issues
2. **Review error messages** carefully
3. **Validate your dataset** structure
4. **Start simple** (2 classes, transfer learning, small dataset)
5. **Check the comments** in the code files

Remember: Machine learning is iterative! It's normal to train multiple times with different settings to get good results.

---

**Happy Training! 🚀**