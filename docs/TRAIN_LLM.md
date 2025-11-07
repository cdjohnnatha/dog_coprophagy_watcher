# LLM Training Script Documentation

## Overview

The `train_llm.py` script is a machine learning training pipeline that uses transfer learning with MobileNetV3Small to train an image classification model. The script is designed for categorical classification tasks, particularly suited for behavior recognition or similar computer vision applications.

## Usage

```bash
python scripts/train_llm.py --train_dir <training_data_path> --val_dir <validation_data_path> [--model_name <name>]
```

### Arguments

- `--train_dir`: Path to training dataset directory (required)
  - Should contain subdirectories for each class
  - Directory structure: `train_dir/class1/`, `train_dir/class2/`, etc.

- `--val_dir`: Path to validation dataset directory (required)
  - Should have the same class structure as training data

- `--model_name`: Name for the saved model (optional, default: "ellie_model")
  - Used as prefix for saved model files

### Examples

```bash
# Basic training
python scripts/train_llm.py --train_dir ./data/train --val_dir ./data/val

# Custom model name
python scripts/train_llm.py --train_dir ./data/train --val_dir ./data/val --model_name dog_behavior_classifier
```

## Model Architecture

### Base Model
- **Architecture**: MobileNetV3Small
- **Input Shape**: 224x224x3 (RGB images)
- **Pre-trained Weights**: ImageNet
- **Trainable**: Base layers frozen during training

### Data Augmentation
- Random horizontal flipping
- Random rotation (±5%)
- Random zoom (±10%)

### Classification Head
- Global average pooling
- Dropout (20%)
- Dense layer with softmax activation
- Number of classes automatically determined from dataset

## Training Configuration

### Hyperparameters
- **Image Size**: 224x224 pixels
- **Batch Size**: 64
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss**: Categorical cross-entropy
- **Metrics**: Accuracy

### Data Pipeline
- **Caching**: Datasets cached in memory for performance
- **Shuffling**: Training data shuffled with buffer size 1000
- **Prefetching**: AUTOTUNE for optimal performance

## Callbacks

### Model Checkpoint
- Saves best model based on validation performance
- Filename: `runs/{model_name}_best.keras`

### TensorBoard Logging
- Logs training metrics and graphs
- Directory: `runs/{model_name}_tb`

### Early Stopping
- Patience: 5 epochs
- Restores best weights on early stopping

## Output Files

Training generates the following files in the `runs/` directory:

1. **Best Model**: `{model_name}_best.keras`
   - Best performing model during training

2. **Final Model**: `{model_name}_final.keras`
   - Model after final epoch

3. **TensorBoard Logs**: `{model_name}_tb/`
   - Training metrics and visualization data

## Dataset Format

### Expected Structure
```
data/
├── train/
│   ├── class_1/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

### Requirements
- Images should be in standard formats (JPEG, PNG, etc.)
- Consistent class names between train and validation sets
- Sufficient data per class for meaningful training

## Dependencies

- **TensorFlow**: 2.x
- **Keras**: Included with TensorFlow
- **NumPy**: Required by TensorFlow
- **Pillow**: For image processing

Install requirements:
```bash
pip install tensorflow numpy pillow
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Training Output
The script provides real-time feedback including:
- Dataset loading information
- Training progress per epoch
- Validation metrics
- Early stopping notifications

## Performance Considerations

- **GPU Acceleration**: Automatically uses GPU if available
- **Memory Usage**: Models cached in memory - ensure sufficient RAM
- **Storage**: Plan for model file sizes (typically 10-50MB for MobileNetV3)
- **Training Time**: ~5-30 minutes depending on dataset size and hardware

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Ensure compatible GPU drivers for TensorFlow GPU
2. **Memory Errors**: Reduce batch size or use smaller images
3. **Class Imbalance**: Ensure balanced representation across classes
4. **Low Accuracy**: Consider more epochs or different architecture

### Validation

- Check that training and validation directories have matching class names
- Verify image formats are supported
- Ensure sufficient data per class (minimum 100-200 images recommended)

## Location

- **File**: `scripts/train_llm.py`
- **Output Directory**: `runs/` (created automatically)

## Integration

This script is part of a larger ML pipeline and can be:
- Called from other scripts
- Integrated into automated training workflows
- Used as a starting point for more complex models
- Extended with additional callbacks or metrics
