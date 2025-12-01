# Model Save/Load Functionality

This document explains how to use the new model saving and loading functionality for the CIFAR-10 and Fashion-MNIST neural network models.

## Overview

The FFNN class now supports saving and loading trained models, including:
- Model weights and biases
- Batch normalization parameters (if enabled)
- Model configuration (architecture, hyperparameters)
- Training history
- Optimizer state (for resuming training)

## Quick Start

### Training and Saving a Model

```bash
# Train CIFAR-10 model and save it
python train_cifar_10.py --save_model --num_epochs 50 --hidden_layers 1024,512,256

# Train Fashion-MNIST model and save it
python train_fashion_10.py --save_model --num_epochs 30 --hidden_layers 512,256,128
```

### Testing a Saved Model

```bash
# Test CIFAR-10 model
python test_cifar_10.py --model_path models/cifar10_model_*.pkl --confusion_matrix

# Test Fashion-MNIST model
python test_fashion_10.py --model_path models/fashion_model_*.pkl --confusion_matrix
```

## Detailed Usage

### 1. Saving Models During Training

#### Training Arguments for Model Saving

- `--save_model`: Enable model saving
- `--model_save_dir`: Directory to save models (default: `models`)
- `--save_weights_only`: Save only weights and biases (lighter file)

#### Examples

```bash
# Save complete model (recommended)
python train_cifar_10.py --save_model --num_epochs 20

# Save only weights (smaller file, but no training history)
python train_cifar_10.py --save_model --save_weights_only --num_epochs 20

# Save to custom directory
python train_cifar_10.py --save_model --model_save_dir my_models
```

### 2. Manual Saving in Code

```python
from model_cifar_10 import FFNN

# Create and train model
model = FFNN(...)
model.train(X_train, y_train, X_val, y_val)

# Save complete model (includes config, history, optimizer state)
model.save_model('path/to/model.pkl')

# Save only weights and biases (lighter file)
model.save_weights_only('path/to/weights.pkl')
```

### 3. Loading Models

#### Load Complete Model

```python
from model_cifar_10 import FFNN

# Load complete model
model = FFNN.load_model('path/to/model.pkl')

# Model is ready for testing
predictions = model.predict(X_test)
accuracy = model.evaluate(X_test, y_test)
```

#### Load Weights Only

```python
# Create new model with same architecture
model = FFNN(
    num_epochs=50,
    hidden_layers=[1024, 512, 256],
    lr=0.001,
    # ... other parameters must match saved model
)

# Load weights into the new model
model.load_weights_only('path/to/weights.pkl')
```

### 4. Testing Scripts

Both testing scripts provide comprehensive evaluation:

#### CIFAR-10 Testing

```bash
python test_cifar_10.py \
  --model_path models/your_model.pkl \
  --confusion_matrix \
  --output_dir test_results \
  --standardize
```

#### Fashion-MNIST Testing

```bash
python test_fashion_10.py \
  --model_path models/your_model.pkl \
  --confusion_matrix \
  --output_dir test_results \
  --standardize
```

#### Testing Arguments

- `--model_path`: Path to saved model (required)
- `--confusion_matrix`: Generate confusion matrix plot
- `--output_dir`: Directory for test results (default: `test_results`)
- `--standardize`: Apply standardization to test data
- `--batch_size`: Batch size for testing (default: 1000)

## File Structure

When you save a model, the following files are created:

```
models/
├── cifar10_model_EXPERIMENT_NAME.pkl     # Complete model file
├── cifar10_config_EXPERIMENT_NAME.txt    # Human-readable config
├── fashion_model_EXPERIMENT_NAME.pkl     # Fashion model file
└── fashion_config_EXPERIMENT_NAME.txt    # Fashion config
```

## What's Saved

### Complete Model (`save_model`)
- ✅ Weights and biases
- ✅ Batch normalization parameters
- ✅ Model configuration (architecture, hyperparameters)
- ✅ Training history (loss and accuracy curves)
- ✅ Optimizer state (for Adam optimizer)

### Weights Only (`save_weights_only`)
- ✅ Weights and biases
- ✅ Batch normalization parameters
- ❌ Training history
- ❌ Optimizer state
- ❌ Configuration metadata

## Example Workflow

### 1. Train and Save

```bash
# Train a model with good hyperparameters
python train_cifar_10.py \
  --save_model \
  --num_epochs 100 \
  --hidden_layers 1024,512,256 \
  --learning_rate 0.001 \
  --optimizer adam \
  --batch_norm \
  --dropout_rate 0.3 \
  --l2_coeff 0.0001
```

### 2. Test the Saved Model

```bash
# Evaluate on test set
python test_cifar_10.py \
  --model_path models/cifar10_model_*.pkl \
  --confusion_matrix \
  --standardize
```

### 3. Use Model in Your Code

```python
from model_cifar_10 import FFNN
import numpy as np

# Load your trained model
model = FFNN.load_model('models/your_best_model.pkl')

# Use for prediction
new_data = np.random.randn(100, 3072)  # Example data
predictions = model.predict(new_data)

# Get prediction probabilities
logits = model.forward(new_data, training=False)
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Make sure the model path is correct
2. **Architecture Mismatch**: When loading weights only, ensure the new model has the same architecture
3. **Memory Issues**: Use `--save_weights_only` for large models if you only need the weights

### Compatibility

- Models can be loaded on different machines/environments
- Python pickle format is used (ensure compatible Python versions)
- All dependencies (numpy, etc.) must be available

## Demo Script

Run the example script to see the functionality in action:

```bash
python example_save_load.py
```

This will train a small model, save it, load it, and verify the functionality works correctly.

## Performance Notes

- **Complete model files**: Larger (includes history and config) but more convenient
- **Weights-only files**: Smaller but require manual architecture specification
- **Loading time**: Models load quickly even for large architectures
- **Memory usage**: Loaded models use the same memory as freshly trained models

## Integration with Existing Code

The save/load functionality is backward compatible. Existing training scripts will work unchanged, and you can add `--save_model` to enable saving without modifying code.