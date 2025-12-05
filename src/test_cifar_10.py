#!/usr/bin/env python3
"""
Test script for evaluating a trained CIFAR-10 model.
This script loads a pre-trained model and evaluates it on the test set.
"""

from load_data import DataLoaderCifar10
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt


bold_text = "\033[1m"
reset_text = "\033[0m"
def parse_arguments():
    """Parse command line arguments for CIFAR-10 testing."""
    parser = argparse.ArgumentParser(description='Test trained FFNN on CIFAR-10 dataset')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Path to CIFAR-10 data directory (default: ../data)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for testing (default: 1000)')
    parser.add_argument('--standardize', action='store_true',
                        help='Apply standardization to test data')
    parser.add_argument('--confusion_matrix', action='store_true',
                        help='Generate and save confusion matrix')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results (default: ./test_results)')
    
    return parser.parse_args()

def test_model(args):
    """Load and test the trained model."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    from model_cifar_10 import FFNN
    # Fallback to regular loading with manual parameters
    model = FFNN.load_model(args.model_path,
        activation='relu',
        optimizer='adam',
        weight_init='random',
        dropout_prob=0.5,
        batch_size=128,
        standardize=True,
        batch_norm=True
    )
    print("Loaded with manual parameters")


    try:
        # Load test data
        data_loader = DataLoaderCifar10()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()
        if args.standardize:
                mean = np.mean(X_train, axis=0)
                std = np.std(X_train, axis=0) + 1e-8  # Add small value to avoid division by zero
                X_train = (X_train - mean) / std
                X_val = (X_val - mean) / std
                X_test = (X_test - mean) / std

    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None

    start_time = time.time()
    
    # Evaluate the model
    test_accuracy = model.evaluate(X_test, y_test)
    
    # Calculate test loss
    test_predictions = model.forward(X_test, training=False)
    test_loss = model.compute_loss(test_predictions, y_test)
    
    eval_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Accuracy: {bold_text}{test_accuracy:.4f} ({test_accuracy*100:.2f}%){reset_text}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Total test samples: {X_test.shape[0]}")
    print(f"{'='*50}")

    # Define CIFAR-10 class names
    cifar_class_names = data_loader.get_class_names()
        
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix_cifar.png')

        # Create a custom confusion matrix plot for CIFAR-10 with class names
        preds = model.predict(X_test)
        true, pred = y_test, preds
        labels = sorted(set(true))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        num_classes = len(labels)
        mat = np.zeros((num_classes, num_classes))

        for t, p in zip(true, pred):
            i = label_to_index[t]
            j = label_to_index[p]
            mat[i, j] += 1

        plt.figure(figsize=(12, 10))
        import seaborn as sns
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", 
                   xticklabels=cifar_class_names, yticklabels=cifar_class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("CIFAR-10 Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        plt.close()
        
    
    # Show training history if available
    if hasattr(model, 'train_loss_history') and model.train_loss_history:
        print("\\nPlotting training history...")
        history_path = os.path.join(args.output_dir, 'training_history_cifar.png')
        model.plot_training_history(save_path=history_path, show_plot=False)
        
        # Print training summary
        summary = model.get_training_summary()
        print("\\nTraining Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Show per-class accuracy
    preds = model.predict(X_test)
    per_class_accuracy = {}

    for class_idx, class_name in enumerate(cifar_class_names):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_preds = preds[class_mask]
            class_true = y_test[class_mask]
            accuracy = np.mean(class_preds == class_true)
            per_class_accuracy[class_name] = accuracy
    
    return 


