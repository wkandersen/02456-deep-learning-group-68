#!/usr/bin/env python3
"""
Test script for evaluating a trained Fashion-MNIST model.
This script loads a pre-trained model and evaluates it on the test set.
"""

from model_fashion_10 import FashionFFNN
from load_data import DataLoaderFashionMNIST
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt

bold_text = "\033[1m"
reset_text = "\033[0m"
def parse_arguments():
    """Parse command line arguments for Fashion-MNIST testing."""
    parser = argparse.ArgumentParser(description='Test trained FFNN on Fashion-MNIST dataset')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model file')
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
    
    try:
        # Load the trained model using the parent FFNN class
        # Since FashionFFNN inherits from FFNN, we can load it as FFNN
        from model_cifar_10 import FFNN
        
        # Try loading with notebook parameters for Fashion-MNIST first
        try:
            model = FFNN.load_model_with_notebook_params(args.model_path, 'fashion')
            print("Loaded with Fashion-MNIST notebook parameters")
        except:
            # Fallback to regular loading with manual parameters
            model = FFNN.load_model(args.model_path,
                activation='leaky_relu',
                optimizer='adam',
                weight_init='xavier',
                dropout_prob=0.3,
                batch_size=128
            )
            print("Loaded with manual parameters")

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    try:
        # Load test data
        data_loader = DataLoaderFashionMNIST()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_data()
            
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
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Total test samples: {X_test.shape[0]}")
    print(f"{'='*50}")
    
    # Define Fashion-MNIST class names
    fashion_class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
        
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix_fashion.png')
        
        # Create a custom confusion matrix plot for Fashion-MNIST with class names
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
                   xticklabels=fashion_class_names, yticklabels=fashion_class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Fashion-MNIST Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # print(f"Confusion matrix saved to {cm_path}")
    
    # Show training history if available
    if hasattr(model, 'train_loss_history') and model.train_loss_history:
        print("\\nPlotting training history...")
        history_path = os.path.join(args.output_dir, 'training_history_fashion.png')
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
    
    for class_idx, class_name in enumerate(fashion_class_names):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_preds = preds[class_mask]
            class_true = y_test[class_mask]
            accuracy = np.mean(class_preds == class_true)
            per_class_accuracy[class_name] = accuracy

    
    return test_accuracy, test_loss

def main():
    args = parse_arguments()
    
    # Test the model
    try:
        result = test_model(args)
        if result is None or result == (None, None):
            print("\nTesting failed - see errors above.")
            return 1
            
        test_accuracy, test_loss = result
        print(f"\nTesting completed successfully!")
        print(f"{bold_text}Final results: Accuracy = {test_accuracy:.4f}, Loss = {test_loss:.4f}{reset_text}")
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())