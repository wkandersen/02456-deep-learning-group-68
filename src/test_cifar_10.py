#!/usr/bin/env python3
"""
Test script for evaluating a trained CIFAR-10 model.
This script loads a pre-trained model and evaluates it on the test set.
"""

from model_cifar_10 import FFNN
from load_data import DataLoaderCifar10
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt

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
    
    print("Loading model...")
    try:
        # Load the trained model
        from model_cifar_10 import FFNN
        
        # Try loading with notebook parameters for CIFAR-10 first
        try:
            model = FFNN.load_model_with_notebook_params(args.model_path, 'cifar')
            print("Loaded with CIFAR-10 notebook parameters")
        except:
            # Fallback to regular loading with manual parameters
            model = FFNN.load_model(args.model_path,
                activation='leaky_relu',
                optimizer='adam', 
                weight_init='xavier',
                dropout_prob=0.2,
                batch_size=64
            )
            print("Loaded with manual parameters")
            
        print(f"Model loaded successfully from {args.model_path}")
        print(f"Model architecture: {model.input_size} -> {' -> '.join(map(str, model.hidden_layers))} -> {model.output_size}")
        print(f"Activation: {model.activation}, Batch norm: {model.batch_norm}, Dropout: {model.dropout_prob}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    print("\nLoading test data...")
    try:
        # Load test data
        data_loader = DataLoaderCifar10(args.data_path)
        X_test, y_test = data_loader.load_test_data()
        
        # Apply standardization if specified
        if args.standardize:
            # Note: In practice, you should save the training statistics and use them here
            # For now, we'll standardize based on the test set itself
            X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
            print("Applied standardization to test data")
        
        print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None
    
    print("\nEvaluating model on test set...")
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
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Results for {args.model_path}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Evaluation time: {eval_time:.2f} seconds\n")
        f.write(f"Total test samples: {X_test.shape[0]}\n")
        f.write(f"Model architecture: {model.input_size} -> {' -> '.join(map(str, model.hidden_layers))} -> {model.output_size}\n")
        f.write(f"Activation: {model.activation}, Batch norm: {model.batch_norm}, Dropout: {model.dropout_}\n")
    
    print(f"Results saved to {results_file}")
    
    # Generate confusion matrix if requested
    if args.confusion_matrix:
        print("\nGenerating confusion matrix...")
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        model.confusion_matrix_scratch_plot(X_test, y_test, cm_path)
        print(f"Confusion matrix saved to {cm_path}")
    
    # Show training history if available
    if hasattr(model, 'train_loss_history') and model.train_loss_history:
        print("\nPlotting training history...")
        history_path = os.path.join(args.output_dir, 'training_history.png')
        model.plot_training_history(save_path=history_path, show_plot=False)
        
        # Print training summary
        summary = model.get_training_summary()
        print("\nTraining Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return test_accuracy, test_loss

def test_in_batches(model, X_test, y_test, batch_size=1000):
    """
    Test the model in batches for memory efficiency.
    
    Args:
        model: Trained FFNN model
        X_test: Test features
        y_test: Test labels
        batch_size: Batch size for processing
    
    Returns:
        tuple: (accuracy, average_loss)
    """
    num_samples = X_test.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    total_correct = 0
    total_loss = 0
    
    print(f"Processing {num_samples} samples in {num_batches} batches...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Get predictions for this batch
        predictions = model.predict(X_batch)
        batch_correct = np.sum(predictions == y_batch)
        total_correct += batch_correct
        
        # Calculate loss for this batch
        logits = model.forward(X_batch, training=False)
        batch_loss = model.compute_loss(logits, y_batch)
        total_loss += batch_loss * len(X_batch)  # Weight by batch size
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Batch {i+1}/{num_batches} completed")
    
    accuracy = total_correct / num_samples
    avg_loss = total_loss / num_samples
    
    return accuracy, avg_loss

def main():
    args = parse_arguments()
    
    print("CIFAR-10 Model Testing")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Standardization: {args.standardize}")
    print("=" * 50)
    
    # Test the model
    try:
        result = test_model(args)
        if result is None or result == (None, None):
            print("\nTesting failed - see errors above.")
            return 1
            
        test_accuracy, test_loss = result
        print(f"\nTesting completed successfully!")
        print(f"Final results: Accuracy = {test_accuracy:.4f}, Loss = {test_loss:.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())