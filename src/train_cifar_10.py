from model_cifar_10 import FFNN
from load_data import DataLoaderCifar10
import numpy as np
from datetime import datetime
import wandb
import getpass
import argparse
import os

np.random.seed(42)
user = getpass.getuser()

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

def parse_arguments():
    """Parse command line arguments for CIFAR-10 training."""
    parser = argparse.ArgumentParser(description='Train FFNN on CIFAR-10 dataset')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer type (default: adam)')
    parser.add_argument('--l2_coeff', type=float, default=0.0001,
                        help='L2 regularization coefficient (default: 0.0001)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    # Model architecture
    parser.add_argument('--hidden_layers', type=str, default='1024,512,256',
                        help='Hidden layers as comma-separated values (default: 1024,512,256)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'],
                        help='Activation function (default: relu)')
    parser.add_argument('--weight_init', type=str, default='he',
                        choices=['xavier', 'he', 'random'],
                        help='Weight initialization method (default: he)')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function (default: cross_entropy)')
    parser.add_argument('--batch_norm', action='store_true',
                        help='Enable batch normalization')
    parser.add_argument('--standardize', action='store_true',
                        help='Enable input standardization')
    
    # Data parameters
    parser.add_argument('--subset_ratio', type=float, default=0.25,
                        help='Fraction of training data to use (default: 0.25)')
    parser.add_argument('--use_subset', action='store_true',
                        help='Use a subset of the training data for quick experiments')
    
    # WandB parameters
    parser.add_argument('--project_name', type=str, default='Deep_learning_project',
                        help='WandB project name (default: Deep_learning_project)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name (default: auto-generated)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    
    # Visualization
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--save_plots', type=str, default='Plots',
                        help='Directory to save plots')
    
    # Model saving
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model to file')
    parser.add_argument('--model_save_dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--save_weights_only', action='store_true',
                        help='Save only weights and biases (lighter file)')
    
    return parser.parse_args()

def train(args=None):
    if args is None:
        args = parse_arguments()
    

    if isinstance(args.hidden_layers, str):
        hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    else:
        hidden_layers = list(args.hidden_layers)
    
    data_loader = DataLoaderCifar10()
    # Get formatted data ready for neural networks
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()

    if args.use_subset:
        X_train, y_train = data_loader.create_subset(split_ratio=args.subset_ratio)

    if args.standardize:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0) + 1e-8  # Add small value to avoid division by zero
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

    input_size = 3072
    
    # Generate experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = f"CIFAR10_{user}_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    # Initialize WandB if not disabled
    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.project_name,
            name=experiment_name,
            config={
                "dataset": "CIFAR-10",
                "input_size": input_size,
                "num_epochs": args.num_epochs,
                "hidden_layers": hidden_layers,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "batch_size": args.batch_size,
                "l2_coeff": args.l2_coeff,
                "weight_init": args.weight_init,
                "activation": args.activation,
                "loss": args.loss,
                "subset_ratio": args.subset_ratio,
                "batch_norm": args.batch_norm,
                "standardize": args.standardize
            },
        )

    # Initialize the model
    model = FFNN(
        num_epochs=args.num_epochs,
        hidden_layers=hidden_layers,
        lr=args.learning_rate,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        l2_coeff=args.l2_coeff,
        weight_init=args.weight_init,
        activation=args.activation,
        _loss=args.loss,
        dropout_prob=args.dropout_rate,
        batch_norm=args.batch_norm,
    )

    # Train the model
    model.train(X_train, y_train, X_val, y_val)
    
    # Plot training history if not disabled
    if not args.no_plots:
        save_path = None
        if args.save_plots:
            # Ensure output directory exists
            os.makedirs(args.save_plots, exist_ok=True)
            save_path = f"{args.save_plots}/training_history_{experiment_name}.png"
        model.plot_training_history(save_path=save_path)

    # Evaluate the model
    test_accuracy = model.evaluate(X_test, y_test)
    
    # Generate confusion matrix and plots
    if not args.no_plots:
        confusion_save_path = None
        if args.save_plots:
            # Ensure output directory exists
            os.makedirs(args.save_plots, exist_ok=True)
            confusion_save_path = f"{args.save_plots}/confusion_matrix_{experiment_name}.png"
            model.confusion_matrix_scratch_plot(X_test, y_test, path=confusion_save_path)
    
    model.log_final_confusion_matrix(X_val, y_val)

    print(f"Test accuracy: {test_accuracy:.4f}")
    
    if not args.no_wandb:
        wandb.log({"final_test_accuracy": test_accuracy})
    
    # Save model if requested
    if args.save_model:
        # Create model save directory
        os.makedirs(args.model_save_dir, exist_ok=True)
        
        # Generate model filename
        model_filename = f"cifar10_model_{experiment_name}.pkl"
        model_path = os.path.join(args.model_save_dir, model_filename)
        
        try:
            if args.save_weights_only:
                model.save_weights_only(model_path)
                print(f"Model weights saved to: {model_path}")
            else:
                model.save_model(model_path)
                print(f"Complete model saved to: {model_path}")
                
            # Also save model configuration as text for reference
            config_filename = f"cifar10_config_{experiment_name}.txt"
            config_path = os.path.join(args.model_save_dir, config_filename)
            
            with open(config_path, 'w') as f:
                f.write(f"CIFAR-10 Model Configuration\n")
                f.write(f"{'='*40}\n")
                f.write(f"Model file: {model_filename}\n")
                f.write(f"Test accuracy: {test_accuracy:.4f}\n")
                f.write(f"Architecture: {model.input_size} -> {' -> '.join(map(str, model.hidden_layers))} -> {model.output_size}\n")
                f.write(f"Activation: {model.activation}\n")
                f.write(f"Optimizer: {model.optimizer}\n")
                f.write(f"Learning rate: {model.lr}\n")
                f.write(f"Batch size: {model.batch_size}\n")
                f.write(f"L2 coefficient: {model.l2_coeff}\n")
                f.write(f"Dropout probability: {model.dropout_prob}\n")
                f.write(f"Batch normalization: {model.batch_norm}\n")
                f.write(f"Weight initialization: {model.weight_init}\n")
                f.write(f"Loss function: {model._loss}\n")
                f.write(f"Epochs trained: {len(model.train_loss_history)}\n")
                
                if model.train_loss_history:
                    f.write(f"Final training loss: {model.train_loss_history[-1]:.4f}\n")
                    f.write(f"Final training accuracy: {model.train_acc_history[-1]:.4f}\n")
                
                val_losses = [loss for loss in model.val_loss_history if loss is not None]
                val_accs = [acc for acc in model.val_acc_history if acc is not None]
                if val_losses:
                    f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")
                    f.write(f"Final validation accuracy: {val_accs[-1]:.4f}\n")
                    f.write(f"Best validation accuracy: {max(val_accs):.4f}\n")
            
            print(f"Model configuration saved to: {config_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
        
    return model, test_accuracy


if __name__ == "__main__":
    train()