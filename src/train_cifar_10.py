from model_cifar_10 import FFNN
from load_data import DataLoaderCifar10
import numpy as np
from datetime import datetime
import wandb
import getpass
import argparse

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
    parser.add_argument('--save_plots', type=str, default=None,
                        help='Directory to save plots')
    
    return parser.parse_args()

def train(args=None):
    if args is None:
        args = parse_arguments()
    
    # Parse hidden layers from string
    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    # Parse hidden layers from string
    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(',')]
    
    data_loader = DataLoaderCifar10()
    # Get formatted data ready for neural networks
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()

    if args.use_subset:
        X_train, y_train = data_loader.create_subset(split_ratio=args.subset_ratio)
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
        standardize=args.standardize

    )

    # Train the model
    model.train(X_train, y_train, X_val, y_val)
    
    # Plot training history if not disabled
    if not args.no_plots:
        save_path = None
        if args.save_plots:
            save_path = f"{args.save_plots}/training_history_{experiment_name}.png"
        model.plot_training_history(save_path=save_path)

    # Evaluate the model
    test_accuracy = model.evaluate(X_test, y_test)
    
    # Generate confusion matrix and plots
    if not args.no_plots:
        confusion_save_path = None
        if args.save_plots:
            confusion_save_path = f"{args.save_plots}/confusion_matrix_{experiment_name}.png"
            model.confusion_matrix_scratch_plot(X_test, y_test)
    
    model.log_final_confusion_matrix(X_val, y_val)

    print(f"Test accuracy: {test_accuracy:.4f}")
    
    if not args.no_wandb:
        wandb.log({"final_test_accuracy": test_accuracy})
        
    return model, test_accuracy


if __name__ == "__main__":
    train()