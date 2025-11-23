from model_fashion_10 import FashionFFNN
from load_data import DataLoaderFashionMNIST
import numpy as np
from datetime import datetime
import wandb
import getpass
import yaml
import os


user = getpass.getuser()

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

# Load sweep configuration from config.yaml
def load_sweep_config():
    try:
        with open('src/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        if config is None:
            raise ValueError("Config file is empty or invalid")
        return config
    except FileNotFoundError:
        print("config.yaml not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print("Available files:", os.listdir('.'))
        raise
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

sweep_configuration = load_sweep_config()


def sweep_objective():
    run = wandb.init()
    config = wandb.config

    # Get parameters from wandb config (from config.yaml)
    num_epochs = config.num_epochs
    hidden_layers = config.n_hidden_units  # This will be one of the lists from config.yaml
    lr = config.learning_rate
    optimizer = config.optimizer
    batch_size = config.batch_size
    l2_coeff = config.l2_coeff
    weight_init = config.weight_init
    activation = config.activation
    loss = config.loss
    dropout = config.dropout
    batch_norm = config.batch_norm
    standardize = config.standardize

    # Load Fashion-MNIST data using formatted data
    data_loader = DataLoaderFashionMNIST()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_data()
    (X_subset, y_subset) = data_loader.create_subset(split_ratio=0.25)

    # Initialize the model
    model = FashionFFNN(
        input_size=784,  # Fashion-MNIST image size (28x28)
        num_epochs=num_epochs,
        hidden_layers=hidden_layers,
        lr=lr,
        optimizer=optimizer,
        batch_size=batch_size,
        l2_coeff=l2_coeff,
        weight_init=weight_init,
        activation=activation,
        _loss=loss,
        dropout_prob=dropout,
        batch_norm=batch_norm,
        standardize=standardize
    )

    # Train the model
    model.train(X_subset, y_subset, X_val, y_val)

    # Evaluate the model
    test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    wandb.log({"test_accuracy": test_accuracy})

    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    
    # Add custom name to sweep configuration
    sweep_configuration["name"] = f"FashionMNIST_Sweep_{user}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    
    # Get project name from config
    project_name = sweep_configuration.get("project", "Deep_learning_project")
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    
    # Run sweep agents
    wandb.agent(sweep_id, function=sweep_objective, count=100)  # Run 50 trials
