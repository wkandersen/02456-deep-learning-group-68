from model import FFNN
from load_data import DataLoader
import numpy as np
from datetime import datetime
import wandb
import getpass
import yaml


user = getpass.getuser()

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

# Load sweep configuration from config.yaml
def load_sweep_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    weight_init = config.weights_init
    activation = config.activation
    loss = config.loss

    # Load CIFAR-10 data using formatted data
    data_loader = DataLoader()
    _, (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()
    (X_subset, y_subset) = data_loader.create_subset(split_ratio=0.25)

    # Initialize the model
    model = FFNN(
        num_epochs=num_epochs,
        hidden_layers=hidden_layers,
        lr=lr,
        optimizer=optimizer,
        batch_size=batch_size,
        l2_coeff=l2_coeff,
        weight_init=weight_init,
        activation=activation,
        _loss=loss
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
    sweep_configuration["name"] = f"Sweep_{user}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=sweep_configuration["project"])
    print(f"Sweep created with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{sweep_configuration['entity']}/{sweep_configuration['project']}/sweeps/{sweep_id}")
    
    # Run sweep agents
    wandb.agent(sweep_id, function=sweep_objective, count=10)  # Run 10 trials
