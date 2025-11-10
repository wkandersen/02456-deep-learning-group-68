from model import FFNN
from load_data import CIFAR10Loader
import numpy as np
from datetime import datetime
import wandb

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")


sweep_configuration = {
    "method": "random",  # Can also use "bayes"
    "name": f"Sweep_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [256,512]},
        "l2_coeff": {"values": [0.001,0.0001]},
        "lr": {"values": [0.1,0.01]},
    }
}


def sweep_objective():
    run = wandb.init()
    config = wandb.config

    num_epochs = 5
    hidden_layers = [512, 256, 128]
    optimizer = 'adam'
    weight_init = 'he'
    activation = 'relu'
    loss = 'mse'

    # Load CIFAR-10 data
    data_loader = CIFAR10Loader()
    train_images, train_labels = data_loader.get_training_data()
    test_images, test_labels = data_loader.get_test_data()
    val_images, val_labels = data_loader.get_validation_data()

    # Initialize the model
    model = FFNN(
        num_epochs=num_epochs,
        hidden_layers=hidden_layers,
        lr=config.lr,
        optimizer=optimizer,
        batch_size=config.batch_size,
        l2_coeff=config.l2_coeff,
        weight_init=weight_init,
        activation=activation,
        _loss=loss
    )



    # Train the model
    model.train(train_images, train_labels, val_images, val_labels)
    # model.plot_training_history()
    test_accuracy = model.evaluate(test_images, test_labels)
    wandb.log({"test_accuracy": test_accuracy})
    run.finish()

# Launch sweep
if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Deep_learning_project")
    wandb.agent(sweep_id, function=sweep_objective, count=16)  # Run 15 trials
