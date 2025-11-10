from model_cifar_10 import FFNN
from load_data import DataLoaderCifar10
import numpy as np
from datetime import datetime
import wandb
import getpass

user = getpass.getuser()

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

def train():
    data_loader = DataLoaderCifar10()
    # Get formatted data ready for neural networks
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()
    X_train, y_train = data_loader.create_subset(split_ratio=0.25)

    input_size = 3072
    num_epochs = 5
    hidden_layers = [1024, 512, 256]
    lr = 0.001
    optimizer = 'adam'
    batch_size = 512
    l2_coeff = 0.0001
    weight_init = 'he'
    activation = 'relu'
    loss = 'cross_entropy'

    run = wandb.init(
        project="Deep_learning_project",
        name=f"Trial_{user}_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        config={
            "num_epochs": num_epochs,
            "hidden_layers": hidden_layers,
            "lr": lr,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "l2_coeff": l2_coeff,
            "weight_init": weight_init,
            "activation": activation,
            "loss": loss,
        },
    )

    # Load CIFAR-10 data

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
    model.train(X_train, y_train, X_val, y_val)
    model.plot_training_history()

    # Evaluate the model
    test_accuracy = model.evaluate(X_test, y_test)
    model.confusion_matrix_plot(X_test, y_test)
    model.log_final_confusion_matrix(X_val, y_val)

    print(f"Test accuracy: {test_accuracy}")
    wandb.log({"test_accuracy": test_accuracy})


if __name__ == "__main__":
    train()