from model import FFNN
from load_data import DataLoader
import numpy as np
from datetime import datetime
import wandb

wandb.login(key="b26660ac7ccf436b5e62d823051917f4512f987a")

num_epochs = 10
hidden_layers = [2048, 1024, 512, 256]
lr = 0.001
optimizer = 'adam'
batch_size = 512
l2_coeff = 0.0001
weight_init = 'he'
activation = 'relu'
loss = 'cross_entropy'

run = wandb.init(
    project="Deep_learning_project",
    name=f"Trial_{datetime.now():%Y-%m-%d_%H-%M-%S}",
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
data_loader = DataLoader()
# train_images, train_labels = data_loader.get_train_data()
# train_images, train_labels = data_loader.get_train_data()
# test_images, test_labels = data_loader.get_test_data()
# val_images, val_labels = data_loader.get_validation_data()


# Get formatted data ready for neural networks
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.get_formatted_data()
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
test_accuracy = model.evaluate(test_images, test_labels)
model.confusion_matrix_plot(test_images, test_labels)
model.log_final_confusion_matrix(val_images, val_labels)

print(f"Test accuracy: {test_accuracy}")
# wandb.log({"test_accuracy": test_accuracy})
