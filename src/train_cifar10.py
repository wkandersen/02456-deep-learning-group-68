from model import FFNN
from load_data import CIFAR10Loader
import numpy as np

# Load CIFAR-10 data
data_loader = CIFAR10Loader()
train_images, train_labels = data_loader.get_training_data()
test_images, test_labels = data_loader.get_test_data()
val_images, val_labels = data_loader.get_validation_data()

# Initialize the model
model = FFNN(
    num_epochs=5,
    hidden_layers=[512, 256, 128],
    lr=0.001,
    optimizer='adam',
    batch_size=512,
    l2_coeff=0.0001,
    weight_init='he',
    activation='relu',
    _loss='mse'
)

# Train the model
model.train(train_images, train_labels, val_images, val_labels)
model.plot_training_history()

# Evaluate the model
test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")
