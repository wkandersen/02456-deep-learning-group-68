from model_cifar_10 import FFNN
import numpy as np

class FashionFFNN(FFNN):
    """
    Feedforward Neural Network model specifically for Fashion-MNIST dataset.
    Inherits from the generic FFNN class and can include dataset-specific methods if needed.
    """
    def __init__(self, *args, **kwargs):
        super(FashionFFNN, self).__init__(*args, **kwargs)
    
    def forward(self, X, training=True):
        self.activations = []
        self.z_values = []
        a = X
        a = a.reshape(a.shape[0], -1)  # Ensure input is 2D
        assert a.shape[1] == self.input_size, f"Expected input size {self.input_size}, but got {a.shape[1]}"
        self.activations.append(a)
        
        # Forward pass through hidden layers
        for i in range(self.num_hidden_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if self.batch_norm:
                gamma = np.ones((1, self.hidden_layers[i]))  # Use actual layer size
                beta = np.zeros((1, self.hidden_layers[i]))   # Use actual layer size
                z = self._batch_normalize(z, gamma, beta)
            # Only apply dropout during training
            if self.dropout_prob > 0 and training:
                z = self._dropout(z, self.dropout_prob)
            a = self._activation(z)
            self.z_values.append(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)  # No activation function at output layer
        return z