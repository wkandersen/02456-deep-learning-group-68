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
        # Ensure training is a boolean value
        training = bool(training)
        
        self.activations = []
        self.z_values = []
        self.bn_cache = []  # Store batch norm intermediate values for backprop
        a = X
        assert a.shape[1] == self.input_size, f"Expected input size {self.input_size}, but got {a.shape[1]}"
        self.activations.append(a)

        # Forward pass through hidden layers
        for i in range(self.num_hidden_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                z, z_norm, mu, var = self._batch_normalize(z, i, training)
                self.bn_cache.append({'z_norm': z_norm, 'mu': mu, 'var': var})
            else:
                self.bn_cache.append(None)
                
            self.z_values.append(z)
            
            # Apply activation
            a = self._activation(z)
            
            # Only apply dropout during training
            if self.dropout_prob > 0 and training:
                a = self._dropout(a, self.dropout_prob)
                
            self.activations.append(a)
        
        # Output layer (no batch norm, no dropout)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)  # No activation function at output layer
        return z