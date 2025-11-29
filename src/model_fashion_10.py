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
        
        # Clear all instance variables to start fresh
        self.activations = []
        self.z_values = []
        self.bn_cache = []  # Store batch norm intermediate values for backprop
        self.dropout_masks = []  # Store dropout masks for backprop
        
        # Ensure input is numpy array
        a = np.asarray(X)
        assert a.shape[1] == self.input_size, f"Expected input size {self.input_size}, but got {a.shape[1]}"
        self.activations.append(a)

        # Forward pass through hidden layers
        for i in range(self.num_hidden_layers):
            # Linear transformation
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z = np.asarray(z)  # Ensure numpy array
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                z_bn, z_norm, mu, var = self._batch_normalize(z, i, training)
                # Ensure all batch norm outputs are numpy arrays
                z_bn = np.asarray(z_bn)
                z_norm = np.asarray(z_norm)
                mu = np.asarray(mu)
                var = np.asarray(var)
                # Store both the original z and the batch norm outputs for backprop
                self.bn_cache.append({'z_norm': z_norm, 'mu': mu, 'var': var, 'z_input': z})
                z = z_bn  # Use the batch normalized values
            else:
                self.bn_cache.append(None)
                
            self.z_values.append(z)
            
            # Apply activation
            a = self._activation(z)
            a = np.asarray(a)  # Ensure numpy array after activation
            
            # Only apply dropout during training
            if self.dropout_prob > 0 and training:
                a_dropped, mask = self._dropout(a, self.dropout_prob)
                a = np.asarray(a_dropped)  # Ensure result is numpy array
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
                
            # Final guarantee that we store a numpy array
            self.activations.append(np.asarray(a))
        
        # Output layer (no batch norm, no dropout)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        z = np.asarray(z)  # Ensure numpy array
        self.z_values.append(z)
        self.activations.append(z)  # No activation function at output layer
        return z