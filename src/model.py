from loss_function import loss_function
import numpy as np

### Feedforward Neural Network class without the use of deep learning frameworks ###
class FFNN:
    def __init__(self, num_epochs, hidden_layers, lr, optimizer, batch_size, l2_coeff, weight_init, activation, _loss, input_size=3072, output_size=10):
        self.num_epochs = num_epochs
        self.hidden_layers = hidden_layers  # Now a list like [512, 128, 64]
        self.num_hidden_layers = len(hidden_layers)  # Number of layers derived from list length
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2_coeff = l2_coeff
        self.weight_init = weight_init
        self.activation = activation
        self._loss = _loss
        self.input_size = input_size  # CIFAR-10: 32*32*3 = 3072
        self.output_size = output_size  # CIFAR-10: 10 classes
        self.weights = []
        self.biases = []
        self.batch_norm = False  # Initialize batch norm flag
        self.dropout_prob = 0.0  # Initialize dropout probability
        self._initialize_weights()
        self.loss_function = loss_function(_loss)
        self.activations = []
        self.z_values = []

    def _initialize_weights(self):
        # Create layer sizes: [input_size, hidden_layer_1, hidden_layer_2, ..., output_size]
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            stddev = np.sqrt(2 / layer_sizes[i])
            W = np.random.normal(0, stddev, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def _batch_normalize(self, X, gamma, beta, eps=1e-5):
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)
        X_norm = (X - mu) / np.sqrt(var + eps)
        out = gamma * X_norm + beta
        return out
    
    def _dropout(self, X, drop_prob):
        mask = (np.random.rand(*X.shape) > drop_prob) / (1.0 - drop_prob)
        return X * mask

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)

    def forward(self, X):
        self.activations = []
        self.z_values = []
        a = X
        self.activations.append(a)
        
        # Forward pass through hidden layers
        for i in range(self.num_hidden_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if self.batch_norm:
                gamma = np.ones((1, self.hidden_layers[i]))  # Use actual layer size
                beta = np.zeros((1, self.hidden_layers[i]))   # Use actual layer size
                z = self._batch_normalize(z, gamma, beta)
            if self.dropout_prob > 0:
                z = self._dropout(z, self.dropout_prob)
            a = self._activation(z)
            self.z_values.append(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)  # No activation function at output layer
        return z