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

    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2

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
    
    def compute_loss(self, predictions, targets):
        return self.loss_function.compute_MSE_loss(predictions, targets, self.weights)
    
    def _to_one_hot(self, y, num_classes=None):
        """Convert class indices to one-hot encoding"""
        if num_classes is None:
            num_classes = self.output_size
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def backward(self, X, y):
        """
        Implement backward pass with backpropagation
        X: input data (batch_size, input_size)
        y: true labels (batch_size,) - class indices OR (batch_size, output_size) - one-hot encoded
        """
        m = X.shape[0]  # batch size
        
        # Convert y to one-hot if it's class indices
        if y.ndim == 1:
            y_one_hot = self._to_one_hot(y)
        else:
            y_one_hot = y
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error (assuming softmax + cross-entropy loss)
        if self._loss == 'mse':
            # For MSE loss
            predictions = self.activations[-1]  # Output layer activations
            dZ = predictions - y_one_hot  # Now both have same shape
        else:
            # For other losses (like cross-entropy), compute gradient manually
            predictions = self.activations[-1]
            # Apply softmax to get probabilities
            exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            dZ = probs - y_one_hot  # Now both have same shape
        
        # Backpropagate through all layers
        for i in reversed(range(len(self.weights))):
            # Current layer activations (input to this layer)
            A_prev = self.activations[i]  # Previous layer activations
            
            # Ensure dZ has the right shape for matrix multiplication
            if dZ.ndim == 1:
                dZ = dZ.reshape(1, -1)
            
            # Compute gradients for weights and biases
            dW[i] = (1/m) * np.dot(A_prev.T, dZ)
            db[i] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            # Add L2 regularization to weight gradients
            if self.l2_coeff > 0:
                dW[i] += self.l2_coeff * self.weights[i]
            
            # Compute gradient for previous layer (if not the input layer)
            if i > 0:
                # Gradient w.r.t. previous layer activations
                dA_prev = np.dot(dZ, self.weights[i].T)
                
                # Gradient w.r.t. previous layer pre-activations (z values)
                z_prev = self.z_values[i-1]  # Pre-activation values of previous layer
                dZ = dA_prev * self._activation_derivative(z_prev)
        
        return dW, db
    
    def update_weights(self, dW, db):
        """
        Update weights and biases using computed gradients
        """
        if self.optimizer == 'sgd':
            # Stochastic Gradient Descent
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
                
        elif self.optimizer == 'adam':
            # Adam optimizer (simplified version)
            # Initialize momentum terms if not already done
            if not hasattr(self, 'v_dW'):
                self.v_dW = [np.zeros_like(w) for w in self.weights]
                self.v_db = [np.zeros_like(b) for b in self.biases]
                self.s_dW = [np.zeros_like(w) for w in self.weights]
                self.s_db = [np.zeros_like(b) for b in self.biases]
                self.t = 0  # time step
            
            self.t += 1
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            
            for i in range(len(self.weights)):
                # Update momentum
                self.v_dW[i] = beta1 * self.v_dW[i] + (1 - beta1) * dW[i]
                self.v_db[i] = beta1 * self.v_db[i] + (1 - beta1) * db[i]
                
                # Update squared gradients
                self.s_dW[i] = beta2 * self.s_dW[i] + (1 - beta2) * (dW[i] ** 2)
                self.s_db[i] = beta2 * self.s_db[i] + (1 - beta2) * (db[i] ** 2)
                
                # Bias correction
                v_dW_corrected = self.v_dW[i] / (1 - beta1 ** self.t)
                v_db_corrected = self.v_db[i] / (1 - beta1 ** self.t)
                s_dW_corrected = self.s_dW[i] / (1 - beta2 ** self.t)
                s_db_corrected = self.s_db[i] / (1 - beta2 ** self.t)
                
                # Update weights
                self.weights[i] -= self.lr * v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon)
                self.biases[i] -= self.lr * v_db_corrected / (np.sqrt(s_db_corrected) + epsilon)
        
        else:
            # Default to SGD if optimizer not recognized
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
        
    def train(self, X_train, y_train):
        num_samples = X_train.shape[0]
        for epoch in range(self.num_epochs):
            # Shuffle the data at the beginning of each epoch
            perm = np.random.permutation(num_samples)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(predictions, y_batch)
                
                # Backward pass
                dW, db = self.backward(X_batch, y_batch)
                
                # Update weights
                self.update_weights(dW, db)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss} Accuracy: {self.evaluate(X_train, y_train)}")
            # Forward pass
            predictions = self.forward(X_batch)
            # Compute loss
            loss = self.compute_loss(predictions, y_batch)
            # Backward pass
            dW, db = self.backward(X_batch, y_batch)
            # Update weights
            self.update_weights(dW, db)

    def validate(self, X_val, y_val):
        preds = self.predict(X_val)
        accuracy = np.mean(preds == y_val)
        loss = self.compute_loss(self.forward(X_val), y_val)
        print(f"Validation Loss: {loss}, Accuracy: {accuracy}")
        return accuracy, loss

    def predict(self, X):
        # with softmax
        logits = self.forward(X)
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy
    
    