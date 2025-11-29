# from slet_til_sidst.loss_function import loss_function
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

### Feedforward Neural Network class without the use of deep learning frameworks ###

class FFNN:
    def __init__(self, num_epochs, hidden_layers, lr, optimizer, batch_size, l2_coeff, weight_init, activation, _loss, input_size=3072, output_size=10, batch_norm=False, dropout_prob=0.0,patience=5):
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
        self.batch_norm = batch_norm  # Initialize batch norm flag
        self.dropout_prob = dropout_prob  # Initialize dropout probability
        self.patience=patience
        
        # Initialize batch normalization parameters if enabled
        if self.batch_norm:
            self.bn_gamma = []  # Scale parameters
            self.bn_beta = []   # Shift parameters
            self.bn_running_mean = []  # Running averages for inference
            self.bn_running_var = []
            self.bn_momentum = 0.9  # Momentum for running averages
            
        self._initialize_weights()
        # self.loss_function = loss_function(_loss)
        self.activations = []
        self.z_values = []
        
        # Initialize history tracking for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def _initialize_weights(self):
        # Create layer sizes: [input_size, hidden_layer_1, hidden_layer_2, ..., output_size]
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            if self.weight_init == 'xavier':
                # Xavier initialization
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            elif self.weight_init == 'random':
                # Random initialization
                W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            else:
                # He initialization
                stddev = np.sqrt(2 / layer_sizes[i])
                W = np.random.normal(0, stddev, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)
            
        # Initialize batch normalization parameters for hidden layers
        if self.batch_norm:
            for i in range(self.num_hidden_layers):
                layer_size = self.hidden_layers[i]
                self.bn_gamma.append(np.ones((1, layer_size)))
                self.bn_beta.append(np.zeros((1, layer_size)))
                self.bn_running_mean.append(np.zeros((1, layer_size)))
                self.bn_running_var.append(np.ones((1, layer_size)))

    def _batch_normalize(self, X, layer_idx, training=True, eps=1e-5):
        """
        Improved batch normalization with learnable parameters and running statistics
        
        Args:
            X: Input tensor (batch_size, features)
            layer_idx: Index of the layer for accessing gamma/beta parameters
            training: Whether in training mode (use batch stats) or inference mode (use running stats)
            eps: Small constant for numerical stability
        """
        # Ensure training is a boolean value
        training = bool(training)
        
        if training:
            # Training mode: use batch statistics
            mu = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)
            
            # Update running averages
            self.bn_running_mean[layer_idx] = (self.bn_momentum * self.bn_running_mean[layer_idx] + 
                                             (1 - self.bn_momentum) * mu)
            self.bn_running_var[layer_idx] = (self.bn_momentum * self.bn_running_var[layer_idx] + 
                                            (1 - self.bn_momentum) * var)
        else:
            # Inference mode: use running statistics
            mu = self.bn_running_mean[layer_idx]
            var = self.bn_running_var[layer_idx]
        
        # Normalize
        X_norm = (X - mu) / np.sqrt(var + eps)
        
        # Scale and shift with learnable parameters
        out = self.bn_gamma[layer_idx] * X_norm + self.bn_beta[layer_idx]
        
        return out, X_norm, mu, var  # Return additional values needed for backprop
    
    def _batch_norm_backward(self, dout, cache, layer_idx, eps=1e-5):
        """
        Backward pass for batch normalization
        
        Args:
            dout: Gradient of loss w.r.t. output of batch norm
            cache: Cache from forward pass containing z_norm, mu, var
            layer_idx: Layer index for accessing gamma/beta
            eps: Small constant for numerical stability
        """
        z_norm = cache['z_norm']
        mu = cache['mu']
        var = cache['var']
        
        m = dout.shape[0]  # batch size
        
        # Gradient w.r.t. gamma and beta
        d_gamma = np.sum(dout * z_norm, axis=0, keepdims=True)
        d_beta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient w.r.t. normalized input
        d_z_norm = dout * self.bn_gamma[layer_idx]
        
        # Gradient w.r.t. variance
        d_var = np.sum(d_z_norm * (cache['z_norm'] * -0.5 * (var + eps)**(-1.5)), axis=0, keepdims=True)
        
        # Gradient w.r.t. mean
        d_mu = np.sum(d_z_norm * (-1.0 / np.sqrt(var + eps)), axis=0, keepdims=True)
        d_mu += d_var * np.mean(-2.0 * (z_norm * np.sqrt(var + eps) + mu - mu), axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        dx = d_z_norm / np.sqrt(var + eps)
        dx += d_var * 2.0 * (z_norm * np.sqrt(var + eps) + mu - mu) / m
        dx += d_mu / m
        
        return dx, d_gamma, d_beta
    
    def _dropout(self, X, drop_prob):
        """
        Apply dropout with inverted scaling during training
        Returns both the output and the mask for use in backpropagation
        """
        mask = (np.random.rand(*X.shape) > drop_prob) / (1.0 - drop_prob)
        return X * mask, mask

    def _activation(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)

    def _activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sigmoid_x = 1 / (1 + np.exp(-x))
            return sigmoid_x * (1 - sigmoid_x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'leaky_relu':
            dx = np.ones_like(x)
            dx[x < 0] = 0.01
            return dx

    def forward(self, X, training=True):
        # Ensure training is a boolean value
        training = bool(training)
        
        self.activations = []
        self.z_values = []
        self.bn_cache = []  # Store batch norm intermediate values for backprop
        self.dropout_masks = []  # Store dropout masks for backprop
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
                a, mask = self._dropout(a, self.dropout_prob)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
                
            self.activations.append(a)
        
        # Output layer (no batch norm, no dropout)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)  # No activation function at output layer
        return z
    
    def compute_loss(self, predictions, targets):
        """
        Compute loss based on the specified loss function
        """
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_one_hot = self._to_one_hot(targets)
        else:
            targets_one_hot = targets
            
        if self._loss == 'cross_entropy':
            # Cross-entropy loss with softmax
            # Apply softmax to predictions
            exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Compute cross-entropy loss
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            loss = -np.mean(np.sum(targets_one_hot * np.log(probs), axis=1))
            
            # Add L2 regularization if specified
            if self.l2_coeff > 0:
                l2_penalty = self.l2_coeff * sum(np.sum(w**2) for w in self.weights)
                loss += l2_penalty
                
        # elif self._loss == 'mse':
        #     # Use the existing MSE loss function
        #     loss = self.loss_function.compute_MSE_loss(predictions, targets, self.weights)
        else:
            raise ValueError(f"Unsupported loss function: {self._loss}")
        
        return loss
    
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
        
        # Initialize batch norm gradients if batch norm is enabled
        if self.batch_norm:
            d_bn_gamma = [np.zeros_like(gamma) for gamma in self.bn_gamma]
            d_bn_beta = [np.zeros_like(beta) for beta in self.bn_beta]
        
        # Compute output layer error based on the loss function being used
        predictions = self.activations[-1]  # Output layer activations (logits)
        
        if self._loss == 'cross_entropy':
            # For cross-entropy with softmax, gradient is: softmax(predictions) - y_one_hot
            exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            dZ = probs - y_one_hot
        elif self._loss == 'mse':
            # For MSE loss, gradient is: predictions - y_one_hot
            # Note: This assumes no activation on output layer for regression
            # For classification with MSE, you might want softmax first
            dZ = predictions - y_one_hot
        else:
            # Default fallback - assume cross-entropy
            exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            dZ = probs - y_one_hot
        
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
                
                # Apply dropout mask if it was used during forward pass
                if self.dropout_prob > 0 and hasattr(self, 'dropout_masks'):
                    layer_idx = i - 1  # Index for the previous layer
                    if layer_idx < len(self.dropout_masks) and self.dropout_masks[layer_idx] is not None:
                        # Apply the same mask that was used during forward pass
                        dA_prev = dA_prev * self.dropout_masks[layer_idx]
                
                # Gradient w.r.t. previous layer pre-activations (z values)
                z_prev = self.z_values[i-1]  # Pre-activation values of previous layer
                dZ_prev = dA_prev * self._activation_derivative(z_prev)
                
                # If batch normalization was applied to the previous layer, backprop through it
                if self.batch_norm and i-1 < self.num_hidden_layers:
                    layer_idx = i-1
                    cache = self.bn_cache[layer_idx]
                    
                    if cache is not None:  # Batch norm was applied
                        # Backprop through batch normalization
                        dZ_prev, d_gamma, d_beta = self._batch_norm_backward(dZ_prev, cache, layer_idx)
                        d_bn_gamma[layer_idx] += d_gamma
                        d_bn_beta[layer_idx] += d_beta
                
                dZ = dZ_prev
        
        if self.batch_norm:
            return dW, db, d_bn_gamma, d_bn_beta
        else:
            return dW, db
    
    def update_weights(self, dW, db, d_bn_gamma=None, d_bn_beta=None):
        """
        Update weights and biases using computed gradients
        """
        if self.optimizer == 'sgd':
            # Stochastic Gradient Descent
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
            
            # Update batch norm parameters if provided
            if d_bn_gamma is not None and d_bn_beta is not None:
                for i in range(len(self.bn_gamma)):
                    self.bn_gamma[i] -= self.lr * d_bn_gamma[i]
                    self.bn_beta[i] -= self.lr * d_bn_beta[i]
                
        elif self.optimizer == 'adam':
            # Adam optimizer (simplified version)
            # Initialize momentum terms if not already done
            if not hasattr(self, 'v_dW'):
                self.v_dW = [np.zeros_like(w) for w in self.weights]
                self.v_db = [np.zeros_like(b) for b in self.biases]
                self.s_dW = [np.zeros_like(w) for w in self.weights]
                self.s_db = [np.zeros_like(b) for b in self.biases]
                self.t = 0  # time step
                
                # Initialize batch norm momentum terms if needed
                if self.batch_norm:
                    self.v_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
                    self.v_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
                    self.s_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
                    self.s_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
            
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
            
            # Update batch norm parameters with Adam if provided
            if d_bn_gamma is not None and d_bn_beta is not None:
                for i in range(len(self.bn_gamma)):
                    # Update momentum for gamma
                    self.v_bn_gamma[i] = beta1 * self.v_bn_gamma[i] + (1 - beta1) * d_bn_gamma[i]
                    self.s_bn_gamma[i] = beta2 * self.s_bn_gamma[i] + (1 - beta2) * (d_bn_gamma[i] ** 2)
                    
                    # Update momentum for beta
                    self.v_bn_beta[i] = beta1 * self.v_bn_beta[i] + (1 - beta1) * d_bn_beta[i]
                    self.s_bn_beta[i] = beta2 * self.s_bn_beta[i] + (1 - beta2) * (d_bn_beta[i] ** 2)
                    
                    # Bias correction
                    v_gamma_corrected = self.v_bn_gamma[i] / (1 - beta1 ** self.t)
                    v_beta_corrected = self.v_bn_beta[i] / (1 - beta1 ** self.t)
                    s_gamma_corrected = self.s_bn_gamma[i] / (1 - beta2 ** self.t)
                    s_beta_corrected = self.s_bn_beta[i] / (1 - beta2 ** self.t)
                    
                    # Update parameters
                    self.bn_gamma[i] -= self.lr * v_gamma_corrected / (np.sqrt(s_gamma_corrected) + epsilon)
                    self.bn_beta[i] -= self.lr * v_beta_corrected / (np.sqrt(s_beta_corrected) + epsilon)
        
        else:
            # Default to SGD if optimizer not recognized
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
            
            # Update batch norm parameters if provided
            if d_bn_gamma is not None and d_bn_beta is not None:
                for i in range(len(self.bn_gamma)):
                    self.bn_gamma[i] -= self.lr * d_bn_gamma[i]
                    self.bn_beta[i] -= self.lr * d_bn_beta[i]
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        num_samples = X_train.shape[0]
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.num_epochs):
            # Shuffle the data at the beginning of each epoch
            perm = np.random.permutation(num_samples)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size]
                
                # Forward pass
                predictions = self.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.compute_loss(predictions, y_batch)
                epoch_loss += loss
                num_batches += 1

                # Backward pass
                if self.batch_norm:
                    dW, db, d_bn_gamma, d_bn_beta = self.backward(X_batch, y_batch)
                else:
                    dW, db = self.backward(X_batch, y_batch)

                # Log gradients
                wandb.log({"batch_gradients": np.mean([np.linalg.norm(dw) for dw in dW])})  # optional summary
                self.log_gradients(dW, db)

                # Update weights
                if self.batch_norm:
                    self.update_weights(dW, db, d_bn_gamma, d_bn_beta)
                else:
                    self.update_weights(dW, db)

                # Optionally log weights at epoch end
                if (i + self.batch_size) >= num_samples:  # end of epoch
                    self.log_weights()

            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches
            
            # Compute training accuracy on a subset for efficiency (or use last batch accuracy)
            # This is much faster than computing on full dataset every epoch
            train_accuracy = self.evaluate_batch(X_batch, y_batch)
            
            # Store training metrics
            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(train_accuracy)
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            wandb.log({"train_loss": avg_loss, "train_accuracy": train_accuracy})
            
            # Validate if validation data is provided
            if X_val is not None and y_val is not None:
                val_accuracy, val_loss = self.validate(X_val, y_val)
                if (epoch+1) % 5 == 0:
                    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                # Store validation metrics
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_accuracy)

                if val_loss < best_val_loss - 1e-6:  # improvement threshold
                                best_val_loss = val_loss
                                best_epoch = epoch + 1
                                epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Check patience
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                    break
                
            else:
                # If no validation data, store None to maintain list alignment
                self.val_loss_history.append(None)
                self.val_acc_history.append(None)

    def validate(self, X_val, y_val):
        preds = self.predict(X_val)
        accuracy = np.mean(preds == y_val)
        loss = self.compute_loss(self.forward(X_val, training=False), y_val)
        wandb.log({"val_loss": loss, "val_accuracy": accuracy})
        return accuracy, loss

    def predict(self, X):
        # with softmax - NO DROPOUT during inference
        logits = self.forward(X, training=False)
        exp_scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy
    
    def evaluate_batch(self, X, y):
        """Evaluate accuracy on a single batch - faster than full dataset"""
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy
    
    def confusion_matrix_plot_sklearn(self, X, y):
        preds = self.predict(X)
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png", dpi=300)
        # plt.show()

    def confusion_matrix_scratch_plot(self, X, y, path):
        preds = self.predict(X)
        true, pred = y, preds
        labels = sorted(set(true))  # include all labels
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        num_classes = len(labels)
        mat = np.zeros((num_classes, num_classes))

        for t, p in zip(true, pred):
            i = label_to_index[t]
            j = label_to_index[p]
            mat[i, j] += 1

        plt.figure(figsize=(8,6))
        sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(path, dpi=300)
        # plt.show()

    def log_final_confusion_matrix(self, X, y):
        preds = self.predict(X)
        wandb.log({
            "final_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y,
                preds=preds,
                class_names=[str(i) for i in range(self.output_size)]
            )
        })

    
    def plot_training_history(self, save_path=None, show_plot=True):
        """
        Plot training and validation loss and accuracy curves
        
        Args:
            save_path (str): Path to save the plot (optional)
            show_plot (bool): Whether to display the plot
        """
        epochs = range(1, len(self.train_loss_history) + 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Loss
        ax1.plot(epochs, self.train_loss_history, 'b-', label='Training Loss', linewidth=2)
        
        # Only plot validation if we have validation data
        if any(loss is not None for loss in self.val_loss_history):
            val_epochs = [i+1 for i, loss in enumerate(self.val_loss_history) if loss is not None]
            val_losses = [loss for loss in self.val_loss_history if loss is not None]
            ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Accuracy
        ax2.plot(epochs, self.train_acc_history, 'b-', label='Training Accuracy', linewidth=2)
        
        # Only plot validation if we have validation data
        if any(acc is not None for acc in self.val_acc_history):
            val_epochs = [i+1 for i, acc in enumerate(self.val_acc_history) if acc is not None]
            val_accs = [acc for acc in self.val_acc_history if acc is not None]
            ax2.plot(val_epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_loss_curve(self, loss_history):
        """Backward compatibility - simple loss plotting"""
        plt.plot(loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    
    def get_training_summary(self):
        """Get a summary of training metrics"""
        if not self.train_loss_history:
            return "No training history available. Train the model first."
        
        summary = {
            'epochs_trained': len(self.train_loss_history),
            'final_train_loss': self.train_loss_history[-1],
            'final_train_accuracy': self.train_acc_history[-1],
            'best_train_accuracy': max(self.train_acc_history),
            'best_train_accuracy_epoch': self.train_acc_history.index(max(self.train_acc_history)) + 1
        }
        
        # Add validation metrics if available
        val_losses = [loss for loss in self.val_loss_history if loss is not None]
        val_accs = [acc for acc in self.val_acc_history if acc is not None]
        
        if val_losses:
            summary.update({
                'final_val_loss': val_losses[-1],
                'final_val_accuracy': val_accs[-1],
                'best_val_accuracy': max(val_accs),
                'best_val_accuracy_epoch': [i+1 for i, acc in enumerate(self.val_acc_history) if acc == max(val_accs)][0]
            })
        
        return summary

    def log_weights(self):
        """Log parameter histograms to WandB."""
        for i, W in enumerate(self.weights):
            wandb.log({f"weights/layer_{i}": wandb.Histogram(W)})
        for i, b in enumerate(self.biases):
            wandb.log({f"biases/layer_{i}": wandb.Histogram(b)})

    def log_gradients(self, dW, db):
        """Log gradient norms to WandB."""
        for i, grad in enumerate(dW):
            wandb.log({f"gradients/layer_{i}": np.linalg.norm(grad)})
        for i, grad_b in enumerate(db):
            wandb.log({f"gradients/bias_layer_{i}": np.linalg.norm(grad_b)})
  