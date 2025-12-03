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
        training = bool(training)

        self.activations = []
        self.z_values = []
        self.bn_cache = []
        self.dropout_masks = []

        a = np.asarray(X)
        assert a.shape[1] == self.input_size
        self.activations.append(a)

        # Hidden layers
        for i in range(self.num_hidden_layers):

            # 1. Linear pre-activation
            z = np.dot(a, self.weights[i]) + self.biases[i]
            raw_z = z.copy()              # save original z BEFORE BN
            self.z_values.append(raw_z)   # store raw z for activation derivative

            # 2. Batch Normalization
            if self.batch_norm:
                out, z_norm, mu, var = self._batch_normalize(z, i, training)

                self.bn_cache.append({
                    'x': raw_z,           # MUST match BN backward
                    'z_norm': z_norm,
                    'mu': mu,
                    'var': var
                })

                z = out
            else:
                self.bn_cache.append(None)

            # 3. Activation
            a = self._activation(z)

            # 4. Dropout
            if self.dropout_prob > 0 and training:
                a, mask = self._dropout(a, self.dropout_prob)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)

            self.activations.append(a)

        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)

        return z