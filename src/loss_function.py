import numpy as np
class loss_function:
    def __init__(self, lam=0.01, weight=1.0, eps=1e-8, use_regularization=False):
        self.lam = lam
        self.weight = weight
        self.eps = eps
        self.use_regularization = use_regularization

    def l2_norm(self, weights):
        return sum((w ** 2).sum() for w in weights)

    def MSE_single(self, prediction, target):
        return (prediction - target) ** 2

    def compute_MSE_loss(self, predictions, targets, weights=None):
        loss = ((predictions[0] - targets[0]) ** 2).mean()

        if self.use_regularization and weights is not None:
            l2_norm = self.l2_norm(weights)
            loss += self.lam * l2_norm

        return loss
    
    def compute_cross_entropy_loss(self, predictions_prob, targets, weights=None):
        n = targets.shape[0]
        k = 10
        y_ij = np.zeros((n, k))
        y_ij[np.arange(n), targets] = 1
        yh_ij = predictions_prob
        loss = -np.sum(y_ij * np.log(yh_ij + self.eps))

        if self.use_regularization and weights is not None:
            l2_norm = self.l2_norm(weights)
            loss += self.lam * l2_norm

        return loss
    
