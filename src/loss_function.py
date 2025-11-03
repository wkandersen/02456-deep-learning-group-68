class loss_function:
    def __init__(self, alpha=1.0, lam=0.01, weight=1.0, eps=1e-8, use_regularization=False):
        self.alpha = alpha
        self.lam = lam
        self.weight = weight
        self.eps = eps
        self.use_regularization = use_regularization

    def l2_norm(self, weights):
        return sum((w ** 2).sum() for w in weights)

    def MSE_single(self, prediction, target):
        return (prediction - target) ** 2

    def compute_MSE_loss(self, predictions, targets, weights=None):
        loss = ((predictions - targets) ** 2).mean()

        if self.use_regularization and weights is not None:
            l2_norm = self.l2_norm(weights)
            loss += self.lam * l2_norm

        return loss
    
