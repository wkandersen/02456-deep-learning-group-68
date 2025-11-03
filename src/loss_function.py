class loss_function:
    def __init__(self, alpha=1.0, lam=0.01, weight=1.0, eps=1e-8, use_regularization=False):
        self.alpha = alpha
        self.lam = lam
        self.weight = weight
        self.eps = eps
        self.use_regularization = use_regularization

    def compute_loss():