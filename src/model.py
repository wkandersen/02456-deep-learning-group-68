from loss_function import loss_function

class FFNN:
    def __init__(self, num_epochs, num_hidden_layers,n_hidden_units,lr,optimizer,batch_size,l2_coeff,weight_init,activation,_loss):
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2_coeff = l2_coeff
        self.weight_init = weight_init
        self.activation = activation
        self._loss = _loss

    def forward_pass():

    def compute_loss(self):
        return loss_function().compute_loss()

    def backward_pass():

    def training_loop():

    def evaluate():
