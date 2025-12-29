import numpy as np
from .activations import Activation
class Sigmoid(Activation):
    def _activate(self, x):
        return 1/(1 + np.exp(-x))
    def _derivative(self):
        return self.out * (1 - self.out)
