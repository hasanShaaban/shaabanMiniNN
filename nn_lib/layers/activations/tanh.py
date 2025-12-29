import numpy as np
from .activations import Activation

class Tanh(Activation):
    def _activate(self, x):
        return np.tanh(x)
    
    def _derivative(self):
        return 1 - self.out**2