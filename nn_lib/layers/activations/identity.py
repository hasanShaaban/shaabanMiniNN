from .activations import Activation
import numpy as np

class Identity(Activation):
    def _activate(self, x):
        return x
    def _derivative(self):
        return np.ones_like(self.out)