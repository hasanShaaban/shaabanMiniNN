from .activations import Activation
import numpy as np


class ReLU(Activation):
    def _activate(self, x):
        return np.maximum(0, x)
    
    def _derivative(self):
        return (self.x > 0).astype(self.x.dtype)
     

