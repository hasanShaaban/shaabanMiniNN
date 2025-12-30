import numpy as np
from .layer import Layer

class Dropout(Layer):

    def __init__(self, p=0.5):
        super().__init__()
        if not 0<= p < 1:
            raise ValueError('p must be between 0 and 1')
        self.p = p
        self.mask = None
        self.training = True

    def froward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (np.random.rand(*x.shape) >= self.p)
        out = x * self.mask / (1-self.p)
        return out
    
    def backward(self, dout):
        if not self.training or self.p == 0:
            return dout
        dx = dout * self.mask / (1-self.p)
        return dx