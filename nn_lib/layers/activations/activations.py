from abc import ABC, abstractmethod
from ..layer import Layer


class Activation(Layer, ABC):
    def __init__(self):
        super().__init__()
        self.x = None
        self.out = None

    def forward(self, x):
        self.x = x
        self.out = self._activate(x)
        return self.out
    
    def backward(self, dout):
        dx = dout * self._derivative()
        return dx
    
    @abstractmethod
    def _activate(self, x):
        pass

    @abstractmethod
    def _derivative(self):
        pass