from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x):
        pass
    def backward(self, dout):
        pass