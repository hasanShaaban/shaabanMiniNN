from abc import ABC, abstractmethod

class Optimizer(ABC):

    def __init__(self, lr = 0.01):
        self.lr = lr

    @abstractmethod
    def update(self, params, grads):
        pass 
        