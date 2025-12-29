from abc import ABC, abstractmethod

class Loss(ABC):
    def __init__(self):
        self.y_pred = None
        self.target = None

    def forward(self, y_pred, target):
        self.y_pred = y_pred
        self.target = target

        return self._loss()
    
    def backward(self):
        return self._gradient()
    
    @abstractmethod
    def _loss(self):
        pass

    @abstractmethod
    def _gradient(self):
        pass