from abc import ABC, abstractmethod

class Metric(ABC):

    @abstractmethod
    def compute(self, y_pred, target):
        pass 
    def reset(self):
        pass 