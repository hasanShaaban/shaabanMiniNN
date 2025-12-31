from .optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]