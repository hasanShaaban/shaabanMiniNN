import numpy as np
from .layer import Layer

class BatchNormalization(Layer):

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.params["gamma"] = np.zeros(dim)
        self.params["beta"] = np.zeros(dim)

        self.grads["gamma"] = np.zeros(dim)
        self.grads["beta"] = np.zeros(dim)

        self.running_mean = np.zeros(dim)
        self.running_var = np.zeros(dim)

        self.cache = None

    def forward(self, x):
        if self.training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            x_hat = (x - mu) / np.sqrt(var + self.eps)
            out = self.params["gamma"] * x_hat + self.params["beta"]

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
            self.cache = (x, x_hat, mu, var)

        else:
            x_hat = (
                (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            )
            out = self.params["gamma"] * x_hat + self.params["beta"]

        return out
    
    def backward(self, dout):
        x, x_hat, mu, var = self.cache
        N, D = x.shape

        gamma = self.params["gamma"]

        self.grads["beta"] = np.sum(dout, axis=0)
        self.grads["gamma"] = np.sum(dout * x_hat, axis=0)

        dx_hat = dout * gamma

        dvar = np.sum(
            dx_hat * (x - mu) * -0.5 * (var + self.eps) ** (-1.5),
            axis=0
        )

        dmu = (
            np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=0) + dvar * np.mean(-2 * (x-mu), axis=0)
        )

        dx = (
            dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mu) / N + dmu / N
        )

        return dx