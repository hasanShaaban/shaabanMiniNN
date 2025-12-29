from .layer import Layer
import numpy as np
class Dense(Layer):
    def __init__(self, input_size, output_size, weight_scale=0.1):
        super().__init__()

        self.params['W'] = weight_scale * np.random.randn(input_size, output_size)
        self.params['b'] = np.zeros(output_size)

        self.X = None
    

    def forward(self, x):
        self.X = x
        W = self.params['W']
        b = self.params['b']

        out = x @ W + b
        return out
    
    def backward(self, dout):
        W = self.params['W']
        x = self.X

        dW = x.T @ dout
        dx = dout @ W.T
        db = np.sum(dout, axis=0)

        self.grads['W'] = dW
        self.grads['b'] = db

        return dx
    
