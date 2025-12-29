class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def predict(self, x):
        return self.forward(x)
    
    def get_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "params") and layer.params:
                params.append(layer.params)
        return params
    
    def get_grads(self):
        grad = []
        for layer in self.layers:
            if hasattr(layer, "grads") and layer.grads:
                grad.append(layer.grads)
        
        return grad