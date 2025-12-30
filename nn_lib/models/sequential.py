class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def params_and_grads(self):
        for layer in self.layers:
            if hasattr(layer, "params") and layer.params:
                yield layer.params, layer.grads

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False