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

    def get_params(self):
        params = []
        for i, layer in enumerate(self.layers):
            if layer.params:
                layer_name = f"{layer.__class__.__name__}_{i}"
                params.append((layer_name, layer.params))
        return params
    
    def summary(self):
        print("Model Parameters Summary")
        print("-" * 40)

        for name, params in self.get_params():
            print(f"\n{name}")
            for key, value in params.items():
                print(f"  {key}: shape={value.shape}")
                print(value)