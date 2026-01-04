import random
from engine import Value

class Neuron:

    def __init__(self, num_in):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(num_in)]
        self.bias = Value(random.uniform(-1,1))
    
    def __call__(self, x, nonlin=True):
        # w*x+b
        activation = sum((w_i * x_i for w_i, x_i in zip(self.weights, x)), self.bias)
        out = activation.tanh() if nonlin else activation
        return out
    
    def parameters(self):
        return self.weights + [self.bias]

class Layer:

    def __init__(self, num_in, num_out):
        self.neurons = [Neuron(num_in) for _ in range(num_out)]
    
    def __call__(self, x, nonlin=True):
        outputs = [n(x, nonlin=nonlin) for n in self.neurons]
        return outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    
    def __init__(self, num_in, num_outs):
        sizes = [num_in] + num_outs
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x, nonlin=(i != len(self.layers)-1))
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
