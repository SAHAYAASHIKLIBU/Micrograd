from engine import Value
from math import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

class Neuron:
    def __init__(self, nin, nonlin = False):
        self.w = [random.uniform(-1, 1) for _ in range(nin)]
        self.b = 0
        self.nonlin = nonlin

    def __call__(self, x):
        out = sum([wi*xi for wi, xi in zip(self.w, x)],self.b)
        return out if not self.nonlin else out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, ndim, nonlin = False):
        self.neurons = [Neuron(nin) for _ in range(ndim)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        return [n.parameters() for n in self.neurons]
          

class MLP:
    def __init__(self, dims):
        self.layers = [Layer(dims[i], dims[i+1], nonlin=False) for i in range(len(dims)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [l.parameters() for l in self.layers]
    