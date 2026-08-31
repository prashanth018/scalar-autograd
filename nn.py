import random
from autograd import Value


class Neuron:
    def __init__(self, nin):
        self.bias = Value(random.uniform(-1, 1))
        self.bias._label = "b"
        self.weights = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]

    def __call__(self, in_vec):
        if len(in_vec) != len(self.weights):
            return
        if isinstance(in_vec[0], (int, float)):
            for w in in_vec:
                w = Value(w)
        return sum([w * x for w, x in zip(self.weights, in_vec)], self.bias)


# Input = [x1, x2, x3, x4]
# A layer of 5 Neurons each of size 4
#  N1   N2   N3   N4   N5
# [n11, n12, n13, n14, n15]
# [n21, n22, n23, n24, n25]
# [n31, n32, n33, n34, n35]
# [n41, n42, n43, n44, n45]
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, in_vec):
        return [neuron(in_vec) for neuron in self.neurons]


class MLP:
    def __init__(self, layer_dims):
        self.layers = [
            Layer(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
        ]

    def __call__(self, vec):
        for layer in self.layers:
            vec = layer(vec)
        return vec
