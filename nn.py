import random
from autograd import Value

LEARNING_RATE = 0.01


class Neuron:
    def __init__(self, nin):
        self.bias = Value(random.uniform(-1, 1))
        self.bias._label = "b"
        self.weights = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]

    def __call__(self, in_vec):
        if len(in_vec) != len(self.weights):
            return
        return sum([w * x for w, x in zip(self.weights, in_vec)], self.bias)

    def parameters(self):
        return self.weights + [self.bias]


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

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class MLP:
    def __init__(self, layer_dims):
        self.layers = [
            Layer(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
        ]

    def __call__(self, vec):
        for layer in self.layers:
            vec = layer(vec)
        # return vec[0] if len(vec) == 1 else vec
        return vec

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        params = self.parameters()
        for n in params:
            n._grad = 0.0

    def update_grads(self):
        params = self.parameters()
        for n in params:
            n._data += -LEARNING_RATE * n._grad


def print_params(params):
    print("######")
    for param in params:
        print(param)
    print("######")


def neuron_test():
    inp = [1.0, 2.0, 3.0]
    n = Neuron(3)
    params = n.parameters()
    print_params(params)
    out = n(inp)
    out.backward()
    print_params(params)


def layer_test():
    inp = [1.0, 2.0, 3.0]
    l = Layer(3, 5)
    out = l(inp)
    for o in out:
        o.backward()
    print("out vals")
    print_params(out)
    print("layer param vals")
    print_params(l.parameters())


def mlp_test():
    inp = [1.0, 2.0, 3.0]
    layer_dims = [3, 5, 5, 1]
    mlp = MLP(layer_dims)
    out = mlp(inp)
    for o in out:
        o.backward()
    print("out val")
    print_params(out)
    print("All MLP param vals")
    print_params(mlp.parameters())


def gradient_descent():
    x = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    y = [1.0, -1.0, -1.0, 1.0]
    layer_dims = [3, 5, 5, 1]
    mlp = MLP(layer_dims)
    epoch = 38
    for e in range(epoch):
        y_pred = [mlp(x_i) for x_i in x]
        loss = sum([(y_i - y_pred_i[0]) ** 2 for y_i, y_pred_i in zip(y, y_pred)])
        print("epoch = ", e, "loss = ", loss)
        mlp.zero_grad()
        loss.backward()
        mlp.update_grads()
    y_pred = [mlp(x_i) for x_i in x]
    for y_pred_i in y_pred:
        print(y_pred_i[0])


if __name__ == "__main__":
    # neuron_test()
    # layer_test()
    # mlp_test()
    gradient_descent()
