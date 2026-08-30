import math


class Value:
    def __init__(self, data, prev=(), op=""):
        self._data = data
        self._grad = 0.0
        self._prev = set(prev)
        self._op = op
        self._backward = lambda: None

    def __add__(self, other):
        # forward pass
        data = self._data + other._data
        out = Value(data, prev=(self, other), op="+")

        # function to call later
        def _backward():
            self._grad += out._grad
            other._grad += out._grad

        # register the call
        out._backward = _backward
        return out

    def __mul__(self, other):
        data = self._data * other._data
        out = Value(data, prev=(self, other), op="*")

        def _backward():
            self._grad += out._grad * other._data
            other._grad += out._grad * self._data

        out._backward = _backward
        return out

    def tanh(self):
        data = (1 - math.exp(-2 * self._data)) / (1 + math.exp(-2 * self._data))
        out = Value(
            data,
            prev=(self,),
            op="tanh",
        )

        def _backward():
            self._grad += out._grad * (1 - out._data**2)

        out._backward = _backward
        return out

    def backward(self):
        self._backward()

    def __repr__(self):
        return f"Value (data = {self._data}, grad = {self._grad})"


if __name__ == "__main__":
    # test 1
    # a = Value(4.0)
    # b = Value(-3.0)
    # e = Value(2.0)
    # c = a * b
    # d = c + e
    # print(c._prev, c._op)
    # print(d._prev, d._op)
    # d._grad = 1.0
    # d.backward()
    # print("####post grad####")
    # print(a)
    # print(b)
    # print(c)
    # print(e)
    # print(d)

    # Implement y = tanh(w1x1 + w2x2 + b)
    w1 = Value(1.0)
    x1 = Value(2.0)
    w1x1 = w1 * x1
    w2 = Value(2.0)
    x2 = Value(-3.0)
    w2x2 = w2 * x2
    b = Value(0.5)
    w1x1w2x2 = w1x1 + w2x2
    y = w1x1w2x2 + b
    # print(y)
    o = y.tanh()
    o._grad = 1
    o.backward()
    print("o = ", o)
    print("y = ", y)
    print("w1x1w2x2 = ", w1x1w2x2)
    print("b = ", b)
    print("w1x1 = ", w1x1)
    print("w2x2 = ", w2x2)
    print("w1 = ", w1)
    print("x1 = ", x1)
    print("w2 = ", w2)
    print("x2 = ", x2)
