import math


class Value:
    def __init__(self, data, prev=(), op=""):
        self._data = data
        self._grad = 0.0
        self._prev = set(prev)
        self._op = op

    def __add__(self, other):
        out = self._data + other._data
        return Value(out, prev=(self, other), op="+")

    def __mul__(self, other):
        out = self._data * other._data
        return Value(out, prev=(self, other), op="*")

    def __repr__(self):
        return f"Value (data = {self._data}, grad = {self._grad})"

    def tanh(self):
        return Value(
            (1 - math.exp(-2 * self._data)) / (1 + math.exp(-2 * self._data)),
            prev=(self,),
            op="tanh",
        )

    def backward(self):
        if self._prev:
            if self._op == "tanh":
                (n,) = self._prev
                n._grad = self._grad * (1 - self._data**2)
                n.backward()
            elif self._op == "+":
                l, r = self._prev
                l._grad = self._grad
                r._grad = self._grad
                l.backward()
                r.backward()
            elif self._op == "*":
                l, r = self._prev
                l._grad = self._grad * r._data
                r._grad = self._grad * l._data
                l.backward()
                r.backward()


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
