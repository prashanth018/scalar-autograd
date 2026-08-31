import math


class Value:
    def __init__(self, data, prev=(), op=""):
        self._data = data
        self._grad = 0.0
        self._prev = set(prev)
        self._op = op
        self._backward = lambda: None
        self._hash = hash(self)
        self._label = ""

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
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

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self._data * other._data
        out = Value(data, prev=(self, other), op="*")

        def _backward():
            self._grad += out._grad * other._data
            other._grad += out._grad * self._data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float supported"
        data = self._data**other
        out = Value(data, prev=(self,), op="pow")

        def _backward():
            self._grad += out._grad * (other * self._data ** (other - 1))

        out._backward = _backward
        return out

    # a/b = a * (1/b) = a * (b**-1)
    def __truediv__(self, other):
        return self * (other**-1)

    def exp(self):
        out = Value(math.exp(self._data), prev=(self,), op="exp")

        def _backward():
            self._grad += out._grad * out._data

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
        self._grad = 1.0
        order = topo_sort(self)
        print("######")
        for ord in order:
            print(ord)
        print("######")
        for node in order:
            node._backward()

    def __repr__(self):
        return f"Value (data = {self._data}, grad = {self._grad}, label = {self._label}, hash = {self._hash})"


# Simple DFS, add the parent only after visiting all the children
def topo_sort(root: Value):
    visited = set()
    order = []
    dfs(root, visited, order)
    order.reverse()
    return order


def dfs(node: Value, visited, order):
    if node._prev == set():
        visited.add(node)
        order.append(node)
        return
    if node in visited:
        return
    for ch in node._prev:
        if ch not in visited:
            dfs(ch, visited, order)
    visited.add(node)
    order.append(node)


# Tree: Implement d = a*b + e
def test1():
    a = Value(4.0)
    a._label = "a"
    b = Value(-3.0)
    b._label = "b"
    e = Value(2.0)
    e._label = "e"
    c = a * b
    c._label = "c"
    d = c + e
    d._label = "d"
    print(c._prev, c._op)
    print(d._prev, d._op)
    d.backward()
    print("####post grad####")
    print("a = ", a)
    print("b = ", b)
    print("c = ", c)
    print("e = ", e)
    print("d = ", d)


# Tree: Implement y = tanh(w1x1 + w2x2 + b)
def test2():
    w1 = Value(1.0)
    w1._label = "w1"
    x1 = Value(2.0)
    x1._label = "x1"
    w1x1 = w1 * x1
    w1x1._label = "w1x1"
    w2 = Value(2.0)
    w2._label = "w2"
    x2 = Value(-3.0)
    x2._label = "x2"
    w2x2 = w2 * x2
    w2x2._label = "w2x2"
    b = Value(0.5)
    b._label = "b"
    w1x1w2x2 = w1x1 + w2x2
    w1x1w2x2._label = "w1x1w2x2"
    y = w1x1w2x2 + b
    y._label = "y"
    o = y.tanh()
    o._label = "o"
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

    # #########
    # o =  Value (data = -0.9981778976111987, grad = 1.0, label = o, hash = 269638817)
    # y =  Value (data = -3.5, grad = 0.003640884720487403, label = y, hash = 269638813)
    # w1x1w2x2 =  Value (data = -4.0, grad = 0.003640884720487403, label = w1x1w2x2, hash = 269638809)
    # b =  Value (data = 0.5, grad = 0.003640884720487403, label = b, hash = 269638805)
    # w1x1 =  Value (data = 2.0, grad = 0.003640884720487403, label = w1x1, hash = 269632309)
    # w2x2 =  Value (data = -6.0, grad = 0.003640884720487403, label = w2x2, hash = 269638801)
    # w1 =  Value (data = 1.0, grad = 0.007281769440974806, label = w1, hash = 269632265)
    # x1 =  Value (data = 2.0, grad = 0.003640884720487403, label = x1, hash = 269632361)
    # w2 =  Value (data = 2.0, grad = -0.010922654161462209, label = w2, hash = 269638793)
    # x2 =  Value (data = -3.0, grad = 0.007281769440974806, label = x2, hash = 269638797)


# DAG: Implement d = a + a*b  (a is reused)
def test3():
    a = Value(3.0)
    a._label = "a"
    b = Value(2.0)
    b._label = "b"
    c = a * b
    c._label = "c"
    d = a + c
    d._label = "d"
    d.backward()
    print("d = ", d)
    print("c = ", c)
    print("b = ", b)
    print("a = ", a)

    # ######
    # d =  Value (data = 9.0, grad = 1.0, label = d, hash = 269192517)
    # c =  Value (data = 6.0, grad = 1.0, label = c, hash = 269192217)
    # b =  Value (data = 2.0, grad = 3.0, label = b, hash = 269192225)
    # a =  Value (data = 3.0, grad = 3.0, label = a, hash = 269184497)


# Implement y = tanh(w1x1 + w2x2 + b)
def test4():
    w1 = Value(1.0)
    w1._label = "w1"
    x1 = Value(2.0)
    x1._label = "x1"
    w1x1 = w1 * x1
    w1x1._label = "w1x1"
    w2 = Value(2.0)
    w2._label = "w2"
    x2 = Value(-3.0)
    x2._label = "x2"
    w2x2 = w2 * x2
    w2x2._label = "w2x2"
    b = Value(0.5)
    b._label = "b"
    w1x1w2x2 = w1x1 + w2x2
    w1x1w2x2._label = "w1x1w2x2"
    y = w1x1w2x2 + b
    y._label = "y"
    exp = y.exp()
    exp._label = "exp"
    val = exp**-2
    val._label = "val"
    o = (1 - val) / (1 + val)
    o._label = "o"
    o.backward()
    print("o = ", o)
    print("val = ", val)
    print("exp = ", exp)
    print("y = ", y)
    print("w1x1w2x2 = ", w1x1w2x2)
    print("b = ", b)
    print("w1x1 = ", w1x1)
    print("w2x2 = ", w2x2)
    print("w1 = ", w1)
    print("x1 = ", x1)
    print("w2 = ", w2)
    print("x2 = ", x2)
    # o =  Value (data = -0.9981778976111987, grad = 1.0, label = o, hash = 273477449)
    # val =  Value (data = 1096.6331584284585, grad = -1.6600285576375993e-06, label = val, hash = 273477405)
    # exp =  Value (data = 0.0301973834223185, grad = 0.12056954304842805, label = exp, hash = 273477401)
    # y =  Value (data = -3.5, grad = 0.0036408847204871184, label = y, hash = 273477397)
    # w1x1w2x2 =  Value (data = -4.0, grad = 0.0036408847204871184, label = w1x1w2x2, hash = 273477393)
    # b =  Value (data = 0.5, grad = 0.0036408847204871184, label = b, hash = 273477389)
    # w1x1 =  Value (data = 2.0, grad = 0.0036408847204871184, label = w1x1, hash = 273477069)
    # w2x2 =  Value (data = -6.0, grad = 0.0036408847204871184, label = w2x2, hash = 273477385)
    # w1 =  Value (data = 1.0, grad = 0.007281769440974237, label = w1, hash = 273476617)
    # x1 =  Value (data = 2.0, grad = 0.0036408847204871184, label = x1, hash = 273477077)
    # w2 =  Value (data = 2.0, grad = -0.010922654161461355, label = w2, hash = 273477377)
    # x2 =  Value (data = -3.0, grad = 0.007281769440974237, label = x2, hash = 273477381)


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    test4()
