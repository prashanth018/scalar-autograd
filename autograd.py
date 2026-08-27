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

    def backward(self):
        if self._prev:
            l, r = self._prev
            if self._op == "+":
                l._grad = self._grad
                r._grad = self._grad
            elif self._op == "*":
                l._grad = self._grad * r._data
                r._grad = self._grad * l._data
            l.backward()
            r.backward()


if __name__ == "__main__":
    a = Value(4.0)
    b = Value(-3.0)
    e = Value(2.0)
    c = a * b
    d = c + e
    print(c._prev, c._op)
    print(d._prev, d._op)
    d._grad = 1.0
    d.backward()
    print("####post grad####")
    print(a)
    print(b)
    print(c)
    print(e)
    print(d)
