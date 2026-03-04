class Value:
    def __init__(self, data, prev = ()):
        self.data = data
        self.prev = set(prev)
        self.grad = 0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Class Value, data = {self.data}"

    def __add__(self, other):
        assert isinstance(other, (int, float, Value)), "only support int and float"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward

        return out
    
    def __mul__(self, other):
        assert isinstance(other, (int, float, Value)), "only support int and float"
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support int and float"
        # other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other, (self,))

        def backward():
            self.grad += (other * (self.data**(other-1))) * out.grad
        out._backward = backward

        return out
    
    def backward(self):
        self.grad = 1
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build(child)
                topo.append(v)
        build(self)
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, other):
        return self.__mul__(other**-1)

    def __sub__(self, other):
        return self.__add__(-other)
    def __rsub__(self, other):
        return other + (-self)
    
    def __radd__(self, other):
        return self.__add__(other)
    def __rmul__(self, other):
        return self * (other ** -1)
    def __rtruediv__(self, other):
        return other * (self ** -1)



