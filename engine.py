import math
import sys
sys.setrecursionlimit(10000) 
e = math.e

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other, (self, other), '**')
        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other # a + b
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __truediv__(self, other):
        return self * (other**(-1))
    
    def __rtruediv__(self, other):
        return other * (self**(-1))
    
    def __repr__(self):
        return f"Value({self.data})"
    
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited and isinstance(v, Value):
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            children = list(node._prev)
            if node._op == '+':
                children[0].grad += node.grad
                children[1].grad += node.grad

            elif node._op == '*':
                children[0].grad += children[1].data * node.grad
                children[1].grad += children[0].data * node.grad

            elif node._op == '**':
                if isinstance(children[0], Value):
                    children[0].grad += children[1] * (children[0].data ** (children[1] - 1)) * node.grad 
                else:
                    children[1].grad += children[0] * (children[1].data ** (children[0] - 1)) * node.grad 
            elif node._op == 'sigmoid':
                children[0].grad += (node.data**(2)) * (e**(-children[0].data) ) * node.grad
            elif node._op == 'ln':
                children[0].grad += (1/children[0].data) * node.grad
            elif node._op == '':
                pass # leaf node                    

