# Scalar Autograd

A scalar-valued autograd engine.

## Intuition

For backprop, a simple recursion wouldn't work. Take `d = a + a*b`: since we are reusing `a`,
both `c` and `d` depend on `a`, so we'd technically have to sum all the grads.

This means we should not process any of `a`'s children (calling `backward()` on `a`) until we have processed
all of `a`'s parents, in order to completely capture the gradients.

**Therefore: topological sort, from root to leaf.**

## Implementation

- `__add__` — defines the behavior for addition of two objects. Likewise, `__mul__` for multiplication.

## Topo sort algorithm
2-pass algorithm

**1. Build the adjacency list** for all nodes in the graph, from leaf to root.

For the expression `d = a + a*b`:

```text
a = [d, c]
b = [c]
c = [d]
d = []
```

**2. Call autograd on each node** that has all the nodes in its dependency list
already visited.

## Learning
- Jacobian of a matrix: A Jacobian is a Gradient at a given input. It's a grid of "how much does each output move when each input wiggles," and backprop is nothing but multiplying by those grids on the way back.

J = [ 10    0   ]
    [  0   0.1  ]

determinant   = 10 * 0.1 = 1
singular vals = 10 and 0.1
sigma_max     = 10

## Questions
What's the intuition behind?:
- Singular value
- Eigen value
- Determinant

## Gaps against the pytorch autograd
- "prev" is multi-node, for example, sum([Values]) creates lot of intermediary nodes which aren't needed. Sum is the cheapest operation in the autograd and just takes prev._grad = out._grad. Efficient autograds take list of nodes and return a single value node. 