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
b = [d]
c = [d]
d = []
```

**2. Call autograd on each node** that has all the nodes in its dependency list
already visited.
