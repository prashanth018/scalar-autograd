# To-Do

Engine works — `Value`, topo sort, backprop, `Neuron`/`Layer`/`MLP`, training loop.

## Open
- [ ] Add nonlinearity in `Neuron.__call__` — right now it returns the raw sum, so a `[3,5,5,1]` MLP is just stacked linear layers (collapses to one linear map). `Value.tanh()` already exists, just wrap the `sum(...)`.
- [ ] Loss blows up to `nan` sometimes in `gradient_descent()`
  - 4 of 6 seeds diverge at lr 0.01. Seeds 3 & 5 hit ~1e+160 before nan. Seeds 1 & 6 are fine.
  - Not a gradient bug — same seeds converge at lr 0.001.
  - Suspects: no nonlinearity (nothing bounds the activations), loss is `sum` not `mean` (4x the gradient), `uniform(-1,1)` init.
  - Seeds that explode start at loss ~95-99. Seeds that survive start ~12-36.
  - Try in order: tanh → mean loss → lower lr.

## Later
- [ ] Fused `sum` op on `Value` — 1 node with n parents instead of an n-deep chain of binary adds. Backward is easy, local grad is 1 for every input. Also caps `dfs` recursion depth on wide layers.
- [ ] `_prev` should be a tuple, not a set. Sets dedupe (`a * a` breaks) and have no order (will bite once ops aren't commutative).
