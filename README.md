# Simulated Annealing for QUBOs

This repo contains a simulated annealing algorithm to solve QUBOs.

## Code

See `python/sim_anneal/` and `cpp/sim_anneal/` for an implementation in
Python and C++, respectively.

### Requirements:

The Python build requires NumPy and Numba and was tested on Python 3.12.

The C++ build was tested with GCC 15.1 (https://gcc.gnu.org/gcc-15/).

### Input:

Each problem instance requires three arguments:

```
Q (n x n matrix): Quadratic matrix representing your QUBO model.

num_restarts (integer): Number of desired restarts.

beta_sched (vector): 1/temperate schedule for each iteration.
```

### Usage:

See `python/README.md` and `cpp/README.md` for instructions.

### Examples:

See `examples/tsp/` for example Traveling Salesman Instances that are
referenced in both `python/examples/` and `cpp/examples`.
