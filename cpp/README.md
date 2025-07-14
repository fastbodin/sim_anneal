# C++ build

This dir contains a simulated annealing algorithm implemented in C++.

## Compile:

Build should compile by running `make` in `sim_anneal/`. This build was tested
with GCC 15.1 (https://gcc.gnu.org/gcc-15/).

## Running:

Having successfully compiled, the code runs with via the following:

```
./build/sim_anneal < {# restarts} < {# iterations per restart} < {beta_schedule} < {# of variables} < {QUBO_matrix}
```

where

1. `beta_schedule` is a file that contains the beta schedule with elements
   separated by a space.
2. `QUBO_matrix` is a file that contains the QUBO matrix with elements
   separated by a space.

See `../problems/tsp/problem_instance/` for relevant examples of these files for a TSP
instance.

## Output:

When run, the program will output the state with the minimum energy found in
simulated anneal to `output/solution` where row $i$ holds the assignment to
variable $i$.

## Example:

See `examples/tsp/` for an example run file and solution found given
the TSP instance in `../problems/tsp/problem_instance/`.

