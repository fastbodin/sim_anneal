# Python build

This dir contains a simulated annealing algorithm implemented in Python.

## Requirements:

The build requires NumPy (tested on 2.3.0) and Numba (tested on 0.61.2) and was
tested on Python 3.12.

## Running:

Import the solver via `from sim_anneal import qubo_dense_solver as qds`.
To run the solver, call `qds.qubo_solve(Q, num_restarts, beta_schedule)`
where

1. Q is a (n x n) numpy matrix of type `np.float64` representing your QUBO
   model.
2. num_restarts is an `integer` # of restarts in simulation.
3. beta_schedule is a (1 x n) numpy matrix of type `np.float64` corresponding
   with your desired 1/temperature schedule.

See `examples/tsp/example.py` for an example solver call given the TSP instance
in `../problems/tsp/problem_instance/`.

## Output:

When run, `qds.qubo_solve(Q, num_restarts, beta_schedule)` will return a (1 x
n) numpy matrix of type `np.bool_` where the $i$ th element is the assignment
to the $i$ th variable.

## Example:

See `examples/tsp/` for an example given the TSP instance in
`../problems/tsp/problem_instance/`.
