# Simulated Annealing for QUBOs

This repo contains a simulated annealing algorithm to solve QUBOs written. See
`python/build/solver.py` for our implementation in Python. See
`python/unit_test.py` for unit tests to check your Python build. Finally, see
`python/run.py` for an example of how to run the solver for a toy example.

### Requirements:

The Python build code requires NumPy and was tested on Python 3.12.

### Input:

Each problem instance requires four arguments:

```
Q (n x n matrix): Quadratic matrix representing your problem.

num_res (integer): Number of desired restarts.

num_iters (integer): Number of iterations per restart.

beta_sched (1 x n matrix): 1/temperate schedule for each iteration.

```

### Output:

Given the input arguments defined above, the solver outputs the
solution with the minimum found energy.

