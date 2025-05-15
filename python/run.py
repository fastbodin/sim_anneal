from build import solver as s
import numpy as np


def main():
    # input quadratic matrix
    n = 20
    Q = np.identity(n)

    # input number of restarts
    num_res = 1

    # input number of iterations
    num_iters = 10

    # input beta schedule for each iteration
    decay_rate = 0.7
    beta_sched = np.exp(-decay_rate * np.linspace(0, 10, num_iters))

    # given quadratic matrix, # of restarts, # of iterations, and beta schedule,
    # run simulated annealing
    x = s.qubo_solve(Q, num_res, num_iters, beta_sched)

    # output solution
    print("Min energy state {}:".format(x.astype(int)))


main()
