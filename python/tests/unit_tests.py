import numpy as np
from sim_anneal import qubo_dense_solver as qds


def delta_energy_test():
    Q = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]])
    if qds.delta_energy(Q, np.array([1, 1, 1], dtype=np.bool_), 2) != -3:
        raise ValueError("Incorrect delta energy computation")
    if qds.delta_energy(Q, np.array([1, 1, 0], dtype=np.bool_), 2) != 3:
        raise ValueError("Incorrect delta energy computation")


def qubo_solve_test():
    n = 20
    Q = np.identity(n)
    num_res = 1
    num_iters = 10
    temp = np.exp(-0.7 * np.linspace(0, 10, num_iters))
    beta_sched = 1 / temp

    x = qds.qubo_solve(Q, num_res, beta_sched)

    if x.any():
        print("Min energy state: {} is not zero vector".format(x.astype(int)))
        print("The chances of this occuring are small but non-zero")


def main():
    delta_energy_test()
    qubo_solve_test()

    print("Unit tests complete")


main()
