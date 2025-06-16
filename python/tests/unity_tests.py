import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../src/")
import qubo_dense_solver as qds


def accept_neighbor_state_test():
    if not qds.accept_neighbor_state(0, 0.1):
        raise ValueError("Incorrect decision")
    if not qds.accept_neighbor_state(-1, 0.1):
        raise ValueError("Incorrect decision")
    if qds.accept_neighbor_state(1, 10000):
        boltz_f = qds.boltzmann_factor(1, 10000)
        print(
            """"Boltzmann factor is {} and delta energy is 1 yet
              solution was accepted""".format(
                boltz_f
            )
        )
        print("The chances of this occuring are small but non-zero")

    count = 0
    num_iter = 100000
    for _ in range(num_iter):
        count += qds.accept_neighbor_state(0.5, 1)
    boltz_f = 0.6065306597126334
    acc_rate = count / num_iter
    if (acc_rate < boltz_f - 0.01) or (acc_rate > boltz_f + 0.01):
        print(
            """Boltzmann factor is {}. After {} iterations, the acceptance
              rate is {}""".format(
                boltz_f, num_iter, acc_rate
            )
        )
        print("The chances of this occuring are small but non-zero")


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

    x = qds.qubo_solve(Q, num_res, num_iters, beta_sched)

    if x.any():
        print("Min energy state: {} is not zero vector".format(x.astype(int)))
        print("The chances of this occuring are small but non-zero")


def main():
    accept_neighbor_state_test()
    delta_energy_test()
    qubo_solve_test()

    print("Unit tests complete")


main()
