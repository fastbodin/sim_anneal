import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../src/")
import qubo_dense_solver as qds


def delta_energy_test():
    Q = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]])
    # qds.energy(Q, np.array([1,1,0]))) = 3
    # qds.energy(Q, np.array([1,1,1]))) = 6
    if qds.delta_energy(Q, np.array([1, 1, 1]), 2) != -3:
        raise ValueError("Incorrect delta energy computation")
    if qds.delta_energy(Q, np.array([1, 1, 0]), 2) != 3:
        raise ValueError("Incorrect delta energy computation")


def energy_test():
    if qds.energy(np.array([[0, 1], [2, 3]]), np.array([5, 7])) != 252:
        raise ValueError("Incorrect energy computation")
    if qds.energy(np.array([[2, 9], [1, -2]]), np.array([-3, 2])) != -50:
        raise ValueError("Incorrect energy computation")


def boltzmann_factor_test():
    # should be 0.36787944117144233
    bolt_f = qds.boltzmann_factor(1, 1)
    if bolt_f >= 0.3678795 or bolt_f <= 0.3678793:
        raise ValueError("Incorrect boltzmann factor")
    # should be 0.0889216174593863
    bolt_f = qds.boltzmann_factor(2.2, 1.1)
    if bolt_f >= 0.088921618 or bolt_f <= 0.088921616:
        raise ValueError("Incorrect boltzmann factor")


def accept_sol_test():
    if not qds.accept_sol(0, 0.1):
        raise ValueError("Incorrect decision")
    if not qds.accept_sol(-1, 0.1):
        raise ValueError("Incorrect decision")
    if qds.accept_sol(1, 10000):
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
        count += qds.accept_sol(0.5, 1)
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


def qubo_solve_test():
    n = 20
    Q = np.identity(n)
    num_res = 1
    num_iters = 10
    decay_rate = 0.7
    temp = np.exp(-decay_rate * np.linspace(0, 10, num_iters))
    beta_sched = 1 / temp

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.linspace(0, 10, num_iters), temp, color="blue")
    axs[0].set_title("Temp")
    axs[1].plot(np.linspace(0, 10, num_iters), beta_sched, color="red")
    axs[1].set_title("Beta")

    x = qds.qubo_solve(Q, num_res, num_iters, beta_sched)

    if x.any():
        print("Min energy state: {} is not zero vector".format(x.astype(int)))
        print("The chances of this occuring are small but non-zero")


def main():
    delta_energy_test()
    energy_test()
    boltzmann_factor_test()
    accept_sol_test()
    qubo_solve_test()

    print("Unit tests complete")


main()
