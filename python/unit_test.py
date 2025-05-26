from build import qubo_solver as s
import numpy as np


def energy_test():
    if s.energy(np.array([[0, 1], [2, 3]]), np.array([5, 7])) != 252:
        raise ValueError("Incorrect energy computation")
    if s.energy(np.array([[2, 9], [1, -2]]), np.array([-3, 2])) != -50:
        raise ValueError("Incorrect energy computation")


def boltzmann_factor_test():
    # should be 0.9139311....
    bolt_f = s.boltzmann_factor(6, 15, 100)
    if bolt_f >= 0.913932 or bolt_f <= 0.913931:
        raise ValueError("Incorrect boltzmann factor")
    # should be 0.7594152...
    bolt_f = s.boltzmann_factor(0, 3.33, 12.1)
    if bolt_f >= 0.75942 or bolt_f <= 0.759414:
        raise ValueError("Incorrect boltzmann factor")


def accept_sol_test():
    if not s.accept_sol(10, 10, 0.1):
        raise ValueError("Incorrect decision")

    count = 0
    num_iter = 100000
    for _ in range(num_iter):
        count += s.accept_sol(10, 10.5, 1)
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
    beta_sched = np.exp(-decay_rate * np.linspace(0, 10, num_iters))

    x = s.qubo_solve(Q, num_res, num_iters, beta_sched)

    if x.any():
        print("Min energy state: {} is not zero vector".format(x.astype(int)))
        print("The chances of this occuring are small but non-zero")


def main():
    energy_test()
    boltzmann_factor_test()
    accept_sol_test()
    qubo_solve_test()


main()
