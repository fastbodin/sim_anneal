import numpy as np
from numpy.typing import NDArray


def delta_energy(Q: NDArray[np.float64], x: NDArray[np.bool_], i: int) -> float:
    """
    Compute the delta energy: (candidate energy) - (previous energy)
    if only the ith bit of x changed
    """
    return 2 * (2 * x[i] - 1) * np.matmul(Q[i], x) - Q[i][i]


def energy(Q: NDArray[np.float64], x: NDArray[np.bool_]) -> float:
    """
    Compute energy: (x Q x^T) given matrix Q and vector x.
    """
    return np.matmul(np.matmul(x, Q), x)


def boltzmann_factor(cur_e: float, can_e: float, beta: float) -> float:
    """
    Compute Boltzmann factor given a current state energy, candidate state
    energy, and beta schedule.
    """
    return np.exp((can_e - cur_e) / (-beta))


def accept_sol(cur_e: float, can_e: float, beta: float) -> bool:
    """
    Accept or decline candidate state given the current state energy, candidate
    state energy, and beta schedule by the Metropolis-Hasting rule.
    """
    if can_e <= cur_e:
        return True
    # accept state with probability equal to Boltzmann factor
    return np.random.random() < boltzmann_factor(cur_e, can_e, beta)


def input_check(
    Q: NDArray[np.float64],
    num_res: int,
    num_iters: int,
    beta_sched: NDArray[np.float64],
) -> None:
    """
    Check inputs of simulated anneal.
    """
    if Q.ndim != 2:
        raise ValueError("Q is not matrix")

    if not np.allclose(Q, Q.T):
        raise ValueError("Q is not symmetric")

    if not isinstance(num_res, int) or num_res < 1:
        raise ValueError("num_res is not positive integer")

    if not isinstance(num_iters, int) or num_iters < 1:
        raise ValueError("num_iters is not positive integer")

    if len(beta_sched) != num_iters:
        raise ValueError("beta_sched is incorrect length")


def qubo_solve(
    Q: NDArray[np.float64],
    num_res: int,
    num_iters: int,
    beta_sched: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Perform simulated anneal for quadratic unconstrained binary optimization

    Args:
        Q: quadratic matrix
        num_res: number of restarts in simulation
        num_iters: number of iterations per restart in simulation
        beta_sched: 1/temperature schedule

    Returns:
        best_sol: minimum energy state found
    """

    input_check(Q, num_res, num_iters, beta_sched)

    n = Q.shape[0]  # each solution comprises n assignments

    # initial solution with min energy state
    best_sol = np.zeros(n, dtype=bool)
    min_energy = float("inf")

    for _ in range(num_res):
        # starting state and associated energy
        x = np.random.randint(2, size=n, dtype=bool)
        cur_e = energy(Q, x)

        for i in range(num_iters):
            beta = beta_sched[i]

            for j in range(n):
                x[j] = not x[j]  # flip spin of node
                can_e = cur_e + delta_energy(Q, x, j)

                if accept_sol(cur_e, can_e, beta):
                    cur_e = can_e
                else:
                    x[j] = not x[j]  # undo flip of spin of node

        if cur_e < min_energy:
            min_energy = cur_e
            best_sol = np.copy(x)

    return best_sol
