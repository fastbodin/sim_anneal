import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit
def consider_neighbor_states(
    Q: NDArray[np.float64],
    n: int,
    x: NDArray[np.bool_],
    xE: float,
    dE: NDArray[np.float64],
    beta: float,
) -> float:
    """
    Given state x, consider neighboring states obtained by flipping
    each node. Accept neighboring states based on Metropolis-Hasting rule.

    Args:
        Q: quadratic matrix
        n: number of variables
        x: state vector
        xE: energy of current state
        dE: delta energies of neighboring states
        beta: 1/temperature for iteration

    Returns:
        xE: Energy of current state
    """

    for i in range(n):
        # Accept or decline candidate state by the Metropolis-Hasting rule.
        if (dE[i] <= 0) or (np.random.random() < np.exp(-dE[i] * beta)):
            x[i] = x[i] ^ True  # flip of spin of node
            xE += dE[i]
            term_sign = 2 * x[i] - 1

            for j in range(n):  # update delta energies
                if j != i:
                    # Given jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j]
                    # computed prior to flipping the spin of node i, it
                    # suffices to add the change to the term x[i] * x[j]
                    dE[j] += term_sign * (2 - 4 * x[j]) * Q[i, j]
            dE[i] *= -1  # re-flipping spin of node i simply flips sign
    return xE


def delta_energy(Q: NDArray[np.float64], x: NDArray[np.bool_], i: int) -> float:
    """
    Compute the delta energy if the ith bit of x is flipped. Given symmetric
    matrix Q and boolean vectors x and e such that x + e = (x with ith bit
    flipped), the delta energy is given by:

    (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
    """

    return (2 - 4 * x[i]) * np.matmul(x, Q[i]) + Q[i, i]


def sim_anneal(
    Q: NDArray[np.float64],
    n: int,
    num_iterations: int,
    beta_sched: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], float]:
    """
    From random starting state, preform simulated anneal.

    Args:
        Q: quadratic matrix
        n: number of variables
        num_iterations: number of iterations
        beta_sched: 1/temperature schedule

    Returns:
        x: minimum energy state found in simulated anneal
        xE: associated energy
    """

    x = np.random.randint(2, size=n, dtype=np.bool_)
    xE = np.matmul(np.matmul(x, Q), x)  # energy of state x: xQx^T
    # delta energies of neighboring states
    dE = np.array([delta_energy(Q, x, i) for i in range(n)], dtype=np.float64)

    for i in range(num_iterations):
        xE = consider_neighbor_states(Q, n, x, xE, dE, beta_sched[i])

    return x, xE


def input_check(
    Q: NDArray[np.float64],
    num_res: int,
    num_iterations: int,
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

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise ValueError("num_iterations is not positive integer")


def qubo_solve(
    Q: NDArray[np.float64],
    num_restarts: int,
    beta_schedule: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Perform simulated anneal for dense quadratic unconstrained binary
    optimization

    Args:
        Q: quadratic matrix
        num_restarts: number of restarts in simulation
        beta_schedule: 1/temperature schedule

    Returns:
        Minimum energy state found
    """
    # np.random.seed(seed=21)  # for testing
    n = Q.shape[0]
    num_iterations = beta_schedule.shape[0]
    input_check(Q, num_restarts, num_iterations)

    min_energy_state = np.empty(n, dtype=np.bool_)
    min_energy = float("inf")
    x = np.empty(n, dtype=np.bool_)
    xE = float("inf")

    for _ in range(num_restarts):
        x, xE = sim_anneal(Q, n, num_iterations, beta_schedule)

        if xE < min_energy:
            min_energy_state = np.copy(x)
            min_energy = xE
            # print(min_energy)

    return min_energy_state
