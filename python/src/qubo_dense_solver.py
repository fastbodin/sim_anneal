import numpy as np
from numpy.typing import NDArray
from numba import njit


@njit
def consider_neighbor_states(
    Q: NDArray[np.float64],
    n: int,
    x: NDArray[np.bool_],
    x_energy: float,
    d_energy: NDArray[np.float64],
    beta: float,
) -> float:
    """
    Given state x, consider neighboring states obtained by flipping
    each node. Accept neighboring states based on Metropolis-Hasting rule.

    Args:
        Q: quadratic matrix
        n: number of variables
        x: state vector
        x_energy: energy of current state
        d_energy: delta energies of neighboring states
        beta: 1/temperature for iteration

    Returns:
        x_energy: Energy of current state
    """

    for i in range(n):
        # Accept or decline candidate state by the Metropolis-Hasting rule.
        if d_energy[i] <= 0 | (
            np.random.random() < np.exp(-d_energy[i] * beta)
        ):
            x[i] = x[i] ^ True  # flip of spin of node
            x_energy += d_energy[i]
            term_sign = 2 * x[i] - 1

            for j in range(n):  # update delta energies
                if j != i:
                    # Given jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j]
                    # computed prior to flipping the spin of node i, it
                    # suffices to add the change to the term x[i] * x[j]
                    d_energy[j] += (2 - 4 * x[j]) * term_sign * Q[i, j]
            d_energy[i] *= -1  # re-flipping spin of node i simply flips sign
    return x_energy


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
    num_iters: int,
    beta_sched: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], float]:
    """
    From random starting state, preform simulated anneal.

    Args:
        Q: quadratic matrix
        n: number of variables
        num_iters: number of iterations
        beta_sched: 1/temperature schedule

    Returns:
        x: minimum energy state found in simulated anneal
        x_energy: associated energy
    """

    x = np.random.randint(2, size=n, dtype=np.bool_)
    x_energy = np.matmul(np.matmul(x, Q), x)  # xQx^T
    # delta energies of neighboring states
    d_energy = np.array(
        [delta_energy(Q, x, i) for i in range(n)], dtype=np.float64
    )

    for i in range(num_iters):
        x_energy = consider_neighbor_states(
            Q, n, x, x_energy, d_energy, beta_sched[i]
        )

    return x, x_energy


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
    Perform simulated anneal for dense quadratic unconstrained binary
    optimization

    Args:
        Q: quadratic matrix
        num_res: number of restarts in simulation
        num_iters: number of iterations per restart in simulation
        beta_sched: 1/temperature schedule

    Returns:
        minimum energy state found
    """

    input_check(Q, num_res, num_iters, beta_sched)

    n = Q.shape[0]
    min_energy_state = np.empty(n, dtype=np.bool_)
    min_energy = float("inf")
    x = np.empty(n, dtype=np.bool_)
    x_energy = float("inf")

    for _ in range(num_res):
        x, x_energy = sim_anneal(Q, n, num_iters, beta_sched)

        if x_energy < min_energy:
            min_energy_state = np.copy(x)
            min_energy = x_energy

    print(np.matmul(np.matmul(min_energy_state, Q), min_energy_state))
    return min_energy_state
