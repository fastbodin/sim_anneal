import numpy as np
from numpy.typing import NDArray
from numba import njit


def delta_energy(Q: NDArray[np.float64], x: NDArray[np.bool_], i: int) -> float:
    """
    Compute the delta energy: (candidate energy) - (current energy) if the ith
    bit of x is flipped.

    Given symmetric matrix Q, boolean vector x, and boolean vector e where e is
    all zeros except the ith bit is 1-2x[i] (x + e = x with ith bit flipped)

    (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i][i]
    """

    return (2 - 4 * x[i]) * np.matmul(x, Q[i]) + Q[i, i]


def energy(Q: NDArray[np.float64], x: NDArray[np.bool_]) -> float:
    """
    Compute: (x Q x^T) given matrix Q and vector x.
    """
    return np.matmul(np.matmul(x, Q), x)


@njit
def boltzmann_factor(delta_energy: float, beta: float) -> float:
    """
    Compute Boltzmann factor given delta energy and beta.
    """
    return np.exp(-delta_energy * beta)


@njit
def accept_neighbor_state(delta_energy: float, beta: float) -> bool:
    """
    Accept or decline candidate state given delta energy and beta schedule by
    the Metropolis-Hasting rule.
    """
    if delta_energy <= 0:
        return True
    # accept state with probability equal to Boltzmann factor
    return np.random.random() < boltzmann_factor(delta_energy, beta)


@njit
def consider_neighbor_states(
    n: int,
    x: NDArray[np.bool_],
    x_energy: float,
    delta_energies: NDArray[np.float64],
    beta: float,
) -> float:
    """
    Given state x, consider neighboring states obtained by flipping
    each node. Accept neighboring states based on Metropolis-Hasting rule.

    Args:
        n: number of variables
        x: state vector
        x_energy: energy of current state
        delta_energies: delta energies of neighboring states
        beta: 1/temperature for iteration

    Returns:
        x_energy: Energy of current state
    """

    for i in range(n):
        if accept_neighbor_state(delta_energies[i], beta):
            x[i] = x[i] ^ True  # flip of spin of node
            x_energy += delta_energies[i]
            for j in range(n):  # update delta energies given accepted state
                if j != i:
                    delta_energies[j] -= delta_energies[i]
            delta_energies[i] = -delta_energies[i]

    return x_energy


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
    x_energy = energy(Q, x)
    # delta energies of neighboring states
    d_energies = np.array(
        [delta_energy(Q, x, i) for i in range(n)], dtype=np.float64
    )

    for i in range(num_iters):
        x_energy = consider_neighbor_states(
            n, x, x_energy, d_energies, beta_sched[i]
        )

    return x, x_energy


@njit
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
        min_energy_state: minimum energy state found
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

    return min_energy_state
