import numpy as np
from numpy.typing import NDArray
from numba import njit
from numba import float64, int64, boolean
from numba.types import Tuple


@njit(float64(float64[:, :], boolean[:], int64))
def delta_energy(Q: NDArray[np.float64], x: NDArray[np.bool_], i: int) -> float:
    """
    Compute the delta energy: (candidate energy) - (current energy) if the ith
    bit of x is flipped
    """
    return (2 - 4 * x[i]) * np.sum(Q[i][x]) + Q[i][i]


def energy(Q: NDArray[np.float64], x: NDArray[np.bool_]) -> float:
    """
    Compute: (x Q x^T) given matrix Q and vector x.
    """
    return np.matmul(np.matmul(x, Q), x)


# @njit
# def energy(Q: NDArray[np.float64], x: NDArray[np.bool_], n: int) -> float:
#    e = 0.0
#    for i in range(n):
#        if x[i]:
#            e += Q[i][i]
#            for j in range(i+1, n):
#                if x[j]:
#                    e += 2*Q[i][j]
#    return e


@njit(float64(float64, float64))
def boltzmann_factor(delta_energy: float, beta: float) -> float:
    """
    Compute Boltzmann factor given delta energy, and beta schedule.

    Given state i (energy e_i), the probability p_j of flipping to state j
    (energy e_j) is given by
    1 / p_j = exp((e_j - e_i) / kT)
    where k is constant and T is temperature. This reduces to
    p_j = exp(-(e_j - e_i) * beta)
    where beta = 1/kT.
    """
    return np.exp(-delta_energy * beta)


@njit(boolean(float64, float64))
def accept_sol(delta_energy: float, beta: float) -> bool:
    """
    Accept or decline candidate state given delta energy and beta schedule by
    the Metropolis-Hasting rule.
    """
    if delta_energy <= 0:
        return True
    # accept state with probability equal to Boltzmann factor
    return np.random.random() < boltzmann_factor(delta_energy, beta)


@njit(
    Tuple((float64, boolean))(
        float64[:, :], boolean[:], float64[:], int64, float64, float64, boolean
    )
)
def iteration(
    Q: NDArray[np.float64],
    x: NDArray[np.bool_],
    delta_energies: NDArray[np.float64],
    n: int,
    beta: float,
    cur_e: float,
    past_state_change: bool,
) -> tuple[float, bool]:

    state_change = False
    for j in range(n):
        if past_state_change:
            delta_energies[j] = delta_energy(Q, x, j)

        if accept_sol(delta_energies[j], beta):
            x[j] = x[j] ^ True  # flip of spin of node
            cur_e += delta_energies[j]
            state_change = True
            past_state_change = True

    return cur_e, state_change


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

    n = Q.shape[0]
    # initial solution with min energy state
    best_sol = np.zeros(n, dtype=bool)
    min_energy = float("inf")

    delta_energies = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=bool)
    state_change = True

    for _ in range(num_res):
        x[:] = np.random.randint(2, size=n, dtype=bool)  # start state
        cur_e = energy(Q, x)  # associated energy
        state_change = True

        for i in range(num_iters):
            cur_e, state_change = iteration(
                Q, x, delta_energies, n, beta_sched[i], cur_e, state_change
            )

        if cur_e < min_energy:
            min_energy = cur_e
            print(cur_e)
            best_sol = np.copy(x)

    return best_sol
