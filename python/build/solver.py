import numpy as np
from numpy.typing import NDArray


def energy(Q: NDArray[np.float_], x: NDArray[np.bool_]) -> float:
    """
    Compute energy: (x Q x^T) given matrix Q and vector x

    Args:
        Q (n x n np.array): quadratic matrix
        x (1 x n np.array): binary vector

    Returns:
        float: energy
    """
    return np.matmul(np.matmul(x, Q), np.transpose(x))


def boltzmann_factor(e_cur: float, e_can: float, beta: float) -> float:
    """
    Compute Boltzmann factor given a current energy state and
    candidate energy state

    Args:
        e_cur: current state energy
        e_can: candidate state energy
        beta: 1 / temperature

    Returns:
        float: Boltzmann factor
    """
    return np.exp((e_can - e_cur) / (-beta))


def accept_sol(e_cur: float, e_can: float, beta: float) -> bool:
    """
    Metropolis-Hasting rule for accepting candidate state

    Args:
        e_cur: current state energy
        e_can: candidate state energy
        beta: 1/temperature

    Returns:
        bool: Truth value of statement 'candidate state is accepted'
    """
    if e_can <= e_cur:  # if new state has lower energy
        return True

    # otherwise, accept state with probability equal to Boltzmann factor
    return np.random.random() < boltzmann_factor(e_cur, e_can, beta)


def input_check(
    Q: NDArray[np.float_],
    num_res: int,
    num_iters: int,
    beta_sched: NDArray[np.float_],
) -> None:
    """
    Check inputs to simulated anneal.

    Args:
        Q (n x n np.array): quadratic matrix
        num_res (int): number of restarts in simulation
        num_iters (int): number of iterations per restart in simulation
        beta_sched (1 x n np.array): 1/temperature schedule

    """
    # Q should be a square 2-dim matrix
    if (Q.ndim != 2) or (Q.shape[0] != Q.shape[1]):
        raise ValueError("Q is not square matrix")

    # num_res is a positive integer
    if type(num_res) != int or num_res < 1:
        raise ValueError("num_res is not positive integer")

    # num_iters is a positive integer
    if type(num_iters) != int or num_iters < 1:
        raise ValueError("num_iters is not positive integer")

    # need a value of beta for each iteration
    if len(beta_sched) != num_iters:
        raise ValueError("beta_sched is incorrect length")


def qubo_solve(
    Q: NDArray[np.float_],
    num_res: int,
    num_iters: int,
    beta_sched: NDArray[np.float_],
) -> NDArray[np.bool_]:
    """
    Perform simulated anneal for quadratic unconstrained binary optimization

    Args:
        Q (n x n np.array): quadratic matrix
        num_res (int): number of restarts in simulation
        num_iters (int): number of iterations per restart in simulation
        beta_sched (1 x n np.array): 1/temperature schedule

    Returns:
        NDArray[np.bool_] (1 x n np.array): minimum energy state
    """

    input_check(Q, num_res, num_iters, beta_sched)  # sanity check

    n = Q.shape[0]  # each solution comprises n assignments

    best_sol = np.random.random(n) < 0.5  # solution with min energy state
    min_energy = float("inf")  # initial min energy state

    for _ in range(num_res):  # given the desired # of resarts
        x = np.random.random(n) < 0.5  # initial solution is random n-bit array
        e_cur = energy(Q, x)  # record energy of current state

        for i in range(num_iters):  # for each desired # of iteration
            beta = beta_sched[i]  # beta is fixed for i

            for j in range(n):  # for each node in solution
                x[j] = not x[j]  # flip spin of node
                e_can = energy(Q, x)  # record candidate energy state

                if accept_sol(e_cur, e_can, beta):  # accept candidate state
                    e_cur = e_can  # change reference energy state
                else:  # reject new state
                    x[j] = not x[j]  # undo flip to spin of node

        if e_cur < min_energy:  # record new minimum energy state solution
            min_energy = e_cur
            best_sol = np.copy(x)

    return best_sol
