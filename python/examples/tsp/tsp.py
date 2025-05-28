import numpy as np
from numpy.typing import NDArray


def city_visit_id(i: int, j: int, n: int) -> int:
    """
    The variable x_{i,j}, which holds the truth statement 'city i is the
    jth city visited on the tour,' is given the returned id.
    """
    return n * i + j


def distance_penalty(
    Q: NDArray[np.float64], n: int, dists: NDArray[np.float64]
) -> None:
    """
    The cost of a TSP solution is the sum of the distances.
    This function updates the matrix Q to reflect the distance
    penalty per tour step.

    Args:
        Q: (n x n) quadratic matrix
        n: number of cities
    """
    for i in range(n):  # for each pair of cities
        for j in range(n):
            if i == j:
                continue
            for k in range(n):  # for each tour step
                # index of x_{i,k} <- city i at step k
                index_0 = city_visit_id(i, k, n)  #
                # index of x_{k,k+1} <- city j at step (k+1) mod n
                index_1 = city_visit_id(j, (k + 1) % n, n)
                # update term: x_{u, j}x_{v, j+1}
                Q[index_0][index_1] += dists[i][j]
                Q[index_1][index_0] += dists[j][i]


def multiple_step_penalty(Q: NDArray[np.float64], n: int, p: float) -> None:
    """
    One of the constraints of a TSP is that one city is visited per
    tour step. This function updates the matrix Q to reflect
    the penalty for not satisfying this constraint.

    Args:
        Q: (n x n) quadratic matrix
        n: number of cities
        p: penalty for not satisfying constraint
    """
    for i in range(n):  # each tour step
        for j in range(n):  # each city
            index_0 = city_visit_id(j, i, n)
            Q[index_0][index_0] -= p  # update linear term: x_{j,i}

            for k in range(j + 1, n):  # for each city not yet considered
                index_1 = city_visit_id(k, i, n)
                Q[index_0][index_1] += p  # update term: x_{j, i}x_{k, i}
                Q[index_1][index_0] += p  # update term: x_{k, i}x_{j, i}


def multiple_visit_penalty(Q: NDArray[np.float64], n: int, p: float) -> None:
    """
    One of the constraints of a TSP is that each city is visited
    exactly once. This function updates the matrix Q to reflect
    the penalty for not satisfying this constraint.

    Args:
        Q: (n x n) quadratic matrix
        n: number of cities
        p: penalty for not satisfying constraint
    """
    for i in range(n):  # each city
        for j in range(n):  # each tour step
            index_0 = city_visit_id(i, j, n)
            Q[index_0][index_0] -= p  # update linear term: x_{i,j}

            for k in range(j + 1, n):  # for each tour step not yet considered
                index_1 = city_visit_id(i, k, n)
                Q[index_0][index_1] += p  # update term: x_{i, j}x_{i, k}
                Q[index_1][index_0] += p  # update term: x_{i, k}x_{i, j}


def constraint_penalty(n: int, dists: NDArray[np.float64]) -> float:
    """
    Returns penalty for not satisfying constraints of TSP
    """
    max_d = np.max(dists)
    min_d = np.min(dists[~np.eye(n, dtype=bool)])  # ignore diagonal
    return n * (max_d - min_d) + 1


def input_check(n: int, dists: NDArray[np.float64]) -> None:
    """
    Check inputs
    """
    if dists.ndim != 2:
        raise ValueError("dists is not matrix")

    if not np.allclose(dists, dists.T):
        raise ValueError("dists is not symmetric")

    if not isinstance(n, int) or n < 1:
        raise ValueError("n is not positive integer")

    if dists.shape[0] != n:
        raise ValueError("dists and n have different dimension")


def get_qubo_matrix(n: int, dists: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Create QUBO matrix given TSP problem

    Args:
        n: number of cities
        dists: (n x n) matrix where dists[i][j] is distance between city i and j

    Returns:
        QUBO matrix of TSP given inputs
    """

    input_check(n, dists)
    Q = np.zeros((n * n, n * n), dtype=float)  # quadratic matrix
    p = constraint_penalty(n, dists)  # penalty for not sat. constraints
    multiple_visit_penalty(Q, n, p)  # update matrix values based on constraint
    multiple_step_penalty(Q, n, p)  # update matrix values based on constraint
    distance_penalty(Q, n, dists)  # update matrix values based on distances
    return Q
