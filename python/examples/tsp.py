import numpy as np
from numpy.typing import NDArray


# returns index of variable x_{i,j} coresponding with city i and step j of tour
def city_stop_index(i: int, j: int, n: int) -> int:
    return n * i + j


# each city should be visited once
def multiple_visit_penalty(Q: NDArray[np.float_], n: int, p: float) -> None:
    for i in range(n):  # for each city
        for j in range(n):  # for each tour step
            index_0 = city_stop_index(i, j, n)
            Q[index_0][index_0] -= p  # update diagonal term: x_{i,j}^2

            for k in range(j + 1, n):  # for each tour step not yet considered
                index_1 = city_stop_index(i, k, n)
                Q[index_0][index_1] += p  # update term: x_{i, j}x_{i, k}
                Q[index_1][index_0] += p  # update term: x_{i, k}x_{i, j}


# one city visited per tour step
def multiple_step_penalty(Q: NDArray[np.float_], n: int, p: float) -> None:
    for j in range(n):  # for each tour step
        for i in range(n):  # for each city
            index_0 = city_stop_index(i, j, n)
            Q[index_0][index_0] -= p  # update diagonal term: x_{i,j}^2

            for k in range(i + 1, n):  # for each city pair not yet considered
                index_1 = city_stop_index(k, j, n)
                Q[index_0][index_1] += p  # update term: x_{i, j}x_{k, j}
                Q[index_1][index_0] += p  # update term: x_{k, j}x_{i, j}


def distance_penalty(
    Q: NDArray[np.float_], n: int, dists: NDArray[np.float_]
) -> None:
    # for each distinct city pair
    for u in range(n):
        for v in range(n):
            if u == v:
                continue

            for j in range(n):  # for each tour step
                # index of x_{u,j} <- city u at step j
                index_0 = city_stop_index(u, j, n)  #
                # index of x_{v,j+1} <- city v at step (j+1) mod n
                index_1 = city_stop_index(v, (j + 1) % n, n)
                # update term: x_{u, j}x_{v, j+1}
                Q[index_0][index_1] += dists[u][v]


def constraint_penality(n: int, dists: NDArray[np.float_]) -> float:
    max_d, min_d = 0, float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist = dists[i][j]
            if max_d < dist:
                max_d = dist
            if min_d > dist:
                min_d = dist
    return n * (max_d - min_d) + 1


# given n cities and known distance between the cities, return quobo matrix
# for TSP instance
def get_qubo_matrix(n: int, dists: NDArray[np.float_]) -> NDArray[np.float_]:
    Q = np.zeros((n * n, n * n), dtype=float)  # initialize matrix Q
    p = constraint_penality(n, dists)  # penalty for not sat. cons. 1 and 2
    multiple_visit_penalty(
        Q, n, p
    )  # update matrix values based on constraint 1
    multiple_step_penalty(Q, n, p)  # update matrix values based on constraint 2
    distance_penalty(Q, n, dists)  # update matrix values based on distances
    return Q
