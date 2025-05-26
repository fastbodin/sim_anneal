import sys

sys.path.append("../build/")
import solver as s

import numpy as np
from numpy.typing import NDArray
import math


# returns index of variable x_{i,j} coresponding with city i and step j of tour
def matrix_index(i: int, j: int, n: int) -> int:
    return n * i + j


# each city should be visited once
def add_cons_1(Q: NDArray[np.float_], n: int, p: float) -> None:
    for i in range(n):  # for each city
        for j in range(n):  # for each tour step
            index_0 = matrix_index(i, j, n)
            Q[index_0][index_0] -= p  # update diagonal term: x_{i,j}^2

            for k in range(j + 1, n):  # for each tour step not yet considered
                index_1 = matrix_index(i, k, n)
                Q[index_0][index_1] += p  # update term: x_{i, j}x_{i, k}
                Q[index_1][index_0] += p  # update term: x_{i, k}x_{i, j}


# each tour step should be defined
def add_cons_2(Q: NDArray[np.float_], n: int, p: float) -> None:
    for j in range(n):  # for each tour step
        for i in range(n):  # for each city
            index_0 = matrix_index(i, j, n)
            Q[index_0][index_0] -= p  # update diagonal term: x_{i,j}^2

            for k in range(i + 1, n):  # for each city pair not yet considered
                index_1 = matrix_index(k, j, n)
                Q[index_0][index_1] += p  # update term: x_{i, j}x_{k, j}
                Q[index_1][index_0] += p  # update term: x_{k, j}x_{i, j}


# return distance between cities u and v
def distance(
    u: int, v: int, x_cor: NDArray[np.float_], y_cor: NDArray[np.float_]
):
    return math.sqrt(
        abs(x_cor[u] - x_cor[v]) ** 2 + abs(y_cor[u] - y_cor[v]) ** 2
    )


def add_dis_weights(
    Q: NDArray[np.float_],
    n: int,
    x_cor: NDArray[np.float_],
    y_cor: NDArray[np.float_],
) -> None:
    # for each distinct city pair
    for u in range(n):
        for v in range(n):
            if u == v:
                continue

            for j in range(n):  # for each tour step
                # index of x_{u,j} <- city u at step j
                index_0 = matrix_index(u, j, n)  #
                # index of x_{v,j+1} <- city v at step (j+1) mod n
                index_1 = matrix_index(v, (j + 1) % n, n)
                # update term: x_{u, j}x_{v, j+1}
                Q[index_0][index_1] += distance(u, v, x_cor, y_cor)


def get_penalty(x_cor: NDArray[np.float_], y_cor: NDArray[np.float_]) -> float:
    n = len(x_cor)
    max_d, min_d = 0, float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(i, j, x_cor, y_cor)
            if max_d < dist:
                max_d = dist
            if min_d > dist:
                min_d = dist
    return n * (max_d - min_d) + 1


def main():
    np.random.seed(seed=2)  # seed random number generator

    n = 3  # number of cities
    x_cor, y_cor = np.random.rand(n), np.random.rand(n)  # coordinates of cities

    # penalty for not satisfying constraints 1 and 2
    p = get_penalty(x_cor, y_cor)

    Q = np.zeros((n * n, n * n), dtype=float)  # initialize matrix q

    add_cons_1(Q, n, p)  # update matrix values based on constraint 1
    add_cons_2(Q, n, p)  # update matrix values based on constraint 2
    add_dis_weights(
        Q, n, x_cor, y_cor
    )  # update matrix values based on distances

    # input number of restarts
    num_res = 2

    # input number of iterations
    num_iters = 20

    # input beta schedule for each iteration
    decay_rate = 0.7
    beta_sched = np.exp(-decay_rate * np.linspace(0, 10, num_iters))

    # given quadratic matrix, # of restarts, # of iterations, and beta schedule,
    # run simulated annealing
    x = s.qubo_solve(Q, num_res, num_iters, beta_sched)

    # output solution
    print("Min energy state {}".format(x.astype(int)))


main()
