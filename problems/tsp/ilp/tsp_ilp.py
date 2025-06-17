import numpy as np
from numpy.typing import NDArray
from numba import njit
import pulp


def ilp(n: int, dists: NDArray[np.float64]):

    ilp = pulp.LpProblem("ILP", pulp.LpMinimize)  # model

    # Variables: x_{i,j} hold the truth value of the statement 'city i is
    # visited directly after city j'
    c_vars = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(n) for j in range(n) if i != j],
        cat="Binary",
    )

    # Variables: t_{i} helps prevent subtours
    t_vars = pulp.LpVariable.dicts(
        "t", [i for i in range(1, n)], lowBound=2, upBound=n, cat="Integer"
    )

    # Objective function
    ilp += pulp.lpSum(
        dists[i][j] * c_vars[(i, j)]
        for i in range(n)
        for j in range(n)
        if i != j
    )

    # Constraints 1 + 2: each city must come before and after exactly one city
    for i in range(n):
        ilp += pulp.lpSum(c_vars[(j, i)] for j in range(n) if i != j) == 1
        ilp += pulp.lpSum(c_vars[(i, j)] for j in range(n) if i != j) == 1

    # Constraint: No subtours
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            # if c_vars[(i,j)] = 1 -> t_vars[i] + 1 <=  t_vars[j]
            # if c_vars[(i,j)] = 0 -> t_vars[i] - t_vars[j] + 2 <= n
            ilp += t_vars[i] - t_vars[j] + 1 <= (n - 1) * (1 - c_vars[(i, j)])

    gurobi_sol = ilp.solve(pulp.getSolver("GUROBI_CMD", msg=False))

    route = []
    for i in range(n):
        for j in range(n):
            if i != j and c_vars[(i, j)].value() == 1:
                route.append((i, j))

    tour = [route[0][0]]
    index = 0
    while index < n - 1:
        tour.append(route[tour[index]][1])
        index += 1

    var_assignment = np.zeros(n * n)
    for index, city in enumerate(tour):
        # City i on tour step j is given index: n * i + j.
        var_assignment[n * city + index] = 1
    np.savetxt("output/solution", var_assignment, delimiter=" ", fmt="%d")


def main():
    dists = np.loadtxt("../problem_instance/distances")
    n = len(dists[0])
    ilp(n, dists)


main()
