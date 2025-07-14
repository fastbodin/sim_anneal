import numpy as np
from sim_anneal import qubo_dense_solver as qds


def main():
    # load instance of tsp
    dir = "../../../problems/tsp/problem_instance/"
    Q = np.loadtxt(dir + "QUBO_matrix")
    beta_schedule = np.loadtxt(dir + "beta_schedule")
    num_restarts = int(np.loadtxt(dir + "solver_data")[0])

    # solve and output solution
    x = qds.qubo_solve(Q, num_restarts, beta_schedule)

    # x = np.loadtxt("../../../problems/tsp/ilp/output/solution")
    # print(np.matmul(np.matmul(x,Q),x))

    np.savetxt("output/solution", x, delimiter=" ", fmt="%d")


main()
