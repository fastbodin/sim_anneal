import numpy as np
from sim_anneal import qubo_dense_solver as qds


def main():
    # load instance of tsp
    Q = np.loadtxt("../../../problems/tsp/problem_instance/QUBO_matrix")
    beta_schedule = np.loadtxt(
        "../../../problems/tsp/problem_instance/beta_schedule"
    )

    run_data = np.loadtxt("../../../problems/tsp/problem_instance/run_data")
    num_restarts = int(run_data[1])

    # solve and output solution
    x = qds.qubo_solve(Q, num_restarts, beta_schedule)

    # x = np.loadtxt("../../../problems/tsp/ilp/output/solution")
    # print(np.matmul(np.matmul(x,Q),x))

    np.savetxt("output/solution", x, delimiter=" ", fmt="%d")


main()
