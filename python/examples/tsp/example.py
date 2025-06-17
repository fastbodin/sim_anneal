import numpy as np
from sim_anneal import qubo_dense_solver as qds


def main():
    # load instance of tsp
    Q = np.loadtxt("../../../examples/tsp/output/QUBO_matrix")
    beta_schedule = np.loadtxt("../../../examples/tsp/output/beta_schedule")

    run_data = np.loadtxt("../../../examples/tsp/output/run_data")
    num_restarts = int(run_data[1])

    # solve and output solution
    x = qds.qubo_solve(Q, num_restarts, beta_schedule)
    np.savetxt("output/solution", x, delimiter=" ", fmt="%d")


main()
