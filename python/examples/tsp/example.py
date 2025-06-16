import numpy as np
from sim_anneal import qubo_dense_solver as qds


def main():
    # load instance of tsp
    Q = np.loadtxt("../../../examples/tsp/output/QUBO_matrix")
    beta_sched = np.loadtxt("../../../examples/tsp/output/beta_schedule")

    run_data = np.loadtxt("../../../examples/tsp/output/run_data")
    num_res = int(run_data[1])
    num_iters = int(run_data[2])

    # solve and output solution
    x = qds.qubo_solve(Q, num_res, num_iters, beta_sched)
    np.savetxt("output/solution", x, delimiter=" ", fmt="%d")


main()
