import numpy as np
import math
from matplotlib import pyplot as plt

import generate_tsp_matrix as gtm


def get_dist(
    u_x_cor: np.float64,
    u_y_cor: np.float64,
    v_x_cor: np.float64,
    v_y_cor: np.float64,
):
    """
    Return distance between cities u and v given (x,y)-coordinates
    """
    return math.sqrt((u_x_cor - v_x_cor) ** 2 + (u_y_cor - v_y_cor) ** 2)


def main():
    """
    Construct a random TSP instance.

    Outputs the associated QUBO matrix and beta schedule.
    """
    np.random.seed(seed=21)  # seed random number generator

    n = 12  # number of cities
    num_res = 1000  # number of restarts
    num_iters = 100000  # number of iterations
    run_data = np.array([n * n, num_res, num_iters])

    # coordinates of cities, distances between cities, and QUBO matrix
    x_cor, y_cor = np.random.rand(n), np.random.rand(n)
    dists = np.array(
        [
            [get_dist(x_cor[u], y_cor[u], x_cor[v], y_cor[v]) for v in range(n)]
            for u in range(n)
        ]
    )
    Q = gtm.construct_qubo_matrix(n, dists)

    temperature = np.exp(-0.5 * np.linspace(0, 10, num_iters))
    beta_schedule = 1 / temperature
    # Visualize beta and temperature schedule
    _, axs = plt.subplots(1, 2)
    axs[0].plot(np.linspace(0, 10, num_iters), temperature, color="blue")
    axs[0].set_title("Temperature Schedule")
    axs[1].plot(np.linspace(0, 10, num_iters), beta_schedule, color="red")
    axs[1].set_title("Beta Schedule")
    plt.tight_layout()
    plt.savefig("problem_instance/beta_and_temperature.png", dpi=300)
    plt.close()

    # save all relevant data for future use
    np.savetxt("problem_instance/run_data", run_data, delimiter=" ", fmt="%d")
    np.savetxt("problem_instance/x_cor", x_cor, delimiter=" ")
    np.savetxt("problem_instance/y_cor", y_cor, delimiter=" ")
    np.savetxt("problem_instance/distances", dists, delimiter=" ")
    np.savetxt("problem_instance/QUBO_matrix", Q, delimiter=" ")
    np.savetxt("problem_instance/beta_schedule", beta_schedule, delimiter=" ")


main()
