import numpy as np
import math
from matplotlib import pyplot as plt

import tsp_matrix


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
    Construct a random instance of the TSP.

    Outputs the associated QUBO matrix and beta schedule.
    """
    np.random.seed(seed=21)  # seed random number generator

    n = 12  # number of cities
    x_cor, y_cor = np.random.rand(n), np.random.rand(n)  # coordinates of cities
    # determine distances between cities
    dists = np.array(
        [
            [get_dist(x_cor[u], y_cor[u], x_cor[v], y_cor[v]) for v in range(n)]
            for u in range(n)
        ]
    )
    Q = tsp_matrix.construct_qubo_matrix(n, dists)
    np.savetxt("output/QUBO_matrix", Q, delimiter=" ")

    num_iters = 100000
    temp = np.exp(-0.5 * np.linspace(0, 10, num_iters))
    beta_sched = 1 / temp
    np.savetxt("output/beta_schedule", beta_sched, delimiter=" ")

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.linspace(0, 10, num_iters), temp, color="blue")
    axs[0].set_title("Temperature Schedule")
    axs[1].plot(np.linspace(0, 10, num_iters), beta_sched, color="red")
    axs[1].set_title("Beta Schedule")

    plt.tight_layout()
    plt.savefig("output/beta_and_temperature.png", dpi=300)
    plt.close()


main()
