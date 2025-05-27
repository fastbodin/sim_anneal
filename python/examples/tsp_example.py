import numpy as np
from numpy.typing import NDArray
import math
import sys
import networkx as nx
import matplotlib.pyplot as plt


import tsp

sys.path.append("../build/")
import qubo_solver as qs


def draw_graph(
    x_cor: NDArray[np.float_],
    y_cor: NDArray[np.float_],
    tour: NDArray[np.int_],
    n: int,
    cost: float,
) -> None:
    G = nx.DiGraph()

    # Add nodes and their positions
    node_pos = {i: (x_cor[i], y_cor[i]) for i in range(n)}
    G.add_nodes_from(node_pos.keys())

    # Add edges for the tour
    edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]
    G.add_edges_from(edges)

    plt.figure(figsize=(8, 6))

    # Draw nodes and labels
    nx.draw_networkx_nodes(
        G, pos=node_pos, node_color="lightblue", node_size=500
    )
    nx.draw_networkx_labels(G, pos=node_pos)

    # Draw edges with arrows
    nx.draw_networkx_edges(
        G,
        pos=node_pos,
        edgelist=edges,
        edge_color="r",
        arrows=True,
        arrowsize=20,
    )

    plt.axis("off")
    plt.title("Tour Cost: {}".format(cost), fontsize=14)
    plt.tight_layout()
    plt.savefig("tsp_sol.png", dpi=300)
    plt.close()


# city i on tour step j is given index: n * i + j
# given input index, return city and tour step
def get_city_and_tour_step(n: int, index: int) -> tuple[int, int]:
    return index // n, index % n


def read_sol(
    n: int, sol: NDArray[np.bool_], dists: NDArray[np.float_]
) -> tuple[NDArray[np.int_], float]:
    tour = np.zeros(n, dtype=int)
    for i in range(len(sol)):
        if sol[i]:
            city, tour_step = get_city_and_tour_step(n, i)
            tour[tour_step] = city

    print("Tour: {}".format(tour))
    cost = sum(dists[tour[i]][tour[(i + 1) % n]] for i in range(n))
    print("Distance: {}".format(cost))
    return tour, cost


# return distance between cities u and v
def get_dist(
    u: int, v: int, x_cor: NDArray[np.float_], y_cor: NDArray[np.float_]
):
    return math.sqrt((x_cor[u] - x_cor[v]) ** 2 + (y_cor[u] - y_cor[v]) ** 2)


def main():
    np.random.seed(seed=21)  # seed random number generator

    n = 12  # number of cities
    x_cor, y_cor = np.random.rand(n), np.random.rand(n)  # coordinates of cities
    # distance between cities
    dists = np.array(
        [[get_dist(u, v, x_cor, y_cor) for v in range(n)] for u in range(n)]
    )
    Q = tsp.get_qubo_matrix(n, dists)  # QUBO matrix given instance of TSP

    num_res = 10  # number of restarts
    num_iters = 10  # number of iterations
    decay_rate = 0.7  # beta schedule for each iteration
    beta_sched = np.exp(-decay_rate * np.linspace(0, 10, num_iters))

    # run simulated annealing
    x = qs.qubo_solve(Q, num_res, num_iters, beta_sched)
    tour, cost = read_sol(n, x, dists)
    draw_graph(x_cor, y_cor, tour, n, cost)


main()
