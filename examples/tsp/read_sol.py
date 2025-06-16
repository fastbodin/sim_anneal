import numpy as np
from numpy.typing import NDArray
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(
    x_cor: NDArray[np.float64],
    y_cor: NDArray[np.float64],
    tour: NDArray[np.int_],
    n: int,
    cost: float,
    lang: str,
) -> None:
    G = nx.DiGraph()

    node_pos = {i: (x_cor[i], y_cor[i]) for i in range(n)}
    G.add_nodes_from(node_pos.keys())

    edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]
    G.add_edges_from(edges)

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(
        G, pos=node_pos, node_color="lightblue", node_size=500
    )
    nx.draw_networkx_labels(G, pos=node_pos)

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
    plt.savefig("solution/tsp_sol_{}.png".format(lang), dpi=300)
    plt.close()


# city i on tour step j is given index: n * i + j
# given input index, return city and tour step
def get_city_and_tour_step(n: int, index: int) -> tuple[int, int]:
    return index // n, index % n


def read_sol(
    n: int, sol: NDArray[np.bool_], dists: NDArray[np.float64]
) -> tuple[NDArray[np.int_], float]:
    tour = np.zeros(n, dtype=int)
    for i in range(len(sol)):
        if sol[i]:
            city, tour_step = get_city_and_tour_step(n, i)
            tour[tour_step] = city

    cost = sum(dists[tour[i]][tour[(i + 1) % n]] for i in range(n))
    print("Tour: {}".format(tour))
    print("Distance: {}".format(cost))
    return tour, cost


def main():
    x_cor = np.loadtxt("output/x_cor")
    y_cor = np.loadtxt("output/y_cor")
    dists = np.loadtxt("output/distances")
    n = len(x_cor)

    print("CPP")
    x = np.loadtxt("../../cpp/examples/tsp/output/solution").astype(np.bool_)
    tour, cost = read_sol(n, x, dists)
    draw_graph(x_cor, y_cor, tour, n, cost, "cpp")

    print("Python")
    x = np.loadtxt("../../python/examples/tsp/output/solution").astype(np.bool_)
    tour, cost = read_sol(n, x, dists)
    draw_graph(x_cor, y_cor, tour, n, cost, "python")


main()
