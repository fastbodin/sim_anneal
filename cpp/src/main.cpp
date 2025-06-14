#include "include.h"

double random_prob() {
   std::random_device rd;  // used to obtain a seed for the random number
   std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
   std::uniform_real_distribution<> dis(0.0, 1.0);
   return dis(gen);
}

bool boltzmann_factor(double delta_energy, double beta) {
  /*/
   * Compute Boltzmann factor given delta energy and beta.
   */
   return std::exp(-delta_energy * beta);
}

bool accept_neighbor_state(double delta_energy, double beta) {
  /*/
   * Accept or decline candidate state given delta energy and beta schedule by
   * the Metropolis-Hasting rule.
   */
   if (delta_energy <= 0) return true;
   // accept state with probability equal to Boltzmann factor
   return (random_prob() < boltzmann_factor(delta_energy, beta));
}

//def consider_neighbor_states(
//    Q: NDArray[np.float64],
//    n: int,
//    x: NDArray[np.bool_],
//    x_energy: float,
//    delta_energies: NDArray[np.float64],
//    beta: float,
//) -> float:
//    """
//    Given state x, consider neighboring states obtained by flipping
//    each node. Accept neighboring states based on Metropolis-Hasting rule.
//
//    Args:
//        Q: quadratic matrix
//        n: number of variables
//        x: state vector
//        x_energy: energy of current state
//        delta_energies: delta energies of neighboring states
//        beta: 1/temperature for iteration
//
//    Returns:
//        x_energy: Energy of current state
//    """
//
//    for i in range(n):
//        if accept_neighbor_state(delta_energies[i], beta):
//            x[i] = x[i] ^ True  # flip of spin of node
//            x_energy += delta_energies[i]
//            # update each delta energy after flipping spin of ith node
//            for j in range(n):
//                if j != i:
//                    # Given the jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j]
//                    # computed prior to flipping the spin of node i, we can
//                    # update it to reflect this flip by adding the change to
//                    # the term x[i] * x[j] in xQx^T
//                    delta_energies[j] += (
//                        (2 - 4 * x[j]) * (2 * x[i] - 1) * Q[i, j]
//                    )
//            # re-flipping spin of ith node simply flips the sign
//            delta_energies[i] *= -1
//    return x_energy


//
//

int main(int argc, char *argv[]) {
  for (int i = 0; i <= 10; i++) {
    std::cout << random_prob() << std::endl;
  }
}
