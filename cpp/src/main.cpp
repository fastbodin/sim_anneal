#include "include.h"
#include <vector>

bool accept_neighbor_state(const double delta_energy, const double beta,
                           Random rng) {
  /*/
   * Accept state given delta energy and beta by the Metropolis-Hasting rule.
   */
  if (delta_energy <= 0)
    return true;
  // accept state with probability equal to Boltzmann factor
  return (rng.getprob() < std::exp(-delta_energy * beta));
}

void consider_neighbor_states(const Dense_Qubo &qubo, State &state,
                              const double beta, Random &rng) {
  /*/
   */
  for (int i = 0; i < qubo.n; ++i) {
    if (accept_neighbor_state(state.d_energies[i], beta, rng)) {
      state.x[i] = !state.x[i]; // flip of spin of node
      state.energy += state.d_energies[i];
      // update delta energies
      for (int j = 0; j < qubo.n; ++j) {
        if (j != i) {
          // Given the jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j]
          // computed prior to flipping the spin of node i, we can
          // update it to reflect this flip by adding the change to
          // the term x[i] * x[j] in xQx^T
          state.d_energies[j] +=
              ((2 - 4 * state.x[j]) * (2 * state.x[i] - 1) * qubo.Q[i][j]);
        }
      }
      state.d_energies[i] *= -1; // flipping spin of ith node simply flips sign
    }
  }
}

State sim_anneal(const Dense_Qubo &qubo, const double beta, Random &rng) {
  /*/
   *   From random starting state, preform simulated anneal.
   *
   *   Args:
   *   	qubo: Instance of Quadratic Unconstrained Binary Optimization.
   *   	rng: Random number generator.
   *
   *   Returns:
   *   	state: Minimum energy state found in simulated anneal.
   */
  State state(qubo.n, rng);
  state.compute_energy(qubo.Q);
  state.compute_d_energies(qubo.Q);
  for (int i = 0; i < qubo.num_iters; ++i) {
    consider_neighbor_states(qubo, state, beta, rng);
  }
  return state;
}

int main(int argc, char *argv[]) {
  Random rng; // random number generator
  for (int i = 0; i <= 10; ++i) {
    std::cout << rng.getprob() << std::endl;
  }
}
