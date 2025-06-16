#include "include.h"

bool accept_neighbor_state(const double d_energy, const double beta,
                           Random rng) {
  /*/
   * Accept state by the Metropolis-Hasting rule.
   *
   * Args:
   * 	d_energy: Delta energy of candidate state.
   *    beta: 1/temperature for iteration.
   *   	rng: Random number generator.
   *
   * Returns
   * 	Truth value of statement 'accept neighboring state.'
   */
  return (d_energy <= 0) || (rng.getprob() < std::exp(-d_energy * beta));
}

void consider_neighbor_states(const Dense_qubo &model, Sol_state &sol,
                              const double beta, Random &rng) {
  /*/
   * Given state x, consider neighboring states obtained by flipping
   * each node. Accept neighboring states based on Metropolis-Hasting rule.
   *
   * Args:
   *   	model: Instance of quadratic unconstrained binary optimization problem.
   *   	sol: Solution state.
   *    beta: 1/temperature for iteration.
   *   	rng: Random number generator.
   */

  for (int i = 0; i < model.n; ++i) {
    if (accept_neighbor_state(sol.d_energy[i], beta, rng)) {
      sol.x[i] = !sol.x[i];               // flip of spin of node
      sol.energy += sol.d_energy[i];      // update state energy
      int term_sign = (2 * sol.x[i] - 1); // for updating delta energies

      for (int j = 0; j < model.n; ++j) {
        if (j != i) {
          // Given jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j] computed prior
          // to flipping the spin of node i, it suffices to add the change to
          // the term x[i] * x[j]
          sol.d_energy[j] += term_sign * ((2 - 4 * sol.x[j]) * model.Q[i][j]);
        }
      }
      sol.d_energy[i] *= -1; // re-flipping spin of node i simply flips sign
    }
  }
}

Sol_state sim_anneal(const Dense_qubo &model, Random &rng) {
  /*/
   *   From random starting sol, preform simulated anneal.
   *
   *   Args:
   *   	model: Instance of quadratic unconstrained binary optimization.
   *   	rng: Random number generator.
   *
   *   Returns:
   *   	sol: Minimum energy sol found in simulated anneal.
   */
  Sol_state sol(model.n, rng); // Initialize random start state.
  sol.compute_energy(model.Q);
  sol.compute_delta_energies(model.Q);
  for (int i = 0; i < model.num_iterations; ++i) {
    consider_neighbor_states(model, sol, model.beta_schedule[i], rng);
  }
  return sol;
}

int main(int argc, char *argv[]) {
  Random rng;                                       // random number generator
  Dense_qubo model = read_qubo(std::atoi(argv[1]),  // # of variables
                               std::atoi(argv[2]),  // # of restarts
                               std::atoi(argv[3])); // # of iterations

  Sol_state best_sol(model.n, rng);
  best_sol.energy = std::numeric_limits<double>::infinity();

  for (int i = 0; i < model.num_restarts; ++i) {
    Sol_state restart_sol = sim_anneal(model, rng);
    if (restart_sol.energy < best_sol.energy) {
      best_sol = restart_sol;
      std::cout << best_sol.energy << std::endl;
    }
  }
  std::cout << best_sol.energy << std::endl;
}
