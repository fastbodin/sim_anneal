#include "include.h"

void consider_neighbor_states(const Dense_qubo &model, Solution_state &sol,
                              const double beta, Random &rng) {
  /*/
   * Given state, consider neighboring states obtained by flipping
   * each node. Accept neighboring states based on Metropolis-Hasting rule.
   *
   * Args:
   *   	model: Instance of quadratic unconstrained binary optimization problem.
   *   	sol: Solution state.
   *    beta: 1/temperature for iteration.
   *   	rng: Random number generator.
   */
  int term_sign;

  for (int i = 0; i < model.n; ++i) {
    // Accept or decline state obtained by flipping sping of ith node by the
    // Metropolis-Hasting rule.
    if ((sol.dE[i] > 0.0) && (rng.getprob() >= std::exp(-sol.dE[i] * beta))) {
      continue;
    }
    sol.x[i] = 1 - sol.x[i];      // flip of spin of node
    sol.E += sol.dE[i];           // update state energy
    term_sign = 2 * sol.x[i] - 1; // for updating delta energies

    for (int j = 0; j < model.n; ++j) {
      if (j != i) {
        // Given jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j] computed prior
        // to flipping the spin of node i, it suffices to add the change to
        // the term x[i] * x[j]
        sol.dE[j] +=
            term_sign * ((2 - 4 * sol.x[j]) * model.Q[i * model.n + j]);
      }
    }
    sol.dE[i] *= -1.0; // re-flipping spin of node i simply flips sign
  }
}

void sim_anneal(const Dense_qubo &model, Random &rng,
                Solution_state &best_sol) {
  /*/
   *   From random starting solution, preform simulated anneal.
   *
   *   Args:
   *   	model: Instance of quadratic unconstrained binary optimization.
   *   	rng: Random number generator.
   *
   *   Returns:
   *   	sol: Minimum energy sol found in simulated anneal.
   */
  Solution_state sol(model.n);
  sol.randomize_x(rng);
  sol.compute_energy(model.Q);
  sol.compute_delta_energies(model.Q);
  for (int i = 0; i < model.num_iterations; ++i) {
    consider_neighbor_states(model, sol, model.beta_schedule[i], rng);
  }

  if (sol.E < best_sol.E) {
    best_sol.x = sol.x;
    best_sol.E = sol.E;
    std::cout << best_sol.E << std::endl;
  }
}

int main(const int argc, const char *argv[]) {
  /*/
   *   Preform simulated anneal.
   *
   *   Command line args:
   *   	pipe in # of variables in model
   *   	pipe in desired # of restarts
   *   	pipe in desired # of iterations per restart
   *
   *   	pipe in QUBO (nxn)-matrix
   *   	pipe in (1xn)-vector beta schedule
   */
  Random rng; // Random number generator

  Dense_qubo model = read_qubo_model();
  Solution_state best_sol(model.n);
  best_sol.E = std::numeric_limits<double>::infinity();

  for (int i = 0; i < model.num_restarts; ++i) {
    sim_anneal(model, rng, best_sol);
  }
  print_solution(best_sol);
}
