#include "checker.hpp"
#include "qubo.hpp"
#include "solver.hpp"
#include "state.hpp"

void consider_neighbor_states(Random &Rng, const DenseQubo &Model,
                              const double beta, State &CurState) {
  /*/
   * Given state, consider neighboring states obtained by flipping
   * each node. Accept neighboring states based on Metropolis-Hasting rule.
   *
   * Args:
   *   	Rng: Random number generator.
   *   	Model: Instance of quadratic unconstrained binary optimization.
   *   	CurState: Solution state.
   *    beta: 1/temperature for iteration.
   */
  int_fast8_t term_sign;

  for (int i = 0; i < Model.n; ++i) {
    // Accept or decline state obtained by flipping sping of ith node by the
    // Metropolis-Hasting rule.
    if ((CurState.delta_energy[i] > 0.0) &&
        (Rng.getprob() >= std::exp(-CurState.delta_energy[i] * beta)))
      continue;
    CurState.x[i] = 1 - CurState.x[i];           // flip of spin of node
    CurState.energy += CurState.delta_energy[i]; // update state energy
    term_sign = 2 * CurState.x[i] - 1;           // for updating delta energies

    for (int j = 0; j < Model.n; ++j) {
      if (j == i)
        continue;
      // Given jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j] computed prior
      // to flipping the spin of node i, it suffices to add the change to
      // the term x[i] * x[j]
      CurState.delta_energy[j] +=
          term_sign * (2 - 4 * CurState.x[j]) * Model.matrix_weight(i, j);
    }
    CurState.delta_energy[i] *= -1.0; // re-flipping spin of node i flips sign
  }
}

void sim_anneal(Random &Rng, const SolverData &Solver, const DenseQubo &Model,
                State &MinState) {
  /*/
   *   From random starting solution, preform simulated anneal.
   *
   *   Args:
   *   	Rng: Random number generator.
   *   	Solver: SolverData parameters
   *   	Model: Instance of quadratic unconstrained binary optimization.
   *   	MinState: Minimum energy state found
   */
  State CurState(Model.n, Rng, Model); // Initialize with random starting state
  for (int i = 0; i < Solver.num_iterations; ++i) {
    consider_neighbor_states(Rng, Model, Solver.beta_schedule[i], CurState);
  }
  if (CurState.energy < MinState.energy) {
    MinState.x = CurState.x;
    MinState.energy = CurState.energy;
    std::cout << MinState.energy << std::endl;
  }
}

int main(const int argc, const char *argv[]) {
  /*/
   *   Preform simulated anneal.
   *
   *   Command line args:
   *   	pipe in desired # of restarts
   *   	pipe in desired # of iterations per restart
   *   	pipe in (1xn)-vector beta schedule
   *
   *   	pipe in # of variables in model
   *   	pipe in QUBO (nxn)-matrix
   */
  Random Rng;              // Random number generator
  SolverData Solver;       // SolverData parameters
  DenseQubo Model;         // Dense QUBO model
  State MinState(Model.n); // Solution state where energy is set to inf
  check_solver_and_qubo(Solver, Model);

  for (int i = 0; i < Solver.num_restarts; ++i) {
    sim_anneal(Rng, Solver, Model, MinState);
  }
  MinState.print_state();
}
