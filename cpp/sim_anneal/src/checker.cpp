#include "qubo.hpp"
#include "solver.hpp"

void check_solver_and_qubo(const SolverData &Solver, const DenseQubo &Model) {
  if (Model.n <= 0) {
    throw_error("Invalid n = " + std::to_string(Model.n));
  }
  if (Solver.num_restarts <= 0) {
    throw_error("Invalid # of restarts = " +
                std::to_string(Solver.num_restarts));
  }
  if (Solver.num_iterations <= 0) {
    throw_error("Invalid # of iterations = " +
                std::to_string(Solver.num_iterations));
  }
  // check that matrix is symmetric
  constexpr double epsilon = 0.000000001;
  for (int i = 0; i < Model.n; ++i) {
    for (int j = 0; j < Model.n; ++j) {
      if ((Model.matrix_weight(i, j) + epsilon < Model.matrix_weight(j, i)) ||
          (Model.matrix_weight(i, j) - epsilon > Model.matrix_weight(j, i))) {
        throw_error("QUBO Matrix is not symmetric");
      }
    }
  }
}
