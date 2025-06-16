#include "include.h"

// For error reporting
void throw_error(std::string error_message) {
  throw std::runtime_error(error_message);
}

void check_qubo(Dense_qubo &model) {
  if (model.n <= 0) {
    throw_error("Invalid n = " + std::to_string(model.n));
  }
  if (model.num_restarts <= 0) {
    throw_error("Invalid # of restarts = " +
                std::to_string(model.num_restarts));
  }
  if (model.num_iterations <= 0) {
    throw_error("Invalid # of iterations = " +
                std::to_string(model.num_iterations));
  }
  // check that matrix Q is symmetric
  double epsilon = 0.000000001;
  for (int i = 0; i < model.n; ++i) {
    for (int j = 0; j < model.n; ++j) {
      if ((model.Q[i][j] + epsilon < model.Q[j][i]) ||
          (model.Q[i][j] - epsilon > model.Q[j][i])) {
      }
    }
  }
}

Dense_qubo read_qubo(const int n, const int num_restarts,
                     const int num_iterations) {
  Dense_qubo model;
  model.n = n;
  model.num_restarts = num_restarts;
  model.num_iterations = num_iterations;

  // Fill matrix Q
  model.Q.resize(n);
  for (int i = 0; i < n; ++i) {
    model.Q[i].resize(n);
    for (int j = 0; j < n; ++j) {
      if (!(std::cin >> model.Q[i][j])) {
        throw_error("Failed to assign value to Q[" + std::to_string(i) + "][" +
                    std::to_string(j) + "] when reading qubo matrix.");
      }
    }
  }

  // Fill beta schedule
  model.beta_schedule.resize(model.num_iterations);
  for (int i = 0; i < model.num_iterations; ++i) {
    if (!(std::cin >> model.beta_schedule[i])) {
      throw_error("Failed to assign value to beta_schedule[" +
                  std::to_string(i) + "] when reading beta schedule.");
    }
  }

  check_qubo(model);
  return model;
}
