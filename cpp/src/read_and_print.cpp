#include "include.h"

void throw_error(const std::string error_message) {
  throw std::runtime_error(error_message);
}

void check_qubo_model(const Dense_qubo &model) {
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
        throw_error("Matrix Q is not symmetric");
      }
    }
  }
}

Dense_qubo read_qubo_model() {
  Dense_qubo model;
  // Read meta data
  if (!(std::cin >> model.n)) {
    throw_error("Failed to assign n");
  }
  if (!(std::cin >> model.num_restarts)) {
    throw_error("Failed to assign # of restarts");
  }
  if (!(std::cin >> model.num_iterations)) {
    throw_error("Failed to assign # of iterations");
  }

  // Fill matrix Q
  model.Q.resize(model.n);
  for (int i = 0; i < model.n; ++i) {
    model.Q[i].resize(model.n);
    for (int j = 0; j < model.n; ++j) {
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

  check_qubo_model(model); // sanity check inputs
  return model;
}

void print_solution(const Solution_state &sol) {
  // std::cout << sol.energy << std::endl;
  for (int i = 0; i < sol.n; ++i) {
    std::cout << sol.x[i] << std::endl;
  }
}
