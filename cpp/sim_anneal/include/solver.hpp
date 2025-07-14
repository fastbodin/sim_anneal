#pragma once
#include <iostream>
#include <vector>

#include "utils.hpp"

class SolverData {
public:
  SolverData() { read_solver_params(); };

  int num_restarts, num_iterations;
  std::vector<double> beta_schedule;

private:
  void read_solver_params() {
    if (!(std::cin >> num_restarts)) {
      throw_error("Failed to read-in # of restarts.");
    }
    if (!(std::cin >> num_iterations)) {
      throw_error("Failed to read-in # of iterations.");
    }
    beta_schedule.resize(num_iterations);
    for (int i = 0; i < num_iterations; ++i) {
      if (!(std::cin >> beta_schedule[i])) {
        throw_error("Failed to read-in beta value: " + std::to_string(i) + ".");
      }
    }
  }
};
