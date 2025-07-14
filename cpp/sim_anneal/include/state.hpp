#pragma once
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "qubo.hpp"

class State {
public:
  // Create partially initialized state
  // Note: The energy is set to infinity.
  State(const int num_var)
      : n(num_var), x(num_var), energy(std::numeric_limits<double>::infinity()),
        delta_energy(num_var){};
  // Fully initialize state
  State(const int num_var, Random &Rng, const DenseQubo &Model)
      : n(num_var), x(num_var), delta_energy(num_var) {
    randomize_x(Rng);
    compute_energy(Model);
    compute_delta_energies(Model);
  };

  int n;                            // number of variables
  std::vector<int_fast8_t> x;       // solution state vector
  double energy;                    // energy of state
  std::vector<double> delta_energy; // delta energies of neighbour states

  void print_state() {
    std::ofstream outFile("output/solution");
    if (outFile.is_open()) {
      for (const auto &var : x) {
        outFile << static_cast<int>(var) << std::endl;
      }
    } else {
      throw_error("Failed to open output file.");
    }
    outFile.close();
  }

private:
  // initialize state vector randomly
  void randomize_x(Random &Rng) {
    for (auto &var : x) {
      var = Rng.getbit();
    }
  }
  // Compute: xQx^T given symmetric matrix Q and state vector x.
  void compute_energy(const DenseQubo &Model) {
    energy = 0.0;
    for (int i = 0; i < n; ++i) {
      energy += Model.matrix_weight(i, i) * x[i];
      for (int j = i + 1; j < n; ++j) {
        energy += 2 * Model.matrix_weight(i, j) * x[i] * x[j];
      }
    }
  }
  // Compute the delta energy if the ith bit of x is flipped. Given symmetric
  // matrix Q and boolean vectors x and e such that x + e = (x with ith bit
  // flipped), the delta energy is given by:
  // (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
  void compute_delta_energies(const DenseQubo &Model) {
    for (int i = 0; i < n; ++i) {
      delta_energy[i] = Model.matrix_weight(i, i);
      for (int j = 0; j < n; ++j) {
        delta_energy[i] += (2 - 4 * x[i]) * Model.matrix_weight(i, j) * x[j];
      }
    }
  }
};
