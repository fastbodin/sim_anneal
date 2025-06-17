#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

class Random {
  std::mt19937 gen; // Standard mersenne_twister_engine
public:
  // Random() : gen(2) {} // set seed for testing purpose
  Random() : gen(std::random_device{}()) {}

  double getprob() { // random real between 0 and 1
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(gen);
  }

  bool getbool() { // random bool
    std::bernoulli_distribution dist(0.5);
    return dist(gen);
  }
};

struct Dense_qubo {
  std::vector<std::vector<double>> Q;
  int n;
  int num_restarts;
  int num_iterations;
  std::vector<double> beta_schedule;
};

struct Solution_state {
  std::vector<bool> x;          // solution state vector
  int n;                        // size of x
  double energy;                // energy of state
  std::vector<double> d_energy; // delta energies of neighbour states

  Solution_state(const int size) : n(size), x(size), d_energy(size){};

  void randomize_x(Random &rng) {
    for (int i = 0; i < n; ++i) {
      x[i] = rng.getbool();
    }
  }

  // Compute: xQx^T given symmetric matrix Q and state vector x.
  void compute_energy(const std::vector<std::vector<double>> &Q) {
    energy = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        energy += Q[i][j] * x[i] * x[j];
      }
    }
  }

  // Compute the delta energy if the ith bit of x is flipped. Given symmetric
  // matrix Q and boolean vectors x and e such that x + e = (x with ith bit
  // flipped), the delta energy is given by:
  // (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
  void compute_delta_energies(const std::vector<std::vector<double>> &Q) {
    for (int i = 0; i < n; ++i) {
      d_energy[i] = Q[i][i];
      for (int j = 0; j < n; ++j) {
        d_energy[i] += 2 * (1 - 2 * x[i]) * Q[i][j] * x[j];
      }
    }
  }
};

// From read_and_print.cpp
void throw_error(const std::string error_message);
void check_qubo_model(const Dense_qubo &model);
Dense_qubo read_qubo_model();
void print_solution(const Solution_state &sol);
