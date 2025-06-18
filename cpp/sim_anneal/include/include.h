#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

class Random {
  std::mt19937 gen; // Standard mersenne_twister_engine
  std::uniform_real_distribution<> dist_real;
  std::bernoulli_distribution dist_ber;

public:
  Random() : gen(std::random_device{}()), dist_real(0.0, 1.0), dist_ber(0.5) {}

  double getprob() { // random real in [0, 1)
    return dist_real(gen);
  }

  uint_fast8_t getbit() { // 0 or 1
    return dist_ber(gen);
  }
};

struct Dense_qubo {
  std::vector<std::vector<double>> Q; // QUBO matrix
  int n;                              // # of variables in model
  int num_restarts;
  int num_iterations;
  std::vector<double> beta_schedule;
};

struct Solution_state {
  std::vector<uint_fast8_t> x; // solution state vector
  int n;                       // size of x
  double E;                    // energy of state
  std::vector<double> dE;      // delta energies of neighbour states

  Solution_state(const int size) : n(size), x(size), dE(size){};

  void randomize_x(Random &rng) {
    for (int i = 0; i < n; ++i) {
      x[i] = rng.getbit();
    }
  }

  // Compute: xQx^T given symmetric matrix Q and state vector x.
  void compute_energy(const std::vector<std::vector<double>> &Q) {
    E = 0;
    for (int i = 0; i < n; ++i) {
      E += Q[i][i] * x[i];
      for (int j = i + 1; j < n; ++j) {
        E += 2 * Q[i][j] * x[i] * x[j];
      }
    }
  }

  // Compute the delta energy if the ith bit of x is flipped. Given symmetric
  // matrix Q and boolean vectors x and e such that x + e = (x with ith bit
  // flipped), the delta energy is given by:
  // (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
  void compute_delta_energies(const std::vector<std::vector<double>> &Q) {
    for (int i = 0; i < n; ++i) {
      dE[i] = Q[i][i];
      for (int j = 0; j < n; ++j) {
        dE[i] += (2 - 4 * x[i]) * Q[i][j] * x[j];
      }
    }
  }
};

// From read_and_print.cpp
void throw_error(const std::string error_message);
void check_qubo_model(const Dense_qubo &model);
Dense_qubo read_qubo_model();
void print_solution(const Solution_state &sol);
