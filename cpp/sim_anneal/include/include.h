#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

class Random {
  std::mt19937 Gen; // Standard mersenne_twister_engine
  std::uniform_real_distribution<> DistReal;
  std::bernoulli_distribution DistBer;

public:
  Random() : Gen(std::random_device{}()), DistReal(0.0, 1.0), DistBer(0.5) {}
  double getprob() { return DistReal(Gen); }    // random real in [0, 1)
  int_fast8_t getbit() { return DistBer(Gen); } // 0 or 1
};

struct Dense_qubo {
  std::vector<double> Q; // flattened QUBO (nxn) matrix, element in row i, col j
                         // is given index i*n + j
  int n;                 // # of variables in model
  int num_restarts;
  int num_iterations;
  std::vector<double> beta_schedule;

  double matrix_weight(const int i, const int j) const { return Q[i * n + j]; }
};

struct Solution_state {
  std::vector<int_fast8_t> x; // solution state vector
  int n;                      // size of x
  double E;                   // energy of state
  std::vector<double> dE;     // delta energies of neighbour states

  Solution_state(const int size) : n(size), x(size), dE(size){};

  void randomize_x(Random &Rng) {
    for (int i = 0; i < n; ++i) {
      x[i] = Rng.getbit();
    }
  }

  // Compute: xQx^T given qubo model and state vector x.
  void compute_energy(const Dense_qubo &model) {
    E = 0;
    for (int i = 0; i < n; ++i) {
      E += model.matrix_weight(i, i) * x[i];
      for (int j = i + 1; j < n; ++j) {
        E += 2 * model.matrix_weight(i, j) * x[i] * x[j];
      }
    }
  }

  // Compute the delta energy if the ith bit of x is flipped. Given symmetric
  // matrix Q and boolean vectors x and e such that x + e = (x with ith bit
  // flipped), the delta energy is given by:
  // (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
  void compute_delta_energies(const Dense_qubo &model) {
    for (int i = 0; i < n; ++i) {
      dE[i] = model.matrix_weight(i, i);
      for (int j = 0; j < n; ++j) {
        dE[i] += (2 - 4 * x[i]) * model.matrix_weight(i, j) * x[j];
      }
    }
  }
};

// From read_and_print.cpp
void throw_error(const std::string error_message);
void check_qubo_model(const Dense_qubo &model);
Dense_qubo read_qubo_model();
void print_solution(const Solution_state &sol);
