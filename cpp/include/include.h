#include <cmath>
#include <iostream>
#include <random>
#include <vector>

class Random {
  std::mt19937 gen; // Standard mersenne_twister_engine
public:
  Random() : gen(2) {} // set seed for testing purpose
  // Random() : gen(std::random_device{}()) {}

  double getprob() {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(gen);
  }

  bool getbool() {
    std::bernoulli_distribution dist(0.5);
    return dist(gen);
  }
};

class Dense_Qubo {
public:
  std::vector<std::vector<double>> Q;
  int n;
  int num_iters;
  std::vector<double> beta_sched;
};

class State {
public:
  std::vector<bool> x;
  int n;
  double energy;
  std::vector<double> d_energies;

  State(int size, Random &rng) : n(size), x(size), d_energies(size) {
    for (int i = 0; i < n; ++i) {
      x[i] = rng.getbool();
    }
  }

  // Compute: xQx^T given symmetric matrix Q and state vector x.
  void compute_energy(const std::vector<std::vector<double>> &Q) {
    energy = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        energy += 2 * Q[i][j] * x[i] * x[j];
      }
    }
  }

  /*/
   * Compute the delta energy if the ith bit of the state vector x is flipped.
   * Given symmetric matrix Q and boolean vector e such that
   * (x + e = x with ith bit flipped), this is computed as follows:
   * (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i].
   */
  void compute_d_energies(const std::vector<std::vector<double>> &Q) {
    for (int i = 0; i < n; ++i) {
      d_energies[i] = Q[i][i];
      for (int j = 0; j < n; ++j) {
        d_energies[i] += 2 * (1 - 2 * x[i]) * Q[i][j] * x[j];
      }
    }
  }
};
