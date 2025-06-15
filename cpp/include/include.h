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

class Dense_qubo {
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

  State(int size, Random &rng) : n(size), x(size) {
    for (int i = 0; i < n; ++i) {
      x[i] = rng.getbool();
    }
  }
};
