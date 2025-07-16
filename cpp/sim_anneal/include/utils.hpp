#pragma once
#include <random>

inline void throw_error(const std::string &error_message) {
  throw std::runtime_error(error_message);
}

class Random {
public:
  Random() : Gen(std::random_device{}()), DistReal(0.0, 1.0), DistBer(0.5) {}

  double getprob() { return DistReal(Gen); }    // random real in [0, 1)
  int_fast8_t getbit() { return DistBer(Gen); } // 0 or 1

private:
  std::mt19937 Gen; // Standard mersenne_twister_engine
  std::uniform_real_distribution<> DistReal;
  std::bernoulli_distribution DistBer;
};
