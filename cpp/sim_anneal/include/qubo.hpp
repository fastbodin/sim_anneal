#pragma once
#include <iostream>
#include <vector>

#include "utils.hpp"

class DenseQubo {
public:
  DenseQubo() { read_qubo(); };

  int n;                      // # of variables in model
  std::vector<double> matrix; // flattened QUBO (nxn) matrix where element in
                              // row i and col j is given index i*n + j
  double matrix_weight(const int i, const int j) const {
    return matrix[i * n + j];
  }

private:
  void read_qubo() {
    if (!(std::cin >> n)) {
      throw_error("Failed to read-in # of variables.");
    }
    matrix.resize(n * n);
    for (int i = 0; i < matrix.size(); ++i) {
      if (!(std::cin >> matrix[i])) {
        int row = i / n, col = i - row * n;
        throw_error("Failed to read-in matrix element in (row, col) = (" +
                    std::to_string(row) + ", " + std::to_string(col) + ").");
      }
    }
  }
};
