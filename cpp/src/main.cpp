#include "include.h"

double random_prob() {
  // std::random_device rd;  // used to obtain a seed for the random number
  std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);
  return dis(gen);
}

bool boltzmann_factor(const double delta_energy, const double beta) {
  /*/
   * Compute Boltzmann factor given delta energy and beta.
   */
  return std::exp(-delta_energy * beta);
}

bool accept_neighbor_state(const double delta_energy, const double beta) {
  /*/
   * Accept or decline candidate state given delta energy and beta schedule by
   * the Metropolis-Hasting rule.
   */
  if (delta_energy <= 0)
    return true;
  // accept state with probability equal to Boltzmann factor
  return (random_prob() < boltzmann_factor(delta_energy, beta));
}

double consider_neighbor_states(const std::vector<std::vector<double>> &Q,
                                const int n, std::vector<bool> &x,
                                double x_energy,
                                std::vector<double> &delta_energies,
                                const double beta) {
  /*/
   *    Given state x, consider neighboring states obtained by flipping
   *    each node. Accept neighboring states based on Metropolis-Hasting rule.
   *
   *    Args:
   *        Q: quadratic matrix
   *        n: number of variables
   *        x: state vector
   *        x_energy: energy of current state
   *        delta_energies: delta energies of neighboring states
   *        beta: 1/temperature for iteration
   *
   *    Returns:
   *        x_energy: Energy of current state
   */
  for (int i = 0; i < n; i++) {
    if (accept_neighbor_state(delta_energies[i], beta)) {
      x[i] = !x[i]; // flip of spin of node
      x_energy += delta_energies[i];
      // update each delta energy after flipping spin of ith node
      for (int j = 0; j < n; j++) {
        if (j != i) {
          // Given the jth delta energy: 2(1-2x[j])Q[j]x^T + Q[j,j]
          // computed prior to flipping the spin of node i, we can
          // update it to reflect this flip by adding the change to
          // the term x[i] * x[j] in xQx^T
          delta_energies[j] += ((2 - 4 * x[j]) * (2 * x[i] - 1) * Q[i][j]);
        }
      }
      // re-flipping spin of ith node simply flips the sign
      delta_energies[i] *= -1;
    }
  }
  return x_energy;
}

double delta_energy(const std::vector<std::vector<double>> &Q, const int n,
                    const std::vector<bool> &x, const int i) {
  /*/
   * Compute the delta energy: (candidate energy) - (current energy) if the ith
   * bit of x is flipped. Given symmetric matrix Q, boolean vector x, and
   * boolean vector e where e is all zeros except the ith bit is
   * 1-2x[i] (x + e = x with ith bit flipped)
   * (x+e)Q(x+e)^T - xQx^t = 2eQx^T + eQe^T = 2(1-2x[i])Q[i]x^T + Q[i,i]
   */
  double value = 0;
  for (int j = 0; j < n; j++) {
    value += Q[i][j] * x[j];
  }
  return 2 * (1 - 2 * x[i]) * value + Q[i][i];
}

double energy(const std::vector<std::vector<double>> &Q, const int n,
              const std::vector<bool> &x) {
  // Compute: xQx^T given symmetric matrix Q and vector x.
  double value = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      value += 2 * Q[i][j] * x[i] * x[j];
    }
  }
  return value;
}

int main(int argc, char *argv[]) {
  for (int i = 0; i <= 10; i++) {
    std::cout << random_prob() << std::endl;
  }
}
