#include <math.h>

#include "activation.h"


double dirac(int i, int j) {
  return (fabs(i - j) > 0) ? 0 : 1;
}

double forwardReLU(double input) {
  return (input > 0.0) ? input : 0.0;
}

double backwardReLU(double input) {
  return (input > 0.0) ? 1.0 : 0.0;
}

void forwardSoftmax(int n, double *input, double *output) {
  int idx;
  double sum = 0.0;

  for (idx = 0; idx < n; ++idx) {
    sum += exp(input[idx]);
  }

  for (idx = 0; idx < n; ++idx) {
    output[idx] = exp(input[idx])/sum;
  }
}

void backwardSoftmax(int n, double *input, double *output) {
  /*
  Backprop for softmax

  The output is n X n.
  */

  int idx_i, idx_j;
  for (idx_j = 0; idx_j < n; ++idx_j) {
    for (idx_i = 0; idx_i < n; ++idx_i) {
      output[idx_j*n + idx_i] = input[idx_i]*(dirac(idx_i, idx_j) - input[idx_j]);
    }
  }
}
