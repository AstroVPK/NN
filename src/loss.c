#include <math.h>
#include <stdio.h>

#include "loss.h"


double crossEntropyForward(int nCat, int M, double *input, double *labels) {
  int idx;
  double loss = 0.0;

  for (idx = 0; idx < nCat*M; ++idx) {
    loss += labels[idx]*log(input[idx]);
  }
  return loss/(double)M;
}

void crossEntropyBackward(int nCat, int M, double *input, double *labels, double *output) {
  int idx;

  for (idx = 0; idx < nCat*M; ++idx) {
    output[idx] = labels[idx]/(M*input[idx]);
  }
}
