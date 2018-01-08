#include "activation.h"

double forwardReLU(double input) {
  return (input > 0.0) ? input : 0.0;
}

double backwardReLU(double input) {
  return (input > 0.0) ? 1.0 : 0.0;
}
