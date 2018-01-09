#pragma once

double forwardReLU(double input);

double backwardReLU(double input);

void forwardSoftmax(int n, double *input, double *output);

void backwardSoftmax();
