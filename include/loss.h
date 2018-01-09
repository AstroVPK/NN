#pragma once

double crossEntropyForward(int nCat, int M, double *input, double *labels);

void crossEntropyBackward(int nCat, int M, double *input, double *labels, double *output);
