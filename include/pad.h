#pragma once

#include "convolution.h"

void malloc_convLayerInputPad(double **A_lMinusPad, double **dA_lMinusPad, convLayerCache *cache);

void free_convLayerInputPad(double *A_lPad, double *dA_lPad);

void malloc_convLayerOutputPad(double **A_lPad, double **dA_lPad, convLayerCache *cache);

void free_convLayerOutputPad(double *A_lMinusPad, double *dA_lMinusPad);

void pad(double *A_l, double *dA_l, double *A_lPad, double *dA_lPad, convLayerCache *cache);

void unpad(double *A_lPad, double *dA_lPad, double *A_l, double *dA_l, convLayerCache *cache);
