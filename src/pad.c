#include <malloc.h>

#include "convolution.h"
#include "pad.h"

/*
void malloc_convLayerInputPad(double **A_lMinusPad, double **dA_lMinusPad, convLayerCache *cache) {
  int m, r, c, k;
  free_convLayerOutput(*A_lMinusPad, *dA_lMinusPad);
  *A_lMinusPad = (double *)malloc(cache->M*(cache->RMinus + 2*cache->P)*(cache->CMinus + 2*cache->P)*cache->KMinus*sizeof(double));
  *dA_lMinusPad = (double *)malloc(cache->M*(cache->RMinus + 2*cache->P)*(cache->CMinus + 2*cache->P)*cache->KMinus*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->RMinus + 2*cache->P; ++r) {
      for (c = 0; c < cache->CMinus + 2*cache->P; ++c) {
        for (k = 0; k < cache->KMinus; ++k) {
          (*A_lMinusPad)[k + cache->KMinus*(c + (cache->CMinus + 2*cache->P)*(r + (cache->RMinus + 2*cache->P)*m))] = 0.0;
          (*dA_lMinusPad)[k + cache->KMinus*(c + (cache->CMinus + 2*cache->P)*(r + (cache->RMinus + 2*cache->P)*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerInputPad(double *A_lMinusPad, double *dA_lMinusPad) {
  if (A_lMinusPad) {
    free(A_lMinusPad);
    A_lMinusPad = NULL;
  }
  if (dA_lMinusPad) {
    free(dA_lMinusPad);
    dA_lMinusPad = NULL;
  }
}

void malloc_convLayerOutputPad(double **A_lPad, double **dA_lPad, convLayerCache *cache) {
  int m, r, c, k;
  free_convLayerOutput(*A_lPad, *dA_lPad);
  *A_lPad = (double *)malloc(cache->M*(cache->R + 2*cache->P)*(cache->C + 2*cache->P)*cache->K*sizeof(double));
  *dA_lPad = (double *)malloc(cache->M*(cache->R + 2*cache->P)*(cache->C + 2*cache->P)*cache->K*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R + 2*cache->P; ++r) {
      for (c = 0; c < cache->C + 2*cache->P; ++c) {
        for (k = 0; k < cache->K; ++k) {
          (*A_lPad)[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))] = 0.0;
          (*dA_lPad)[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerOutputPad(double *A_lPad, double *dA_lPad) {
  if (A_lPad) {
    free(A_lPad);
    A_lPad = NULL;
  }
  if (dA_lPad) {
    free(dA_lPad);
    dA_lPad = NULL;
  }
}

void pad(double *A_l, double *dA_l, double *A_lPad, double *dA_lPad, convLayerCache *cache) {
  int m, r, c, k;
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          A_lPad[k + cache->K*(c + cache->P + (cache->C + 2*cache->P)*(r + cache->P + (cache->R + 2*cache->P)*m))] = A_l[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))];
          dA_lPad[k + cache->K*(c + cache->P + (cache->C + 2*cache->P)*(r + cache->P + (cache->R + 2*cache->P)*m))] = dA_l[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))];
        }
      }
    }
  }
}

void unpad(double *A_lPad, double *dA_lPad, double *A_l, double *dA_l, convLayerCache *cache) {
  int m, r, c, k;
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          A_l[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))] = A_lPad[k + cache->K*(c + cache->P + (cache->C + 2*cache->P)*(r + cache->P + (cache->R + 2*cache->P)*m))];
          dA_l[k + cache->K*(c + (cache->C + 2*cache->P)*(r + (cache->R + 2*cache->P)*m))] = dA_lPad[k + cache->K*(c + cache->P + (cache->C + 2*cache->P)*(r + cache->P + (cache->R + 2*cache->P)*m))];
        }
      }
    }
  }
}
*/
