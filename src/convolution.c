#include <math.h>
#include <stdio.h>

#include "convolutions.h"
#include "activations.h"


void init_convLayerCache(int M, int RMinus, int CMinus, int F, int KMinus, int K, int S, int P, convLayerCache *cache) {
  cache->M = M;
  cache->RMinus = RMinus;
  cache->CMinus = CMinus;
  cache->F = F;
  cache->KMinus = KMinus;
  cache->K = K;
  cache->S = S;
  cache->P = P;
  cache->R = (int)floor(((double)(cache->RMinus + 2*cache->P - cache->F)/cache->S) + 1.0);
  cache->C = (int)floor(((double)(cache->CMinus + 2*cache->P - cache->F)/cache->S) + 1.0);
  cache->W = NULL;
  cache->b = NULL;
  cache->Z = NULL;
  cache->dZ = NULL;
}

void malloc_convLayerCache(convLayerCache *cache) {
  int f1, f2, kMinus, k, m, r, c;
  free_convLayerCache(cache);
  cache->W = (double *)malloc(cache->F*cache->F*cache->KMinus*cache->K*sizeof(double));
  for (f1 = 0; f1 < cache->F; ++f1) {
    for (f2 = 0; f2 < cache->F; ++f2) {
      for (kMinus = 0; kMinus < cache->KMinus; ++kMinus) {
        for (k = 0; k < cache->K; ++k) {
          cache->W[k + cache->K*(kMinus + cache->kMinus*(f2 + cache->F*f1))] = 0.0; // Really this should be better initialized!
        }
      }
    }
  }
  cache->b = (double *)malloc(K*sizeof(double));
  for (k = 0; k < cache->K; ++k) {
    cache->b[k] = 0.0;
  }
  cache->Z = (double *)malloc(cache->M*cache->R*cache->C*cache->K*sizeof(double));
  cache->dZ = (double *)malloc(cache->M*cache->R*cache->C*cache->K*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
          cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerCache(convLayerCache *cache) {
  if (cache->W) {
    free(cache->W);
    cache->W = NULL;
  }
  if (cache->b) {
    free(cache->b);
    cache->b = NULL;
  }
  if (cache->Z) {
    free(cache->Z);
    cache->Z = NULL;
  }
  if (cache->dZ) {
    free(cache->dZ);
    cache->dZ = NULL;
  }
}

void malloc_convLayerOutput(double **A_l, convLayerCache *cache) {
  int m, r, c, k;
  free_convLayerOutput(*A_l);
  *A_l = (double *)malloc(cache->R*cache->C*cache->K*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          (*A_l)[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerOutput(double *A_l) {
  if (A_l) {
    free(A_l);
    A_l = NULL;
  }
}

void forwardConvolution(double *A_lMinus, double *A_l, ConvLayerCache *cache) {
  int m, r, c, k, kMinus, f1, f2;

  for (m = 0, m < cache->M, ++m) {
    for (r = 0, r < cache->R, ++r) {
      for (c = 0, c < cache->C, ++c) {
        for (k = 0, c < cache->K, ++k) {

          for (f1 = 0; f1 < cache->F; ++f1) {
            for (f2 = 0; f2 < cache->F; ++f2) {
              for (kMinus = 0; kMinus < KMinus; ++kMinus) {
                r*stride + f1, c*stride + f2, kMinus
                Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += W[k + cache->K*(kMinus + cache->kMinus*(f2 + cache->F*f1))]*A_lMinus1[kMinus + cache->KMinus*((c*cache->S + f2) + cache->C*((r*cache->S + f1) + cache->R*m))];
              }
            }
          }
          Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += b[k];
          A_l[k + cache->K*(c + cache->C*(r + cache->R*m))] = forwardReLU(Z[k + cache->K*(c + cache->C*(r + cache->R*m))]);
        }
      }
    }
  }
}

void backwardConvolution(double *dA_l, double *dA_lMinus, dW_l, db_l, ConvLayerCache *cache) {
  // First compute dL/dZ^[l] = (dL/dA^[l])*(dg^[l](Z^[l])/dZ)
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))] = backwardReLU(Z[k + cache->K*(c + cache->C*(r + cache->R*m))])*dA_l[k + cache->K*(c + cache->C*(r + cache->R*m))];
        }
      }
    }
  }

  // Now compute dA^[l-1] i.e. dL/dA^[l-1] =

}
