#include <math.h>
#include <malloc.h>
#include <stdio.h>

#include "activation.h"
#include "convolution.h"


void init_convLayerCache(int M, int RMinus, int CMinus, int KMinus, int F, int K, int S, convLayerCache *cache) {
  cache->M = M;
  cache->RMinus = RMinus;
  cache->CMinus = CMinus;
  cache->F = F;
  cache->KMinus = KMinus;
  cache->K = K;
  cache->S = S;
  cache->R = (int)floor(((double)(cache->RMinus - cache->F)/cache->S) + 1.0);
  cache->C = (int)floor(((double)(cache->CMinus - cache->F)/cache->S) + 1.0);
  cache->W = NULL;
  cache->dW = NULL;
  cache->b = NULL;
  cache->db = NULL;
  cache->Z = NULL;
  cache->dZ = NULL;
}

void malloc_convLayerCache(convLayerCache *cache) {
  int f1, f2, kMinus, k, m, r, c;
  free_convLayerCache(cache);
  cache->W = (double *)malloc(cache->F*cache->F*cache->KMinus*cache->K*sizeof(double));
  cache->dW = (double *)malloc(cache->F*cache->F*cache->KMinus*cache->K*sizeof(double));
  for (f1 = 0; f1 < cache->F; ++f1) {
    for (f2 = 0; f2 < cache->F; ++f2) {
      for (kMinus = 0; kMinus < cache->KMinus; ++kMinus) {
        for (k = 0; k < cache->K; ++k) {
          cache->W[k + cache->K*(kMinus + cache->KMinus*(f2 + cache->F*f1))] = 0.0; // Really this should be better initialized!
          cache->dW[k + cache->K*(kMinus + cache->KMinus*(f2 + cache->F*f1))] = 0.0;
        }
      }
    }
  }
  cache->b = (double *)malloc(cache->K*sizeof(double));
  cache->db = (double *)malloc(cache->K*sizeof(double));
  for (k = 0; k < cache->K; ++k) {
    cache->b[k] = 0.0;
    cache->db[k] = 0.0;
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
  if (cache->dW) {
    free(cache->dW);
    cache->dW = NULL;
  }
  if (cache->b) {
    free(cache->b);
    cache->b = NULL;
  }
  if (cache->db) {
    free(cache->db);
    cache->db = NULL;
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

void malloc_convLayerOutput(double **A_l, double **dA_l, convLayerCache *cache) {
  int m, r, c, k;
  free_convLayerOutput(*A_l, *dA_l);

  (*A_l) = (double *)malloc(cache->M*cache->R*cache->C*cache->K*sizeof(double));
  (*dA_l) = (double *)malloc(cache->M*cache->R*cache->C*cache->K*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {
          (*A_l)[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
          (*dA_l)[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerOutput(double *A_l, double *dA_l) {
  if (A_l) {
    free(A_l);
    A_l = NULL;
  }
  if (dA_l) {
    free(dA_l);
    dA_l = NULL;
  }
}

void malloc_convLayerInput(double **A_lMinus, double **dA_lMinus, convLayerCache *cache) {
  int m, r, c, k;
  free_convLayerOutput(*A_lMinus, *dA_lMinus);
  *A_lMinus = (double *)malloc(cache->M*cache->RMinus*cache->CMinus*cache->KMinus*sizeof(double));
  *dA_lMinus = (double *)malloc(cache->M*cache->RMinus*cache->CMinus*cache->KMinus*sizeof(double));
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->RMinus; ++r) {
      for (c = 0; c < cache->CMinus; ++c) {
        for (k = 0; k < cache->KMinus; ++k) {
          (*A_lMinus)[k + cache->KMinus*(c + cache->CMinus*(r + cache->RMinus*m))] = 0.0;
          (*dA_lMinus)[k + cache->KMinus*(c + cache->CMinus*(r + cache->RMinus*m))] = 0.0;
        }
      }
    }
  }
}

void free_convLayerInput(double *A_lMinus, double *dA_lMinus) {
  if (A_lMinus) {
    free(A_lMinus);
    A_lMinus = NULL;
  }
  if (dA_lMinus) {
    free(dA_lMinus);
    dA_lMinus = NULL;
  }
}

void forwardConvolution1X1(double *A_lMinus, double *A_l, convLayerCache *cache) {
  int m, r, c, k, kMinus;

  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {

          cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] = 0.0;
          for (kMinus = 0; kMinus < cache->KMinus; ++kMinus) {
            cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += cache->W[k + cache->K*kMinus]*A_lMinus[kMinus + cache->KMinus*(c*cache->S + cache->C*(r*cache->S + cache->R*m))];
          }

          cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += cache->b[k];
          A_l[k + cache->K*(c + cache->C*(r + cache->R*m))] = forwardReLU(cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))]);
        }
      }
    }
  }
}

void backwardConvolution1X1(double *A_lMinus, double *dA_l, double *dA_lMinus, convLayerCache *cache) {
  int m, r, c, k, p, q, pMinus;

  // First compute dL/dZ^[l] = (dL/dA^[l])*(dg^[l](Z^[l])/dZ) and then compute dA^[l-1], dW^[l], & db^[l]
  for (r = 0; r < cache->R; ++r) {
    for (c = 0; c < cache->C; ++c) {
      for (k = 0; k < cache->K; ++k) {

        //for (q = 0; q < cache->M; ++q) {
        for (m = 0; m < cache->M; ++m) {
          q = m;  // dZ/dA_lMinus = 0 if q ne m
          cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))] = backwardReLU(cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))])*dA_l[k + cache->K*(c + cache->C*(r + cache->R*m))];

          //for (p = 0; p < cache->K; ++p) {
          p = k; // dZ/dW = 0 if p ne k
          for (pMinus = 0; pMinus < cache->KMinus; ++pMinus) {

            dA_lMinus[pMinus + cache->KMinus*(c*cache->S + cache->CMinus*(r*cache->S + cache->RMinus*q))] += cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))]*cache->W[k + cache->K*pMinus];

            cache->dW[p + cache->K*pMinus] += cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))]*A_lMinus[pMinus + cache->KMinus*(c*cache->S + cache->CMinus*(r*cache->S + cache->RMinus*m))];
          } // end for pMinus
          //} // end for p

          cache->db[p] += cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))];

        } // end for m
        //} end for q
      } // end for k
    } // end for c
  } // end for r
}

void forwardConvolution(double *A_lMinus, double *A_l, convLayerCache *cache) {
  int m, r, c, k, kMinus, vert, horiz, vert_start, vert_end, horiz_start, horiz_end, f1, f2;

  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; c < cache->K; ++k) {

          vert_start = r*cache->S;
          vert_end = vert_start + cache->F;
          horiz_start = c*cache->S;
          horiz_end = horiz_start + cache->F;

          for (vert = vert_start; vert < vert_end; ++vert) {
            for (horiz = horiz_start; horiz < horiz_end; ++horiz) {
              f1 = vert - vert_start;
              f2 = horiz - horiz_start;

              for (kMinus = 0; kMinus < cache->KMinus; ++kMinus) {
                cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += cache->W[k + cache->K*(kMinus + cache->KMinus*(f2 + cache->F*f1))]*A_lMinus[kMinus + cache->KMinus*((c*cache->S + f2) + cache->C*((r*cache->S + f1) + cache->R*m))];
              }
            }
          }
          cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))] += cache->b[k];
          A_l[k + cache->K*(c + cache->C*(r + cache->R*m))] = forwardReLU(cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))]);
        }
      }
    }
  }
}

/*
void backwardConvolution(double *A_lMinus, double *dA_l, double *dA_lMinus, convLayerCache *cache) {
  int m, r, c, k;

  // First compute dL/dZ^[l] = (dL/dA^[l])*(dg^[l](Z^[l])/dZ) and then compute dA^[l-1], dW^[l], & db^[l]
  int vert_start, vert_end, horiz_start, horiz_end;
  int vert, horiz, kMinus, f1, f2;
  for (m = 0; m < cache->M; ++m) {
    for (r = 0; r < cache->R; ++r) {
      for (c = 0; c < cache->C; ++c) {
        for (k = 0; k < cache->K; ++k) {

          cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))] = backwardReLU(cache->Z[k + cache->K*(c + cache->C*(r + cache->R*m))])*dA_l[k + cache->K*(c + cache->C*(r + cache->R*m))];

          vert_start = r*cache->S;
          vert_end = vert_start + cache->F;
          horiz_start = c*cache->S;
          horiz_end = horiz_start + cache->F;
          for (vert = vert_start; vert < vert_end; ++vert) {
            for (horiz = horiz_start; horiz < horiz_end; ++horiz) {
              f1 = vert - vert_start;
              f2 = horiz - horiz_start;
              for (kMinus = 0; kMinus < cache->KMinus; ++kMinus) {
                dA_lMinus[kMinus + cache->K*(horiz + cache->C*(vert + cache->R*m))] += cache->W[k + cache->K*(kMinus + cache->KMinus*(f2 + cache->F*f1))]*cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))];

                cache->dW[k + cache->K*(kMinus + cache->KMinus*(f2 + cache->F*f1))] += A_lMinus[kMinus + cache->K*(horiz + cache->C*(vert + cache->R*m))]*cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))];

                cache->db[k] += cache->dZ[k + cache->K*(c + cache->C*(r + cache->R*m))];
              }
            }
          }
        }
      }
    }
  }
}
*/
