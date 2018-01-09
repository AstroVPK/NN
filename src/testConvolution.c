#include <malloc.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>

#include "activation.h"
#include "convolution.h"
//#include "pad.h"
#include "loss.h"

#define TEST


int main() {
  convLayerCache cache;
  double *A_0, *dA_0, *A_1, *dA_1;
  A_0 = NULL;
  dA_0 = NULL;
  A_1 = NULL;
  dA_1 = NULL;

  // Make a convLayerCache for 1X1X8 convolution with pad 0 & stride 0 of 4X4X3 images with a minibatch size of 2.
  init_convLayerCache(10, 4, 4, 3, 1, 8, 1, &cache);
  malloc_convLayerCache(&cache);

  malloc_convLayerInput(&A_0, &dA_0, &cache);
  malloc_convLayerOutput(&A_1, &dA_1, &cache);

  /*
  We shall use a 2nd order centered finite difference scheme.
  Compute A_l with the current cache. Then compute loss L.
  Compute the gradient G.
  Compute A_l_Plus with the current cache + 1 input changed by +epsilon. Compute L_plus.
  Compute A_l_Plus with the current cache + 1 input changed by -epsilon. Compute L_minus.
  Compute (L_plus - L_minus)/2*epsilon and check against G.
  */

#ifdef TEST

  double epsilon = 1.0e-9, loss, loss_minus, loss_plus;

  const gsl_rng_type *rngT;
  gsl_rng *rngR;
  gsl_rng_env_setup();
  rngT = gsl_rng_default;
  rngR = gsl_rng_alloc(rngT);

  int m, r, c, k, kMinus;

  // Initialize filter bank & bias bank
  for (k = 0; k < cache.K; ++k) {
    for (kMinus = 0; kMinus < cache.KMinus; ++kMinus) {
      cache.W[k + cache.K*kMinus] = gsl_rng_uniform(rngR);
    }
    cache.b[k] = gsl_rng_uniform(rngR);
  }
  // Initialize input.
  for (m = 0; m < cache.M; ++m) {
    for (r = 0; r < cache.RMinus; ++r) {
      for (c = 0; c < cache.CMinus; ++c) {
        for (k = 0; k < cache.KMinus; ++k) {
          A_0[k + cache.KMinus*(c + cache.CMinus*(r + cache.RMinus*m))] = gsl_rng_uniform(rngR);
        }
      }
    }
  }

  // Initialize labelList.
  double *labelList = (double *)malloc(cache.M*cache.R*cache.C*cache.K*sizeof(double));
  for (m = 0; m < cache.M; ++m) {
    for (r = 0; r < cache.R; ++r) {
      for (c = 0; c < cache.C; ++c) {
        for (k = 0; k < cache.K; ++k) {
          labelList[k + cache.K*(c + cache.C*(r + cache.R*m))] = 0.0;
        }
      }
    }
    labelList[m*cache.R*cache.C*cache.K + 0] = 1.0; // Every train vector is of the same category!
  }

  // Test dW
  //Perform forward convolution & compute loss.
  forwardConvolution1X1(A_0, A_1, &cache);
  loss = crossEntropyForward(cache.R*cache.C*cache.K, cache.M, A_1, labelList);
  crossEntropyBackward(cache.R*cache.C*cache.K, cache.M, A_1, labelList, dA_1);
  backwardConvolution1X1(A_0, dA_1, dA_0, &cache);
  double bpGrad = cache.dW[0 + cache.K*1];
  printf(" Loss: %17.16e\n", loss);

  // Now lets perturb W[1,0]
  cache.W[0 + cache.K*1] += epsilon;
  forwardConvolution1X1(A_0, A_1, &cache);
  loss_plus = crossEntropyForward(cache.R*cache.C*cache.K, cache.M, A_1, labelList);
  printf("Loss+: %17.16e\n", loss_plus);

  // Now lets perturb W[1,0] again
  cache.W[0 + cache.K*1] -= 2.0*epsilon;
  forwardConvolution1X1(A_0, A_1, &cache);
  loss_minus = crossEntropyForward(cache.R*cache.C*cache.K, cache.M, A_1, labelList);
  printf("Loss-: %17.16e\n", loss_minus);

  double fdGrad = (loss_plus - loss_minus)/(2.0*epsilon);
  printf("  Back Prop Grad: %17.16e\n", bpGrad);
  printf("Finite Diff Grad: %17.16e\n", fdGrad);

  // Free labelList
  free(labelList);

  gsl_rng_free(rngR);

#endif

  /*
  Testing complete!!!
  */

  free_convLayerOutput(A_1, dA_1);
  free_convLayerInput(A_0, dA_0);

  free_convLayerCache(&cache);
}
