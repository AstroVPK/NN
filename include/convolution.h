#pragma once

typedef struct convLayerCache_ {
  int      M;              // Mini-batch size
  int      RMinus;         // Number of rows in l-1 th layer activations
  int      R;              // Number of rows in l th layer activations
  int      CMinus;         // Number of columns in l-1 th layer activations
  int      C;              // Number of columns in l th layer activations
  int      F;              // Filter dimension.
  int      KMinus;         // Number of filters in layer l-1
  int      K;              // Number of filters in layer l
  int      S;              // Stride
  double  *W;              // Filter bank          : F X F X KMinus X K dimensional
  double  *dW;             // Filter bank gradient : F X F X KMinus X K dimensional
  double  *b;              // Bias bank            : K dimensional
  double  *db;             // Bias bank gradient   : K dimensional
  double  *Z;              // Intermediate         : R X C X K
  double  *dZ;             // Intermediate         : R X C X K
} convLayerCache;

void init_convLayerCache(int M, int RMinus, int CMinus, int KMinus, int F, int K, int S, convLayerCache *cache);

void malloc_convLayerCache(convLayerCache *cache);

void free_convLayerCache(convLayerCache *cache);

void malloc_convLayerOutput(double **A_l, double **dA_l, convLayerCache *cache);

void free_convLayerOutput(double *A_l, double *dA_l);

void malloc_convLayerInput(double **A_lMinus, double **dA_lMinus, convLayerCache *cache);

void free_convLayerInput(double *A_lMinus, double *dA_lMinus);

void forwardConvolution1X1(double *A_lMinus, double *A_l, convLayerCache *cache);

void backwardConvolution1X1(double *A_lMinus, double *dA_l, double *dA_lMinus, convLayerCache *cache);

void forwardConvolution(double *A_lMinus, double *A_l, convLayerCache *cache);

void backwardConvolution(double *A_lMinus, double *dA_l, double *dA_lMinus, convLayerCache *cache);
