#include <malloc.h>
#include <stdio.h>

#include "activation.h"
#include "convolution.h"
#include "pad.h"

int main() {

convLayerCache cache;
double *A_lMinus, *dA_lMinus, *A_l, *dA_l, *A_lMinusPad, *dA_lMinusPad;

init_convLayerCache(10, 4, 4, 3, 2, 8, 2, 2, &cache);
malloc_convLayerCache(&cache);

malloc_convLayerInput(&A_lMinus, &dA_lMinus, &cache);
malloc_convLayerOutput(&A_l, &dA_l, &cache);

malloc_convLayerInputPad(&A_lMinusPad, &dA_lMinusPad, &cache);

free_convLayerOutputPad(A_lMinusPad, dA_lMinusPad);

free_convLayerOutput(A_l, dA_l);
free_convLayerInput(A_lMinus, dA_lMinus);

free_convLayerCache(&cache);
}
