#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

int main (void) {
  
  const double lr = 0.1f;

  double hiddenLayer[NUM_HIDDEN_NODES];
  double outputLayer[NUM_OUTPUTS];

  double hiddenLayerBias[NUM_HIDDEN_NODES];
  double outputLayerBias[NUM_OUTPUTS];

  double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];
  double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];

  
}