#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Add for seeding rand()

#define NUM_INPUTS 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

// Random initializer
double init_weights() {
  return ( (double) rand() ) / ( (double) RAND_MAX );
}

// Activation function: "Squashes" any input into a probability-like range
double sigmoid(double x) {
  return 1.0 / ( 1.0 + exp(-x) );
}

// derivative of sigmoid to back-propogate
double d_sigmoid(double x) {
  return x * ( 1.0 - x );
}

void shuffle(int *array, size_t n) {

  if (n > 1) {
    for (size_t i = 0; i < ( n - 1 ); i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int main (void) {
  
  srand(time(NULL)); // Seed the random number generator

  // Neural Network definitions
  const double lr = 0.1;

  double hiddenLayer[NUM_HIDDEN_NODES];
  double outputLayer[NUM_OUTPUTS];

  double hiddenLayerBias[NUM_HIDDEN_NODES];
  double outputLayerBias[NUM_OUTPUTS];

  double hiddenWeights[NUM_INPUTS][NUM_HIDDEN_NODES];
  double outputWeights[NUM_HIDDEN_NODES][NUM_OUTPUTS];

  // Training datasets
  double trainingInputs[NUM_TRAINING_SETS][NUM_INPUTS] = 
    {
      {0.0f,0.0f},
      {1.0f,0.0f},
      {0.0f,1.0f},
      {1.0f,1.0f},
    };
  double trainingOutputs[NUM_TRAINING_SETS][NUM_OUTPUTS] = 
    {
      {0.0f},
      {1.0f},
      {1.0f},
      {0.0f},
    };
  
  // Initialize initial weights
  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
      hiddenWeights[i][j] = init_weights(); 
    }
  }
  
  for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      outputWeights[i][j] = init_weights(); 
    }
  }

  for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
    hiddenLayerBias[i] = init_weights();
  }

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    outputLayerBias[i] = init_weights();
  }

  int trainingSetOrder[] = {0, 1, 2, 3};

  int numberOfEpochs = 10000;

  for (int epoch = 0; epoch < numberOfEpochs; epoch++) {

    shuffle(trainingSetOrder, NUM_TRAINING_SETS);

    for (int x = 0; x < NUM_TRAINING_SETS; x++) {

      int i = trainingSetOrder[x];

      // Forward pass

      // Compute hidden layer activation
      for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
        
        double activation = hiddenLayerBias[j];

        for (int k = 0; k < NUM_INPUTS; k++) {
          activation += trainingInputs[i][k] * hiddenWeights[k][j];
        }

        hiddenLayer[j] = sigmoid(activation);
        
      }
      
      // Compute output layer activation
      for (int j = 0; j < NUM_OUTPUTS; j++) {
        
        double activation = outputLayerBias[j];

        for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
          activation += hiddenLayer[k] * outputWeights[k][j];
        }

        outputLayer[j] = sigmoid(activation);
        
      }

      printf("Input: %g %g    Output: %g    Predicted output: %g\n",
        trainingInputs[i][0],
        trainingInputs[i][1],
        outputLayer[0],
        trainingOutputs[i][0]
        );

      // Back propagation

      // Compute change in output weights

      double deltaOutput[NUM_OUTPUTS];
      for (int j = 0; j < NUM_OUTPUTS; j++) {
        
        double error = (trainingOutputs[i][j] - outputLayer[j]);
        deltaOutput[j] = error * d_sigmoid(outputLayer[j]);

      }
      
      double deltaHidden[NUM_HIDDEN_NODES];
      for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
        double error = 0.0f;
        for (int k = 0; k < NUM_OUTPUTS; k++) {
          error += deltaOutput[k] * outputWeights[j][k];
        }
        deltaHidden[j] = error * d_sigmoid(hiddenLayer[j]); 
      }
      
      // Apply change in output weights
      for (int j = 0; j < NUM_OUTPUTS; j++) {
        outputLayerBias[j] += deltaOutput[j] * lr;
        for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
          outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
        }
      }
      
      // Apply change in hidden weights
      for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
        hiddenLayerBias[j] += deltaHidden[j] * lr;
        for (int k = 0; k < NUM_INPUTS; k++) {
          hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * lr;
        }
      }

      
    }
    
  }

  // final weights after done training
  printf("\n\nFinal Hidden Weights\n");
  for (int i = 0; i < NUM_INPUTS; i++)
  {
    for (int j = 0; j < NUM_HIDDEN_NODES; j++)
    {
      printf("%f ", hiddenWeights[i][j]);          
    }  
    printf("\n");      
  }

  printf("\nFinal Hidden Biases\n");
  for (int i = 0; i < NUM_HIDDEN_NODES; i++)
  {
    printf("%f ", hiddenLayerBias[i]);
  }
  
  // final weights after done training
  printf("\n\nFinal Output Weights\n");
  for (int i = 0; i < NUM_HIDDEN_NODES; i++)
  {
    for (int j = 0; j < NUM_OUTPUTS; j++)
    {
      printf("%f ", outputWeights[i][j]);          
    }  
    printf("\n");      
  }

  printf("\nFinal Output Biases\n");
  for (int i = 0; i < NUM_OUTPUTS; i++)
  {
    printf("%f ", outputLayerBias[i]);
  }

  printf("\n");
  

  printf("\n\n>>> Main Ends\n");

  return 0;
}