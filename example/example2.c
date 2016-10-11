#include <stdio.h>
#include <time.h>
#include <string.h>

#include "tinydnnc.h"

clock_t tick;

float inputDataset[8] = {0, 0, 0, 1, 1, 0, 1, 1};
float outputDataset[4] = {0, 1, 1, 0};
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define SET_SIZE 4

long epoch = 0;

void epochCb(DNN_Network *net, void *data)
{
  float error = DNN_GetLoss(net,
                            DNN_LOSS_MSE,
                            inputDataset,
                            outputDataset,
                            SET_SIZE,
                            NUM_INPUTS,
                            NUM_OUTPUTS);
  clock_t diff = clock() - tick;
  float elapsed = (float)diff / CLOCKS_PER_SEC;
  if (!(epoch % 100))
    printf("Epoch: %03ld; Error: %f; Elapsed(s): %f\n",epoch,error,elapsed);
  epoch += 1;
  tick = clock();
}

void useNetwork(DNN_Network *net) {
    float inputs[2];
    float outputs[1];
    for (int j = 0; j < SET_SIZE; j++) {
        memcpy(inputs,inputDataset+j*2,sizeof(float)*2);
        DNN_Predict(net,inputs,outputs,NUM_INPUTS,NUM_OUTPUTS);
        printf("%f XOR %f -> %f\n", inputs[0], inputs[1], outputs[0]);
    }
}

int main(int argc, char* argv[])
{
  DNN_Network *net = DNN_SequentialNetwork();

  DNN_Layer *fc = DNN_FullyConnectedLayer(DNN_ACTIVATION_SIGMOID,
                                          NUM_INPUTS,
                                          NUM_INPUTS,1,
                                          DNN_BACKEND_TINYDNN);
  DNN_Layer *fc2 = DNN_FullyConnectedLayer(DNN_ACTIVATION_SIGMOID,
                                          NUM_INPUTS,
                                          NUM_OUTPUTS,1,
                                          DNN_BACKEND_TINYDNN);

  DNN_SequentialAdd(net,fc);
  DNN_SequentialAdd(net,fc2);

  DNN_Optimizer *optimizer = DNN_AdamOptimizer(0.001,0.9,0.999,0.9,0.999);
  //DNN_Optimizer *optimizer = DNN_SGDOptimizer(0.01,0.0);

  tick = clock();
  epoch = 0;

  DNN_Fit(net, optimizer, DNN_LOSS_MSE,
          inputDataset, outputDataset,
          SET_SIZE, NUM_INPUTS, NUM_OUTPUTS, 2 /* minibatch */, 10000,
          NULL, epochCb, NULL,
          0, 2, NULL);

  useNetwork(net);

  DNN_NetworkDelete(net);
  DNN_LayerDelete(fc);
  DNN_LayerDelete(fc2);
  DNN_OptimizerDelete(optimizer);
  printf("DONE\n");
  return 0;
}

