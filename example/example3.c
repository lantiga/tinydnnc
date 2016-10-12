
#include "stdio.h"
#include "tinydnnc.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

clock_t tick;
long epoch = 0;

void epochCb(DNN_Network *net, void *data)
{
  clock_t diff = clock() - tick;
  float elapsed = (float)diff / CLOCKS_PER_SEC;
  printf("Epoch: %03ld; Elapsed(s): %f\n", epoch, elapsed);
  epoch += 1;
  tick = clock();
}

int main(int argc, char* argv[])
{
  int inputdim = 2;
  int nclasses = 3;

  int ntrain = 1000;
  int ntest = 100;

  int ret;

  float *train_data = (float *)malloc(inputdim*ntrain*sizeof(float));
  long *train_labels = (long *)malloc(ntrain*sizeof(long));

  float k = 0.0;
  long errors = 0;

  const char *charset = ".0/ ";
  long c;
  float x, y;
  char cmap[80][80];
  for (int y=0; y<80; y++) {
    for (int x=0; x<80; x++) {
      cmap[x][y] = 3;
    }
  }
  for (int n=0; n<ntrain; n++) {
    c = rand()%3;
    switch (c) {
    case 0:
      x = sin(k) / 2.0 + (float)rand()/RAND_MAX / 10.0;
      y = cos(k) / 2.0 + (float)rand()/RAND_MAX / 10.0;
      break;
    case 1:
      x = sin(k) / 3.0 + 0.4 + (float)rand()/RAND_MAX / 8.0;
      y = cos(k) / 4.0 + 0.4 + (float)rand()/RAND_MAX / 6.0;
      break;
    case 2:
      x = sin(k) / 3.0 - 0.5 + (float)rand()/RAND_MAX / 30.0;
      y = cos(k) / 3.0 + (float)rand()/RAND_MAX / 40.0;
      break;
    }
    cmap[(int)((x+1)*40)][(int)((y+1)*40)] = c;
    train_data[n*inputdim] = x;
    train_data[n*inputdim+1] = y;
    train_labels[n] = c;
    k += 3.141592653589793 * 2 / ntrain;
  }
  for (int y=0; y<80; y++) {
    for (int x=0; x<80; x++) {
      printf("%c",charset[cmap[x][y]]);
    }
    printf("\n");
  }

  DNN_Network *net = DNN_SequentialNetwork();

  DNN_Layer *fc = DNN_FullyConnectedLayer(DNN_ACTIVATION_SIGMOID,
                                          inputdim,
                                          20,1,
                                          DNN_BACKEND_TINYDNN);
  DNN_Layer *fc2 = DNN_FullyConnectedLayer(DNN_ACTIVATION_SIGMOID,
                                          20,
                                          nclasses,0,
                                          DNN_BACKEND_TINYDNN);

  DNN_SequentialAdd(net,fc);
  DNN_SequentialAdd(net,fc2);

  //DNN_Optimizer *optimizer = DNN_AdamOptimizer(0.001,0.9,0.999,0.9,0.999);
  //DNN_Optimizer *optimizer = DNN_AdamOptimizer(0.1,0.9,0.999,0.9,0.999);
  //DNN_Optimizer *optimizer = DNN_SGDOptimizer(0.01,0.0);
  DNN_Optimizer *optimizer = DNN_SGDOptimizer(0.5,0.0);

  tick = clock();
  epoch = 0;
  DNN_Train(net, optimizer, DNN_LOSS_CROSSENTROPY_MULTICLASS,
            train_data, train_labels,
            ntrain, inputdim, 40, 40,
            NULL, epochCb, NULL,
            1, 1, NULL);

  float input[2];
  long label;
  for (int y=0; y<80; y++) {
    for (int x=0; x<80; x++) {
      input[0] = (x-40.0) / 40.0;
      input[1] = (y-40.0) / 40.0;
      label = DNN_PredictLabel(net, input, inputdim);
      printf("%c",charset[label]);
    }
    printf("\n");
  }

  DNN_NetworkDelete(net);

  DNN_LayerDelete(fc);
  DNN_LayerDelete(fc2);

  DNN_OptimizerDelete(optimizer);

  free(train_data);
  free(train_labels);

  return 0;
}

