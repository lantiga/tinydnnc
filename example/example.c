
#include "stdio.h"
#include "tinydnnc.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

typedef struct CallbackData {
  float **test_images;
  long *test_labels;
  long nsamples;
  long sample_size;
} CallbackData;

void batchCb(DNN_Network* net, void *data)
{
}

long epoch = 0;

void epochCb(DNN_Network* net, void *data)
{
  CallbackData *cbdata = (CallbackData *)data;
  float error = DNN_GetError(net,
                             cbdata->test_images,
                             cbdata->test_labels,
                             cbdata->nsamples,
                             cbdata->sample_size);
  printf("Epoch: %03ld; Error: %f.\n",epoch,error);
  epoch += 1;
}

int main(int argc, char* argv[])
{
  int imgsize = 28;
  int nclasses = 10;

  mnist_data *mnist_train_data;
  mnist_data *mnist_test_data;
  unsigned int nimages;
  unsigned int n_test_images;
  int ret;

  // MEMO
  //typedef struct mnist_data {
  //  MNIST_DATA_TYPE data[28][28]; /* 28x28 data for the image */
  //  unsigned int label; /* label : 0 to 9 */
  //} mnist_data;

  ret = mnist_load("../deps/tiny-dnn/data/train-images.idx3-ubyte",
                   "../deps/tiny-dnn/data/train-labels.idx1-ubyte",
                   &mnist_train_data, &nimages);

  nimages = 10000;

  if (ret) {
    printf("Error reading mnist data: %d\n", ret);
    exit(1);
  }

  float **train_images = (float **)malloc(nimages*sizeof(float *));
  long *train_labels = (long *)malloc(nimages*sizeof(long));

  for (int n=0; n<nimages; n++) {
    train_images[n] = (float *)malloc(imgsize*imgsize*sizeof(float *));
    for (int i=0; i<imgsize; i++) {
      for (int j=0; j<imgsize; j++) {
        train_images[n][i*imgsize+j] = mnist_train_data[n].data[i][j];
      }
    }
    train_labels[n] = mnist_train_data[n].label;
  }

  free(mnist_train_data);

  ret = mnist_load("../deps/tiny-dnn/data/t10k-images.idx3-ubyte",
                   "../deps/tiny-dnn/data/t10k-labels.idx1-ubyte",
                   &mnist_test_data, &n_test_images);

  if (ret) {
    printf("Error reading mnist data: %d\n", ret);
    exit(1);
  }

  float **test_images = (float **)malloc(n_test_images*sizeof(float *));
  long *test_labels = (long *)malloc(n_test_images*sizeof(long));

  for (int n=0; n<n_test_images; n++) {
    test_images[n] = (float *)malloc(imgsize*imgsize*sizeof(float *));
    for (int i=0; i<imgsize; i++) {
      for (int j=0; j<imgsize; j++) {
        test_images[n][i*imgsize+j] = mnist_test_data[n].data[i][j];
      }
    }
    test_labels[n] = mnist_test_data[n].label;
  }

  free(mnist_test_data);

  CallbackData cbdata;
  cbdata.test_images = test_images;
  cbdata.test_labels = test_labels;
  cbdata.nsamples = nimages;
  cbdata.sample_size = imgsize*imgsize;

  DNN_Network *net = DNN_SequentialNetwork();

#define FC_EXAMPLE

#ifdef FC_EXAMPLE
  DNN_Layer *fc = DNN_FullyConnectedLayer(DNN_ACTIVATION_SOFTMAX,
                                          imgsize*imgsize*1,
                                          nclasses,0,
                                          DNN_BACKEND_TINYDNN);
  DNN_SequentialAdd(net,fc);
#endif

#ifdef CNN_EXAMPLE
  DNN_Layer *conv1 = DNN_ConvolutionalLayer(DNN_ACTIVATION_RELU,
                                            imgsize,imgsize,
                                            5,5,1,8,DNN_PADDING_SAME,
                                            1,1,1,
                                            DNN_BACKEND_TINYDNN);
  DNN_Layer *maxpool1 = DNN_MaxPoolLayer(DNN_ACTIVATION_IDENTITY,
                                         imgsize,imgsize,
                                         8,2,2,
                                         DNN_BACKEND_TINYDNN);
  DNN_Layer *fc = DNN_FullyConnectedLayer(DNN_ACTIVATION_SOFTMAX,
                                          imgsize/2*imgsize/2*16,
                                          nclasses,0,
                                          DNN_BACKEND_TINYDNN);

  DNN_SequentialAdd(net,conv1);
  DNN_SequentialAdd(net,maxpool1);
  DNN_SequentialAdd(net,fc);
#endif

#ifdef CNN2_EXAMPLE
  DNN_Layer *conv1 = DNN_ConvolutionalLayer(DNN_ACTIVATION_RELU,
                                            imgsize,imgsize,
                                            5,5,1,8,DNN_PADDING_SAME,
                                            1,1,1,
                                            DNN_BACKEND_TINYDNN);
  DNN_Layer *maxpool1 = DNN_MaxPoolLayer(DNN_ACTIVATION_IDENTITY,
                                         imgsize,imgsize,
                                         8,2,2,
                                         DNN_BACKEND_TINYDNN);
  DNN_Layer *conv2 = DNN_ConvolutionalLayer(DNN_ACTIVATION_RELU,
                                            imgsize/2,imgsize/2,
                                            5,5,8,16,DNN_PADDING_SAME,
                                            1,1,1,
                                            DNN_BACKEND_TINYDNN);
  DNN_Layer *maxpool2 = DNN_MaxPoolLayer(DNN_ACTIVATION_IDENTITY,
                                         imgsize/2,imgsize/2,
                                         16,2,2,
                                         DNN_BACKEND_TINYDNN);
  DNN_Layer *fc = DNN_FullyConnectedLayer(DNN_ACTIVATION_SOFTMAX,
                                          imgsize/4*imgsize/4*16,
                                          nclasses,0,
                                          DNN_BACKEND_TINYDNN);

  DNN_SequentialAdd(net,conv1);
  DNN_SequentialAdd(net,maxpool1);
  DNN_SequentialAdd(net,conv2);
  DNN_SequentialAdd(net,maxpool2);
  DNN_SequentialAdd(net,fc);
#endif

  DNN_Optimizer *optimizer = DNN_AdamOptimizer(0.001,0.9,0.999,0.9,0.999);

  epoch = 0;
  DNN_Train(net, optimizer, DNN_LOSS_CROSSENTROPY_MULTICLASS,
            train_images, train_labels,
            nimages, imgsize*imgsize, 20, 20,
            batchCb, epochCb, &cbdata,
            0, 2, NULL);

  DNN_NetworkDelete(net);

#ifdef FC_EXAMPLE
  DNN_LayerDelete(fc);
#endif

#ifdef CNN_EXAMPLE
  DNN_LayerDelete(conv1);
  DNN_LayerDelete(maxpool1);
  DNN_LayerDelete(fc);
#endif

#ifdef CNN2_EXAMPLE
  DNN_LayerDelete(conv1);
  DNN_LayerDelete(maxpool1);
  DNN_LayerDelete(conv2);
  DNN_LayerDelete(maxpool2);
  DNN_LayerDelete(fc);
#endif

  DNN_OptimizerDelete(optimizer);

  for (int n=0; n<nimages; n++) {
    free(train_images[n]);
  }
  free(train_images);
  free(train_labels);

  for (int n=0; n<n_test_images; n++) {
    free(test_images[n]);
  }
  free(test_images);
  free(test_labels);

  printf("DONE\n");
  return 0;
}

