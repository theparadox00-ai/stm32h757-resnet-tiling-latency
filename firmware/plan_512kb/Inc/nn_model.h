#ifndef NN_MODEL_H
#define NN_MODEL_H
#include "nn_engine.h"
#define NN_MODEL_NAME "model.tflite"
#define NN_N_LAYERS 16
#define NN_N_TENSORS 17
#define NN_ARENA_FLOATS 122880u
#define NN_IN_T 0
#define NN_OUT_T 16
#define NN_IN_FLOATS 3072u
#define NN_OUT_FLOATS 10u
extern const nn_tensor_t nn_tensors[NN_N_TENSORS];
extern const nn_layer_t nn_layers[NN_N_LAYERS];
extern const int16_t nn_inputs[];
extern const char *const nn_layer_names[NN_N_LAYERS];
#endif
