#include "nn_model.h"

const nn_tensor_t nn_tensors[NN_N_TENSORS] = {
    { 58880u, 3072u, 32, 32, 3, 32, -1 },
    { 47360u, 40960u, 32, 32, 40, 3, 0 },
    { 51200u, 40960u, 32, 32, 40, 3, 1 },
    { 61952u, 40960u, 32, 32, 40, 1, 2 },
    { 40960u, 40960u, 32, 32, 40, 5, 3 },
    { 55040u, 20480u, 16, 16, 80, 3, 4 },
    { 0u, 20480u, 16, 16, 80, 16, 5 },
    { 20480u, 20480u, 16, 16, 80, 16, 6 },
    { 40960u, 20480u, 16, 16, 80, 16, 7 },
    { 0u, 10240u, 8, 8, 160, 8, 8 },
    { 10240u, 10240u, 8, 8, 160, 8, 9 },
    { 0u, 10240u, 8, 8, 160, 8, 10 },
    { 20480u, 10240u, 8, 8, 160, 8, 11 },
    { 0u, 160u, 1, 1, 160, 1, 12 },
    { 0u, 160u, 1, 1, 160, 1, 13 },
    { 160u, 10u, 1, 1, 10, 1, 14 },
    { 0u, 10u, 1, 1, 10, 1, 15 },
};

const int16_t nn_inputs[] = { 0 };

const nn_layer_t nn_layers[NN_N_LAYERS] = {
    { NN_OP_CONV, 1, NN_LOC_FLASH, NN_LOC_FLASH, 0, -1, 1, 0, 0, 0u, 1080u, 3, 3, 1, 1, 1, 1, 1, 1, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 1, NN_LOC_FLASH, NN_LOC_FLASH, 1, -1, 2, 0, 0, 1120u, 15520u, 3, 3, 1, 1, 1, 1, 1, 1, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 0, NN_LOC_FLASH, NN_LOC_FLASH, 2, -1, 3, 0, 0, 15560u, 29960u, 3, 3, 1, 1, 1, 1, 1, 1, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_ADD, 1, NN_LOC_NONE, NN_LOC_NONE, 1, 3, 4, 0, 0, 0u, 0u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 1, NN_LOC_FLASH, NN_LOC_FLASH, 4, -1, 5, 0, 0, 30000u, 58800u, 3, 3, 2, 2, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 0, NN_LOC_FLASH, NN_LOC_FLASH, 5, -1, 6, 0, 0, 58880u, 116480u, 3, 3, 1, 1, 1, 1, 1, 1, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 0, NN_LOC_FLASH, NN_LOC_FLASH, 4, -1, 7, 0, 0, 116560u, 119760u, 1, 1, 2, 2, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 2 },
    { NN_OP_ADD, 1, NN_LOC_NONE, NN_LOC_NONE, 7, 6, 8, 0, 0, 0u, 0u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 1, NN_LOC_FLASH, NN_LOC_FLASH, 8, -1, 9, 0, 0, 119840u, 235040u, 3, 3, 2, 2, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 0, NN_LOC_QSPI, NN_LOC_QSPI, 9, -1, 10, 0, 0, 0u, 230400u, 3, 3, 1, 1, 1, 1, 1, 1, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_CONV, 0, NN_LOC_QSPI, NN_LOC_QSPI, 8, -1, 11, 0, 0, 230560u, 243360u, 1, 1, 2, 2, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_ADD, 1, NN_LOC_NONE, NN_LOC_NONE, 11, 10, 12, 0, 0, 0u, 0u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_AVGPOOL, 0, NN_LOC_NONE, NN_LOC_NONE, 12, -1, 13, 0, 0, 0u, 0u, 8, 8, 8, 8, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_RESHAPE, 0, NN_LOC_NONE, NN_LOC_NONE, 13, -1, 14, 0, 0, 0u, 0u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_FC, 0, NN_LOC_QSPI, NN_LOC_QSPI, 14, -1, 15, 0, 0, 243520u, 245120u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 0.0f, 1 },
    { NN_OP_SOFTMAX, 0, NN_LOC_NONE, NN_LOC_NONE, 15, -1, 16, 0, 0, 0u, 0u, 1, 1, 1, 1, 1, 1, 0, 0, { 0, 0, 0, 0, 0, 0 }, 1.0f, 1 },
};

const nn_group_t nn_groups[NN_N_GROUPS] = {
    { 0, 6, 0 },
    { 7, 7, 1 },
    { 8, 8, 1 },
    { 9, 9, 1 },
    { 10, 10, 1 },
    { 11, 11, 1 },
    { 12, 12, 1 },
    { 13, 13, 1 },
    { 14, 14, 1 },
    { 15, 15, 1 },
};

const char *const nn_layer_names[NN_N_LAYERS] = {
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "AVERAGE_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
};
