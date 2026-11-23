#ifndef NN_ENGINE_H
#define NN_ENGINE_H
#include <stdint.h>

enum {
    NN_OP_CONV = 1, NN_OP_DWCONV, NN_OP_FC, NN_OP_ADD, NN_OP_SUB, NN_OP_MUL,
    NN_OP_AVGPOOL, NN_OP_MAXPOOL, NN_OP_MEAN_HW, NN_OP_RESHAPE, NN_OP_SOFTMAX,
    NN_OP_RELU, NN_OP_RELU6, NN_OP_LOGISTIC, NN_OP_TANH, NN_OP_LEAKY_RELU, NN_OP_HARD_SWISH,
    NN_OP_CONCAT, NN_OP_PAD
};

enum { NN_ACT_NONE = 0, NN_ACT_RELU = 1, NN_ACT_RELU_N1_TO_1 = 2, NN_ACT_RELU6 = 3, NN_ACT_TANH = 4 };

enum { NN_LOC_NONE = 0, NN_LOC_FLASH = 1, NN_LOC_QSPI = 2 };

typedef struct {
    uint32_t off;
    uint32_t size;
    uint16_t h, w, c;
} nn_tensor_t;

typedef struct {
    uint8_t op, act, wloc, bloc;
    int16_t in0, in1, out;
    int16_t ins_off, n_ins;
    uint32_t woff, boff;
    int16_t kh, kw, sh, sw, dh, dw, pt, pl;
    int16_t p[6];
    float f0;
} nn_layer_t;

void nn_init(void);
float *nn_input(void);
const float *nn_output(void);
void nn_run(uint64_t (*now)(void), uint64_t *layer_cycles);

#endif
