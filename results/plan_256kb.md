# ResNet float32 — 256 KB plan firmware (row-streaming engine)

Date: 23 Nov 2026. CubeIDE project for this plan: separate project (same setup as S12KB, see h7_board_setup_log.md).
**Rule from Surya: everything identical to the 512 KB firmware (G1) except the planning.** Same clock (Direct SMPS, VOS1, 400 MHz), caches (invalidate then enable), DTCM weight staging + zero buffer, assembly 4×4 kernel, flash/QSPI weight split (QSPI file unchanged), DWT + SysTick 64-bit timing, UART COM5, weight checksums, HardFault reporter, single run caches ON.

## The 256 KB plan (blueprint "256 KB RAM · fastest")
- RAM budget 262,144 B, planned peak 252,928 B (layers #0–#6), no SDRAM, no SD.
- Group 1 = layers #0–#6 **row-streamed**. Held whole: t0 (input, 12 KB), t27 and t28 (outputs of #5 and #6, 80 KB each, cross into #7).
  Ring buffers (rows × 5120 B): t22 = 3, t23 = 3, t24 = 1, t25 = 5, t26 = 3 → 75 KB.
  Rows per step: #0–#5 = 1, #6 = 2.
- Groups 2–10 = layers #7–#15, each whole (same as the 512 KB plan).
- Generated arena = **252,928 B — exactly the planned peak**.

## What changed vs G1 (planning only)
- `nn_tensor_t` gains `cap` (rows held in RAM; = H for whole tensors) and `prod` (producing layer). `nn_layer_t` gains `step` (rows per step). New `nn_group_t {first, last, whole}` table.
- Every row-capable kernel (conv, depthwise, add/sub/mul, pools, activations) takes an output row range [r0, r1) and reads/writes rows through `RP(t, r) = base + (r % cap) · W · C` (ring buffer). For whole tensors cap = H, so the 512 KB path is unchanged arithmetic.
- `nn_run` walks the groups: whole groups run each layer once (as G1); streamed groups use a scheduler that repeatedly lets each layer produce its next `step` rows when (a) the input rows it needs exist and (b) the output ring slot it overwrites is no longer needed by any consumer in the group. No progress → returns an error ("SCHEDULE ERROR" printed).
- In a streamed group the conv kernel is called once per row step, so each layer's weights are re-staged into DTCM for every output row (the 7 layers' weights, 477 KB, cannot sit in DTCM together). This is the real cost of the smaller RAM budget.
- Generator `nn_gen2.py`: reads groups, ring rows and steps from the blueprint, simulates the exact same scheduler on the PC (fails if it would deadlock; retries with 1-row steps), plans the arena with ring sizes and group-span lifetimes, emits the group table. Rejects blueprints with SDRAM/SD spills (not yet supported). `--auto-stream` = test mode that streams every run of row-capable layers with minimal ring buffers.
- main.c: prints the plan (groups, streamed), reports SCHEDULE ERROR if nn_run fails. Everything else identical.

## Verification (PC)
- 256 KB plan vs TFLite: x86 and qemu-arm, rel err 7.9e-7.
- Board LCG input: output **bit-identical** to the 512 KB result (13471 147 11360 159454 601976 4 7624 10479 195436 44, argmax 4).
- 512 KB plan through the new engine: still OK (rel 7.9e-7).
- Test models A–D with --auto-stream (streamed groups, minimal rings): all OK vs TFLite.
- M7 build (arm-none-eabi-gcc -O3, default .ld): no warnings, text 973,088 B, bss 255,016 B.

## Board result
| Plan | Latency (-O3, caches ON, 400 MHz) |
|---|---|
| 512 KB (G1) | 325.338 ms |
| 256 KB | 504.264 ms (201,705,877 cycles), argmax 4, output matches 512 KB |

## Board run log — 23 Nov 2026, 14:31 (project 256Kb, exact PuTTY output)
```
model model.tflite: 16 layers, arena 491520 B in AXI SRAM
CPU 400 MHz; weights: 940800 B internal flash, 980520 B QSPI
internal weights sum 0xA26CBA10 (expect 0xA26CBA10) OK
QSPI weights sum     0xE1805E51 (expect 0xE1805E51) OK
first QSPI words: 0x3E8FD9E4 0x3DF74F9D
step A: caches off at reset, wiping them (invalidate)
step B: caches wiped
step C: enabling caches
step D: caches enabled

=== caches ON ===
run 1: 129998263 cycles = 324.995 ms
min 324.995 ms  avg 324.995 ms  max 324.995 ms
per layer (last run):
  #0   CONV_2D            10.333 ms
  #1   CONV_2D            59.293 ms
  #2   CONV_2D            58.427 ms
  #3   ADD                1.793 ms
  #4   CONV_2D            30.335 ms
  #5   CONV_2D            56.877 ms
  #6   CONV_2D            4.148 ms
  #7   ADD                0.898 ms
  #8   CONV_2D            30.760 ms
  #9   CONV_2D            66.858 ms
  #10  CONV_2D            4.398 ms
  #11  ADD                0.452 ms
  #12  AVERAGE_POOL_2D    0.192 ms
  #13  RESHAPE            0.000 ms
  #14  FULLY_CONNECTED    0.222 ms
  #15  SOFTMAX            0.004 ms
output (x1e6, first 10 of 10): 13471 147 11360 159454 601976 4 7624 10479 195436 44
argmax 4

done
```
**Note: this run was NOT the 256 KB plan.** It reports `arena 491520 B` (the 512 KB plan's arena) and has no `plan: … groups, … row-streamed` line, which the 256 KB main.c prints. So the 256Kb project still contained the 512 KB-plan files (G1 main.c + nn_model.c). It is effectively a second run of the 512 KB plan: 324.995 ms (G1 first run: 325.338 ms; ADD layers slightly faster). The real 256 KB run must show `arena 252928 B` and `plan: 10 groups, 1 row-streamed`.

## Board run log — 23 Nov 2026, 14:32 (project 256Kb, REAL 256 KB plan, exact PuTTY output)
```
model model.tflite: 16 layers, arena 252928 B in AXI SRAM
CPU 400 MHz; weights: 940800 B internal flash, 980520 B QSPI
plan: 10 groups, 1 row-streamed
internal weights sum 0xA26CBA10 (expect 0xA26CBA10) OK
QSPI weights sum     0xE1805E51 (expect 0xE1805E51) OK
first QSPI words: 0x3E8FD9E4 0x3DF74F9D
step A: caches off at reset, wiping them (invalidate)
step B: caches wiped
step C: enabling caches
step D: caches enabled

=== caches ON ===
run 1: 201705877 cycles = 504.264 ms
min 504.264 ms  avg 504.264 ms  max 504.264 ms
per layer (last run):
  #0   CONV_2D            32.125 ms
  #1   CONV_2D            91.159 ms
  #2   CONV_2D            89.977 ms
  #3   ADD                6.115 ms
  #4   CONV_2D            55.157 ms
  #5   CONV_2D            99.971 ms
  #6   CONV_2D            9.449 ms
  #7   ADD                3.077 ms
  #8   CONV_2D            36.727 ms
  #9   CONV_2D            71.663 ms
  #10  CONV_2D            6.524 ms
  #11  ADD                1.497 ms
  #12  AVERAGE_POOL_2D    0.565 ms
  #13  RESHAPE            0.000 ms
  #14  FULLY_CONNECTED    0.246 ms
  #15  SOFTMAX            0.005 ms
output (x1e6, first 10 of 10): 13471 147 11360 159454 601976 4 7624 10479 195436 44
argmax 4

done
```
Result: **256 KB plan = 504.264 ms** (201,705,877 cycles). Output identical to the 512 KB plan (argmax 4), so the row-streamed schedule is correct. 1.55x slower than 512 KB (325.338 ms).

Functional units: identical to the 512 KB G1 build — 400 MHz clock (SMPS, VOS1, PLL1), I+D caches, flash 2 WS, DTCM WBUF/ZBUF weight staging, assembly mk4x4 FPU kernel (32 vfma.f32), QSPI memory-mapped weights (940,800 B flash + 980,520 B QSPI), DWT cycle counter, USART1 115200. main.c differs only by the "plan:" print and SCHEDULE ERROR check. Only change: kernels address tensors through RP() ring-buffer rows + row scheduler (the memory plan).

## Code — nn_engine.h
```c
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
    uint16_t cap;
    int16_t prod;
} nn_tensor_t;

typedef struct {
    uint8_t op, act, wloc, bloc;
    int16_t in0, in1, out;
    int16_t ins_off, n_ins;
    uint32_t woff, boff;
    int16_t kh, kw, sh, sw, dh, dw, pt, pl;
    int16_t p[6];
    float f0;
    uint8_t step;
} nn_layer_t;

typedef struct {
    int16_t first, last;
    uint8_t whole;
} nn_group_t;

void nn_init(void);
float *nn_input(void);
const float *nn_output(void);
int nn_run(uint64_t (*now)(void), uint64_t *layer_cycles);

#endif
```

## Code — nn_engine.c
```c
#include "nn_engine.h"
#include "nn_model.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

__attribute__((aligned(32))) float nn_arena[NN_ARENA_FLOATS];

#define WBUF_FLOATS 24576u
#define ZBUF_FLOATS 8192u
#ifdef MODEL_HOST
static float wbuf_host[WBUF_FLOATS];
static float zbuf_host[ZBUF_FLOATS];
#define WBUF wbuf_host
#define ZBUF zbuf_host
#else
#define WBUF ((float *)0x20000000u)
#define ZBUF ((float *)0x20018000u)
#endif

#define T(i) (&nn_tensors[(i)])
#define TP(i) (nn_arena + nn_tensors[(i)].off)
#define RP(i, r) (nn_arena + nn_tensors[(i)].off + (size_t)((uint32_t)(r) % nn_tensors[(i)].cap) * nn_tensors[(i)].w * nn_tensors[(i)].c)

static const float *wptr(uint8_t loc, uint32_t off)
{
    if (loc == NN_LOC_FLASH) return (const float *)&model_wi[off];
    if (loc == NN_LOC_QSPI) return (const float *)&model_wq[off];
    return 0;
}

void nn_init(void)
{
    memset(ZBUF, 0, ZBUF_FLOATS * sizeof(float));
}

float *nn_input(void) { return TP(NN_IN_T); }
const float *nn_output(void) { return TP(NN_OUT_T); }

static inline float act_apply(float v, int act)
{
    switch (act) {
    case NN_ACT_RELU: return v < 0.0f ? 0.0f : v;
    case NN_ACT_RELU6: return v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
    case NN_ACT_RELU_N1_TO_1: return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    case NN_ACT_TANH: return tanhf(v);
    default: return v;
    }
}

#if defined(MODEL_HOST) && !defined(__ARM_ARCH)
static void mk4x4(float *acc, const float *w, const float *x0, const float *x1,
                  const float *x2, const float *x3, int pairs)
{
    for (int i = 0; i < pairs * 2; i++) {
        for (int j = 0; j < 4; j++) {
            float wv = w[i * 4 + j];
            acc[j * 4 + 0] += x0[i] * wv;
            acc[j * 4 + 1] += x1[i] * wv;
            acc[j * 4 + 2] += x2[i] * wv;
            acc[j * 4 + 3] += x3[i] * wv;
        }
    }
}
#else
static inline void mk4x4(float *acc, const float *w, const float *x0, const float *x1,
                         const float *x2, const float *x3, int pairs)
{
    __asm volatile(
        "vldmia %[acc], {s16-s31}\n"
        "1:\n"
        "vldmia %[w]!, {s0-s7}\n"
        "vldmia %[x0]!, {s8-s9}\n"
        "vldmia %[x1]!, {s10-s11}\n"
        "vldmia %[x2]!, {s12-s13}\n"
        "vldmia %[x3]!, {s14-s15}\n"
        "vfma.f32 s16, s8, s0\n"
        "vfma.f32 s17, s10, s0\n"
        "vfma.f32 s18, s12, s0\n"
        "vfma.f32 s19, s14, s0\n"
        "vfma.f32 s20, s8, s1\n"
        "vfma.f32 s21, s10, s1\n"
        "vfma.f32 s22, s12, s1\n"
        "vfma.f32 s23, s14, s1\n"
        "vfma.f32 s24, s8, s2\n"
        "vfma.f32 s25, s10, s2\n"
        "vfma.f32 s26, s12, s2\n"
        "vfma.f32 s27, s14, s2\n"
        "vfma.f32 s28, s8, s3\n"
        "vfma.f32 s29, s10, s3\n"
        "vfma.f32 s30, s12, s3\n"
        "vfma.f32 s31, s14, s3\n"
        "vfma.f32 s16, s9, s4\n"
        "vfma.f32 s17, s11, s4\n"
        "vfma.f32 s18, s13, s4\n"
        "vfma.f32 s19, s15, s4\n"
        "vfma.f32 s20, s9, s5\n"
        "vfma.f32 s21, s11, s5\n"
        "vfma.f32 s22, s13, s5\n"
        "vfma.f32 s23, s15, s5\n"
        "vfma.f32 s24, s9, s6\n"
        "vfma.f32 s25, s11, s6\n"
        "vfma.f32 s26, s13, s6\n"
        "vfma.f32 s27, s15, s6\n"
        "vfma.f32 s28, s9, s7\n"
        "vfma.f32 s29, s11, s7\n"
        "vfma.f32 s30, s13, s7\n"
        "vfma.f32 s31, s15, s7\n"
        "subs %[n], %[n], #1\n"
        "bne 1b\n"
        "vstmia %[acc], {s16-s31}\n"
        : [w] "+r"(w), [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3), [n] "+r"(pairs)
        : [acc] "r"(acc)
        : "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31", "memory", "cc");
}
#endif

static inline void run4x4(float *acc, const float *w, const float *x0, const float *x1,
                          const float *x2, const float *x3, int len)
{
    int pairs = len >> 1;
    if (pairs > 0) mk4x4(acc, w, x0, x1, x2, x3, pairs);
    if (len & 1) {
        int i = len - 1;
        for (int j = 0; j < 4; j++) {
            float wv = w[i * 4 + j];
            acc[j * 4 + 0] += x0[i] * wv;
            acc[j * 4 + 1] += x1[i] * wv;
            acc[j * 4 + 2] += x2[i] * wv;
            acc[j * 4 + 3] += x3[i] * wv;
        }
    }
}

static void pack_conv(float *dst, const float *w, int OC, int oc0, int n4, int KH, int KW, int C, int c0, int cn)
{
    int per = KH * KW * cn;
    for (int g = 0; g < n4; g += 4) {
        float *d = dst + g * per;
        for (int j = 0; j < 4; j++) {
            int oc = oc0 + g + j;
            if (oc >= OC) {
                for (int t = 0; t < per; t++) d[t * 4 + j] = 0.0f;
                continue;
            }
            const float *src = w + (size_t)oc * KH * KW * C;
            int t = 0;
            for (int k = 0; k < KH * KW; k++) {
                const float *s = src + k * C + c0;
                for (int c = 0; c < cn; c++, t++) d[t * 4 + j] = s[c];
            }
        }
    }
}

static void conv2d(const nn_layer_t *L, int r0, int r1)
{
    const nn_tensor_t *ti = T(L->in0), *to = T(L->out);
    const float *w = wptr(L->wloc, L->woff);
    const float *b = wptr(L->bloc, L->boff);
    int H = ti->h, W = ti->w, C = ti->c, OW = to->w, OC = to->c;
    int KH = L->kh, KW = L->kw, SH = L->sh, SW = L->sw, DH = L->dh, DW = L->dw, PT = L->pt, PL = L->pl;

    int cc = C;
    if (4 * KH * KW * cc > (int)WBUF_FLOATS) cc = (int)WBUF_FLOATS / (4 * KH * KW);
    if (cc > (int)ZBUF_FLOATS) cc = (int)ZBUF_FLOATS;
    if (cc < 1) cc = 1;

    __attribute__((aligned(8))) float acc[16];
    const float *xp[4];

    for (int c0 = 0; c0 < C; c0 += cc) {
        int cn = C - c0 < cc ? C - c0 : cc;
        int first = (c0 == 0), last = (c0 + cn >= C);
        int per = KH * KW * cn;
        int ob = ((int)WBUF_FLOATS / per) & ~3;
        int oc4 = (OC + 3) & ~3;
        if (ob > oc4) ob = oc4;
        int whole_row = (cn == C) && (DW == 1);

        for (int oc0 = 0; oc0 < OC; oc0 += ob) {
            int n = OC - oc0 < ob ? OC - oc0 : ob;
            int n4 = (n + 3) & ~3;
            pack_conv(WBUF, w, OC, oc0, n4, KH, KW, C, c0, cn);

            for (int oy = r0; oy < r1; oy++) {
                int iy0 = oy * SH - PT;
                for (int ox = 0; ox < OW; ox += 4) {
                    int nv = OW - ox < 4 ? OW - ox : 4;
                    int ixs = ox * SW - PL;
                    int interior = whole_row && nv == 4 && ixs >= 0 && (ixs + 3 * SW + KW - 1) < W;
                    float *ybase = RP(L->out, oy) + (size_t)ox * OC;

                    for (int g = 0; g < n4; g += 4) {
                        for (int j = 0; j < 4; j++) {
                            int oc = oc0 + g + j;
                            for (int p = 0; p < 4; p++) {
                                float v = 0.0f;
                                if (oc < OC && p < nv) v = first ? (b ? b[oc] : 0.0f) : ybase[p * OC + oc];
                                acc[j * 4 + p] = v;
                            }
                        }
                        const float *wg = WBUF + g * per;
                        for (int ky = 0; ky < KH; ky++) {
                            int iy = iy0 + ky * DH;
                            if (iy < 0 || iy >= H) continue;
                            const float *row = RP(L->in0, iy);
                            if (interior) {
                                const float *b0 = row + (size_t)ixs * C;
                                run4x4(acc, wg + ky * KW * cn * 4, b0, b0 + SW * C, b0 + 2 * SW * C, b0 + 3 * SW * C, KW * C);
                            } else {
                                for (int kx = 0; kx < KW; kx++) {
                                    for (int p = 0; p < 4; p++) {
                                        int ix = ixs + p * SW + kx * DW;
                                        xp[p] = (p < nv && ix >= 0 && ix < W) ? row + (size_t)ix * C + c0 : ZBUF;
                                    }
                                    run4x4(acc, wg + (ky * KW + kx) * cn * 4, xp[0], xp[1], xp[2], xp[3], cn);
                                }
                            }
                        }
                        for (int j = 0; j < 4; j++) {
                            int oc = oc0 + g + j;
                            if (oc >= OC) break;
                            for (int p = 0; p < nv; p++) {
                                float v = acc[j * 4 + p];
                                ybase[p * OC + oc] = last ? act_apply(v, L->act) : v;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void dwconv2d(const nn_layer_t *L, int r0, int r1)
{
    const nn_tensor_t *ti = T(L->in0), *to = T(L->out);
    const float *w = wptr(L->wloc, L->woff);
    const float *b = wptr(L->bloc, L->boff);
    int H = ti->h, W = ti->w, C = ti->c, OW = to->w, OC = to->c;
    int KH = L->kh, KW = L->kw, SH = L->sh, SW = L->sw, DH = L->dh, DW = L->dw, PT = L->pt, PL = L->pl;
    int M = L->p[0];
    int wn = KH * KW * OC;
    if (wn <= (int)WBUF_FLOATS) {
        memcpy(WBUF, w, (size_t)wn * sizeof(float));
        w = WBUF;
    }
    for (int oy = r0; oy < r1; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            float *yo = RP(L->out, oy) + (size_t)ox * OC;
            for (int o = 0; o < OC; o++) yo[o] = b ? b[o] : 0.0f;
            for (int ky = 0; ky < KH; ky++) {
                int iy = oy * SH - PT + ky * DH;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < KW; kx++) {
                    int ix = ox * SW - PL + kx * DW;
                    if (ix < 0 || ix >= W) continue;
                    const float *xi = RP(L->in0, iy) + (size_t)ix * C;
                    const float *wk = w + (ky * KW + kx) * OC;
                    if (M == 1) {
                        for (int c = 0; c < C; c++) yo[c] += xi[c] * wk[c];
                    } else {
                        for (int c = 0; c < C; c++) {
                            float v = xi[c];
                            for (int m = 0; m < M; m++) yo[c * M + m] += v * wk[c * M + m];
                        }
                    }
                }
            }
            if (L->act) for (int o = 0; o < OC; o++) yo[o] = act_apply(yo[o], L->act);
        }
    }
}

static void fully_connected(const nn_layer_t *L)
{
    const nn_tensor_t *ti = T(L->in0), *to = T(L->out);
    const float *x = TP(L->in0);
    float *y = TP(L->out);
    const float *w = wptr(L->wloc, L->woff);
    const float *b = wptr(L->bloc, L->boff);
    int N = (int)ti->size, O = to->c;
    int o = 0;
    for (; o + 3 < O; o += 4) {
        const float *w0 = w + (size_t)o * N, *w1 = w0 + N, *w2 = w1 + N, *w3 = w2 + N;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int i = 0; i < N; i++) {
            float v = x[i];
            a0 += v * w0[i]; a1 += v * w1[i]; a2 += v * w2[i]; a3 += v * w3[i];
        }
        if (b) { a0 += b[o]; a1 += b[o + 1]; a2 += b[o + 2]; a3 += b[o + 3]; }
        y[o] = act_apply(a0, L->act); y[o + 1] = act_apply(a1, L->act);
        y[o + 2] = act_apply(a2, L->act); y[o + 3] = act_apply(a3, L->act);
    }
    for (; o < O; o++) {
        const float *w0 = w + (size_t)o * N;
        float a = 0.0f;
        for (int i = 0; i < N; i++) a += x[i] * w0[i];
        if (b) a += b[o];
        y[o] = act_apply(a, L->act);
    }
}

static void eltwise(const nn_layer_t *L, int r0, int r1)
{
    const nn_tensor_t *ta = T(L->in0), *tb = T(L->in1), *to = T(L->out);
    int op = L->op, act = L->act;
    int same = (ta->h == to->h && ta->w == to->w && ta->c == to->c && tb->h == to->h && tb->w == to->w && tb->c == to->c);
    for (int h = r0; h < r1; h++) {
        const float *ar = RP(L->in0, ta->h == 1 ? 0 : h);
        const float *br = L->wloc ? wptr(L->wloc, L->woff) + (size_t)(tb->h == 1 ? 0 : h) * tb->w * tb->c : RP(L->in1, tb->h == 1 ? 0 : h);
        float *yr = RP(L->out, h);
        if (same) {
            int n = to->w * to->c;
            if (op == NN_OP_ADD)      for (int i = 0; i < n; i++) yr[i] = act_apply(ar[i] + br[i], act);
            else if (op == NN_OP_SUB) for (int i = 0; i < n; i++) yr[i] = act_apply(ar[i] - br[i], act);
            else                      for (int i = 0; i < n; i++) yr[i] = act_apply(ar[i] * br[i], act);
            continue;
        }
        for (int w = 0; w < to->w; w++) {
            const float *a = ar + (size_t)(ta->w == 1 ? 0 : w) * ta->c;
            const float *b = br + (size_t)(tb->w == 1 ? 0 : w) * tb->c;
            float *y = yr + (size_t)w * to->c;
            int sa = ta->c == 1 ? 0 : 1, sb = tb->c == 1 ? 0 : 1;
            for (int c = 0; c < to->c; c++) {
                float u = a[c * sa], v = b[c * sb], r;
                if (op == NN_OP_ADD) r = u + v; else if (op == NN_OP_SUB) r = u - v; else r = u * v;
                y[c] = act_apply(r, act);
            }
        }
    }
}

static void pool2d(const nn_layer_t *L, int r0, int r1)
{
    const nn_tensor_t *ti = T(L->in0), *to = T(L->out);
    int H = ti->h, W = ti->w, C = ti->c, OW = to->w;
    int is_max = (L->op == NN_OP_MAXPOOL);
    for (int oy = r0; oy < r1; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            float *yo = RP(L->out, oy) + (size_t)ox * C;
            int y0 = oy * L->sh - L->pt, x0 = ox * L->sw - L->pl;
            int ya = y0 < 0 ? 0 : y0, yb = y0 + L->kh > H ? H : y0 + L->kh;
            int xa = x0 < 0 ? 0 : x0, xb = x0 + L->kw > W ? W : x0 + L->kw;
            int cnt = (yb - ya) * (xb - xa);
            for (int c = 0; c < C; c++) yo[c] = is_max ? -INFINITY : 0.0f;
            for (int iy = ya; iy < yb; iy++) {
                for (int ix = xa; ix < xb; ix++) {
                    const float *xi = RP(L->in0, iy) + (size_t)ix * C;
                    if (is_max) { for (int c = 0; c < C; c++) if (xi[c] > yo[c]) yo[c] = xi[c]; }
                    else        { for (int c = 0; c < C; c++) yo[c] += xi[c]; }
                }
            }
            for (int c = 0; c < C; c++) {
                float v = is_max ? yo[c] : (cnt > 0 ? yo[c] / (float)cnt : 0.0f);
                yo[c] = act_apply(v, L->act);
            }
        }
    }
}

static void mean_hw(const nn_layer_t *L)
{
    const nn_tensor_t *ti = T(L->in0);
    const float *x = TP(L->in0);
    float *y = TP(L->out);
    int HW = ti->h * ti->w, C = ti->c;
    for (int c = 0; c < C; c++) y[c] = 0.0f;
    for (int p = 0; p < HW; p++) {
        const float *xi = x + (size_t)p * C;
        for (int c = 0; c < C; c++) y[c] += xi[c];
    }
    for (int c = 0; c < C; c++) y[c] /= (float)HW;
}

static void softmax(const nn_layer_t *L)
{
    const nn_tensor_t *ti = T(L->in0);
    const float *x = TP(L->in0);
    float *y = TP(L->out);
    int C = ti->c, outer = (int)(ti->size / (uint32_t)C);
    float beta = L->f0;
    for (int r = 0; r < outer; r++) {
        const float *xr = x + (size_t)r * C;
        float *yr = y + (size_t)r * C;
        float mx = xr[0];
        for (int i = 1; i < C; i++) if (xr[i] > mx) mx = xr[i];
        float s = 0.0f;
        for (int i = 0; i < C; i++) { yr[i] = expf((xr[i] - mx) * beta); s += yr[i]; }
        for (int i = 0; i < C; i++) yr[i] /= s;
    }
}

static void unary(const nn_layer_t *L, int r0, int r1)
{
    uint32_t n = (uint32_t)T(L->out)->w * T(L->out)->c;
    for (int r = r0; r < r1; r++) {
    const float *x = RP(L->in0, r);
    float *y = RP(L->out, r);
    switch (L->op) {
    case NN_OP_RELU:     for (uint32_t i = 0; i < n; i++) y[i] = x[i] < 0.0f ? 0.0f : x[i]; break;
    case NN_OP_RELU6:    for (uint32_t i = 0; i < n; i++) { float v = x[i]; y[i] = v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v); } break;
    case NN_OP_LOGISTIC: for (uint32_t i = 0; i < n; i++) y[i] = 1.0f / (1.0f + expf(-x[i])); break;
    case NN_OP_TANH:     for (uint32_t i = 0; i < n; i++) y[i] = tanhf(x[i]); break;
    case NN_OP_LEAKY_RELU: for (uint32_t i = 0; i < n; i++) { float v = x[i]; y[i] = v < 0.0f ? v * L->f0 : v; } break;
    case NN_OP_HARD_SWISH: for (uint32_t i = 0; i < n; i++) { float v = x[i], r = v + 3.0f; r = r < 0.0f ? 0.0f : (r > 6.0f ? 6.0f : r); y[i] = v * r / 6.0f; } break;
    default: break;
    }
    }
}

static void concat(const nn_layer_t *L)
{
    const nn_tensor_t *to = T(L->out);
    float *y = TP(L->out);
    int axis = L->p[0];
    int od[4] = {1, to->h, to->w, to->c};
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= od[i];
    for (int i = axis + 1; i < 4; i++) inner *= od[i];
    int ystep = od[axis] * inner;
    int pos = 0;
    for (int k = 0; k < L->n_ins; k++) {
        int ti = nn_inputs[L->ins_off + k];
        const nn_tensor_t *t = T(ti);
        int id[4] = {1, t->h, t->w, t->c};
        int blk = id[axis] * inner;
        const float *x = TP(ti);
        for (int o = 0; o < outer; o++)
            memcpy(y + (size_t)o * ystep + pos, x + (size_t)o * blk, (size_t)blk * sizeof(float));
        pos += blk;
    }
}

static void pad(const nn_layer_t *L)
{
    const nn_tensor_t *ti = T(L->in0), *to = T(L->out);
    const float *x = TP(L->in0);
    float *y = TP(L->out);
    memset(y, 0, (size_t)to->size * sizeof(float));
    int top = L->p[0], left = L->p[2], cf = L->p[4];
    for (int h = 0; h < ti->h; h++)
        for (int w = 0; w < ti->w; w++)
            memcpy(y + ((size_t)(h + top) * to->w + (w + left)) * to->c + cf,
                   x + ((size_t)h * ti->w + w) * ti->c, (size_t)ti->c * sizeof(float));
}

static void reshape(const nn_layer_t *L)
{
    float *y = TP(L->out);
    const float *x = TP(L->in0);
    if (y != x) memmove(y, x, (size_t)T(L->out)->size * sizeof(float));
}

static void exec_rows(const nn_layer_t *L, int r0, int r1)
{
    switch (L->op) {
    case NN_OP_CONV:     conv2d(L, r0, r1); break;
    case NN_OP_DWCONV:   dwconv2d(L, r0, r1); break;
    case NN_OP_FC:       fully_connected(L); break;
    case NN_OP_ADD:
    case NN_OP_SUB:
    case NN_OP_MUL:      eltwise(L, r0, r1); break;
    case NN_OP_AVGPOOL:
    case NN_OP_MAXPOOL:  pool2d(L, r0, r1); break;
    case NN_OP_MEAN_HW:  mean_hw(L); break;
    case NN_OP_RESHAPE:  reshape(L); break;
    case NN_OP_SOFTMAX:  softmax(L); break;
    case NN_OP_CONCAT:   concat(L); break;
    case NN_OP_PAD:      pad(L); break;
    default:             unary(L, r0, r1); break;
    }
}

static int windowed(const nn_layer_t *L)
{
    return L->op == NN_OP_CONV || L->op == NN_OP_DWCONV || L->op == NN_OP_AVGPOOL || L->op == NN_OP_MAXPOOL;
}

static int need_hi(const nn_layer_t *L, int t, int rl)
{
    int H = T(t)->h;
    if (H == 1) return 0;
    if (!windowed(L)) return rl;
    int hi = rl * L->sh - L->pt + (L->kh - 1) * L->dh;
    return hi > H - 1 ? H - 1 : hi;
}

static int need_lo(const nn_layer_t *L, int t, int r)
{
    if (T(t)->h == 1) return 0;
    if (!windowed(L)) return r;
    int lo = r * L->sh - L->pt;
    return lo < 0 ? 0 : lo;
}

static int16_t nxt[NN_N_LAYERS];

static int avail_rows(int t, int first, int last)
{
    int p = nn_tensors[t].prod;
    if (p >= first && p <= last) return nxt[p];
    return T(t)->h;
}

static int input_ok(const nn_layer_t *L, int t, int rl, int first, int last)
{
    if (t < 0) return 1;
    return need_hi(L, t, rl) < avail_rows(t, first, last);
}

static int space_ok(int li, int rl, int first, int last)
{
    const nn_layer_t *L = &nn_layers[li];
    const nn_tensor_t *to = T(L->out);
    int evict = rl - (int)to->cap;
    if (evict < 0) return 1;
    for (int c = first; c <= last; c++) {
        const nn_layer_t *Lc = &nn_layers[c];
        if (nxt[c] >= T(Lc->out)->h) continue;
        if (Lc->in0 == L->out && evict >= need_lo(Lc, L->out, nxt[c])) return 0;
        if (!Lc->wloc && Lc->in1 == L->out && evict >= need_lo(Lc, L->out, nxt[c])) return 0;
    }
    return 1;
}

static int run_stream(int first, int last, uint64_t (*now)(void), uint64_t *lc)
{
    for (int i = first; i <= last; i++) nxt[i] = 0;
    for (;;) {
        int progress = 0, done = 1;
        for (int i = first; i <= last; i++) {
            const nn_layer_t *L = &nn_layers[i];
            int OH = T(L->out)->h;
            while (nxt[i] < OH) {
                int r = nxt[i];
                int n = L->step ? L->step : 1;
                if (n > OH - r) n = OH - r;
                int rl = r + n - 1;
                if (!input_ok(L, L->in0, rl, first, last)) break;
                if (!L->wloc && !input_ok(L, L->in1, rl, first, last)) break;
                if (!space_ok(i, rl, first, last)) break;
                uint64_t t = now();
                exec_rows(L, r, r + n);
                lc[i] += now() - t;
                nxt[i] = (int16_t)(r + n);
                progress = 1;
            }
            if (nxt[i] < OH) done = 0;
        }
        if (done) return 0;
        if (!progress) return -1;
    }
}

int nn_run(uint64_t (*now)(void), uint64_t *lc)
{
    for (int i = 0; i < NN_N_LAYERS; i++) lc[i] = 0;
    for (int g = 0; g < NN_N_GROUPS; g++) {
        const nn_group_t *G = &nn_groups[g];
        if (G->whole) {
            for (int i = G->first; i <= G->last; i++) {
                const nn_layer_t *L = &nn_layers[i];
                uint64_t t = now();
                exec_rows(L, 0, T(L->out)->h);
                lc[i] = now() - t;
            }
        } else if (run_stream(G->first, G->last, now, lc) != 0) {
            return -1 - g;
        }
    }
    return 0;
}
```

## Code — main.c
```c
#include "stm32h7xx.h"
#include "nn_engine.h"
#include "nn_model.h"
#include "model_weights.h"
#include <stdio.h>

#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DWT_LAR    (*(volatile uint32_t *)0xE0001FB0u)

#define CPU_HZ 400000000u
#define CYC_PER_US (CPU_HZ / 1000000u)

static void uart_init(uint32_t bus_hz)
{
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
    (void)RCC->APB2ENR;

    GPIOB->MODER &= ~((3u << (14u * 2u)) | (3u << (15u * 2u)));
    GPIOB->MODER |=  ((2u << (14u * 2u)) | (2u << (15u * 2u)));
    GPIOB->AFR[1] &= ~((0xFu << ((14u - 8u) * 4u)) | (0xFu << ((15u - 8u) * 4u)));
    GPIOB->AFR[1] |=  ((4u << ((14u - 8u) * 4u)) | (4u << ((15u - 8u) * 4u)));

    USART1->CR1 = 0u;
    USART1->BRR = bus_hz / 115200u;
    USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while ((USART1->ISR & USART_ISR_TXE_TXFNF) == 0u) {}
    USART1->TDR = (uint8_t)c;
}

static void uart_wait_done(void)
{
    while ((USART1->ISR & USART_ISR_TC) == 0u) {}
}

int __io_putchar(int ch)
{
    if (ch == '\n') uart_putc('\r');
    uart_putc((char)ch);
    return ch;
}

static void raw_puts(const char *p)
{
    while (*p) { if (*p == '\n') uart_putc('\r'); uart_putc(*p++); }
}

static void raw_hex(uint32_t v)
{
    static const char h[] = "0123456789ABCDEF";
    uart_putc('0'); uart_putc('x');
    for (int i = 28; i >= 0; i -= 4) uart_putc(h[(v >> i) & 0xFu]);
}

void fault_report(uint32_t *sp)
{
    raw_puts("\n\n*** HARD FAULT ***\nPC   = "); raw_hex(sp[6]);
    raw_puts("\nLR   = "); raw_hex(sp[5]);
    raw_puts("\nCFSR = "); raw_hex(SCB->CFSR);
    raw_puts("\nHFSR = "); raw_hex(SCB->HFSR);
    raw_puts("\nMMFAR= "); raw_hex(SCB->MMFAR);
    raw_puts("\nBFAR = "); raw_hex(SCB->BFAR);
    raw_puts("\nSP   = "); raw_hex((uint32_t)sp);
    raw_puts("\n");
    while (1) {}
}

__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4      \n"
        "ite eq          \n"
        "mrseq r0, msp   \n"
        "mrsne r0, psp   \n"
        "b fault_report  \n");
}

__attribute__((naked)) void MemManage_Handler(void)  { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void BusFault_Handler(void)   { __asm volatile("b HardFault_Handler"); }
__attribute__((naked)) void UsageFault_Handler(void) { __asm volatile("b HardFault_Handler"); }

static uint64_t now(void);

static uint64_t now_verbose(void)
{
    static uint32_t calls;
    calls++;
    if ((calls & 1u) == 0u) uart_putc('.');
    return now();
}

static void clock_init(void)
{
    PWR->CR3 = (PWR->CR3 & ~(PWR_CR3_LDOEN | PWR_CR3_BYPASS)) | PWR_CR3_SMPSEN;
    while ((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) == 0u) {}

    PWR->D3CR = (PWR->D3CR & ~PWR_D3CR_VOS) | PWR_D3CR_VOS;
    while ((PWR->D3CR & PWR_D3CR_VOSRDY) == 0u) {}

    FLASH->ACR = FLASH_ACR_LATENCY_2WS | FLASH_ACR_WRHIGHFREQ_1;
    while ((FLASH->ACR & FLASH_ACR_LATENCY) != FLASH_ACR_LATENCY_2WS) {}

    RCC->CR |= RCC_CR_HSEON;
    while ((RCC->CR & RCC_CR_HSERDY) == 0u) {}

    RCC->PLLCKSELR = RCC_PLLCKSELR_PLLSRC_HSE | (5u << RCC_PLLCKSELR_DIVM1_Pos);
    RCC->PLLCFGR = (RCC->PLLCFGR & ~(RCC_PLLCFGR_PLL1RGE | RCC_PLLCFGR_PLL1VCOSEL | RCC_PLLCFGR_PLL1FRACEN))
                 | (2u << RCC_PLLCFGR_PLL1RGE_Pos)
                 | RCC_PLLCFGR_DIVP1EN | RCC_PLLCFGR_DIVQ1EN | RCC_PLLCFGR_DIVR1EN;
    RCC->PLL1DIVR = ((160u - 1u) << RCC_PLL1DIVR_N1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_P1_Pos)
                  | ((4u - 1u)   << RCC_PLL1DIVR_Q1_Pos)
                  | ((2u - 1u)   << RCC_PLL1DIVR_R1_Pos);
    RCC->CR |= RCC_CR_PLL1ON;
    while ((RCC->CR & RCC_CR_PLL1RDY) == 0u) {}

    uart_wait_done();

    RCC->D1CFGR = RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2;
    RCC->D2CFGR = RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC->D3CFGR = RCC_D3CFGR_D3PPRE_DIV2;
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL1;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL1) {}

    uart_init(100000000u);
}

static volatile uint32_t cyc_hi;
static volatile uint32_t cyc_last;

static uint64_t now(void)
{
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    uint32_t c = DWT_CYCCNT;
    if (c < cyc_last) cyc_hi++;
    cyc_last = c;
    uint64_t r = ((uint64_t)cyc_hi << 32) | c;
    __set_PRIMASK(pm);
    return r;
}

void SysTick_Handler(void)
{
    (void)now();
}

static void dwt_init(void)
{
    DEMCR |= (1u << 24);
    DWT_LAR = 0xC5ACCE55u;
    DWT_CYCCNT = 0u;
    cyc_hi = 0u;
    cyc_last = 0u;
    DWT_CTRL |= 1u;

    SysTick->LOAD = 0x00FFFFFFu;
    SysTick->VAL = 0u;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
}

static void pin_af(GPIO_TypeDef *port, uint32_t pin, uint32_t af, uint32_t pullup)
{
    port->MODER   = (port->MODER   & ~(3u << (pin * 2u))) | (2u << (pin * 2u));
    port->OSPEEDR |= (3u << (pin * 2u));
    port->OTYPER  &= ~(1u << pin);
    port->PUPDR   = (port->PUPDR   & ~(3u << (pin * 2u))) | ((pullup ? 1u : 0u) << (pin * 2u));
    if (pin < 8u) port->AFR[0] = (port->AFR[0] & ~(0xFu << (pin * 4u))) | (af << (pin * 4u));
    else          port->AFR[1] = (port->AFR[1] & ~(0xFu << ((pin - 8u) * 4u))) | (af << ((pin - 8u) * 4u));
}

static void qspi_wait_idle(void)
{
    while (QUADSPI->SR & QUADSPI_SR_BUSY) {}
}

static void qspi_cmd(uint32_t instr, uint32_t imode)
{
    qspi_wait_idle();
    QUADSPI->FCR = QUADSPI_FCR_CTCF;
    QUADSPI->CCR = (imode << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos);
    while ((QUADSPI->SR & QUADSPI_SR_TCF) == 0u) {}
    QUADSPI->FCR = QUADSPI_FCR_CTCF;
    qspi_wait_idle();
}

static void delay_cycles(uint32_t n)
{
    uint32_t s = DWT_CYCCNT;
    while ((DWT_CYCCNT - s) < n) {}
}

static void qspi_init(void)
{
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN | RCC_AHB4ENR_GPIOFEN |
                    RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN;
    RCC->AHB3ENR |= RCC_AHB3ENR_QSPIEN;
    (void)RCC->AHB3ENR;
    RCC->AHB3RSTR |= RCC_AHB3RSTR_QSPIRST;
    RCC->AHB3RSTR &= ~RCC_AHB3RSTR_QSPIRST;

    pin_af(GPIOB, 2u,  9u,  0u);
    pin_af(GPIOG, 6u,  10u, 1u);
    pin_af(GPIOC, 11u, 9u,  1u);
    pin_af(GPIOF, 8u,  10u, 0u);
    pin_af(GPIOH, 2u,  9u,  0u);
    pin_af(GPIOF, 9u,  10u, 0u);
    pin_af(GPIOH, 3u,  9u,  0u);
    pin_af(GPIOF, 7u,  9u,  0u);
    pin_af(GPIOG, 9u,  9u,  0u);
    pin_af(GPIOF, 6u,  9u,  0u);
    pin_af(GPIOG, 14u, 9u,  0u);

    QUADSPI->CR = 0u;
    QUADSPI->DCR = (26u << QUADSPI_DCR_FSIZE_Pos) | (3u << QUADSPI_DCR_CSHT_Pos);
    QUADSPI->CR = (2u << QUADSPI_CR_PRESCALER_Pos) | (3u << QUADSPI_CR_FTHRES_Pos) |
                  QUADSPI_CR_SSHIFT | QUADSPI_CR_DFM | QUADSPI_CR_EN;

    qspi_cmd(0x66u, 3u);
    qspi_cmd(0x99u, 3u);
    delay_cycles(400000u);
    qspi_cmd(0x66u, 1u);
    qspi_cmd(0x99u, 1u);
    delay_cycles(400000u);

    qspi_wait_idle();
    QUADSPI->CCR = (3u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (0x6Bu << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (1u << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (3u << QUADSPI_CCR_DMODE_Pos) | (8u << QUADSPI_CCR_DCYC_Pos);
}

static void qspi_abort(void)
{
    QUADSPI->CR |= QUADSPI_CR_ABORT;
    while (QUADSPI->CR & QUADSPI_CR_ABORT) {}
    qspi_wait_idle();
    QUADSPI->FCR = QUADSPI_FCR_CTEF | QUADSPI_FCR_CTCF | QUADSPI_FCR_CSMF | QUADSPI_FCR_CTOF;
}

static void qspi_read_ind(uint32_t instr, uint32_t admode, uint32_t dmode, uint32_t dcyc,
                          uint32_t addr, uint8_t *buf, uint32_t n)
{
    qspi_abort();
    QUADSPI->DLR = n - 1u;
    QUADSPI->CCR = (1u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (admode << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (dmode << QUADSPI_CCR_DMODE_Pos) | (dcyc << QUADSPI_CCR_DCYC_Pos);
    if (admode) QUADSPI->AR = addr;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t t = 0;
        while ((QUADSPI->SR & (QUADSPI_SR_FTF | QUADSPI_SR_TCF)) == 0u) { if (++t > 4000000u) { printf(" [timeout]"); return; } }
        buf[i] = *(volatile uint8_t *)&QUADSPI->DR;
    }
    qspi_abort();
}

static void print_bytes(const char *name, const uint8_t *b, uint32_t n)
{
    printf("%-28s:", name);
    for (uint32_t i = 0; i < n; i++) printf(" %02X", b[i]);
    printf("\n");
}

static void qspi_mm(uint32_t instr, uint32_t admode, uint32_t dmode, uint32_t dcyc)
{
    qspi_abort();
    QUADSPI->CCR = (3u << QUADSPI_CCR_FMODE_Pos) |
                   (1u << QUADSPI_CCR_IMODE_Pos) | (instr << QUADSPI_CCR_INSTRUCTION_Pos) |
                   (admode << QUADSPI_CCR_ADMODE_Pos) | (2u << QUADSPI_CCR_ADSIZE_Pos) |
                   (dmode << QUADSPI_CCR_DMODE_Pos) | (dcyc << QUADSPI_CCR_DCYC_Pos);
    __DSB();
    printf("mm 0x%02lX admode %lu dmode %lu dummy %2lu : 0x%08lX 0x%08lX\n",
           (unsigned long)instr, (unsigned long)admode, (unsigned long)dmode, (unsigned long)dcyc,
           (unsigned long)model_wq[0], (unsigned long)model_wq[1]);
}

static void qspi_diag(void)
{
    uint8_t b[16];
    printf("\n--- QSPI diagnostic ---\n");
    printf("expected bytes at 0x90000000   : E4 D9 8F 3E 9D 4F F7 3D\n");
    printf("expected words                 : 0x3E8FD9E4 0x3DF74F9D\n");
    printf("QSPI CR 0x%08lX DCR 0x%08lX SR 0x%08lX\n", (unsigned long)QUADSPI->CR, (unsigned long)QUADSPI->DCR, (unsigned long)QUADSPI->SR);

    qspi_read_ind(0x9Fu, 0u, 1u, 0u, 0u, b, 8u);
    print_bytes("JEDEC ID 0x9F (2 chips)", b, 8u);

    qspi_read_ind(0x03u, 1u, 1u, 0u, 0u, b, 16u);
    print_bytes("read 0x03 1-1-1 addr 0", b, 16u);

    qspi_read_ind(0x0Bu, 1u, 1u, 8u, 0u, b, 16u);
    print_bytes("read 0x0B 1-1-1 dummy 8", b, 16u);

    qspi_read_ind(0x6Bu, 1u, 3u, 8u, 0u, b, 16u);
    print_bytes("read 0x6B 1-1-4 dummy 8", b, 16u);

    qspi_read_ind(0x6Cu, 1u, 3u, 8u, 0u, b, 16u);
    print_bytes("read 0x6C 4-byte cmd dummy 8", b, 16u);

    qspi_mm(0x6Bu, 1u, 3u, 6u);
    qspi_mm(0x6Bu, 1u, 3u, 8u);
    qspi_mm(0x6Bu, 1u, 3u, 10u);
    qspi_mm(0xEBu, 3u, 3u, 10u);
    qspi_mm(0x03u, 1u, 1u, 0u);
    printf("--- end diagnostic ---\n");
}

static int check_weights(void)
{
    uint32_t si = 0u, sq = 0u;
    for (uint32_t i = 0; i < MODEL_WI_USED; i++) si += model_wi[i];
    for (uint32_t i = 0; i < MODEL_WQ_COUNT; i++) sq += model_wq[i];
    printf("internal weights sum 0x%08lX (expect 0x%08lX) %s\n", (unsigned long)si, (unsigned long)MODEL_WI_SUM, si == MODEL_WI_SUM ? "OK" : "FAIL");
    printf("QSPI weights sum     0x%08lX (expect 0x%08lX) %s\n", (unsigned long)sq, (unsigned long)MODEL_WQ_SUM, sq == MODEL_WQ_SUM ? "OK" : "FAIL");
    if (MODEL_WQ_COUNT > 1u) printf("first QSPI words: 0x%08lX 0x%08lX\n", (unsigned long)model_wq[0], (unsigned long)model_wq[1]);
    return (si == MODEL_WI_SUM) && (sq == MODEL_WQ_SUM);
}

static uint32_t lcg_state;

static float lcg(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}

static void load_input(void)
{
    float *in = nn_input();
    lcg_state = 12345u;
    for (uint32_t i = 0; i < NN_IN_FLOATS; i++) in[i] = lcg();
    if (SCB->CCR & SCB_CCR_DC_Msk) SCB_CleanDCache();
}

static void print_u64(uint64_t v)
{
    if (v >= 1000000000ull) printf("%lu%09lu", (unsigned long)(v / 1000000000ull), (unsigned long)(v % 1000000000ull));
    else printf("%lu", (unsigned long)v);
}

static void print_ms(uint64_t cyc)
{
    uint64_t us = cyc / CYC_PER_US;
    printf("%lu.%03lu ms", (unsigned long)(us / 1000u), (unsigned long)(us % 1000u));
}

static uint64_t run_once(uint64_t *lc)
{
    load_input();
    if (nn_run(now, lc) != 0) printf("SCHEDULE ERROR\n");
    uint64_t total = 0;
    for (uint32_t i = 0; i < NN_N_LAYERS; i++) total += lc[i];
    return total;
}

static void print_output(void)
{
    const float *o = nn_output();
    uint32_t n = NN_OUT_FLOATS;
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (o[i] > o[best]) best = i;
    printf("output (x1e6, first %lu of %lu):", (unsigned long)(n < 16u ? n : 16u), (unsigned long)n);
    for (uint32_t i = 0; i < n && i < 16u; i++) printf(" %ld", (long)(o[i] * 1000000.0f));
    printf("\nargmax %lu\n", (unsigned long)best);
}

static void bench(const char *name, uint32_t warmup, uint32_t runs)
{
    static uint64_t lc[NN_N_LAYERS];
    uint64_t sum = 0, mn = ~0ull, mx = 0;

    raw_puts("\n=== "); raw_puts(name); raw_puts(" ===\n");
    for (uint32_t i = 0; i < warmup; i++) {
        load_input();
        if (nn_run(now_verbose, lc) != 0) printf("SCHEDULE ERROR\n");
    }

    for (uint32_t r = 0; r < runs; r++) {
        uint64_t c = run_once(lc);
        sum += c;
        if (c < mn) mn = c;
        if (c > mx) mx = c;
        printf("run %lu: ", (unsigned long)(r + 1u));
        print_u64(c);
        printf(" cycles = ");
        print_ms(c);
        printf("\n");
    }

    printf("min "); print_ms(mn);
    printf("  avg "); print_ms(sum / runs);
    printf("  max "); print_ms(mx);
    printf("\nper layer (last run):\n");
    for (uint32_t i = 0; i < NN_N_LAYERS; i++) {
        printf("  #%-3lu %-18s ", (unsigned long)i, nn_layer_names[i]);
        print_ms(lc[i]);
        printf("\n");
    }
    print_output();
}

__attribute__((noinline)) static void app(void)
{
    uart_init(64000000u);
    clock_init();
    dwt_init();

    printf("\n\nmodel %s: %d layers, arena %lu B in AXI SRAM\n", NN_MODEL_NAME, NN_N_LAYERS, (unsigned long)(NN_ARENA_FLOATS * 4u));
    printf("CPU %lu MHz; weights: %lu B internal flash, %lu B QSPI\n", (unsigned long)(CPU_HZ / 1000000u),
           (unsigned long)(MODEL_WI_USED * 4u), (unsigned long)(MODEL_WQ_COUNT * 4u));
    {
        int streamed = 0;
        for (int g = 0; g < NN_N_GROUPS; g++) if (!nn_groups[g].whole) streamed++;
        printf("plan: %d groups, %d row-streamed\n", NN_N_GROUPS, streamed);
    }

    nn_init();
    if (MODEL_WQ_COUNT > 0u) qspi_init();
    if (!check_weights()) {
        printf("weights check FAILED\n");
        if (MODEL_WQ_COUNT > 0u) qspi_diag();
        while (1) {}
    }

    raw_puts("step A: caches off at reset, wiping them (invalidate)\n");
    SCB_InvalidateDCache();
    SCB_InvalidateICache();
    raw_puts("step B: caches wiped\n");

    raw_puts("step C: enabling caches\n");
    SCB_EnableICache();
    SCB_EnableDCache();
    raw_puts("step D: caches enabled\n");
    bench("caches ON", 0u, 1u);

    printf("\ndone\n");
    while (1) {}
}

int main(void)
{
    SCB->CPACR |= (0xFu << 20);
    __DSB();
    __ISB();
    app();
    while (1) {}
}
```

## Code — nn_model.c (generated for the 256 KB plan)
```c
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
```

## Code — nn_gen2.py
```python
#!/usr/bin/env python3
import sys, os, re, json, html, argparse
import numpy as np
import tflite

OPS = {
    'CONV_2D': 'NN_OP_CONV', 'DEPTHWISE_CONV_2D': 'NN_OP_DWCONV', 'FULLY_CONNECTED': 'NN_OP_FC',
    'ADD': 'NN_OP_ADD', 'SUB': 'NN_OP_SUB', 'MUL': 'NN_OP_MUL',
    'AVERAGE_POOL_2D': 'NN_OP_AVGPOOL', 'MAX_POOL_2D': 'NN_OP_MAXPOOL', 'MEAN': 'NN_OP_MEAN_HW',
    'RESHAPE': 'NN_OP_RESHAPE', 'SQUEEZE': 'NN_OP_RESHAPE', 'EXPAND_DIMS': 'NN_OP_RESHAPE',
    'SOFTMAX': 'NN_OP_SOFTMAX', 'RELU': 'NN_OP_RELU', 'RELU6': 'NN_OP_RELU6', 'LOGISTIC': 'NN_OP_LOGISTIC',
    'TANH': 'NN_OP_TANH', 'LEAKY_RELU': 'NN_OP_LEAKY_RELU', 'HARD_SWISH': 'NN_OP_HARD_SWISH',
    'CONCATENATION': 'NN_OP_CONCAT', 'PAD': 'NN_OP_PAD', 'PADV2': 'NN_OP_PAD',
}
ALIAS_OPS = {'RESHAPE', 'SQUEEZE', 'EXPAND_DIMS'}
OPNAME = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith('_')}
TTYPE = {v: k for k, v in tflite.TensorType.__dict__.items() if not k.startswith('_')}


def die(msg):
    print('ERROR: ' + msg)
    sys.exit(1)


def opt(op, cls):
    o = op.BuiltinOptions()
    if o is None:
        return None
    c = cls()
    c.Init(o.Bytes, o.Pos)
    return c


def same_pad(inp, out, k, s, d):
    eff = (k - 1) * d + 1
    tot = max((out - 1) * s + eff - inp, 0)
    return tot // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tflite')
    ap.add_argument('--out', default='gen_out')
    ap.add_argument('--flash-kb', type=float, default=960.0, help='internal flash budget for weights (KB)')
    ap.add_argument('--blueprint', default=None)
    ap.add_argument('--auto-stream', action='store_true', help='test mode: stream every run of row-capable layers with minimal ring buffers')
    a = ap.parse_args()

    buf = open(a.tflite, 'rb').read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    if m.SubgraphsLength() != 1:
        die('only single-subgraph models are supported')
    g = m.Subgraphs(0)

    def tensor(i):
        return g.Tensors(i)

    def shape(i):
        return [int(v) for v in tensor(i).ShapeAsNumpy()] if tensor(i).ShapeLength() else []

    def const_data(i):
        if i < 0:
            return None
        b = m.Buffers(tensor(i).Buffer())
        if b is None or b.DataLength() == 0:
            return None
        return b.DataAsNumpy().tobytes()

    def dims4(i):
        s = shape(i)
        if len(s) > 4:
            die(f'tensor {i} rank {len(s)} > 4')
        s = [1] * (4 - len(s)) + s
        if s[0] != 1:
            die(f'tensor {i} batch {s[0]} != 1')
        return s

    for i in range(g.TensorsLength()):
        t = tensor(i)
        if TTYPE[t.Type()] not in ('FLOAT32', 'INT32'):
            die(f'tensor {i} type {TTYPE[t.Type()]}: only float32 models are supported')

    ops = []
    unsupported = set()
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        code = m.OperatorCodes(op.OpcodeIndex())
        name = OPNAME[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        if name not in OPS:
            unsupported.add(name)
        ops.append((name, op, [int(v) for v in op.InputsAsNumpy()], [int(v) for v in op.OutputsAsNumpy()]))
    if unsupported:
        die('unsupported ops: ' + ', '.join(sorted(unsupported)))

    in_t = int(g.InputsAsNumpy()[0])
    out_t = int(g.OutputsAsNumpy()[0])
    if g.InputsLength() != 1 or g.OutputsLength() != 1:
        die('only models with one input and one output are supported')

    weights = []
    wmap = {}

    def add_weight(ti):
        if ti < 0:
            return None
        if ti in wmap:
            return wmap[ti]
        d = const_data(ti)
        if d is None:
            die(f'tensor {ti} expected constant')
        arr = np.frombuffer(d, dtype=np.float32)
        wmap[ti] = len(weights)
        weights.append((ti, arr))
        return wmap[ti]

    layers = []
    concat_inputs = []
    act_tensors = set([in_t, out_t])
    alias = {}

    for idx, (name, op, ins, outs) in enumerate(ops):
        L = dict(name=name, op=OPS[name], act=0, in0=-1, in1=-1, out=outs[0], w=None, b=None, kh=1, kw=1,
                 sh=1, sw=1, dh=1, dw=1, pt=0, pl=0, p=[0] * 6, f0=0.0, ins_off=0, n_ins=0)
        o4 = dims4(outs[0])
        act_tensors.add(outs[0])
        if name in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            x, w = ins[0], ins[1]
            b = ins[2] if len(ins) > 2 else -1
            i4 = dims4(x)
            ws = shape(w)
            if name == 'CONV_2D':
                c = opt(op, tflite.Conv2DOptions)
                KH, KW = ws[1], ws[2]
                if ws[3] != i4[3]:
                    die(f'layer {idx}: grouped conv not supported')
            else:
                c = opt(op, tflite.DepthwiseConv2DOptions)
                KH, KW = ws[1], ws[2]
                M = o4[3] // i4[3]
                if M * i4[3] != o4[3]:
                    die(f'layer {idx}: bad depth multiplier')
                L['p'][0] = M
            SH, SW, DH, DW = c.StrideH(), c.StrideW(), c.DilationHFactor(), c.DilationWFactor()
            L.update(act=c.FusedActivationFunction(), in0=x, kh=KH, kw=KW, sh=SH, sw=SW, dh=DH, dw=DW)
            if c.Padding() == 0:
                L['pt'] = same_pad(i4[1], o4[1], KH, SH, DH)
                L['pl'] = same_pad(i4[2], o4[2], KW, SW, DW)
            L['w'] = add_weight(w)
            L['b'] = add_weight(b) if b >= 0 and const_data(b) is not None else None
            act_tensors.add(x)
        elif name == 'FULLY_CONNECTED':
            c = opt(op, tflite.FullyConnectedOptions)
            if c is not None and c.WeightsFormat() != 0:
                die(f'layer {idx}: shuffled FC weights not supported')
            x, w = ins[0], ins[1]
            b = ins[2] if len(ins) > 2 else -1
            ws = shape(w)
            n_in = int(np.prod(shape(x)))
            if ws[1] != n_in or ws[0] != o4[3] or int(np.prod(o4)) != ws[0]:
                die(f'layer {idx}: FC with batch > 1 not supported')
            L.update(act=c.FusedActivationFunction() if c else 0, in0=x)
            L['w'] = add_weight(w)
            L['b'] = add_weight(b) if b >= 0 and const_data(b) is not None else None
            act_tensors.add(x)
        elif name in ('ADD', 'SUB', 'MUL'):
            cls = {'ADD': tflite.AddOptions, 'SUB': tflite.SubOptions, 'MUL': tflite.MulOptions}[name]
            c = opt(op, cls)
            x0, x1 = ins[0], ins[1]
            k0, k1 = const_data(x0) is not None, const_data(x1) is not None
            if k0 and not k1:
                if name == 'SUB':
                    die(f'layer {idx}: SUB with constant first operand not supported')
                x0, x1 = x1, x0
                k0, k1 = k1, k0
            if k0:
                die(f'layer {idx}: both operands constant')
            for t in (x0, x1):
                d = dims4(t)
                for ax in range(1, 4):
                    if d[ax] != o4[ax] and d[ax] != 1:
                        die(f'layer {idx}: unsupported broadcast')
            L.update(act=c.FusedActivationFunction() if c else 0, in0=x0, in1=x1)
            act_tensors.add(x0)
            if k1:
                L['w'] = add_weight(x1)
            else:
                act_tensors.add(x1)
        elif name in ('AVERAGE_POOL_2D', 'MAX_POOL_2D'):
            c = opt(op, tflite.Pool2DOptions)
            x = ins[0]
            i4 = dims4(x)
            L.update(act=c.FusedActivationFunction(), in0=x, kh=c.FilterHeight(), kw=c.FilterWidth(),
                     sh=c.StrideH(), sw=c.StrideW())
            if c.Padding() == 0:
                L['pt'] = same_pad(i4[1], o4[1], c.FilterHeight(), c.StrideH(), 1)
                L['pl'] = same_pad(i4[2], o4[2], c.FilterWidth(), c.StrideW(), 1)
            act_tensors.add(x)
        elif name == 'MEAN':
            x = ins[0]
            r = len(shape(x))
            axes = sorted(int(v) % r for v in np.frombuffer(const_data(ins[1]), dtype=np.int32))
            if r != 4 or axes != [1, 2]:
                die(f'layer {idx}: MEAN only over H,W of a 4D tensor is supported (axes {axes})')
            L.update(in0=x)
            act_tensors.add(x)
        elif name in ALIAS_OPS:
            L.update(in0=ins[0])
            act_tensors.add(ins[0])
            alias[outs[0]] = ins[0]
        elif name == 'SOFTMAX':
            c = opt(op, tflite.SoftmaxOptions)
            L.update(in0=ins[0], f0=float(c.Beta()))
            act_tensors.add(ins[0])
        elif name == 'LEAKY_RELU':
            c = opt(op, tflite.LeakyReluOptions)
            L.update(in0=ins[0], f0=float(c.Alpha()))
            act_tensors.add(ins[0])
        elif name in ('RELU', 'RELU6', 'LOGISTIC', 'TANH', 'HARD_SWISH'):
            L.update(in0=ins[0])
            act_tensors.add(ins[0])
        elif name == 'CONCATENATION':
            c = opt(op, tflite.ConcatenationOptions)
            if c.FusedActivationFunction() != 0:
                die(f'layer {idx}: fused activation on CONCATENATION not supported')
            r = len(shape(outs[0]))
            axis = c.Axis() % r + (4 - r)
            for t in ins:
                if const_data(t) is not None:
                    die(f'layer {idx}: constant concat input not supported')
                act_tensors.add(t)
            L.update(in0=ins[0], ins_off=len(concat_inputs), n_ins=len(ins))
            L['p'][0] = axis
            L['_cat'] = list(ins)
            concat_inputs.extend(ins)
        elif name in ('PAD', 'PADV2'):
            if name == 'PADV2' and len(ins) > 2:
                cv = np.frombuffer(const_data(ins[2]), dtype=np.float32)
                if cv.size and float(cv[0]) != 0.0:
                    die(f'layer {idx}: PADV2 with non-zero value not supported')
            pads = np.frombuffer(const_data(ins[1]), dtype=np.int32).reshape(-1, 2)
            pads = np.vstack([np.zeros((4 - len(pads), 2), dtype=np.int32), pads])
            if pads[0].any():
                die(f'layer {idx}: batch padding not supported')
            L.update(in0=ins[0])
            L['p'] = [int(pads[1][0]), int(pads[1][1]), int(pads[2][0]), int(pads[2][1]), int(pads[3][0]), int(pads[3][1])]
            act_tensors.add(ins[0])
        layers.append(L)

    nL = len(layers)
    prod = {}
    for i, L in enumerate(layers):
        prod[L['out']] = i
    def consumers(t):
        r = []
        for i, L in enumerate(layers):
            ins = [L['in0']] + ([L['in1']] if L['in1'] >= 0 and L['w'] is None else []) + L.get('_cat', [])
            if t in ins:
                r.append(i)
        return r
    ROWOPS = {'NN_OP_CONV', 'NN_OP_DWCONV', 'NN_OP_ADD', 'NN_OP_SUB', 'NN_OP_MUL', 'NN_OP_AVGPOOL', 'NN_OP_MAXPOOL',
              'NN_OP_RELU', 'NN_OP_RELU6', 'NN_OP_LOGISTIC', 'NN_OP_TANH', 'NN_OP_LEAKY_RELU', 'NN_OP_HARD_SWISH'}
    groups = [dict(first=i, last=i, whole=True) for i in range(nL)]
    cap = {}
    for L in layers:
        L['step'] = 1
    plan_note = None
    if a.blueprint:
        sbp = open(a.blueprint, encoding='utf-8').read()
        mm = re.search(r'Machine-readable plan.*?<pre>(.*?)</pre>', sbp, re.S)
        jbp = json.loads(html.unescape(mm.group(1)))
        if len(jbp['layers']) != nL:
            die(f"blueprint has {len(jbp['layers'])} layers, model has {nL}")
        if jbp.get('spilled') or jbp.get('sd_transfers'):
            die('blueprint spills tensors to SDRAM/SD: not supported by this engine yet')
        groups = []
        for G in jbp['groups']:
            gg = dict(first=G['first_layer'], last=G['last_layer'], whole=bool(G['whole']))
            if not gg['whole'] and gg['first'] == gg['last'] and not G['buffers']:
                gg['whole'] = True
            if not gg['whole']:
                for li in range(gg['first'], gg['last'] + 1):
                    if layers[li]['op'] not in ROWOPS:
                        die(f"blueprint streams layer {li} ({layers[li]['name']}) which this engine cannot run row by row")
                    layers[li]['step'] = int(G['rows_per_step'].get(str(li), 1)) or 1
                for b in G['buffers']:
                    cap[b['tensor']] = max(cap.get(b['tensor'], 0), int(b['rows']))
            groups.append(gg)
        plan_note = f"blueprint plan: RAM budget {jbp['ram_budget']} B, planned peak {jbp['totals']['peak_ram']} B, {len(groups)} groups ({sum(not g['whole'] for g in groups)} row-streamed)"
        ram_budget = jbp['ram_budget']
    elif a.auto_stream:
        groups = []
        i = 0
        while i < nL:
            if layers[i]['op'] in ROWOPS:
                j = i
                while j + 1 < nL and layers[j + 1]['op'] in ROWOPS:
                    j += 1
                groups.append(dict(first=i, last=j, whole=(i == j)))
                i = j + 1
            else:
                groups.append(dict(first=i, last=i, whole=True))
                i += 1
    gid = {}
    for k, G in enumerate(groups):
        for li in range(G['first'], G['last'] + 1):
            gid[li] = k

    def d4(t):
        return dims4(t)
    def windowed(L):
        return L['op'] in ('NN_OP_CONV', 'NN_OP_DWCONV', 'NN_OP_AVGPOOL', 'NN_OP_MAXPOOL')
    def need_hi(L, t, rl):
        H = d4(t)[1]
        if H == 1: return 0
        if not windowed(L): return rl
        return min(H - 1, rl * L['sh'] - L['pt'] + (L['kh'] - 1) * L['dh'])
    def need_lo(L, t, r):
        if d4(t)[1] == 1: return 0
        if not windowed(L): return r
        return max(0, r * L['sh'] - L['pt'])

    def ring_tensors(G):
        rt = set()
        for li in range(G['first'], G['last'] + 1):
            o = layers[li]['out']
            cs = consumers(o)
            if cs and all(G['first'] <= c <= G['last'] for c in cs) and o != out_t:
                rt.add(o)
        return rt

    def simulate(G, caps):
        f, l = G['first'], G['last']
        nxt = {i: 0 for i in range(f, l + 1)}
        def avail(t):
            p = prod.get(t, -1)
            return nxt[p] if f <= p <= l else d4(t)[1]
        def ins_of(L):
            return [L['in0']] + ([L['in1']] if L['in1'] >= 0 and L['w'] is None else [])
        while True:
            progress = False
            done = True
            for i in range(f, l + 1):
                L = layers[i]
                OH = d4(L['out'])[1]
                while nxt[i] < OH:
                    r = nxt[i]
                    n = min(L['step'], OH - r)
                    rl = r + n - 1
                    if any(need_hi(L, t, rl) >= avail(t) for t in ins_of(L)):
                        break
                    cp = caps.get(L['out'], OH)
                    ev = rl - cp
                    ok = True
                    if ev >= 0:
                        for c in range(f, l + 1):
                            Lc = layers[c]
                            if nxt[c] >= d4(Lc['out'])[1]:
                                continue
                            if L['out'] in ins_of(Lc) and ev >= need_lo(Lc, L['out'], nxt[c]):
                                ok = False
                                break
                    if not ok:
                        break
                    nxt[i] = r + n
                    progress = True
                if nxt[i] < OH:
                    done = False
            if done:
                return True
            if not progress:
                return False

    for G in groups:
        if G['whole']:
            continue
        rts = ring_tensors(G)
        caps = {t: min(cap.get(t, d4(t)[1]), d4(t)[1]) for t in rts}
        if a.auto_stream and not a.blueprint:
            for t in rts:
                need = 1
                for c in consumers(t):
                    Lc = layers[c]
                    k = ((Lc['kh'] - 1) * Lc['dh'] + 1 + Lc['sh'] * (Lc['step'] - 1)) if windowed(Lc) else Lc['step']
                    need = max(need, k)
                caps[t] = min(need, d4(t)[1])
            while not simulate(G, caps):
                grown = False
                for t in caps:
                    if caps[t] < d4(t)[1]:
                        caps[t] += 1
                        grown = True
                if not grown:
                    die(f"group {G['first']}-{G['last']}: cannot schedule")
        else:
            if not simulate(G, caps):
                saved = {li: layers[li]['step'] for li in range(G['first'], G['last'] + 1)}
                for li in saved:
                    layers[li]['step'] = 1
                if not simulate(G, caps):
                    die(f"group {G['first']}-{G['last']}: the planned ring buffers deadlock")
        for t, c in caps.items():
            cap[t] = c
        G['caps'] = caps

    def root(t):
        while t in alias:
            t = alias[t]
        return t

    life = {}
    size = {}
    def tcap(t):
        return cap.get(t, dims4(t)[1]) if t in cap else dims4(t)[1]
    for t in act_tensors:
        r = root(t)
        d = dims4(t)
        size[r] = max(size.get(r, 0), tcap(t) * d[2] * d[3])
    def touch(t, i):
        r = root(t)
        if 0 <= i < nL:
            G = groups[gid[i]]
            lo, hi = G['first'], G['last']
        else:
            lo = hi = i
        s, e = life.get(r, (lo, hi))
        life[r] = (min(s, lo), max(e, hi))
    touch(in_t, -1)
    for i, L in enumerate(layers):
        for t in [L['in0'], L['in1']] + L.get('_cat', []):
            if t >= 0 and t in act_tensors and not (L['w'] is not None and t == L['in1'] and L['op'] in ('NN_OP_ADD', 'NN_OP_SUB', 'NN_OP_MUL')):
                touch(t, i)
        touch(L['out'], i)
    touch(out_t, nL)

    ALIGN = 8
    placed = []
    offs = {}
    for r in sorted(life, key=lambda r: -size[r]):
        s, e = life[r]
        sz = (size[r] + ALIGN - 1) // ALIGN * ALIGN
        conflicts = sorted((o, o2) for (rr, o, o2, s2, e2) in placed if not (e < s2 or e2 < s))
        off = 0
        for (o, o2) in conflicts:
            if off + sz <= o:
                break
            off = max(off, o2)
        offs[r] = off
        placed.append((r, off, off + sz, s, e))
    arena = max((p[2] for p in placed), default=0)

    tlist = sorted(set(list(act_tensors) + [L['in1'] for L in layers if L['in1'] >= 0]))
    tidx = {t: i for i, t in enumerate(tlist)}

    budget = int(a.flash_kb * 1024)
    used = 0
    wi_words, wq_words = [], []
    wloc = {}
    spill = False
    for wid, (ti, arr) in enumerate(weights):
        nbytes = arr.nbytes
        if not spill and used + nbytes > budget:
            spill = True
        if not spill:
            wloc[wid] = ('NN_LOC_FLASH', len(wi_words))
            wi_words.extend(arr.view(np.uint32).tolist())
            used += nbytes
        else:
            wloc[wid] = ('NN_LOC_QSPI', len(wq_words))
            wq_words.extend(arr.view(np.uint32).tolist())
    si = sum(wi_words) & 0xFFFFFFFF
    sq = sum(wq_words) & 0xFFFFFFFF

    os.makedirs(os.path.join(a.out, 'Src'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'Inc'), exist_ok=True)
    mname = os.path.basename(a.tflite)

    with open(os.path.join(a.out, 'Inc', 'model_weights.h'), 'w') as f:
        f.write('#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n#include <stdint.h>\n')
        f.write(f'#define MODEL_WI_COUNT {max(len(wi_words),1)}u\n#define MODEL_WQ_COUNT {len(wq_words)}u\n')
        f.write(f'#define MODEL_WI_USED {len(wi_words)}u\n')
        f.write(f'#define MODEL_WI_SUM 0x{si:08X}u\n#define MODEL_WQ_SUM 0x{sq:08X}u\n')
        f.write('extern const uint32_t model_wi[MODEL_WI_COUNT];\n')
        f.write('#define WQ_ADDR 0x90000000u\n#ifndef model_wq\n#define model_wq ((const uint32_t *)WQ_ADDR)\n#endif\n#endif\n')
    with open(os.path.join(a.out, 'Src', 'model_weights.c'), 'w') as f:
        f.write('#include "model_weights.h"\n\n__attribute__((aligned(32))) const uint32_t model_wi[MODEL_WI_COUNT] = {\n')
        ws = wi_words if wi_words else [0]
        for i in range(0, len(ws), 8):
            f.write('    ' + ', '.join('0x%08Xu' % v for v in ws[i:i + 8]) + ',\n')
        f.write('};\n')
    np.array(wq_words, dtype=np.uint32).astype('<u4').tofile(os.path.join(a.out, 'qspi_weights.bin'))

    in_n = int(np.prod(dims4(in_t)))
    out_n = int(np.prod(dims4(out_t)))
    with open(os.path.join(a.out, 'Inc', 'nn_model.h'), 'w') as f:
        f.write('#ifndef NN_MODEL_H\n#define NN_MODEL_H\n#include "nn_engine.h"\n')
        f.write(f'#define NN_MODEL_NAME "{mname}"\n')
        f.write(f'#define NN_N_LAYERS {nL}\n#define NN_N_TENSORS {len(tlist)}\n#define NN_N_GROUPS {len(groups)}\n')
        f.write(f'#define NN_ARENA_FLOATS {max(arena, 8)}u\n')
        f.write(f'#define NN_IN_T {tidx[in_t]}\n#define NN_OUT_T {tidx[out_t]}\n')
        f.write(f'#define NN_IN_FLOATS {in_n}u\n#define NN_OUT_FLOATS {out_n}u\n')
        f.write('extern const nn_tensor_t nn_tensors[NN_N_TENSORS];\n')
        f.write('extern const nn_layer_t nn_layers[NN_N_LAYERS];\n')
        f.write('extern const nn_group_t nn_groups[NN_N_GROUPS];\n')
        f.write('extern const int16_t nn_inputs[];\n')
        f.write('extern const char *const nn_layer_names[NN_N_LAYERS];\n#endif\n')
    with open(os.path.join(a.out, 'Src', 'nn_model.c'), 'w') as f:
        f.write('#include "nn_model.h"\n\n')
        f.write('const nn_tensor_t nn_tensors[NN_N_TENSORS] = {\n')
        for t in tlist:
            d = dims4(t)
            off = offs.get(root(t), 0)
            f.write(f'    {{ {off}u, {int(np.prod(d))}u, {d[1]}, {d[2]}, {d[3]}, {tcap(t)}, {prod.get(t, -1)} }},\n')
        f.write('};\n\nconst int16_t nn_inputs[] = { ')
        f.write(', '.join(str(tidx[t]) for t in concat_inputs) if concat_inputs else '0')
        f.write(' };\n\nconst nn_layer_t nn_layers[NN_N_LAYERS] = {\n')
        for L in layers:
            wl, wo = wloc[L['w']] if L['w'] is not None else ('NN_LOC_NONE', 0)
            bl, bo = wloc[L['b']] if L['b'] is not None else ('NN_LOC_NONE', 0)
            in0 = tidx[L['in0']] if L['in0'] >= 0 else -1
            in1 = tidx[L['in1']] if L['in1'] >= 0 else -1
            pp = ', '.join(str(v) for v in L['p'])
            f.write(f"    {{ {L['op']}, {L['act']}, {wl}, {bl}, {in0}, {in1}, {tidx[L['out']]}, {L['ins_off']}, {L['n_ins']}, "
                    f"{wo}u, {bo}u, {L['kh']}, {L['kw']}, {L['sh']}, {L['sw']}, {L['dh']}, {L['dw']}, {L['pt']}, {L['pl']}, "
                    f"{{ {pp} }}, {L['f0']!r}f, {L['step']} }},\n")
        f.write('};\n\nconst nn_group_t nn_groups[NN_N_GROUPS] = {\n')
        for G in groups:
            f.write(f"    {{ {G['first']}, {G['last']}, {1 if G['whole'] else 0} }},\n")
        f.write('};\n\nconst char *const nn_layer_names[NN_N_LAYERS] = {\n')
        for L in layers:
            f.write(f'    "{L["name"]}",\n')
        f.write('};\n')

    rep = []
    rep.append(f'model            : {mname}')
    rep.append(f'layers           : {nL}')
    rep.append(f'input / output   : {dims4(in_t)} -> {dims4(out_t)}')
    rep.append(f'arena (RAM)      : {arena * 4} B ({arena * 4 / 1024:.1f} KB)')
    rep.append(f'weights flash    : {len(wi_words) * 4} B ({len(wi_words) * 4 / 1024:.1f} KB)  sum 0x{si:08X}')
    rep.append(f'weights QSPI     : {len(wq_words) * 4} B ({len(wq_words) * 4 / 1024:.1f} KB)  sum 0x{sq:08X}  -> qspi_weights.bin @ 0x90000000')
    if plan_note:
        rep.append(plan_note)
        if arena * 4 > ram_budget:
            rep.append(f'WARNING          : arena {arena * 4} B is larger than the blueprint RAM budget {ram_budget} B')
    for G in groups:
        if not G['whole']:
            rep.append(f"stream group     : layers {G['first']}-{G['last']}, ring rows " + ', '.join(f't{t}={c}' for t, c in sorted(G['caps'].items())) + ', steps ' + ', '.join(f"#{li}={layers[li]['step']}" for li in range(G['first'], G['last'] + 1)))
    open(os.path.join(a.out, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))


if __name__ == '__main__':
    main()
```
