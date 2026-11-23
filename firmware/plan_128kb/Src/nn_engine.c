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

#if NN_HAS_SDRAM
#ifdef MODEL_HOST
static uint32_t sdram_host[NN_SDRAM_FLOATS];
#define SDRAM ((volatile uint32_t *)sdram_host)
#else
#define SDRAM ((volatile uint32_t *)0xD0000000u)
#endif
#endif

uint64_t nn_sd_cycles;

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
static int16_t ld[NN_N_TENSORS];

static void sd_put(int t, int r0, int r1)
{
#if NN_HAS_SDRAM
    const nn_tensor_t *tt = T(t);
    uint32_t n = (uint32_t)tt->w * tt->c;
    for (int r = r0; r < r1; r++) {
        const uint32_t *s = (const uint32_t *)RP(t, r);
        volatile uint32_t *d = SDRAM + tt->ext + (uint32_t)r * n;
        for (uint32_t i = 0; i < n; i++) d[i] = s[i];
    }
#else
    (void)t; (void)r0; (void)r1;
#endif
}

static void sd_get(int t, int r0, int r1)
{
#if NN_HAS_SDRAM
    const nn_tensor_t *tt = T(t);
    uint32_t n = (uint32_t)tt->w * tt->c;
    for (int r = r0; r < r1; r++) {
        uint32_t *d = (uint32_t *)RP(t, r);
        const volatile uint32_t *s = SDRAM + tt->ext + (uint32_t)r * n;
        for (uint32_t i = 0; i < n; i++) d[i] = s[i];
    }
#else
    (void)t; (void)r0; (void)r1;
#endif
}

static int uses(const nn_layer_t *L, int t)
{
    return L->in0 == t || (!L->wloc && L->in1 == t);
}

static int first_user(int t, int first, int last)
{
    for (int c = first; c <= last; c++) if (uses(&nn_layers[c], t)) return c;
    return -1;
}

static int avail_rows(int t, int first, int last)
{
    if (T(t)->sd == NN_SD_IN) return ld[t];
    int p = nn_tensors[t].prod;
    if (p >= first && p <= last) return nxt[p];
    return T(t)->h;
}

static int input_ok(const nn_layer_t *L, int t, int rl, int first, int last)
{
    if (t < 0) return 1;
    return need_hi(L, t, rl) < avail_rows(t, first, last);
}

static int room_ok(int t, int rl, int first, int last)
{
    int evict = rl - (int)T(t)->cap;
    if (evict < 0) return 1;
    for (int c = first; c <= last; c++) {
        const nn_layer_t *Lc = &nn_layers[c];
        if (nxt[c] >= T(Lc->out)->h) continue;
        if (uses(Lc, t) && evict >= need_lo(Lc, t, nxt[c])) return 0;
    }
    return 1;
}

static int space_ok(int li, int rl, int first, int last)
{
    return room_ok(nn_layers[li].out, rl, first, last);
}

static int run_stream(int first, int last, uint64_t (*now)(void), uint64_t *lc)
{
    for (int i = first; i <= last; i++) nxt[i] = 0;
    for (int t = 0; t < NN_N_TENSORS; t++) ld[t] = 0;
    for (;;) {
        int progress = 0, done = 1;
        for (int t = 0; t < NN_N_TENSORS; t++) {
            if (T(t)->sd != NN_SD_IN) continue;
            int u = first_user(t, first, last);
            if (u < 0) continue;
            while (ld[t] < T(t)->h && room_ok(t, ld[t], first, last)) {
                uint64_t c0 = now();
                sd_get(t, ld[t], ld[t] + 1);
                uint64_t d = now() - c0;
                lc[u] += d;
                nn_sd_cycles += d;
                ld[t]++;
                progress = 1;
            }
        }
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
                if (T(L->out)->sd == NN_SD_OUT) {
                    uint64_t c0 = now();
                    sd_put(L->out, r, r + n);
                    nn_sd_cycles += now() - c0;
                }
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
    nn_sd_cycles = 0;
    for (int g = 0; g < NN_N_GROUPS; g++) {
        const nn_group_t *G = &nn_groups[g];
        if (G->whole) {
            for (int t = 0; t < NN_N_TENSORS; t++) {
                if (T(t)->sd != NN_SD_IN) continue;
                int u = first_user(t, G->first, G->last);
                if (u < 0) continue;
                uint64_t c0 = now();
                sd_get(t, 0, T(t)->h);
                uint64_t d = now() - c0;
                lc[u] += d;
                nn_sd_cycles += d;
            }
            for (int i = G->first; i <= G->last; i++) {
                const nn_layer_t *L = &nn_layers[i];
                uint64_t t = now();
                exec_rows(L, 0, T(L->out)->h);
                if (T(L->out)->sd == NN_SD_OUT) {
                    uint64_t c0 = now();
                    sd_put(L->out, 0, T(L->out)->h);
                    nn_sd_cycles += now() - c0;
                }
                lc[i] += now() - t;
            }
        } else if (run_stream(G->first, G->last, now, lc) != 0) {
            return -1 - g;
        }
    }
    return 0;
}
