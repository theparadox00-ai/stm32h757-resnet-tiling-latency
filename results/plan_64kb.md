# pretrainedResnet_large_float — 64 KB RAM plan firmware (STM32H757I-EVAL, CM7)

Status: generated + verified on PC (x86 and qemu-arm, rel err 7.9e-07 vs TFLite), cross-compiled and linked with the user's default .ld. **Board result: 501.930 ms** (23 Nov 2026, 21:02).

## Plan (blueprint pretrainedResnet_large_float_blueprint-64-KB-RAM-fastest.html)
- RAM budget 65,536 B, planned peak 53,248 B. Generated arena: **53,248 B** (exact match).
- 7 groups, 3 row-streamed (1 row/step each):
  - group 0 = layers 0–3: rings t22=3, t23=3, t24=1, t25=1 (out-sd)
  - group 1 = layers 4–7: t25 read copy=3 (in-sd), t26=3, t27=1, t28=1, t29=1 (out-sd)
  - group 2 = layers 8–11: t29 read copy=3 (in-sd), t30=3, t31=1, t32=1, t33=1 (out-sd)
  - group 3 = layer 12 whole, reads all 8 rows of t33 from SDRAM; groups 4–6 = layers 13, 14, 15 whole
- SDRAM spills (0xD0000000 + offset): t25 163,840 B @ 81,920; t29 81,920 B @ 0; t33 40,960 B @ 81,920 (re-uses t25's space after t25 is dead). SDRAM test covers 245,760 B.
- Weights: same split, same model_weights.c and qspi_weights.bin as 512/256/128 KB.
- Negative test (SDRAM copies disabled) → wrong output (rel err 1.6), so data really goes through SDRAM.

## Files
- Same as the 128Kb project (fw3 main.c, nn_engine.c, nn_engine.h, model_weights.c/.h); only **nn_model.c** and **nn_model.h** change (r64).
- Build: -O3, defines STM32H757xx and CORE_CM7.
- Expected: `arena 53248 B`, `plan: 7 groups, 3 row-streamed`, `SDRAM test OK (245760 B at 0xD0000000)`.

## Board result
| Plan | Latency |
|---|---|
| 64 KB | 501.930 ms (200,772,070 cycles), of which SDRAM copies 5.230 ms, argmax 4, output matches all other plans |

## Invalid run — 23 Nov 2026, 20:59 (64Kb project still had the old 256 KB main.c / nn_engine.c)
No "SDRAM test" line and no "of which SDRAM copies" line → old fw2 main.c without SDRAM; output wrong (argmax 9, expected 4). Total printed 496.051 ms — NOT a valid 64 KB result.
Cause: project was copied from 256Kb (fw2 files); first build also showed "excess elements in struct initializer" (old nn_engine.h). Also had to add Src + Inc as source folders (Paths and Symbols → Source Location).
```
output (x1e6, first 10 of 10): 314 24 0 2100 0 0 2 56 3289 994211
argmax 9
```

## Valid board run — 23 Nov 2026, 21:02 (64Kb project, fw3 main.c/nn_engine.c/nn_engine.h + r64 nn_model.c/.h, exact PuTTY output)
```
model model.tflite: 16 layers, arena 53248 B in AXI SRAM
CPU 400 MHz; weights: 940800 B internal flash, 980520 B QSPI
plan: 7 groups, 3 row-streamed
SDRAM test OK (245760 B at 0xD0000000)
internal weights sum 0xA26CBA10 (expect 0xA26CBA10) OK
QSPI weights sum     0xE1805E51 (expect 0xE1805E51) OK
first QSPI words: 0x3E8FD9E4 0x3DF74F9D
step A: caches off at reset, wiping them (invalidate)
step B: caches wiped
step C: enabling caches
step D: caches enabled

=== caches ON ===
run 1: 200772070 cycles = 501.930 ms
min 501.930 ms  avg 501.930 ms  max 501.930 ms
of which SDRAM copies: 5.230 ms
per layer (last run):
  #0   CONV_2D            11.643 ms
  #1   CONV_2D            69.399 ms
  #2   CONV_2D            68.684 ms
  #3   ADD                2.316 ms
  #4   CONV_2D            42.412 ms
  #5   CONV_2D            75.984 ms
  #6   CONV_2D            5.310 ms
  #7   ADD                1.173 ms
  #8   CONV_2D            49.384 ms
  #9   CONV_2D            164.193 ms
  #10  CONV_2D            9.803 ms
  #11  ADD                0.585 ms
  #12  AVERAGE_POOL_2D    0.811 ms
  #13  RESHAPE            0.000 ms
  #14  FULLY_CONNECTED    0.221 ms
  #15  SOFTMAX            0.004 ms
output (x1e6, first 10 of 10): 13471 147 11360 159454 601976 4 7624 10479 195436 44
argmax 4

done
```
Observation: 64 KB (501.9 ms) is faster than 128 KB (642.4 ms) and about equal to 256 KB (504.3 ms). Layers 0–2 are much faster here (#0 11.6 vs 32.0 ms, #1 69.4 vs 91.1 ms) — in this plan group 0 has only 4 layers instead of 7–8, so fewer layers are interleaved row by row. Cause not yet measured (likely cache/weight re-pack effects of interleaving).

## Latency so far (same functional units)
| Plan | Latency |
|---|---|
| 512 KB | 325.338 ms |
| 256 KB | 504.264 ms |
| 128 KB | 642.361 ms |
| 64 KB | 501.930 ms |
