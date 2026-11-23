# pretrainedResnet_large_float — 128 KB RAM plan firmware (STM32H757I-EVAL, CM7)

Status: code complete, verified on PC (x86 + qemu-arm, rel err 7.9e-07 vs TFLite), cross-compiled + linked with the user's default STM32H757XIHX_FLASH.ld (no warnings). **Board result: 642.361 ms** (23 Nov 2026).

## Plan (from blueprint pretrainedResnet_large_float_blueprint-128-KB-RAM-fastest.html)
- RAM budget 131,072 B, planned peak 94,208 B. Generated arena: **94,208 B** (exact match).
- 6 groups, 2 row-streamed:
  - group 0 = layers 0–7 streamed, 1 row/step, rings t22=3, t23=3, t24=1, t25=3, t26=3, t27=1, t28=1, t29=1 (out-sd write buffer)
  - group 1 = layers 8–11 streamed, 1 row/step, rings t29(read copy, engine tensor 38)=3 (in-sd), t30=3, t31=1, t32=1
  - groups 2–5 = layers 12, 13, 14, 15 whole
- Spill: t29 (16x16x80 float = 81,920 B), output of layer 7 (ADD), goes to **SDRAM at 0xD0000000 + 0** (partition "SDRAM activations"). Group 0 writes each row out as it is made; group 1 reads rows back into a 3-row ring as layers 8 and 10 need them.
- Weights split unchanged: 940,800 B internal flash (sum 0xA26CBA10) + 980,520 B QSPI (sum 0xE1805E51). qspi_weights.bin is identical to the 512/256 KB one, so QSPI needs no reflash.

## What changed vs the 256 KB build (everything else identical)
Same functional units: 400 MHz (SMPS, VOS1, PLL1), I+D cache, flash 2 WS, DTCM WBUF/ZBUF, assembly mk4x4 FPU kernel, QSPI memory-mapped weights, DWT timing, USART1.
New, only because the plan spills:
1. **FMC SDRAM** (IS42S32800J, 32 MB, bank 2 @ 0xD0000000), bare-metal register init in main.c, values from ST's STM32H747I-EVAL BSP:
   - pins AF12, very high speed, pull-up: PD0,1,8,9,10,14,15; PE0,1,7–15; PF0–5,11–15; PG0–5,8,15; PH5–15; PI0–7,9,10 (no clash with QSPI or USART1 pins)
   - SDCR1: SDCLK = HCLK/2 (100 MHz), read burst on, RPIPE 0. SDCR2: 9 col bits, 12 row bits, 32-bit, 4 banks, CAS 3
   - SDTR: TMRD 2, TXSR 7, TRAS 4, TRC 7, TWR 2, TRP 2, TRCD 2
   - sequence: clock enable → 1 ms → precharge all → 8 auto-refresh → load mode 0x230 (BL1, sequential, CAS3, single write burst) → refresh count 0x603
   - FMCEN set in BCR1. Startup test writes and reads back the 81,920 B spill area ("SDRAM test OK").
   - 0xD0000000 is Device memory in the default M7 map (not cached), so copies are 32-bit word loops; no MPU or cache maintenance needed.
2. **Engine (fw3)**: nn_tensor_t gained `int32_t ext; uint8_t sd;` (NN_SD_OUT = write-through to SDRAM, NN_SD_IN = loaded from SDRAM). New sd_put / sd_get, room_ok (generalised space_ok), and load steps in run_stream. Copy time is added to the producing/consuming layer's time (so the total includes it) and also printed as "of which SDRAM copies".
3. **Generator nn_gen3.py**: reads `spilled` + partitions; splits the spilled tensor into a write-side buffer (producer group) and one read-side clone per consumer group with the blueprint's in-sd rows; the scheduler simulation models loads. SD-card spills are still rejected.
- Negative test: with the SDRAM copy loops disabled, the output is wrong (rel err 1.5), which proves the data really goes through SDRAM.
- Regression: test models A–D still pass with fw3/nn_gen3.

## Files (7) for the 128Kb CubeIDE project
- Core/Src: main.c, nn_engine.c (fw3), nn_model.c, model_weights.c (r128)
- Core/Inc: nn_engine.h (fw3), nn_model.h, model_weights.h (r128)
- Build: -O3, defines STM32H757xx and CORE_CM7.
- Expected first lines: `arena 94208 B`, `plan: 6 groups, 2 row-streamed`, `SDRAM test OK (81920 B at 0xD0000000)`.

## Board result
| Plan | Latency |
|---|---|
| 128 KB | 642.361 ms (256,944,565 cycles), of which SDRAM copies 2.396 ms, argmax 4, output matches 512/256 KB |

## Board run log — 23 Nov 2026, 19:42 (project 128Kb, exact PuTTY output)
```
model model.tflite: 16 layers, arena 94208 B in AXI SRAM
CPU 400 MHz; weights: 940800 B internal flash, 980520 B QSPI
plan: 6 groups, 2 row-streamed
SDRAM test OK (81920 B at 0xD0000000)
internal weights sum 0xA26CBA10 (expect 0xA26CBA10) OK
QSPI weights sum     0xE1805E51 (expect 0xE1805E51) OK
first QSPI words: 0x3E8FD9E4 0x3DF74F9D
step A: caches off at reset, wiping them (invalidate)
step B: caches wiped
step C: enabling caches
step D: caches enabled

=== caches ON ===
run 1: 256944565 cycles = 642.361 ms
min 642.361 ms  avg 642.361 ms  max 642.361 ms
of which SDRAM copies: 2.396 ms
per layer (last run):
  #0   CONV_2D            32.034 ms
  #1   CONV_2D            91.098 ms
  #2   CONV_2D            89.960 ms
  #3   ADD                6.190 ms
  #4   CONV_2D            55.122 ms
  #5   CONV_2D            99.985 ms
  #6   CONV_2D            10.558 ms
  #7   ADD                3.756 ms
  #8   CONV_2D            70.503 ms
  #9   CONV_2D            168.911 ms
  #10  CONV_2D            11.909 ms
  #11  ADD                1.515 ms
  #12  AVERAGE_POOL_2D    0.562 ms
  #13  RESHAPE            0.000 ms
  #14  FULLY_CONNECTED    0.245 ms
  #15  SOFTMAX            0.005 ms
output (x1e6, first 10 of 10): 13471 147 11360 159454 601976 4 7624 10479 195436 44
argmax 4

done
```
Notes:
- Before this run the debugger failed once ("Target is not responding … Fail starting session"); the retry ran fine.
- SDRAM copies cost only 2.396 ms (0.4%). Most of the extra time vs 256 KB is in layers 8–10, which are now streamed 1 row at a time: #8 36.7→70.5 ms, #9 71.7→168.9 ms, #10 6.5→11.9 ms. In row mode each conv call re-packs its weights into DTCM for every row (layer 9's 921,600 B of weights come from QSPI), so the weight re-reads grow with the number of rows.

## Latency so far (all same functional units)
| Plan | Latency |
|---|---|
| 512 KB | 325.338 ms |
| 256 KB | 504.264 ms |
| 128 KB | 642.361 ms |
