# ResNet tiling plans on STM32H757 latency study

Inference latency of a float32 ResNet (`pretrainedResnet_large_float.tflite`, 32×32×3 input, 10 classes, 16 layers, 1.92 MB of weights) on the **STM32H757I-EVAL** board (Cortex-M7 @ 400 MHz, bare metal), for memory plans made with Tiling Lab at different RAM budgets.

Each plan decides how much RAM the activations may use, which layers run row by row, and which tensors go to external SDRAM. The firmware runs exactly that plan and measures latency with the DWT cycle counter (method based on EntoBench).

## Results

| Plan | RAM budget | Activation arena (AXI SRAM) | Groups (row-streamed) | SDRAM used | SDRAM copy time | Latency |
|---|---|---|---|---|---|---|
| 512 KB | 524,288 B | 491,520 B | 16 (0) | 0 | – | **325.338 ms** |
| 256 KB | 262,144 B | 252,928 B | 10 (1) | 0 | – | **504.264 ms** |
| 128 KB | 131,072 B | 94,208 B | 6 (2) | 81,920 B | 2.396 ms | **642.361 ms** |
| 64 KB | 65,536 B | 53,248 B | 7 (3) | 245,760 B | 5.230 ms | **501.930 ms** |

- -O3, I+D caches on, 400 MHz, one timed run per plan.
- Every plan gives the same output: `13471 147 11360 159454 601976 4 7624 10479 195436 44` (×1e-6), argmax 4. It matches TFLite on PC (relative error 8e-7).
- Each arena matches the blueprint's planned peak RAM exactly.

Per-layer times and the full board logs are in `results/`.

## Memory map

| Memory | Address | Holds | 512 KB | 256 KB | 128 KB | 64 KB |
|---|---|---|---|---|---|---|
| Internal flash (M7 bank) | 0x08000000 | weights, part 1 | 940,800 B | 940,800 B | 940,800 B | 940,800 B |
| QSPI flash (2× MT25TL01G) | 0x90000000 | weights, part 2 | 980,520 B | 980,520 B | 980,520 B | 980,520 B |
| DTCM | 0x20000000 | packed weights of the current layer (96 KB) + zero buffer (32 KB) | 128 KB | 128 KB | 128 KB | 128 KB |
| AXI SRAM | 0x24000000 | activation arena | 491,520 B | 252,928 B | 94,208 B | 53,248 B |
| SDRAM (FMC, IS42S32800J) | 0xD0000000 | activations that do not fit | 0 | 0 | 81,920 B | 245,760 B |

Weights always live in flash + QSPI. Only the activations change between plans: 512 KB keeps every tensor whole; 256 KB streams layers #0–#6 row by row through small ring buffers; 128 KB and 64 KB also write some layer outputs to SDRAM and read them back row by row when the next group needs them.

## Hardware used by the firmware
- Clock: Direct SMPS, VOS1, PLL1 from the 25 MHz HSE → 400 MHz CPU, 200 MHz AXI, 100 MHz APB.
- Caches: I-cache and D-cache on (invalidated first).
- FPU: single-precision FMA. Hand-written assembly kernel `mk4x4` (4 pixels × 4 output channels, 16 accumulators, 32 `vfma.f32` per step).
- DTCM: weight packing buffer + zero buffer for padding.
- QUADSPI: dual-flash, memory-mapped, 1-1-4 read at 66.7 MHz.
- FMC SDRAM (128/64 KB plans only): 32-bit, 100 MHz, CAS 3.
- DWT cycle counter (64-bit with SysTick overflow), USART1 115200 → ST-LINK virtual COM port.
- Default CubeIDE linker file, CM7 only (the CM4 and its flash bank are not used).

## Folder layout
```
firmware/
  common/      model_weights.c/.h   (same for every plan)
  plan_512kb/  main.c, nn_engine.c/.h, nn_model.c/.h   (engine G1: whole layers)
  plan_256kb/  main.c, nn_engine.c/.h, nn_model.c/.h   (engine fw2: + row streaming)
  plan_128kb/  main.c, nn_engine.c/.h, nn_model.c/.h   (engine fw3: + SDRAM)
  plan_64kb/   main.c, nn_engine.c/.h, nn_model.c/.h   (engine fw3)
qspi/          qspi_weights.bin   (write to 0x90000000 once)
model/         pretrainedResnet_large_float.tflite
results/       board logs and per-layer times for each plan
```
Each plan folder holds exactly the code that produced its result. The fw3 engine (plan_128kb / plan_64kb) can also run the 512 KB and 256 KB plans; only `nn_model.c/.h` changes.

## Board setup
- USB into **CN23** (ST-LINK). **JP10**, **JP7** and **JP8** on **STlk** (JP7/JP8 on CON send the UART to the DB9 port instead of the PC).
- UART output: ST-LINK virtual COM port, 115200 baud.

## CubeIDE project (one per plan)
1. File → New → STM32 Project → STM32H757I-EVAL → Empty project. Use the `_CM7` sub-project only.
2. Copy CMSIS headers into `Inc`: everything from [cmsis-core](https://github.com/STMicroelectronics/cmsis-core) `CMSIS/Core/Include` (incl. `m-profile`), and `stm32h7xx.h`, `stm32h757xx.h`, `system_stm32h7xx.h` from [cmsis-device-h7](https://github.com/STMicroelectronics/cmsis-device-h7) `Include`.
3. Copy the firmware:
   - `Src`: the plan's `main.c`, `nn_engine.c`, `nn_model.c` + `firmware/common/Src/model_weights.c`
   - `Inc`: the plan's `nn_engine.h`, `nn_model.h` + `firmware/common/Inc/model_weights.h`
4. Properties → C/C++ Build → Settings → MCU/MPU GCC Compiler: define symbols `STM32H757xx` and `CORE_CM7`, optimization **-O3**.
5. The default linker file `STM32H757XIHX_FLASH.ld` is used unchanged.

## Write the QSPI weights (once per board)
PowerShell, with STM32CubeIDE 2.2.0 installed:
```
$cli = Get-ChildItem "C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins" -Recurse -Filter STM32_Programmer_CLI.exe | Select-Object -First 1
$el = Get-ChildItem $cli.Directory.FullName -Recurse -Filter "MT25TL01G*H747I-EVAL*.stldr" | Select-Object -First 1
& $cli.FullName -c port=SWD mode=UR -el $el.FullName -w "qspi\qspi_weights.bin" 0x90000000 -v
```
At start-up the firmware checks both weight sums (flash 0xA26CBA10, QSPI 0xE1805E51) and, for the 128/64 KB plans, tests the SDRAM.

## Run
Build, run, open the COM port at 115200. The firmware prints the plan, runs one inference and prints the total latency, per-layer times and the output.

## How latency is measured
- DWT CYCCNT at 400 MHz, extended to 64 bits with SysTick.
- The timed window is the inference only (layer #0 → softmax), including SDRAM copies. Clock, QSPI/SDRAM init, weight checks and input generation are outside it.
- latency_ms = cycles / 400,000
- Input: fixed pseudo-random values (LCG, seed 12345), so every plan computes the same thing.

## Acknowledgment
Firmware was developed with assistance from Claude (Anthropic). Hardware bring-up, all board measurements and analysis were done by the author.
