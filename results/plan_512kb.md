# 512 KB plan — board result

Engine: G1 (every layer run whole, tensors in AXI SRAM). Arena 491,520 B. No SDRAM.

**Latency: 325.338 ms** (130,135,561 cycles, -O3, caches ON, 400 MHz). Output identical to TFLite (argmax 4).
A second run (same files, built in another project) gave 324.995 ms (129,998,263 cycles).

| Layer | Op | ms |
|---|---|---|
| #0 | CONV_2D | 10.333 |
| #1 | CONV_2D | 59.297 |
| #2 | CONV_2D | 58.433 |
| #3 | ADD | 1.894 |
| #4 | CONV_2D | 30.357 |
| #5 | CONV_2D | 56.921 |
| #6 | CONV_2D | 4.151 |
| #7 | ADD | 0.947 |
| #8 | CONV_2D | 30.818 |
| #9 | CONV_2D | 66.872 |
| #10 | CONV_2D | 4.419 |
| #11 | ADD | 0.470 |
| #12 | AVERAGE_POOL_2D | 0.192 |
| #13 | RESHAPE | 0.000 |
| #14 | FULLY_CONNECTED | 0.222 |
| #15 | SOFTMAX | 0.004 |

Output (x1e6): 13471 147 11360 159454 601976 4 7624 10479 195436 44, argmax 4
