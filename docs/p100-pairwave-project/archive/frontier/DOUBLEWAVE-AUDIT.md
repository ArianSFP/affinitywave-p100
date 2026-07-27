# DoubleWave exact double-buffer audit

## Hypothesis

BroadWave uses a 256-thread M64xN128 CTA with 32 KiB shared memory. A single
dequantized B buffer requires a producer/consumer barrier and a refill
barrier for every K32 stage. DoubleWave splits the 64 output rows between two
256-thread cohorts in one 512-thread CTA:

- each thread retains 4x4 accumulators on each original split-K2 rail;
- A and fully dequantized B both use two shared buffers;
- every split-K2 FP32 FMA and final rail add remains in its original order;
- the CTA uses one synchronization per K32 stage;
- no arithmetic, Q8 conversion, or weight traffic is duplicated.

The expected resource point was one CTA/SM with 16 resident warps, matching
the 16 total warps from two resident BroadWave CTAs.

## Compiled resources and correctness

The sm_60 kernel compiles with:

- 106 registers/thread;
- 49,152 bytes shared memory;
- zero stack and local memory;
- one 512-thread CTA/SM.

The four-GPU coalesced edge-route oracle checked 16,777,216 BF16 values per
GPU. Every GPU reported zero mismatches.

## Measured result

| kernel | gate | up | down | service stage |
| --- | ---: | ---: | ---: | ---: |
| BroadWave matched pool-four | about 5.27 ms | about 5.27 ms | about 5.15 ms | about 16.66 ms |
| DoubleWave | 6.32-6.34 ms | 6.32-6.34 ms | 6.33-6.37 ms | 19.94-20.01 ms |

DoubleWave regresses the exact projection by about 20%, despite removing one
barrier per K32 stage.

The SASS attribution explains the result. BroadWave has 192 static shared
loads per 256-thread CTA, or 49,152 CTA-scaled shared-load instructions.
DoubleWave has 160 per thread but 512 threads, or 81,920 CTA-scaled
shared-load instructions. The two row cohorts must reread the same B values,
raising shared-load issue work by 66.7%. The barrier saving cannot repay that
fanout.

DoubleWave is closed. A future synchronization redesign must retain
BroadWave's B-consumer fanout; adding row cohorts to fund a second float B
buffer is not viable.

Evidence:

- `doublewave-v1-service-p2048-ub2048.out`
- `doublewave-v1-service-p2048-ub2048.err`
- `doublewave-sass-summary-v1.txt`
