# VectorWave exact-Q8 microtile audit

## Hypothesis

BroadWave maps one thread to eight rows and four strided output columns. Per
K value each thread issues two vector A loads and four scalar B loads from
shared memory.

VectorWave maps each warp to two logical half-warps. Each thread owns four
rows and eight contiguous output columns. The exact two K16 accumulator
rails and final FP32 add are unchanged. The intended gain was to replace six
shared-load instructions per K value with one vector A load and two vector B
loads.

## Compiled result

Both kernels use 256 threads, 122 registers/thread, 32 KiB shared memory, no
stack, and no local memory. Static SASS per thread is:

| instruction | BroadWave | VectorWave |
| --- | ---: | ---: |
| total | 1,956 | 1,734 |
| FFMA | 1,024 | 1,024 |
| LDS | 192 | 96 |
| STS | 48 | 48 |
| LDG | 8 | 8 |
| BAR | 4 | 4 |
| signed conversion | 24 | 24 |
| FMUL | 32 | 32 |
| FADD | 32 | 32 |

The compiler produced the intended instruction stream without spills.

## Measured result

The four-cell edge-route oracle checked 16,777,216 BF16 owner values/GPU
with zero mismatches.

| stage | BroadWave | VectorWave |
| --- | ---: | ---: |
| gate | 5.2622 ms | 5.6445 ms |
| up | about 5.262 ms | 5.6420 ms |
| down | 5.1598 ms | 5.7788 ms |
| complete instrumented service | 16.6480 ms | 18.0292 ms |

VectorWave is 8.30% slower end to end despite eliminating 96 static shared
loads. A 16-byte shared load from every lane spans four banks; the contiguous
half-warp mapping creates the corresponding multi-transaction bank demand.
BroadWave's four scalar loads are individually conflict-free. The instruction
count falls, but the shared-memory transactions do not.

VectorWave is closed. A swizzle cannot remove the lower bound of 128 B values
per warp and K value through 32 four-byte banks; it can only redistribute
those transactions.

Evidence:

- `vectorwave-v1-service-p2048-ub2048.out`
- `vectorwave-v1-service-p2048-ub2048.err`
- `ggml/src/ggml-cuda/affinity-wave.cu`
