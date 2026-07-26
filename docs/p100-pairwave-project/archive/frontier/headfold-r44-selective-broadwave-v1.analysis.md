# Mapped trace: `headfold-r44-selective-broadwave-v1.nsys-rep`

Window: 3648.335 ms. Critical GPU: 3. Perfect-balance device-work floor: 2799.343 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2668.798 | 979.537 |
| 1 | 2756.668 | 891.667 |
| 2 | 2833.714 | 814.621 |
| 3 | 2938.194 | 710.141 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2863.623 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.857 |
| 1 | 150 | 0 | 715.731 |
| 2 | 150 | 0 | 716.429 |
| 3 | 150 | 0 | 715.607 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| memcpy | 10800 | 731.925 | 738.198 |
| target-fp16-sgemm | 150 | 715.607 | 715.607 |
| exact-q8-projection | 1440 | 711.407 | 711.407 |
| other-affinitywave | 1480 | 523.425 | 523.425 |
| flash-attention | 30 | 439.695 | 439.695 |
| other-graph | 1333 | 242.217 | 242.233 |
| convert | 310 | 48.559 | 48.559 |
| dequant | 310 | 9.859 | 9.859 |
| memset | 140 | 0.406 | 0.406 |
