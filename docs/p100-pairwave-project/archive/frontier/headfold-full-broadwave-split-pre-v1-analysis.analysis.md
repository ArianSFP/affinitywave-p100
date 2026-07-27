# Mapped trace: `headfold-full-broadwave-split-pre-v1.nsys-rep`

Window: 4084.020 ms. Critical GPU: 3. Perfect-balance device-work floor: 2966.824 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2703.017 | 1381.003 |
| 1 | 2852.530 | 1231.490 |
| 2 | 3117.684 | 966.336 |
| 3 | 3194.065 | 889.954 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2862.378 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.377 |
| 1 | 150 | 0 | 715.510 |
| 2 | 150 | 0 | 715.350 |
| 3 | 150 | 0 | 716.141 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 360 | 832.877 | 832.877 |
| target-fp16-sgemm | 150 | 716.141 | 716.141 |
| memcpy | 5784 | 635.363 | 637.198 |
| flash-attention | 30 | 440.985 | 440.985 |
| other-affinitywave | 640 | 249.991 | 249.991 |
| other-graph | 1333 | 242.121 | 242.138 |
| owner-reduction | 40 | 53.484 | 53.484 |
| convert | 310 | 48.554 | 48.554 |
| dequant | 310 | 9.744 | 9.744 |
| owner-sum | 40 | 5.585 | 5.585 |
| memset | 140 | 0.377 | 0.377 |
