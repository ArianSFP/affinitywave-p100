# Mapped trace: `cohortrail-pairwave-candidate-trace-pp2048.nsys-rep`

Window: 1596.349 ms. Critical GPU: 3. Perfect-balance device-work floor: 783.893 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 784.041 | 812.307 |
| 1 | 784.652 | 811.697 |
| 2 | 778.838 | 817.511 |
| 3 | 788.043 | 808.306 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.066 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.759 |
| 1 | 70 | 0 | 133.839 |
| 2 | 70 | 0 | 133.649 |
| 3 | 70 | 0 | 133.820 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 294.259 | 294.259 |
| target-fp16-sgemm | 70 | 133.820 | 133.820 |
| other-graph | 1513 | 125.031 | 125.038 |
| other-affinitywave | 1480 | 107.886 | 107.886 |
| memcpy | 5264 | 80.339 | 80.888 |
| flash-attention | 20 | 33.344 | 33.344 |
| convert | 310 | 12.781 | 12.781 |
| dequant | 310 | 9.202 | 9.202 |
| memset | 80 | 0.122 | 0.122 |
