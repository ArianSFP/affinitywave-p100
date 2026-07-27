# Mapped trace: `cohortrail-grid43-trace-pp2048.nsys-rep`

Window: 1611.993 ms. Critical GPU: 3. Perfect-balance device-work floor: 793.840 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 793.500 | 818.493 |
| 1 | 794.518 | 817.476 |
| 2 | 787.877 | 824.116 |
| 3 | 799.464 | 812.529 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.096 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.816 |
| 1 | 70 | 0 | 133.686 |
| 2 | 70 | 0 | 133.872 |
| 3 | 70 | 0 | 133.723 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 305.750 | 305.750 |
| target-fp16-sgemm | 70 | 133.723 | 133.723 |
| other-graph | 1513 | 125.162 | 125.162 |
| other-affinitywave | 1480 | 107.746 | 107.746 |
| memcpy | 5264 | 79.855 | 80.538 |
| flash-attention | 20 | 33.290 | 33.290 |
| convert | 310 | 12.781 | 12.781 |
| dequant | 310 | 9.200 | 9.200 |
| memset | 80 | 0.113 | 0.113 |
