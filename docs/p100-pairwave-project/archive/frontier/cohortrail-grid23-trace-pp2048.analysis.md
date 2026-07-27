# Mapped trace: `cohortrail-grid23-trace-pp2048.nsys-rep`

Window: 1619.069 ms. Critical GPU: 3. Perfect-balance device-work floor: 798.070 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 798.544 | 820.525 |
| 1 | 796.660 | 822.409 |
| 2 | 793.826 | 825.243 |
| 3 | 803.252 | 815.817 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.522 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.272 |
| 1 | 70 | 0 | 134.070 |
| 2 | 70 | 0 | 133.551 |
| 3 | 70 | 0 | 133.630 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 309.335 | 309.335 |
| target-fp16-sgemm | 70 | 133.630 | 133.630 |
| other-graph | 1513 | 125.094 | 125.094 |
| other-affinitywave | 1480 | 107.785 | 107.785 |
| memcpy | 5264 | 80.389 | 80.992 |
| flash-attention | 20 | 33.363 | 33.363 |
| convert | 310 | 12.785 | 12.785 |
| dequant | 310 | 9.198 | 9.198 |
| memset | 80 | 0.124 | 0.124 |
