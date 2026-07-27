# Mapped trace: `cohortrail-grid33-skip-trace-pp2048.nsys-rep`

Window: 1583.724 ms. Critical GPU: 3. Perfect-balance device-work floor: 782.904 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 782.596 | 801.128 |
| 1 | 782.535 | 801.190 |
| 2 | 778.746 | 804.978 |
| 3 | 787.737 | 795.988 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.516 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.642 |
| 1 | 70 | 0 | 133.518 |
| 2 | 70 | 0 | 133.760 |
| 3 | 70 | 0 | 133.597 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 293.388 | 293.388 |
| target-fp16-sgemm | 70 | 133.597 | 133.597 |
| other-graph | 1513 | 125.220 | 125.231 |
| other-affinitywave | 1480 | 107.891 | 107.891 |
| memcpy | 5264 | 80.436 | 81.054 |
| flash-attention | 20 | 33.363 | 33.363 |
| convert | 310 | 12.787 | 12.787 |
| dequant | 310 | 9.204 | 9.204 |
| memset | 80 | 0.123 | 0.123 |
