# Mapped trace: `cohortrail-prefetch-trace-pp2048.nsys-rep`

Window: 1556.954 ms. Critical GPU: 3. Perfect-balance device-work floor: 767.760 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 768.369 | 788.584 |
| 1 | 767.044 | 789.910 |
| 2 | 763.761 | 793.193 |
| 3 | 771.868 | 785.085 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.431 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.758 |
| 1 | 70 | 0 | 133.671 |
| 2 | 70 | 0 | 134.194 |
| 3 | 70 | 0 | 133.807 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 280.694 | 280.694 |
| target-fp16-sgemm | 70 | 133.807 | 133.807 |
| other-graph | 1513 | 125.838 | 125.838 |
| other-affinitywave | 1280 | 104.429 | 104.429 |
| memcpy | 5264 | 79.753 | 80.452 |
| flash-attention | 20 | 33.348 | 33.348 |
| convert | 310 | 12.784 | 12.784 |
| dequant | 310 | 9.200 | 9.200 |
| memset | 80 | 0.120 | 0.120 |
