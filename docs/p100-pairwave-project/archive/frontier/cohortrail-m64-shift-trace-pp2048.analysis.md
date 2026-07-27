# Mapped trace: `cohortrail-m64-shift-trace-pp2048.nsys-rep`

Window: 1498.138 ms. Critical GPU: 3. Perfect-balance device-work floor: 739.672 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 741.630 | 756.508 |
| 1 | 738.228 | 759.910 |
| 2 | 735.716 | 762.422 |
| 3 | 743.114 | 755.024 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.806 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.876 |
| 1 | 70 | 0 | 133.653 |
| 2 | 70 | 0 | 133.762 |
| 3 | 70 | 0 | 133.516 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 252.286 | 252.286 |
| target-fp16-sgemm | 70 | 133.516 | 133.516 |
| other-graph | 1513 | 125.140 | 125.140 |
| other-affinitywave | 1280 | 104.700 | 104.700 |
| memcpy | 5264 | 80.462 | 81.076 |
| flash-attention | 20 | 33.378 | 33.378 |
| convert | 310 | 12.780 | 12.780 |
| dequant | 310 | 9.215 | 9.215 |
| memset | 80 | 0.133 | 0.133 |
