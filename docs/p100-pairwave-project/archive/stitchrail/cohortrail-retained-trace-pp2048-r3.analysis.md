# Mapped trace: `cohortrail-retained-trace-pp2048-r3.nsys-rep`

Window: 1512.829 ms. Critical GPU: 3. Perfect-balance device-work floor: 738.445 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 741.919 | 770.909 |
| 1 | 735.020 | 777.808 |
| 2 | 733.290 | 779.539 |
| 3 | 743.552 | 769.277 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.499 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.661 |
| 1 | 70 | 0 | 133.446 |
| 2 | 70 | 0 | 133.753 |
| 3 | 70 | 0 | 133.640 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 252.268 | 252.268 |
| target-fp16-sgemm | 70 | 133.640 | 133.640 |
| other-graph | 1513 | 125.827 | 125.827 |
| other-affinitywave | 1280 | 104.626 | 104.626 |
| memcpy | 5264 | 79.797 | 80.431 |
| flash-attention | 20 | 33.344 | 33.344 |
| convert | 310 | 12.776 | 12.776 |
| dequant | 310 | 9.211 | 9.211 |
| memset | 80 | 0.125 | 0.125 |
