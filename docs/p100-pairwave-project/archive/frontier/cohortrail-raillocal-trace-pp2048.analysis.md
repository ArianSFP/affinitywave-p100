# Mapped trace: `cohortrail-raillocal-trace-pp2048.nsys-rep`

Window: 1525.293 ms. Critical GPU: 3. Perfect-balance device-work floor: 752.090 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 752.813 | 772.479 |
| 1 | 751.682 | 773.611 |
| 2 | 748.301 | 776.991 |
| 3 | 755.565 | 769.727 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.805 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.630 |
| 1 | 70 | 0 | 133.577 |
| 2 | 70 | 0 | 133.939 |
| 3 | 70 | 0 | 133.659 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 265.542 | 265.542 |
| target-fp16-sgemm | 70 | 133.659 | 133.659 |
| other-graph | 1513 | 125.188 | 125.188 |
| other-affinitywave | 1280 | 104.485 | 104.485 |
| memcpy | 5264 | 79.621 | 80.255 |
| flash-attention | 20 | 33.366 | 33.366 |
| convert | 310 | 12.787 | 12.787 |
| dequant | 310 | 9.211 | 9.211 |
| memset | 80 | 0.112 | 0.112 |
