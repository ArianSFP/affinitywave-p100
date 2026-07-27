# Mapped trace: `cohortrail-retained-trace-pp2048-r1.nsys-rep`

Window: 1495.555 ms. Critical GPU: 3. Perfect-balance device-work floor: 739.376 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 741.073 | 754.482 |
| 1 | 738.045 | 757.510 |
| 2 | 735.684 | 759.871 |
| 3 | 742.701 | 752.854 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.086 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.792 |
| 1 | 70 | 0 | 133.876 |
| 2 | 70 | 0 | 133.960 |
| 3 | 70 | 0 | 133.458 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 252.231 | 252.231 |
| target-fp16-sgemm | 70 | 133.458 | 133.458 |
| other-graph | 1513 | 125.077 | 125.077 |
| other-affinitywave | 1280 | 104.661 | 104.661 |
| memcpy | 5264 | 80.011 | 80.707 |
| flash-attention | 20 | 33.319 | 33.319 |
| convert | 310 | 12.781 | 12.781 |
| dequant | 310 | 9.202 | 9.202 |
| memset | 80 | 0.111 | 0.111 |
