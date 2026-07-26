# Mapped trace: `cohortrail-p2-fixed-trace-pp2048.nsys-rep`

Window: 1517.116 ms. Critical GPU: 3. Perfect-balance device-work floor: 749.066 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 750.758 | 766.358 |
| 1 | 747.833 | 769.283 |
| 2 | 744.926 | 772.190 |
| 3 | 752.747 | 764.368 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.646 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.691 |
| 1 | 70 | 0 | 133.788 |
| 2 | 70 | 0 | 133.689 |
| 3 | 70 | 0 | 133.479 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 261.978 | 261.978 |
| target-fp16-sgemm | 70 | 133.479 | 133.479 |
| other-graph | 1513 | 124.916 | 124.931 |
| other-affinitywave | 1280 | 104.608 | 104.608 |
| memcpy | 5264 | 80.129 | 80.846 |
| flash-attention | 20 | 33.379 | 33.379 |
| convert | 310 | 12.785 | 12.785 |
| dequant | 310 | 9.203 | 9.203 |
| memset | 80 | 0.132 | 0.132 |
