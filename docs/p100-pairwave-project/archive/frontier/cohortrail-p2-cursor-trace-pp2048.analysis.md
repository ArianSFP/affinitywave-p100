# Mapped trace: `cohortrail-p2-cursor-trace-pp2048.nsys-rep`

Window: 1507.761 ms. Critical GPU: 3. Perfect-balance device-work floor: 744.836 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 745.278 | 762.483 |
| 1 | 743.655 | 764.106 |
| 2 | 741.117 | 766.644 |
| 3 | 749.292 | 758.469 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.388 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.423 |
| 1 | 70 | 0 | 133.589 |
| 2 | 70 | 0 | 133.608 |
| 3 | 70 | 0 | 133.768 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 258.069 | 258.069 |
| target-fp16-sgemm | 70 | 133.768 | 133.768 |
| other-graph | 1513 | 125.341 | 125.356 |
| other-affinitywave | 1280 | 104.566 | 104.566 |
| memcpy | 5264 | 80.469 | 81.132 |
| flash-attention | 20 | 33.344 | 33.344 |
| convert | 310 | 12.784 | 12.784 |
| dequant | 310 | 9.198 | 9.198 |
| memset | 80 | 0.119 | 0.119 |
