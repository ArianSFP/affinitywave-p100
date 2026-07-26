# Mapped trace: `cohortrail-p2-raila-trace-pp2048.nsys-rep`

Window: 1514.995 ms. Critical GPU: 3. Perfect-balance device-work floor: 747.234 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 748.584 | 766.411 |
| 1 | 745.870 | 769.124 |
| 2 | 743.577 | 771.418 |
| 3 | 750.904 | 764.091 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.831 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.731 |
| 1 | 70 | 0 | 133.742 |
| 2 | 70 | 0 | 134.366 |
| 3 | 70 | 0 | 133.993 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 260.033 | 260.033 |
| target-fp16-sgemm | 70 | 133.993 | 133.993 |
| other-graph | 1513 | 125.333 | 125.333 |
| other-affinitywave | 1280 | 104.424 | 104.424 |
| memcpy | 5264 | 80.159 | 80.841 |
| flash-attention | 20 | 33.314 | 33.314 |
| convert | 310 | 12.779 | 12.779 |
| dequant | 310 | 9.200 | 9.200 |
| memset | 80 | 0.112 | 0.112 |
