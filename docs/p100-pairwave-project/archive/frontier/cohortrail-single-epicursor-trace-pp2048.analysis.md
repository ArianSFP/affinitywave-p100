# Mapped trace: `cohortrail-single-epicursor-trace-pp2048.nsys-rep`

Window: 1497.433 ms. Critical GPU: 3. Perfect-balance device-work floor: 739.789 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 739.346 | 758.086 |
| 1 | 737.388 | 760.045 |
| 2 | 737.021 | 760.412 |
| 3 | 745.402 | 752.031 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.342 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.405 |
| 1 | 70 | 0 | 133.402 |
| 2 | 70 | 0 | 133.846 |
| 3 | 70 | 0 | 133.690 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 254.137 | 254.137 |
| target-fp16-sgemm | 70 | 133.690 | 133.690 |
| other-graph | 1513 | 125.678 | 125.688 |
| other-affinitywave | 1280 | 104.540 | 104.540 |
| memcpy | 5264 | 79.903 | 80.470 |
| flash-attention | 20 | 33.343 | 33.343 |
| convert | 310 | 12.780 | 12.780 |
| dequant | 310 | 9.207 | 9.207 |
| memset | 80 | 0.128 | 0.128 |
