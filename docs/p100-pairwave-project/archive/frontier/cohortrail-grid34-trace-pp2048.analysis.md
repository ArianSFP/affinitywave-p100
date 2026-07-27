# Mapped trace: `cohortrail-grid34-trace-pp2048.nsys-rep`

Window: 1586.416 ms. Critical GPU: 3. Perfect-balance device-work floor: 782.792 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 782.199 | 804.217 |
| 1 | 782.098 | 804.318 |
| 2 | 778.418 | 807.998 |
| 3 | 788.452 | 797.964 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.424 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.782 |
| 1 | 70 | 0 | 133.759 |
| 2 | 70 | 0 | 133.780 |
| 3 | 70 | 0 | 134.102 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 293.434 | 293.434 |
| target-fp16-sgemm | 70 | 134.102 | 134.102 |
| other-graph | 1513 | 125.338 | 125.338 |
| other-affinitywave | 1480 | 107.865 | 107.865 |
| memcpy | 5264 | 80.257 | 80.846 |
| flash-attention | 20 | 33.309 | 33.309 |
| convert | 310 | 12.781 | 12.781 |
| dequant | 310 | 9.206 | 9.206 |
| memset | 80 | 0.120 | 0.120 |
