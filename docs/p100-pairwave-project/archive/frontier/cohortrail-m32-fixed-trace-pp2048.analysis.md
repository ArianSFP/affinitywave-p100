# Mapped trace: `cohortrail-m32-fixed-trace-pp2048.nsys-rep`

Window: 1554.309 ms. Critical GPU: 3. Perfect-balance device-work floor: 747.127 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 748.198 | 806.111 |
| 1 | 746.184 | 808.125 |
| 2 | 743.083 | 811.226 |
| 3 | 751.042 | 803.267 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.794 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.537 |
| 1 | 70 | 0 | 133.850 |
| 2 | 70 | 0 | 133.647 |
| 3 | 70 | 0 | 133.760 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 260.032 | 260.032 |
| target-fp16-sgemm | 70 | 133.760 | 133.760 |
| other-graph | 1513 | 125.672 | 125.672 |
| other-affinitywave | 1280 | 104.575 | 104.575 |
| memcpy | 5264 | 80.179 | 80.830 |
| flash-attention | 20 | 33.339 | 33.339 |
| convert | 310 | 12.778 | 12.778 |
| dequant | 310 | 9.207 | 9.207 |
| memset | 80 | 0.122 | 0.122 |
