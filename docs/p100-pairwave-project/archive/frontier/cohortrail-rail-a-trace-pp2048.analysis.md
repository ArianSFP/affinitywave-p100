# Mapped trace: `cohortrail-rail-a-trace-pp2048.nsys-rep`

Window: 1588.766 ms. Critical GPU: 3. Perfect-balance device-work floor: 783.364 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 783.892 | 804.874 |
| 1 | 782.316 | 806.450 |
| 2 | 779.686 | 809.081 |
| 3 | 787.561 | 801.205 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.985 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 134.188 |
| 1 | 70 | 0 | 133.491 |
| 2 | 70 | 0 | 133.700 |
| 3 | 70 | 0 | 133.606 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 293.795 | 293.795 |
| target-fp16-sgemm | 70 | 133.606 | 133.606 |
| other-graph | 1513 | 124.636 | 124.639 |
| other-affinitywave | 1480 | 107.838 | 107.838 |
| memcpy | 5264 | 80.173 | 80.885 |
| flash-attention | 20 | 33.346 | 33.346 |
| convert | 310 | 12.779 | 12.779 |
| dequant | 310 | 9.210 | 9.210 |
| memset | 80 | 0.123 | 0.123 |
