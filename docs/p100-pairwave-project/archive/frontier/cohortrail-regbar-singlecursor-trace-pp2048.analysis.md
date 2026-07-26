# Mapped trace: `cohortrail-regbar-singlecursor-trace-pp2048.nsys-rep`

Window: 1493.128 ms. Critical GPU: 3. Perfect-balance device-work floor: 739.513 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 741.195 | 751.933 |
| 1 | 738.278 | 754.850 |
| 2 | 734.997 | 758.131 |
| 3 | 743.583 | 749.545 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.493 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.628 |
| 1 | 70 | 0 | 133.795 |
| 2 | 70 | 0 | 133.555 |
| 3 | 70 | 0 | 133.515 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 253.376 | 253.376 |
| target-fp16-sgemm | 70 | 133.515 | 133.515 |
| other-graph | 1513 | 125.603 | 125.603 |
| other-affinitywave | 1280 | 104.436 | 104.436 |
| memcpy | 5264 | 79.727 | 80.324 |
| flash-attention | 20 | 33.310 | 33.310 |
| convert | 310 | 12.784 | 12.784 |
| dequant | 310 | 9.202 | 9.202 |
| memset | 80 | 0.113 | 0.113 |
