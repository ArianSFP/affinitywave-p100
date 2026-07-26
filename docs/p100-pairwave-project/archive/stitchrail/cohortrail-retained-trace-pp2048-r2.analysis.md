# Mapped trace: `cohortrail-retained-trace-pp2048-r2.nsys-rep`

Window: 1498.992 ms. Critical GPU: 3. Perfect-balance device-work floor: 740.090 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 741.289 | 757.704 |
| 1 | 738.619 | 760.373 |
| 2 | 735.886 | 763.106 |
| 3 | 744.567 | 754.425 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.532 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.628 |
| 1 | 70 | 0 | 133.709 |
| 2 | 70 | 0 | 133.435 |
| 3 | 70 | 0 | 133.759 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 253.928 | 253.928 |
| target-fp16-sgemm | 70 | 133.759 | 133.759 |
| other-graph | 1513 | 125.063 | 125.063 |
| other-affinitywave | 1280 | 104.686 | 104.686 |
| memcpy | 5264 | 80.213 | 80.877 |
| flash-attention | 20 | 33.308 | 33.308 |
| convert | 310 | 12.779 | 12.779 |
| dequant | 310 | 9.211 | 9.211 |
| memset | 80 | 0.129 | 0.129 |
