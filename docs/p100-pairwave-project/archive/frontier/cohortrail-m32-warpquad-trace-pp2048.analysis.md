# Mapped trace: `cohortrail-m32-warpquad-trace-pp2048.nsys-rep`

Window: 1513.423 ms. Critical GPU: 3. Perfect-balance device-work floor: 746.841 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 747.115 | 766.308 |
| 1 | 746.452 | 766.971 |
| 2 | 743.040 | 770.383 |
| 3 | 750.759 | 762.663 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.106 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.480 |
| 1 | 70 | 0 | 133.894 |
| 2 | 70 | 0 | 134.046 |
| 3 | 70 | 0 | 133.685 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 260.009 | 260.009 |
| target-fp16-sgemm | 70 | 133.685 | 133.685 |
| other-graph | 1513 | 125.055 | 125.062 |
| other-affinitywave | 1280 | 104.520 | 104.520 |
| memcpy | 5264 | 80.261 | 80.820 |
| flash-attention | 20 | 33.307 | 33.307 |
| convert | 310 | 12.782 | 12.782 |
| dequant | 310 | 9.208 | 9.208 |
| memset | 80 | 0.112 | 0.112 |
