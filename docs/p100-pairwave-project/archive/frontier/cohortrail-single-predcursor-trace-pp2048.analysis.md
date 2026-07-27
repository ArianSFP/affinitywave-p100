# Mapped trace: `cohortrail-single-predcursor-trace-pp2048.nsys-rep`

Window: 1495.794 ms. Critical GPU: 3. Perfect-balance device-work floor: 739.089 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 738.068 | 757.726 |
| 1 | 738.008 | 757.786 |
| 2 | 736.809 | 758.985 |
| 3 | 743.470 | 752.324 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.675 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.572 |
| 1 | 70 | 0 | 133.670 |
| 2 | 70 | 0 | 134.178 |
| 3 | 70 | 0 | 133.255 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 252.999 | 252.999 |
| target-fp16-sgemm | 70 | 133.255 | 133.255 |
| other-graph | 1513 | 125.048 | 125.058 |
| other-affinitywave | 1280 | 104.555 | 104.555 |
| memcpy | 5264 | 80.327 | 80.885 |
| flash-attention | 20 | 33.375 | 33.375 |
| convert | 310 | 12.783 | 12.783 |
| dequant | 310 | 9.200 | 9.200 |
| memset | 80 | 0.123 | 0.123 |
