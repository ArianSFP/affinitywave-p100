# Mapped trace: `cohortrail-m64-qprefetch-trace-pp2048.nsys-rep`

Window: 1506.878 ms. Critical GPU: 3. Perfect-balance device-work floor: 743.470 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 745.767 | 761.111 |
| 1 | 742.444 | 764.434 |
| 2 | 738.792 | 768.086 |
| 3 | 746.876 | 760.002 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.418 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.643 |
| 1 | 70 | 0 | 133.401 |
| 2 | 70 | 0 | 133.769 |
| 3 | 70 | 0 | 133.606 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 256.922 | 256.922 |
| target-fp16-sgemm | 70 | 133.606 | 133.606 |
| other-graph | 1513 | 124.820 | 124.823 |
| other-affinitywave | 1280 | 104.524 | 104.524 |
| memcpy | 5264 | 80.479 | 81.186 |
| flash-attention | 20 | 33.341 | 33.341 |
| convert | 310 | 12.782 | 12.782 |
| dequant | 310 | 9.207 | 9.207 |
| memset | 80 | 0.114 | 0.114 |
