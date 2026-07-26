# Mapped trace: `cohortrail-m64-cursor-trace-pp2048.nsys-rep`

Window: 1506.204 ms. Critical GPU: 3. Perfect-balance device-work floor: 744.693 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 745.964 | 760.240 |
| 1 | 744.260 | 761.943 |
| 2 | 739.680 | 766.524 |
| 3 | 748.867 | 757.337 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.737 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.585 |
| 1 | 70 | 0 | 133.591 |
| 2 | 70 | 0 | 133.708 |
| 3 | 70 | 0 | 133.853 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 257.857 | 257.857 |
| target-fp16-sgemm | 70 | 133.853 | 133.853 |
| other-graph | 1513 | 125.220 | 125.225 |
| other-affinitywave | 1280 | 104.586 | 104.586 |
| memcpy | 5264 | 80.193 | 80.784 |
| flash-attention | 20 | 33.301 | 33.301 |
| convert | 310 | 12.784 | 12.784 |
| dequant | 310 | 9.197 | 9.197 |
| memset | 80 | 0.112 | 0.112 |
