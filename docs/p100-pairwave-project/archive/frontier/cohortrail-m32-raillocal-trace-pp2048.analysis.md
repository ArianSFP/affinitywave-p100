# Mapped trace: `cohortrail-m32-raillocal-trace-pp2048.nsys-rep`

Window: 1544.440 ms. Critical GPU: 3. Perfect-balance device-work floor: 763.386 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 764.560 | 779.880 |
| 1 | 764.688 | 779.752 |
| 2 | 759.323 | 785.117 |
| 3 | 764.971 | 779.470 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.011 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.628 |
| 1 | 70 | 0 | 133.652 |
| 2 | 70 | 0 | 134.157 |
| 3 | 70 | 0 | 133.574 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 274.470 | 274.470 |
| target-fp16-sgemm | 70 | 133.574 | 133.574 |
| other-graph | 1513 | 125.151 | 125.151 |
| other-affinitywave | 1280 | 104.606 | 104.606 |
| memcpy | 5264 | 80.004 | 80.566 |
| flash-attention | 20 | 33.335 | 33.335 |
| convert | 310 | 12.783 | 12.783 |
| dequant | 310 | 9.197 | 9.197 |
| memset | 80 | 0.133 | 0.133 |
