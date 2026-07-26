# Mapped trace: `cohortrail-m64-cursor-ldg128-trace-pp2048.nsys-rep`

Window: 1511.266 ms. Critical GPU: 3. Perfect-balance device-work floor: 744.382 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 744.443 | 766.823 |
| 1 | 741.538 | 769.729 |
| 2 | 740.966 | 770.300 |
| 3 | 750.579 | 760.688 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.553 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.407 |
| 1 | 70 | 0 | 133.600 |
| 2 | 70 | 0 | 133.875 |
| 3 | 70 | 0 | 133.671 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 258.865 | 258.865 |
| target-fp16-sgemm | 70 | 133.671 | 133.671 |
| other-graph | 1513 | 125.537 | 125.537 |
| other-affinitywave | 1280 | 104.654 | 104.654 |
| memcpy | 5264 | 80.430 | 81.011 |
| flash-attention | 20 | 33.323 | 33.323 |
| convert | 310 | 12.783 | 12.783 |
| dequant | 310 | 9.210 | 9.210 |
| memset | 80 | 0.114 | 0.114 |
