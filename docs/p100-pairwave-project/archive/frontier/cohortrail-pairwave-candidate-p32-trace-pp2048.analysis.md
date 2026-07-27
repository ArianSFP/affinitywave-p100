# Mapped trace: `cohortrail-pairwave-candidate-p32-trace-pp2048.nsys-rep`

Window: 1674.042 ms. Critical GPU: 3. Perfect-balance device-work floor: 826.324 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 825.228 | 848.813 |
| 1 | 827.227 | 846.815 |
| 2 | 821.109 | 852.932 |
| 3 | 831.733 | 842.309 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.171 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.604 |
| 1 | 70 | 0 | 133.542 |
| 2 | 70 | 0 | 133.547 |
| 3 | 70 | 0 | 133.478 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 337.364 | 337.364 |
| target-fp16-sgemm | 70 | 133.478 | 133.478 |
| other-graph | 1513 | 125.562 | 125.562 |
| other-affinitywave | 1480 | 107.868 | 107.868 |
| memcpy | 5264 | 80.221 | 80.904 |
| flash-attention | 20 | 33.378 | 33.378 |
| convert | 310 | 12.782 | 12.782 |
| dequant | 310 | 9.195 | 9.195 |
| memset | 80 | 0.118 | 0.118 |
