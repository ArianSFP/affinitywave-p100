# Mapped trace: `diagonal-panel512-p2048-trace-v1.nsys-rep`

Window: 1194.301 ms. Critical GPU: 0. Perfect-balance device-work floor: 893.359 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 898.070 | 296.232 |
| 1 | 885.525 | 308.777 |
| 2 | 895.364 | 298.938 |
| 3 | 894.480 | 299.822 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.572 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 134.027 |
| 1 | 70 | 0 | 133.411 |
| 2 | 70 | 0 | 133.615 |
| 3 | 70 | 0 | 133.519 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 2880 | 451.064 | 451.064 |
| target-fp16-sgemm | 70 | 134.027 | 134.027 |
| other-graph | 1543 | 125.964 | 126.037 |
| memcpy | 1360 | 86.313 | 86.349 |
| gdn | 30 | 34.538 | 34.538 |
| flash-attention | 20 | 33.320 | 33.320 |
| other-affinitywave | 800 | 19.736 | 19.736 |
| convert | 310 | 12.801 | 12.801 |
| owner-reduction | 640 | 12.323 | 12.323 |
| dequant | 310 | 9.322 | 9.322 |
| owner-sum | 160 | 4.594 | 4.594 |
| memset | 200 | 0.409 | 0.409 |
