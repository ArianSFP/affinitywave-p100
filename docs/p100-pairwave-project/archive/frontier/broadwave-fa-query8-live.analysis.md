# Mapped trace: `broadwave-fa-query8-live.nsys-rep`

Window: 3352.912 ms. Critical GPU: 3. Perfect-balance device-work floor: 2624.965 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2424.576 | 928.335 |
| 1 | 2496.718 | 856.194 |
| 2 | 2736.396 | 616.516 |
| 3 | 2842.172 | 510.740 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2863.432 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.921 |
| 1 | 150 | 0 | 715.655 |
| 2 | 150 | 0 | 716.001 |
| 3 | 150 | 0 | 715.854 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 826.515 | 826.515 |
| target-fp16-sgemm | 150 | 715.854 | 715.854 |
| flash-attention | 30 | 532.375 | 532.375 |
| memcpy | 804 | 286.910 | 287.240 |
| other-graph | 1303 | 241.876 | 241.923 |
| gdn | 30 | 144.886 | 144.886 |
| owner-reduction | 160 | 53.916 | 53.916 |
| other-affinitywave | 640 | 48.836 | 48.836 |
| convert | 310 | 48.548 | 48.548 |
| dequant | 310 | 9.798 | 9.798 |
| owner-sum | 40 | 5.573 | 5.573 |
| memset | 260 | 0.868 | 0.868 |
