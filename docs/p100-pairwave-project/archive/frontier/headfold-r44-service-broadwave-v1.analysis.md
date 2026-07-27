# Mapped trace: `headfold-r44-service-broadwave-v1.nsys-rep`

Window: 3868.218 ms. Critical GPU: 3. Perfect-balance device-work floor: 3016.830 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2869.818 | 998.400 |
| 1 | 2954.860 | 913.358 |
| 2 | 3093.347 | 774.872 |
| 3 | 3149.297 | 718.921 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2863.213 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.975 |
| 1 | 150 | 0 | 715.941 |
| 2 | 150 | 0 | 715.786 |
| 3 | 150 | 0 | 715.511 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| memcpy | 10960 | 1031.023 | 1037.261 |
| target-fp16-sgemm | 150 | 715.511 | 715.511 |
| exact-q8-projection | 1440 | 711.095 | 711.095 |
| other-affinitywave | 1440 | 446.455 | 446.455 |
| flash-attention | 30 | 439.929 | 439.929 |
| other-graph | 1333 | 242.658 | 242.673 |
| convert | 310 | 48.562 | 48.562 |
| dequant | 310 | 9.862 | 9.862 |
| memset | 140 | 0.413 | 0.413 |
