# Mapped trace: `headfold-r44-prefetch-broadwave-v1.nsys-rep`

Window: 4050.923 ms. Critical GPU: 3. Perfect-balance device-work floor: 3063.621 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2899.573 | 1151.350 |
| 1 | 2982.782 | 1068.141 |
| 2 | 3182.683 | 868.240 |
| 3 | 3189.447 | 861.476 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2862.948 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.560 |
| 1 | 150 | 0 | 716.136 |
| 2 | 150 | 0 | 715.661 |
| 3 | 150 | 0 | 715.591 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| memcpy | 11040 | 1117.930 | 1122.390 |
| exact-q8-projection | 1440 | 796.811 | 796.811 |
| target-fp16-sgemm | 150 | 715.591 | 715.591 |
| flash-attention | 30 | 440.883 | 440.883 |
| other-affinitywave | 800 | 262.443 | 262.443 |
| other-graph | 1333 | 242.445 | 242.458 |
| convert | 310 | 48.559 | 48.559 |
| owner-reduction | 320 | 39.407 | 39.407 |
| dequant | 310 | 9.858 | 9.858 |
| owner-sum | 40 | 5.587 | 5.587 |
| memset | 140 | 0.410 | 0.410 |
