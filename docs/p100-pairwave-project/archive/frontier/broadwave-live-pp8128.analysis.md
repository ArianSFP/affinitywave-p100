# Mapped trace: `broadwave-live-pp8128.nsys-rep`

Window: 3187.831 ms. Critical GPU: 3. Perfect-balance device-work floor: 2574.837 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2412.790 | 775.040 |
| 1 | 2467.438 | 720.393 |
| 2 | 2672.358 | 515.473 |
| 3 | 2746.762 | 441.069 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2862.482 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.876 |
| 1 | 150 | 0 | 715.531 |
| 2 | 150 | 0 | 715.634 |
| 3 | 150 | 0 | 715.441 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 824.097 | 824.097 |
| target-fp16-sgemm | 150 | 715.441 | 715.441 |
| flash-attention | 30 | 440.133 | 440.133 |
| memcpy | 804 | 289.773 | 290.095 |
| other-graph | 1303 | 242.024 | 242.076 |
| gdn | 30 | 145.152 | 145.152 |
| owner-reduction | 160 | 53.877 | 53.877 |
| other-affinitywave | 640 | 48.804 | 48.804 |
| convert | 310 | 48.549 | 48.549 |
| dequant | 310 | 9.782 | 9.782 |
| owner-sum | 40 | 5.587 | 5.587 |
| memset | 260 | 0.918 | 0.918 |
