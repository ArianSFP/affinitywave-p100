# Mapped trace: `halfpipe-control-trace-v1.nsys-rep`

Window: 3186.562 ms. Critical GPU: 3. Perfect-balance device-work floor: 2563.441 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2400.067 | 786.495 |
| 1 | 2455.735 | 730.828 |
| 2 | 2660.611 | 525.951 |
| 3 | 2737.353 | 449.209 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2862.045 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 715.540 |
| 1 | 150 | 0 | 715.355 |
| 2 | 150 | 0 | 715.476 |
| 3 | 150 | 0 | 715.674 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 812.763 | 812.763 |
| target-fp16-sgemm | 150 | 715.674 | 715.674 |
| flash-attention | 30 | 440.680 | 440.680 |
| memcpy | 804 | 290.396 | 290.737 |
| other-graph | 1303 | 242.004 | 242.056 |
| gdn | 30 | 144.938 | 144.938 |
| owner-reduction | 160 | 53.893 | 53.893 |
| other-affinitywave | 640 | 48.673 | 48.673 |
| convert | 310 | 48.556 | 48.556 |
| dequant | 310 | 9.794 | 9.794 |
| owner-sum | 40 | 5.574 | 5.574 |
| memset | 260 | 0.897 | 0.897 |
