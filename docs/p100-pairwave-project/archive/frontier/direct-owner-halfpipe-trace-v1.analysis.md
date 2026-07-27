# Mapped trace: `direct-owner-halfpipe-trace-v1.nsys-rep`

Window: 3294.855 ms. Critical GPU: 3. Perfect-balance device-work floor: 2633.494 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2465.287 | 829.568 |
| 1 | 2523.533 | 771.322 |
| 2 | 2729.234 | 565.621 |
| 3 | 2815.922 | 478.933 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2863.396 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 716.299 |
| 1 | 150 | 0 | 715.908 |
| 2 | 150 | 0 | 715.556 |
| 3 | 150 | 0 | 715.633 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 812.297 | 812.297 |
| target-fp16-sgemm | 150 | 715.633 | 715.633 |
| flash-attention | 30 | 440.662 | 440.662 |
| other-graph | 1303 | 241.808 | 241.840 |
| memcpy | 644 | 184.029 | 184.029 |
| gdn | 30 | 144.862 | 144.862 |
| owner-sum | 40 | 115.405 | 115.405 |
| owner-reduction | 160 | 53.710 | 53.710 |
| convert | 310 | 48.563 | 48.563 |
| other-affinitywave | 640 | 48.543 | 48.543 |
| dequant | 310 | 9.771 | 9.771 |
| memset | 260 | 0.658 | 0.658 |
