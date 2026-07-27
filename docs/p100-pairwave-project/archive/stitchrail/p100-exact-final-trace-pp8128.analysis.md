# Mapped trace: `p100-exact-final-trace-pp8128.nsys-rep`

Window: 2814.809 ms. Critical GPU: 3. Perfect-balance device-work floor: 2227.638 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2141.684 | 673.124 |
| 1 | 2131.424 | 683.385 |
| 2 | 2279.599 | 535.209 |
| 3 | 2357.843 | 456.966 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 240 |
| Mapped target launches | 240 |
| Target SGEMM summed time | 24.309 ms |
| Mapped SGEMM summed time | 24.309 ms |
| Duration attribution | 100.000% |
| Logical GEMM ranges | 3120 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 60 | 60 | 6.090 |
| 1 | 60 | 60 | 6.071 |
| 2 | 60 | 60 | 6.078 |
| 3 | 60 | 60 | 6.070 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|
| recurrent/recurrent-beta-alpha | 480 | 24.309 | 2.218 | 0.000 | 0.128 |
| full-attention/attention | 240 | 0.000 | 4.594 | 0.000 | 6.136 |
| full-attention/other | 240 | 0.000 | 1.086 | 1.593 | 1.023 |
| full-attention/recurrent-output | 80 | 0.000 | 2.043 | 12.055 | 2.727 |
| full-attention/router | 160 | 0.000 | 0.000 | 0.000 | 0.171 |
| recurrent/attention | 240 | 0.000 | 6.124 | 0.000 | 8.182 |
| recurrent/other | 720 | 0.000 | 3.261 | 4.782 | 3.068 |
| recurrent/recurrent-output | 240 | 0.000 | 6.066 | 36.168 | 8.182 |
| recurrent/recurrent-qkv | 240 | 0.000 | 11.966 | 0.000 | 16.364 |
| recurrent/router | 480 | 0.000 | 0.000 | 0.000 | 0.513 |

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| other-graph | 1333 | 870.013 | 870.097 |
| exact-q8-projection | 1440 | 725.927 | 725.927 |
| flash-attention | 30 | 440.238 | 440.238 |
| memcpy | 804 | 294.415 | 294.957 |
| gdn | 60 | 141.014 | 141.014 |
| other-affinitywave | 640 | 30.685 | 30.685 |
| owner-reduction | 160 | 22.319 | 22.319 |
| convert | 160 | 17.513 | 17.513 |
| dequant | 310 | 9.346 | 9.346 |
| target-fp16-sgemm | 60 | 6.070 | 6.070 |
| owner-sum | 40 | 5.674 | 5.674 |
| memset | 260 | 0.975 | 0.975 |
