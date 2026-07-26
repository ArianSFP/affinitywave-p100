# Mapped trace: `mapped-reference-pp8128.nsys-rep`

Window: 3261.149 ms. Critical GPU: 3. Perfect-balance device-work floor: 2639.797 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2476.583 | 784.566 |
| 1 | 2519.530 | 741.619 |
| 2 | 2751.726 | 509.423 |
| 3 | 2811.348 | 449.801 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 600 |
| Target SGEMM summed time | 2862.303 ms |
| Mapped SGEMM summed time | 2862.303 ms |
| Duration attribution | 100.000% |
| Logical GEMM ranges | 3120 |
| Mapping collisions | 0 |
| Acceptance | PASS |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 150 | 715.476 |
| 1 | 150 | 150 | 715.469 |
| 2 | 150 | 150 | 715.847 |
| 3 | 150 | 150 | 715.511 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|
| recurrent/recurrent-qkv | 240 | 1093.965 | 12.003 | 18.226 | 16.364 |
| recurrent/attention | 240 | 560.812 | 6.137 | 18.229 | 8.182 |
| recurrent/recurrent-output | 240 | 557.946 | 6.002 | 36.134 | 8.182 |
| full-attention/attention | 240 | 364.559 | 4.779 | 18.213 | 6.136 |
| full-attention/recurrent-output | 80 | 186.152 | 2.031 | 12.056 | 2.727 |
| recurrent/other | 720 | 74.212 | 4.411 | 41.244 | 3.068 |
| full-attention/other | 240 | 24.657 | 1.475 | 13.751 | 1.023 |
| full-attention/router | 160 | 0.000 | 0.000 | 0.000 | 0.171 |
| recurrent/recurrent-beta-alpha | 480 | 0.000 | 2.144 | 36.447 | 0.128 |
| recurrent/router | 480 | 0.000 | 0.000 | 0.000 | 0.513 |

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 1440 | 902.971 | 902.971 |
| target-fp16-sgemm | 150 | 715.511 | 715.511 |
| flash-attention | 30 | 439.352 | 439.352 |
| memcpy | 804 | 285.532 | 285.831 |
| other-graph | 1303 | 241.971 | 242.010 |
| gdn | 30 | 145.271 | 145.271 |
| owner-reduction | 160 | 53.861 | 53.861 |
| convert | 310 | 48.570 | 48.570 |
| other-affinitywave | 640 | 37.332 | 37.332 |
| dequant | 310 | 9.738 | 9.738 |
| owner-sum | 40 | 5.578 | 5.578 |
| memset | 260 | 0.878 | 0.878 |
