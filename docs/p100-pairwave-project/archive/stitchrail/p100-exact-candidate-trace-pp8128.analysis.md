# Mapped trace: `p100-exact-candidate-trace-pp8128.nsys-rep`

Window: 2833.258 ms. Critical GPU: 3. Perfect-balance device-work floor: 2239.573 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2152.821 | 680.437 |
| 1 | 2143.090 | 690.168 |
| 2 | 2292.280 | 540.978 |
| 3 | 2370.101 | 463.158 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 0 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 0.000 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | n/a% |
| Logical GEMM ranges | 3120 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.000 |
| 1 | 0 | 0 | 0.000 |
| 2 | 0 | 0 | 0.000 |
| 3 | 0 | 0 | 0.000 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|
| full-attention/attention | 240 | 0.000 | 4.717 | 0.000 | 6.136 |
| full-attention/other | 240 | 0.000 | 1.471 | 1.590 | 1.023 |
| full-attention/recurrent-output | 80 | 0.000 | 2.044 | 12.054 | 2.727 |
| full-attention/router | 160 | 0.000 | 0.000 | 0.000 | 0.171 |
| recurrent/attention | 240 | 0.000 | 6.074 | 0.000 | 8.182 |
| recurrent/other | 720 | 0.000 | 4.418 | 4.772 | 3.068 |
| recurrent/recurrent-beta-alpha | 480 | 0.000 | 2.212 | 0.000 | 0.128 |
| recurrent/recurrent-output | 240 | 0.000 | 6.041 | 36.162 | 8.182 |
| recurrent/recurrent-qkv | 240 | 0.000 | 11.869 | 0.000 | 16.364 |
| recurrent/router | 480 | 0.000 | 0.000 | 0.000 | 0.513 |

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| other-graph | 1533 | 891.581 | 891.652 |
| exact-q8-projection | 1440 | 725.582 | 725.582 |
| flash-attention | 30 | 440.273 | 440.273 |
| memcpy | 804 | 291.648 | 292.234 |
| gdn | 60 | 140.819 | 140.819 |
| other-affinitywave | 640 | 30.671 | 30.671 |
| owner-reduction | 160 | 22.403 | 22.403 |
| convert | 80 | 13.645 | 13.645 |
| dequant | 310 | 9.698 | 9.698 |
| owner-sum | 40 | 5.650 | 5.650 |
| memset | 300 | 1.162 | 1.162 |
