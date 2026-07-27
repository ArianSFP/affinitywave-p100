# Mapped trace: `p100-exact-baseline-diagonal-trace-pp8128.nsys-rep`

Window: 2897.116 ms. Critical GPU: 3. Perfect-balance device-work floor: 2306.215 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2223.112 | 674.003 |
| 1 | 2208.640 | 688.475 |
| 2 | 2355.853 | 541.263 |
| 3 | 2437.253 | 459.863 |

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
| full-attention/attention | 240 | 0.000 | 4.767 | 18.209 | 6.136 |
| full-attention/other | 240 | 0.000 | 1.471 | 13.751 | 1.023 |
| full-attention/recurrent-output | 80 | 0.000 | 2.036 | 12.054 | 2.727 |
| full-attention/router | 160 | 0.000 | 0.000 | 0.000 | 0.171 |
| recurrent/attention | 240 | 0.000 | 6.115 | 18.227 | 8.182 |
| recurrent/other | 720 | 0.000 | 4.430 | 41.230 | 3.068 |
| recurrent/recurrent-beta-alpha | 480 | 0.000 | 2.235 | 36.421 | 0.128 |
| recurrent/recurrent-output | 240 | 0.000 | 5.995 | 36.134 | 8.182 |
| recurrent/recurrent-qkv | 240 | 0.000 | 11.975 | 18.223 | 16.364 |
| recurrent/router | 480 | 0.000 | 0.000 | 0.000 | 0.513 |

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| other-graph | 1453 | 883.117 | 883.239 |
| exact-q8-projection | 2880 | 730.317 | 730.317 |
| flash-attention | 30 | 439.044 | 439.044 |
| memcpy | 804 | 286.586 | 287.149 |
| gdn | 30 | 144.706 | 144.706 |
| other-affinitywave | 1280 | 52.877 | 52.877 |
| convert | 310 | 48.555 | 48.555 |
| owner-reduction | 160 | 37.979 | 37.979 |
| dequant | 310 | 9.756 | 9.756 |
| owner-sum | 40 | 9.725 | 9.725 |
| memset | 300 | 1.066 | 1.066 |
