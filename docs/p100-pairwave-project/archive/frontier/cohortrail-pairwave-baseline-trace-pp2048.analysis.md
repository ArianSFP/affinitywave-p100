# Mapped trace: `cohortrail-pairwave-baseline-trace-pp2048.nsys-rep`

Window: 1648.104 ms. Critical GPU: 3. Perfect-balance device-work floor: 805.196 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 806.027 | 842.077 |
| 1 | 804.133 | 843.971 |
| 2 | 802.188 | 845.917 |
| 3 | 808.438 | 839.666 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 534.917 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 133.772 |
| 1 | 70 | 0 | 133.901 |
| 2 | 70 | 0 | 133.450 |
| 3 | 70 | 0 | 133.794 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 720 | 319.326 | 319.326 |
| target-fp16-sgemm | 70 | 133.794 | 133.794 |
| other-graph | 1513 | 124.473 | 124.478 |
| other-affinitywave | 1240 | 103.896 | 103.896 |
| memcpy | 5264 | 80.253 | 80.881 |
| flash-attention | 20 | 33.336 | 33.336 |
| convert | 310 | 12.778 | 12.778 |
| dequant | 310 | 9.206 | 9.206 |
| memset | 80 | 0.121 | 0.121 |
