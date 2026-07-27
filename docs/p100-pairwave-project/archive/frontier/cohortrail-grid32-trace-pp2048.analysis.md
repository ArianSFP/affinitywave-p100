# Mapped trace: `cohortrail-grid32-trace-pp2048.nsys-rep`

Window: 1614.981 ms. Critical GPU: 3. Perfect-balance device-work floor: 795.781 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 796.017 | 818.964 |
| 1 | 795.325 | 819.656 |
| 2 | 791.261 | 823.720 |
| 3 | 800.521 | 814.460 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 280 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 535.968 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 70 | 0 | 134.117 |
| 1 | 70 | 0 | 133.986 |
| 2 | 70 | 0 | 133.995 |
| 3 | 70 | 0 | 133.870 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 306.453 | 306.453 |
| target-fp16-sgemm | 70 | 133.870 | 133.870 |
| other-graph | 1513 | 125.478 | 125.479 |
| other-affinitywave | 1480 | 107.763 | 107.763 |
| memcpy | 5264 | 80.178 | 80.732 |
| flash-attention | 20 | 33.323 | 33.323 |
| convert | 310 | 12.781 | 12.781 |
| dequant | 310 | 9.203 | 9.203 |
| memset | 80 | 0.122 | 0.122 |
