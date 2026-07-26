# Mapped trace: `tailwave-live-pp8128.nsys-rep`

Window: 3302.748 ms. Critical GPU: 3. Perfect-balance device-work floor: 2677.660 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2515.680 | 787.068 |
| 1 | 2554.220 | 748.528 |
| 2 | 2789.193 | 513.555 |
| 3 | 2851.549 | 451.199 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 600 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 2862.819 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | 0.000% |
| Logical GEMM ranges | 0 |
| Mapping collisions | 0 |
| Acceptance | FAIL |

| GPU | Target launches | Mapped | Summed time (ms) |
|---:|---:|---:|---:|
| 0 | 150 | 0 | 716.278 |
| 1 | 150 | 0 | 715.552 |
| 2 | 150 | 0 | 715.662 |
| 3 | 150 | 0 | 715.326 |

## Logical operation families

| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |
|---|---:|---:|---:|---:|---:|

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| exact-q8-projection | 960 | 728.376 | 728.376 |
| target-fp16-sgemm | 150 | 715.326 | 715.326 |
| flash-attention | 30 | 439.895 | 439.895 |
| memcpy | 804 | 287.489 | 287.798 |
| other-affinitywave | 1600 | 252.625 | 252.625 |
| other-graph | 1303 | 242.024 | 242.059 |
| gdn | 30 | 145.131 | 145.131 |
| owner-reduction | 160 | 53.816 | 53.816 |
| convert | 310 | 48.558 | 48.558 |
| dequant | 310 | 9.768 | 9.768 |
| owner-sum | 40 | 5.572 | 5.572 |
| memset | 260 | 0.867 | 0.867 |
