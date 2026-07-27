# Mapped trace: `diagonal-panel2048-halfpipe-sync-dense-exact-p8128-trace-v1.nsys-rep`

Window: 2958.189 ms. Critical GPU: 3. Perfect-balance device-work floor: 2382.213 ms.

## Device work

| GPU | Busy union (ms) | Idle in window (ms) |
|---:|---:|---:|
| 0 | 2303.934 | 654.255 |
| 1 | 2282.364 | 675.825 |
| 2 | 2429.251 | 528.938 |
| 3 | 2513.303 | 444.887 |

## Dense mapping acceptance

| Metric | Result |
|---|---:|
| Target SGEMM launches | 0 |
| Mapped target launches | 0 |
| Target SGEMM summed time | 0.000 ms |
| Mapped SGEMM summed time | 0.000 ms |
| Duration attribution | n/a% |
| Logical GEMM ranges | 0 |
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

## Critical-GPU families

| Family | Count | Union (ms) | Summed time (ms) |
|---|---:|---:|---:|
| other-graph | 1453 | 883.425 | 883.545 |
| exact-q8-projection | 1440 | 814.727 | 814.727 |
| flash-attention | 30 | 440.055 | 440.055 |
| memcpy | 804 | 285.338 | 285.902 |
| gdn | 30 | 144.675 | 144.675 |
| other-affinitywave | 800 | 49.390 | 49.390 |
| convert | 310 | 48.554 | 48.554 |
| owner-reduction | 160 | 37.956 | 37.956 |
| owner-sum | 40 | 9.727 | 9.727 |
| dequant | 310 | 9.689 | 9.689 |
| memset | 300 | 1.072 | 1.072 |
