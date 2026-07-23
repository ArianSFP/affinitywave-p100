# AffinityWave production performance recovery

Date: 2026-07-23

## Result

The final accurate, real-output pp8128/ub8128 samples are 2518.216 and
2516.122 tok/s, for a mean of 2517.169 tok/s and 0.04% half-range spread.
The corresponding internal wave totals are 3176.664 and 3178.244 ms.

The formal c512 comparison against the matched normal ub128 logits is
unchanged:

| Metric | Result |
| --- | ---: |
| PPL delta | -0.002294 (PASS) |
| KLD | 0.000880 |
| RMS probability delta | 0.748% |
| Same top token | 99.608% |

The final allowed-precision CUDA model-path suite passed 221/221 cases. The
earlier broad focused suite passed 2,172/2,172 cases. No sub-Q8 model or
sub-Q8 optimization was evaluated.

## Performance recovery

The retained 2827.6 tok/s checkpoint is a service-only result with output
withheld and unvalidated model numerics. It is useful as a service target but
is not an end-to-end production comparator.

The accurate production path was recovered by:

- making deterministic route construction parallel instead of serial;
- replacing singleton NCCL calls with a custom BF16 owner sum that preserves
  the accepted NCCL accumulation order;
- pre-capturing and reusing exact CUDA graphs and persistent state snapshots;
- moving safe convolution and GDN state transfer ahead of the corridor;
- retaining FP32 activation transport where reduced-precision variants failed
  the PPL gate;
- replacing quadratic causal-mask setup with a guarded monotonic recurrence.

The mask probe localized 137.6 ms of the reused production graph's outer
interval to `set_inputs()`. The KQ mask accounted for essentially all of it.
At pp8128 the guarded recurrence reduced mask setup from about 144.3 ms to
12.6 ms and raised end-to-end throughput from 2423.9 to 2516-2518 tok/s.

## Prompt-length matrix

| Prompt/ubatch | End-to-end tok/s | Internal tok/s |
| ---: | ---: | ---: |
| 512 | 1185.465 | 1193.3 |
| 1024 | 1715.424 | 1728.5 |
| 2048 | 2253.077 | 2276.5 |
| 4096 | 2575.968 | 2610.6 |
| 8128 | 2517.169 | 2558.1 |

This is a real prompt-ingestion optimization. It does not accelerate
token-by-token decode. Throughput peaks near 4096 tokens on this rig; by 8128
tokens, attention and memory traffic begin to outweigh further fixed-cost
amortization.

## Rejected experiments

- BF16 wire: PPL delta -0.005803.
- FP16 wire: PPL delta -0.017781.
- Power-of-two-scaled FP16 wire (256 and 512): PPL delta -0.008980.
- Forced M64 variants: either failed the +/-0.003 PPL gate or regressed
  performance.
- Four-cell grouping: 7.2% performance regression.
- Direct/shared M16 variants: neutral or slower.

The raw Nsight reports, SQLite exports, logits, and activation captures remain
local and untracked because they total about 1.8 GB. The final trace is
`trace-final-accurate.nsys-rep`; its aggregate kernel report is
`trace-final-accurate.stats.csv`.
