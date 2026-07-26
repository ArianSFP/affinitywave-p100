# ExpandWave exact-weight expansion audit

Date: 2026-07-24

## Question

BroadWave converts signed Q8 bytes to FP32 and multiplies by the per-block
scale while staging every K32 weight tile. ExpandWave tests whether that
conversion work, rather than the ordered FP32 FMA chain, is the Pascal
throughput ceiling.

The diagnostic expands one complete 64-expert down-projection set from exact
T64 Q8_0 into row-major FP32. It then runs an N128 kernel with BroadWave's
same M64 tile, split-K2 partition, per-output FMA order, final add, 32 KiB
shared-memory footprint, and two-CTA/SM residency. M32 and M16 tails remain on
the exact Q8 kernels. The expanded form adds 268,438,528 bytes/GPU to the
service benchmark.

## SASS and resources

| kernel | registers | shared | FFMA | FMUL | signed I2F | global loads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BroadWave Q8 N128 | 122 | 32 KiB | 1,024 | 32 | 24 | 7 |
| ExpandWave FP32 N128 | 123 | 32 KiB | 1,024 | 0 | 0 | 19 |

Both kernels have zero stack/local spill and retain two CTAs/SM. The FP32
variant removes Q8 conversion and scale arithmetic without changing the
irreducible FMA count.

## Matched result

Four-GPU, four-cell coalesced edge-route service, pp2048/cell:

| quantity | BroadWave Q8 | ExpandWave FP32 | change |
| --- | ---: | ---: | ---: |
| down projection | 5.164076 ms | 5.035814 ms | -0.128262 ms (-2.48%) |
| full instrumented service | 16.657062 ms | 16.505006 ms | -0.152056 ms |
| expansion | 0 | 1.095088 ms | +1.095088 ms |
| check mismatches | 0 | 0 | bitwise |

Including expansion, the mechanism regresses by approximately 0.967 ms per
service. Even granting perfectly hidden expansion and applying the full 2.48%
kernel benefit to the live 824.097 ms Q8 slice produces only about 20.5 ms
of pp8128 ceiling. The actual gain is smaller because only M64 work used the
expanded path.

The older N64 FP32-weight control was worse: down increased from 6.294762 to
7.051438 ms before expansion. That control changed geometry and is not used
for the main verdict.

## Decision

ExpandWave is closed. Native Q8 byte conversion and scale multiplication are
not the principal BroadWave bottleneck. The ordered FP32 FMA work dominates.
No gate/up expansion, double-buffered 512 MiB scratch, or full 768 MiB
three-projection cache is justified.

This also closes FP32 weight streaming as an architecture-scale answer under
the user's low-memory requirement. Future exact expert work must reduce
duplicated output work, exploit algebra that is bitwise compatible, or move
critical-path arithmetic across GPUs; merely changing the stored weight
representation cannot meet the target.

Evidence:

- `expandwave-broadwave-control-v2-service-p2048-ub2048.out`
- `expandwave-n128-f32down-v1-service-p2048-ub2048.out`
- `expandwave-legacy-q8-v1-service-p2048-ub2048.out`
- `expandwave-legacy-f32down-v1-service-p2048-ub2048.out`
- `expandwave-sass-summary-v1.txt`
