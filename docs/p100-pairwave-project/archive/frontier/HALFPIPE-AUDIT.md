# HalfPipe exact BroadWave staging audit

Date: 2026-07-24

Status: `halfpipe_sync` retained as a small exact kernel improvement;
`halfpipe_bar` closed. Neither is an architectural contributor.

## Question

BroadWave computes each M64xN128 output tile as two ordered K16 FP32
accumulator rails. Its original loop prefetches a complete next K32 compact
weight stage into registers, computes both rails, synchronizes the CTA,
expands the complete next stage into shared memory, and synchronizes again.

HalfPipe tests whether whole-warp producer cohorts can refill one 8-KiB
expanded-B half while the other cohort computes the opposite accumulator
rail. It retains:

- exact T64/K32 Q8_0 weights;
- M64xN128 output geometry;
- 256 threads and 32 KiB shared memory;
- two resident CTAs/SM;
- 64 FP32 accumulators/thread;
- 1,024 ordered FFMAs/thread; and
- the final `acc0 + acc1` rounding boundary.

It adds no persistent or transient VRAM.

## `halfpipe_sync`

The safe diagnostic uses full-CTA barriers to establish all buffer lifetimes.
Warps 0-3 produce every output column of B half 0 and warps 4-7 produce every
output column of B half 1. After all warps finish rail 0, the first cohort
refills the next half 0 while the second cohort starts rail 1. At the next
stage, the second cohort refills half 1 while the first cohort starts rail 0.

The compiled resource shape is:

| input | registers/thread | shared memory | stack/local | resident CTAs/SM |
| --- | ---: | ---: | ---: | ---: |
| F32 | 124 | 32,768 bytes | 0 | 2 |
| BF16 | 128 | 32,768 bytes | 0 | 2 |

BroadWave uses 122 and 124 registers/thread respectively. HalfPipe remains
inside the two-CTA resource boundary, including the exact 128-register BF16
case.

The static F32 SASS grows from 1,956 to 2,058 instructions because the two
producer paths are explicit. Shared compute loads remain 64 `LDS.U.128` and
128 `LDS.U.32`; the optimization changes their temporal placement rather than
their count.

### Isolated exact service

Four-cell coalesced edge routing contains 241 M64, 16 M32, and 42 M16 tiles
per projection. All four devices reproduce the 16,777,216-value BF16 owner
oracle with zero mismatches.

| engine | critical instrumented service (ms) | M64 trace median (ms) |
| --- | ---: | ---: |
| BroadWave | 16.7567 | 4.6267 |
| `halfpipe_sync` | 16.5666 | 4.5154 |

The M64 kernel improves 2.41%, but unchanged M32/M16 and service work reduce
the full isolated saving to 0.1901 ms, or 1.13%.

### Full pp8128 graph

Two matched runs per engine give:

| engine | run 1 (ms) | run 2 (ms) | mean (ms) | mean tok/s |
| --- | ---: | ---: | ---: | ---: |
| BroadWave | 3,157.108 | 3,158.552 | 3,157.830 | 2,573.919 |
| `halfpipe_sync` | 3,148.841 | 3,145.569 | 3,147.205 | 2,582.610 |

The measured end-to-end saving is 10.626 ms, or 0.34%. This is real and
consistent with the isolated M64 result, but it is two orders of magnitude
below the 1.116-second gap from this result to 4,000 tok/s.

## `halfpipe_bar`

The second diagnostic replaces gang barriers with four named PTX CTA
barriers:

- half-0 free;
- half-0 filled;
- half-1 free; and
- half-1 filled.

Each 128-thread producer cohort waits only for readers of the half it will
overwrite. The other cohort is allowed to advance to the opposite rail or
the next stage. The implementation is warp-convergent, uses
`bar.arrive <id>, 256` for non-waiting cohorts, and preserves the same output
oracle.

It compiles at 127 F32 or 128 BF16 registers/thread and 32 KiB shared memory,
with no spills. It is nevertheless slower:

| engine | critical instrumented service (ms) | gate/up/down range (ms) |
| --- | ---: | ---: |
| `halfpipe_sync` | 16.5666 | 5.1366-5.2152 |
| `halfpipe_bar` | 17.8876 | 5.5895-5.6321 |

Independent barrier phases and the deliberately skewed half-CTA cohorts
increase scheduler and barrier pressure across the two resident CTAs. The
extra freedom does not create extra FP32 issue capacity. `halfpipe_bar` is
closed.

## Decision

Retain `GGML_CUDA_AW_Q8_ENGINE=halfpipe_sync` as a default-off exact
optimization and as evidence that compact-weight staging has a small
remaining scheduling cost. Do not promote it to the production default at
this stage. Close further split-barrier variants: even perfect recovery of
the safe-form residual cannot provide an architectural-scale saving.

The measured 10.626 ms can be credited in future dependency replays, but it
does not make the retained bounded-memory architecture pass the 1.98-second
integration gate.

## Evidence

- `halfpipe-sync-v1-coalesced-service-p2048-ub2048.out`
- `halfpipe-broad-control-v1-coalesced-service-p2048-ub2048.out`
- `halfpipe-sync-trace-v1.nsys-rep`
- `halfpipe-broad-trace-v1.nsys-rep`
- `halfpipe-sync-resource-v1.out`
- `halfpipe-sync-sass-v1.out`
- `halfpipe-bar-v1-coalesced-service-p2048-ub2048.out`
- `halfpipe-bar-resource-v1.out`
- `halfpipe-sync-live-v1-bench-p8128-ub8128.out`
- `halfpipe-sync-live-v2-bench-p8128-ub8128.out`
- `halfpipe-broad-live-control-v1-bench-p8128-ub8128.out`
- `halfpipe-broad-live-control-v2-bench-p8128-ub8128.out`
