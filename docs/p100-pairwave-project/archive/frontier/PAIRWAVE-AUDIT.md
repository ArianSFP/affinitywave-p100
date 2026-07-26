# PairWave audit

## Decision boundary

PairWave is a default-off placement and scheduling experiment for the exact
Qwen3.6 prefill path. It does not alter router arithmetic, quantization,
expert arithmetic, BF16 owner boundaries, or canonical group order.

The fixed pairs are `(GPU0,GPU1)` and `(GPU2,GPU3)`. Layers 0-19 alternate
between them. Layer 20 repeats pair 1, after which layers 20-39 alternate
pair 1 and pair 0. Each pair owns 20 layers, including 15 recurrent and
five attention layers.

The pp8128 prompt is divided into two chronological 4064-token panels. Each
panel consists of two 2032-token lanes. Every expert weight is stored once:
the active pair holds all 256 experts for its layer, with two complete
canonical 64-expert groups on each active GPU.

## Measured replay

`pairwave-replay.py` uses the accepted diagonal trace, the measured
BroadWave service attribution, and the final captured route pass. The
current replay output is `pairwave-replay-v3.json`.

It reproduces 7,593,472,000 bytes of traffic:

- 2,661,130,240 bytes of F32 expert requests;
- 2,402,127,872 bytes of canonical BF16 returns;
- 2,530,213,888 bytes of inter-pair hidden handoffs.

The first replay concatenated the two lane route lists before counting
M64/M32/M16 work. That is not numerically valid: the live pooled service
changes 174 layer-0 values and fails the established whole-model accuracy
precedent. `pairwave-replay-v1.json` and its 3016.620 tok/s estimate are
therefore superseded.

The current replay tiles each original 2,032-token lane independently. Its
main boundaries are:

| scenario | wall | rate |
| --- | ---: | ---: |
| literal N256, serialized copies | 3,066.557 ms | 2,650.530 tok/s |
| PairFold N512, 50% copy exposure | 2,817.016 ms | 2,885.323 tok/s |
| PairFold N512, 25% copy exposure | 2,782.101 ms | 2,921.533 tok/s |
| PairFold N512, zero-copy bound | 2,747.179 ms | 2,958.672 tok/s |
| PairFold N2048, zero-copy/tax bound | 2,737.309 ms | 2,969.340 tok/s |

The 3,000 tok/s wall target is 2,709.333 ms. Even the zero-copy,
zero-panel-tax arithmetic bound misses it by 27.975 ms. Full PairFold
scheduler integration is closed until an exact placement or arithmetic
mechanism removes that deficit with useful margin.

The N512 phase-reused arena is 131,017,728 bytes/GPU and reclaims
1,201,193,984 bytes/GPU relative to the legacy full live state.

## Service probes

The 128-descriptor pooled pair shape was measured with two coalesced cells
and the captured edge route distribution. These measurements remain useful
for N-panel arithmetic and copy-engine overlap only; they are not valid
whole-service timings for the retained 256-descriptor lane-exact path.

- Unpanelled HalfPipe control: 17.811386 ms critical instrumented time.
- Strict N256: 20.124117 ms, a 2.312731 ms tax. Closed for PairWave.
- N512 repeats: 18.046387 ms and 18.012384 ms, conservatively a
  0.235001 ms tax.
- N512 plus the exact 16.65 MB/GPU pair-return schedule: 18.003489 ms and
  18.022650 ms. The matched second-run exposure is 0.010266 ms.
- All result checks report zero mismatches.

For every one of the 80 captured layer panels, lane-exact tiles whose rows
are already resident provide more work than the last dispatch: minimum
4.5492x, median 5.5153x, maximum 6.0486x. No tile class is changed. This
supports a ready-tile queue; it does not justify splitting the two canonical
groups into serial service calls.

## Manifest and one-copy loader

`pairwave-laneexact-v3.manifest` is generated from the lane-exact
`pairfold_n512_25pct_copy` replay scenario. Each row records the layer pair,
the physical owner of each canonical group, and the two panel orientations.
The earlier `pairwave-n512-v1.manifest` differs at layer 4 because it was
selected with pooled tile costs.

`GGML_CUDA_AW_PAIRWAVE_MANIFEST=<file>` is default-off. The loader:

1. keeps `GGML_CUDA_MOE_EPLB_MAP` on the router unchanged;
2. composes the manifest group ownership with that router map for expert
   weights only;
3. creates alternating `128/128/0/0` or `0/0/128/128` expert slices;
4. repacks only nonempty slices in place to exact T64.

The serialized load gate
`pairwave-loader-v3-load-aw-p512-ub512` passes. Both the meta loader and
CUDA validator accept all 40 manifest rows. Every GPU registers exactly 60
T64 tensors, equal to 20 owned layers times three projections. This proves
one-copy ownership and a balanced catalog without executing the unfinished
PairWave scheduler.

## Current implementation gate

The pair-local exact primitive now passes, but its measured lane-exact cost
raises the architecture floor above 3,000 tok/s. The next gate is not a
larger scheduler implementation. It is an offline placement replay that may
stream a bounded set of whole exact-T64 experts to the partner GPU and move
only complete original lane tiles. The replay must preserve canonical BF16
slots, charge any route-result traffic required by split execution, include
measured weight-copy exposure, and retain approximately 1 GiB/GPU of the
memory saving.

No implementation proceeds unless that conservative replay is at most
2,709.333 ms with enough margin to absorb runtime effects.

## Real-route exact service gate

The first 128-descriptor implementation pooled the two lanes before exact
tiling. It is rejected. Layer-0 inputs, route IDs, and route weights match
the accepted reference, but 174 of 1,048,576 reduced F32 values change,
with maximum absolute difference 6.103515625e-05.

`pairwave-promotion-audit.py` maps every route back to its original and
pooled tile class. All 137 tokens containing a changed output value select
at least one promoted route; zero changed tokens lie outside that set. The
1,966 route transitions are:

- 860 M16 -> M32;
- 71 M16 -> M64;
- 100 M32 -> M16;
- 910 M32 -> M64;
- 25 M64 -> M32.

The machine-readable result is `pairwave-promotion-audit-v1.json`.

The retained service uses 256 descriptors per active GPU: two lanes times
two canonical groups times 64 experts. Descriptors and pointers are
duplicated, but weights remain single-copy. The device queue still combines
all ready logical tiles in one launch while each lane retains its original
route order and M64/M32/M16 boundaries.

This lane-exact engine passes:

- all four layer-0 inputs, IDs, weights, and reduced outputs byte-for-byte;
- all 160 c512 service boundaries byte-for-byte;
- the complete c512 saved-logits gate, SHA-256
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

That hash is identical to both the current HeadFold and decoupled GroupWave
accepted files. PairWave therefore has a one-copy, exact service primitive.

## Tile-preserving fusion experiments

Two exact fusion kernels tested whether lanes can share weight staging
without merging route rows.

`2xM32` places two unchanged 128-thread M32 cohorts in one 256-thread CTA
and stages B once. It pairs 69.07% of c512 M32 tiles and passes the layer-0
byte-identity gate. The compiled kernel uses 128 registers/thread and
32 KiB shared memory. It is slower at pp2048:

- plain lane-exact: 1555.161945 and 1554.077587 ms;
- 2xM32: 1593.354802 and 1592.923220 ms.

`2xM16` retains the existing two 64-thread cohorts, shares only the B
stage, and reduces resources from 93 registers/20 KiB to
87 registers/12 KiB. It has no stack or local allocation, permits five
CTAs/SM, pairs 70.83% of c512 M16 tails, and is byte-identical at layer 0.
It nevertheless regresses:

- five CTAs/SM: 1631.849784 and 1633.352619 ms;
- three CTAs/SM: 1671.986727 ms.

Both fusions are closed. On GP100, independently issuing the two B loads
inside the existing cohorts is cheaper than coupling their progress behind
larger CTA-wide stage barriers. PairWave must use measured plain
lane-exact tile costs; it cannot claim the earlier route-pooling gain.

## Lane-exact service trace

`pairwave-laneexact-plain-pp2048-trace-v1.nsys-rep` captures two mapping
passes and one timed pass of the retained plain service. The three wave
intervals are 1,820.714, 1,596.165, and 1,574.159 ms. Exact-Q8 aggregate
device work is stable after the first warm-up:

| pass | M64 | M32 | M16 | total |
| --- | ---: | ---: | ---: | ---: |
| mapping 0 | 432.170 ms | 231.060 ms | 619.252 ms | 1,282.483 ms |
| mapping 1 | 425.269 ms | 226.928 ms | 610.845 ms | 1,263.042 ms |
| timed | 425.419 ms | 226.888 ms | 610.837 ms | 1,263.144 ms |

These are sums across four devices, not wall time. The tail classes account
for 837.725 ms, or 66.3% of exact-Q8 device work. Since the exact 2xM32 and
2xM16 kernels regress, future placement models must use this measured
lane-exact work rather than the pooled 128-descriptor service.
