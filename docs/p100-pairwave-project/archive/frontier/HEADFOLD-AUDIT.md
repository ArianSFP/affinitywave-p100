# HeadFold exact placement audit

Date: 2026-07-24

## Decision

HeadFold is retained as the first replacement placement in this campaign
with a measured, exact path through 3,000 tok/s. It assigns eight complete
Gated DeltaNet heads to each GPU while keeping the base hidden activation
sequence-sharded. The chronological recurrence is unchanged. Exact ring
transport is overlapped with output-channel QKV panels, and each completed
2032-token recurrence segment is redistributed while the next segment runs.

The calibrated central replay predicts:

- 2,645.432 ms;
- 3,072.47 tok/s;
- about 165.2 MB/GPU of conservative peak ephemeral storage;
- no persistent expert replica cache.

The deliberately conservative replay predicts 2,744.306 ms, or 2,961.77
tok/s. A full implementation is therefore justified for a production
3,000 tok/s attempt, but not yet for 3,500 or 4,000 tok/s.

## Exact local recurrence gate

The first complete-head layout used the current two-column GDN kernel:

| layout | kernel time |
| --- | ---: |
| current H32/T2032 | 5.46481 ms |
| HeadFold H8/T8128, two columns/warp | 10.35341 ms |
| HeadFold H8/T8128, one column/warp | 7.50614 ms |

The one-column kernel doubles the number of independent CTAs and removes
most of the long-recurrence underfill. Splitting the same recurrence into
four H8/T2032 launches improves it further:

| implementation | time |
| --- | ---: |
| monolithic H8/T8128 | 7.53016 ms |
| four H8/T2032 segments | 7.25270 ms |

The split is bitwise exact. Both runs used the same deterministic test
input, and their complete 33,816,576-byte output plus final-state images
have SHA-256:

`2bd7407fef38e21b52cd4ff9a81f2a37e4e43c19811090c917711684a3a160dc`

An earlier pair of dumps differed because the stock backend test initializes
each process from `std::random_device`. That comparison is invalid, is not
used in any decision, and is retained as a harness lesson. A default-off
fixed-seed test selector repaired the control.

## Layout-faithful transport

The first exact transport probe moved the actual tensor byte counts and
performed the required token/head interleave:

| operation | time |
| --- | ---: |
| FP32 hidden all-gather | 7.96908 ms |
| staged head redistribution plus transpose | 4.14824 ms |
| pitched direct head redistribution | 3.98209 ms |
| direct peer-load gather kernel | 2.42872 ms |

All modes reported zero errors. The naive best corridor is 10.39144 ms per
GDN layer, which is too expensive. Its useful result is the direct peer
gather and the exact tensor layout; it is not the retained schedule.

## Exact QKV ring pipeline

The production-shaped dense audit had already established that Pascal
algorithm 6 produces four 2048-output panels that are bitwise identical to
the monolithic 8192-output selector 99. HeadFold assigns one such panel to
each GPU.

A three-step FP32 ring all-gather was composed with four sequential
2032-token panel GEMMs per GPU:

| quantity | time |
| --- | ---: |
| ring alone | 4.03777 ms |
| four exact panel GEMMs | 9.25414 ms |
| ring plus panel pipeline | 9.28147 ms |
| exposed transport | 0.02734 ms |

Validation covered:

- 66,584,576 gathered hidden values, zero errors;
- 66,584,576 FP32 QKV output values against monolithic selector 99, zero
  errors.

The minimal reusable pipeline scratch is 58,261,504 bytes/GPU. Each GPU
stores only its 8,388,608-byte dense weight panel in the probe. The
production design can shard these existing dense weights instead of adding
a cache.

## Exact recurrent-output pipeline

The recurrence was specialized only for measurement with the same
one-column FP32 operation order as the production kernel. Each head GPU
records a completion event after every 2032-token segment. The destination
sequence GPU uses pitched peer copies to write each 1024-feature head shard
directly into the 4096-feature recurrent-output layout while the next
segment scans.

| quantity | time |
| --- | ---: |
| four recurrence segments | 7.43165 ms |
| recurrence plus output redistribution | 9.67703 ms |
| exposed output transport | 2.24537 ms |

Validation covered:

- 33,816,576 monolithic-versus-segmented recurrence/output-state values,
  zero errors;
- 33,292,288 redistributed output values, zero errors.

Combining the measured production operator recurrence, 7.25270 ms, with the
measured exposed output cost gives a replay value of 9.49807 ms/layer.

## Bulk-synchronous HeadFold replay

The replay uses the measured BroadWave layer attribution and the per-layer
token route model. A layer consists of:

1. concurrent sequence-local dense work;
2. exact HeadFold GDN or balanced-attention epoch;
3. same-layer exact R16 expert pooling;
4. measured serialized Q8/PCIe service;
5. concurrent sequence-local post work.

It does not credit unmeasured panel release or cross-layer overlap. Exact
dense selectors are retained. Attention uses the measured 34.664323 ms
bitwise epoch. R16 uses the measured route distribution, 803.635 ms total
Q8 model, 163.521 ms total communication model, and long-corridor weight
exposure centrally.

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R0 central | 2,923.824 ms | 2,779.92 tok/s |
| HeadFold R16 central | 2,645.432 ms | 3,072.47 tok/s |
| HeadFold R16 conservative | 2,744.306 ms | 2,961.77 tok/s |
| exact-Q8 at P100 FP32 peak bound | 2,340.915 ms | 3,472.15 tok/s |
| exact-Q8 at P100 FP16 peak bound | 2,090.021 ms | 3,888.96 tok/s |
| zero-cost Q8 bound | 1,841.797 ms | 4,413.08 tok/s |

The conservative case keeps the worst measured attention residual, applies
the 2.36% balanced-layout token imbalance to attention-layer service, and
charges the short R16 streaming exposure at all attention layers.

The central target requirements are:

| target | required exact-Q8 factor | speedup | effective throughput |
| --- | ---: | ---: | ---: |
| 3,000 tok/s | 1.0795 | already inside central model | 5.35 TF/s |
| 3,500 tok/s | 0.5979 | 1.673x | 9.66 TF/s |
| 4,000 tok/s | 0.2367 | 4.225x | 24.40 TF/s |

The 4,000 requirement is above even the P100 FP16 arithmetic peak if only
the expert term changes. The next program must therefore improve both exact
expert arithmetic and the 1.842-second non-Q8 floor.

## Memory boundary

The 1.992 GiB/GPU persistent R16 cache from the route model is rejected.
Only the measured ephemeral form is retained:

- HeadFold QKV ring scratch: 58,261,504 bytes/GPU;
- active R16 exact-T64 panel: 53,477,376 bytes/GPU;
- double-buffered R16 exact-T64 panels: 106,954,752 bytes/GPU;
- conservative simultaneous peak: 165,216,256 bytes/GPU.

Buffers are phase-reusable. There is no multi-GiB cache and no expert down
weight replication.

## Evidence

- `headfold-gdn-h8-t8128-v2.out`
- `headfold-gdn-h8-t8128-onecol-v1.out`
- `headfold-gdn-h8-t2032-onecol-v1.out`
- `headfold-gdn-h8-t8128-monolithic-fixedseed-v1.out`
- `headfold-gdn-h8-t8128-segmented-fixedseed-v1.out`
- `headfold-gdn-monolithic-fixedseed-v1.bin`
- `headfold-gdn-segmented-fixedseed-v1.bin`
- `headfold-transport-probe.cu`
- `headfold-transport-v1.out`
- `headfold-qkv-panel-selector-v1.out`
- `headfold-qkv-pipeline-probe.cu`
- `headfold-qkv-ring-pipeline-v1.out`
- `headfold-gdn-output-pipeline-probe.cu`
- `headfold-gdn-output-pipeline-v1.out`
- `headfold-replay.py`
- `headfold-replay-v1.json`

## Wider ephemeral placement after HeadFold

The original R24 closure assumed that a streamed expert panel had only one
9.1 ms QKV projection in which to arrive. HeadFold makes layer execution
bulk-synchronous and exposes a measured 32.6 ms recurrent pre-phase and
48.1 ms attention pre-phase. That is materially new scheduling evidence, so
R24 and larger ephemeral panels were retested. Persistent replication remains
closed.

The first control used one production-shaped M8192/N2032/K2048 FP32-accumulate
GEMM:

| panel | exact-T64 bytes/GPU | copy | compute | concurrent | exposed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| R24 | 80,216,064 | 12.778 ms | 9.120 ms | 12.440 ms | 3.319 ms |
| R32 | 106,954,752 | 16.962 ms | 9.139 ms | 17.417 ms | 8.279 ms |
| R40 | 133,693,440 | 18.858 ms | 9.120 ms | 17.884 ms | 8.764 ms |
| R44 | 147,062,784 | 23.051 ms | 9.115 ms | 20.352 ms | 11.237 ms |

This reproduces the old short-corridor rejection. The HeadFold-length control
then used three sequential exact production-shaped GEMMs, 27.35-27.38 ms:

| panel | copy | compute | concurrent | exposed | validation errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| R24 | 12.016 ms | 27.375 ms | 27.400 ms | 0.025 ms | 0 |
| R32 | 16.064 ms | 27.345 ms | 27.407 ms | 0.062 ms | 0 |
| R40 | 20.197 ms | 27.357 ms | 27.422 ms | 0.065 ms | 0 |
| R44 | 18.863 ms | 27.380 ms | 27.396 ms | 0.016 ms | 0 |

The captured placement's worst source skew was also reconstructed rather than
assuming balanced sources:

| panel | source expert counts | copy | compute | concurrent | exposed |
| ---: | --- | ---: | ---: | ---: | ---: |
| R32 | 27, 33, 16, 52 | 17.146 ms | 27.370 ms | 27.371 ms | 0.001 ms |
| R44 | 31, 38, 43, 64 | 23.695 ms | 27.369 ms | 27.412 ms | 0.043 ms |

All destination regions passed first/last-byte validation. The full R44
owner-partial oracle then checked all 40 layers at 2,048 columns:

- 2,663,383,040 owner values;
- 665,845,760 final values;
- 294,174 primary groups;
- 688,279 cached groups;
- 193,316 remote groups;
- zero assignment errors;
- zero owner mismatches;
- zero final mismatches.

R44 is retained as an ephemeral upper boundary, not a persistent cache. It
uses 147,062,784 bytes/GPU active or 294,125,568 bytes/GPU double buffered.
Including the 58,261,504-byte HeadFold ring scratch gives a conservative
simultaneous peak of 352,387,072 bytes/GPU. Buffers remain phase-reusable.

The provisional row-uniform v2 replay was:

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R16 central | 2,645.432 ms | 3,072.47 tok/s |
| HeadFold R32 central | 2,521.626 ms | 3,223.32 tok/s |
| HeadFold R44 central | 2,472.221 ms | 3,287.73 tok/s |
| HeadFold R44 conservative | 2,546.511 ms | 3,191.82 tok/s |
| R44 exact-Q8 at nominal FP32 peak | 2,198.196 ms | 3,697.58 tok/s |
| R44 exact-Q8 at nominal FP16 peak | 1,972.426 ms | 4,120.81 tok/s |

The conservative replay charges 0.5 ms/layer of exposed streaming, over
eleven times the worst measured full-pre value, and retains the worst
attention residual plus attention-layout service imbalance.

After R44, 3,500 tok/s requires about 7.29 effective exact-Q8 TFLOP/s, a
1.262x improvement over BroadWave. Reaching 4,000 by Q8 alone would still
require 14.76 TFLOP/s. R44 therefore changes the 3,500 feasibility boundary,
but 4,000 still requires faster exact arithmetic plus another reduction in
the fixed floor.

### Tail-calibrated correction

The row-uniform Q8 model was then replaced with an overdetermined live fit.
Uniform, tail-mix, and captured edge-mix routes were measured with one-, two-,
and four-cell pooling. All runs passed the exact service oracle. A least
squares fit over 36 device observations gives these three-projection costs:

- M64: 57.666 us/tile, or 0.901 us/issued row;
- M32: 38.904 us/tile, or 1.216 us/issued row;
- M16: 25.717 us/tile, or 1.607 us/issued row;
- fit RMSE: 0.096 ms per complete three-projection service.

R44 creates more cached-expert tails, so its Q8 estimate rises from 723.163
to 749.770 ms. The v3 replay supersedes v2:

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R16 central | 2,656.346 ms | 3,059.84 tok/s |
| HeadFold R32 central | 2,541.352 ms | 3,198.30 tok/s |
| HeadFold R44 central | 2,498.828 ms | 3,252.73 tok/s |
| HeadFold R44 conservative | 2,573.271 ms | 3,158.63 tok/s |
| R44 exact-Q8 at nominal FP32 peak | 2,214.721 ms | 3,669.99 tok/s |
| R44 zero-Q8 bound | 1,749.058 ms | 4,647.07 tok/s |

The corrected 3,500 boundary is 7.55 effective TFLOP/s, or 1.308x
BroadWave. The Q8-only 4,000 requirement is 15.31 TFLOP/s and remains
impossible with exact FP32 FMA chains.

Budgets through R96 were also modeled with the corrected tail costs. The
combined Q8 plus communication stage reaches 819.9, 783.9, 774.0, and
774.1 ms at R44, R64, R80, and R96. R64-R96 therefore buy only another
36-46 ms while raising conservative transient peak storage to
486-700 MB/GPU. Tail-aware enumeration of all 24 physical owner
permutations saves only 0.675 ms at R44. R44 is the measured memory/performance
knee; wider ephemeral placement is closed.

## Exact BroadWave arithmetic ceiling

A progressive CUDA-event probe retained BroadWave's 64 FP32 accumulators and
1,024 ordered FFMAs per K32 stage. It then added shared operands, barriers,
and staging in isolation. The no-spill results on the slowest GPU were:

| mechanism | effective throughput |
| --- | ---: |
| register operand control | 7.19 TFLOP/s |
| shared operands plus barriers | 7.70 TFLOP/s |
| shared operands plus exact-size staging | 7.01 TFLOP/s |

The barrier result being faster than the register control is a compiler
scheduling effect, not a claim that barriers accelerate arithmetic. The
relevant comparison is 7.70 versus 7.01 TFLOP/s: removing every staging
store leaves only about 10% headroom, below the corrected 1.308x requirement.

RegisterBWave tested the remaining radical alternative. It kept the exact
M64xN128 accumulator order and double-buffered shared A tile, but loaded and
decoded compact T64 weights separately in each row warp. This removed
two-thirds of the shared loads, half the barriers, and reduced shared memory
to 16 KiB. It compiled at 127 registers with no spills and passed
16,777,216 BF16 comparisons per GPU with zero mismatches.

The duplicated weight decode lost decisively:

| engine | coalesced three-projection stage |
| --- | ---: |
| BroadWave | about 16.6 ms |
| RegisterBWave | 18.72-18.85 ms |

RegisterBWave is closed. Together with ExpandWave and RailWave, the result
shows that exact BroadWave is already close to the practical Pascal FP32
dataflow ceiling. The 4,000 program must lower the 1.46-second HeadFold
pre-phase as well as improve Q8.

Additional evidence:

- `streamcache-r24-dense-m8192-headfold-v1.out`
- `streamcache-r32-dense-m8192-headfold-v1.out`
- `streamcache-r40-dense-m8192-headfold-v1.out`
- `streamcache-r44-dense-m8192-headfold-v1.out`
- `streamcache-r24-dense3-m8192-headfold-v1.out`
- `streamcache-r32-dense3-m8192-headfold-v1.out`
- `streamcache-r40-dense3-m8192-headfold-v1.out`
- `streamcache-r44-dense3-m8192-headfold-v1.out`
- `streamcache-r32-skew52-dense3-m8192-headfold-v1.out`
- `streamcache-r44-skew64-dense3-m8192-headfold-v1.out`
- `streamcache-owner-r32-all-c2048-v1.out`
- `streamcache-owner-r44-all-c2048-v1.out`
- `headfold-replay-v2.json`
- `headfold-replay-v3.json`
- `cache-relay-model-frequency-wide-v1.json`
- `broadwave-cal-uniform-pool1-v1-service-p2048-ub2048.out`
- `broadwave-cal-uniform-pool2-v1-service-p2048-ub2048.out`
- `broadwave-cal-uniform-pool4-v1-service-p2048-ub2048.out`
- `broadwave-cal-tail-mix-pool1-v1-service-p2048-ub2048.out`
- `broadwave-cal-tail-mix-pool2-v1-service-p2048-ub2048.out`
- `broadwave-cal-tail-mix-pool4-v1-service-p2048-ub2048.out`
- `broadwave-ceiling-probe.cu`
- `broadwave-ceiling-v3.out`
- `regbwave-v1-coalesced-service-p2048-ub2048.out`

## Fine-grained exact recurrent output publication

The original HeadFold output probe published four 2,032-token recurrent
segments. A runtime sweep retained the same chronological FP32 recurrence
and the same four 2,032-token sequence destinations, but divided each
destination into one, two, four, or eight publication parts. Each source
head panel is copied only after its corresponding chronological segment
event; no arithmetic is reordered.

| parts/destination | total segments | tokens/segment | recurrence | recurrence + output | exposed output |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 2,032 | 7.597 ms | 10.204 ms | 2.607 ms |
| 2 | 8 | 1,016 | 7.297 ms | 8.654 ms | 1.358 ms |
| 4 | 16 | 508 | 7.392 ms | 8.189 ms | 0.796 ms |
| 8 | 32 | 254 | 7.623 ms | 8.177 ms | 0.554 ms |

Two 200-iteration confirmations of the 16-segment form measured 8.183 and
8.199 ms complete. The 32-segment form measured 8.162 ms, only 0.029 ms
below the 16-segment mean and inside run variation while doubling launches
and events. The retained boundary is therefore 16 segments, or four
508-token publications per destination.

Every run checked the complete recurrent output, final state, and
redistributed sequence output:

- 33,816,576 recurrence/state values per run;
- 33,292,288 transported values per run;
- zero bitwise mismatches.

The retained 8.191 ms value lowers the modeled 30-layer recurrent path by
39.2 ms relative to replay v3. Tail-calibrated replay v4 is:

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R16 central | 2,617.143 ms | 3,105.68 tok/s |
| HeadFold R16 conservative | 2,716.079 ms | 2,992.55 tok/s |
| HeadFold R44 central | 2,459.625 ms | 3,304.57 tok/s |
| HeadFold R44 conservative | 2,534.068 ms | 3,207.49 tok/s |
| R44 exact-Q8 at nominal FP32 peak | 2,175.518 ms | 3,736.12 tok/s |
| R44 zero-Q8 bound | 1,709.855 ms | 4,753.62 tok/s |

At R44, 3,500 tok/s now requires 7.07 effective exact-Q8 TFLOP/s, or a
1.224x speedup over the calibrated BroadWave rate. Reaching 4,000 tok/s by
Q8 alone still requires 13.44 TFLOP/s. The fixed floor must fall as part of
the architecture.

Evidence:

- `headfold-gdn-output-pipeline-p1-v2.out`
- `headfold-gdn-output-pipeline-p2-v2.out`
- `headfold-gdn-output-pipeline-p4-v2.out`
- `headfold-gdn-output-pipeline-p8-v2.out`
- `headfold-gdn-output-pipeline-p4a-v2.out`
- `headfold-gdn-output-pipeline-p4b-v2.out`
- `headfold-gdn-output-pipeline-p8a-v2.out`
- `headfold-replay-v4.json`

## P100 clock and power audit

The arithmetic ceiling is not an accidental board-power ceiling. During an
unperturbed 1,000-repeat ceiling run all four P100s reached 1,328 MHz, stayed
well below the 250 W power limit, and produced these slowest-device rates:

| mechanism | sustained rate |
| --- | ---: |
| register-only ordered FP32 FMA | 7.874 TFLOP/s |
| shared operands plus barriers | 7.705 TFLOP/s |
| shared operands plus exact-size staging | 7.011 TFLOP/s |

Rapid `nvidia-smi` polling was found to perturb this probe severely and is not
valid while timing. The affected artifact is not used in any replay.

Forcing application clocks to 715/1,328 MHz made the four devices more uniform,
but did not materially accelerate the measured BroadWave stages. The locked
run measured 16.580-16.589 ms of instrumented stage time. Two default-clock
runs measured 16.583-16.739 and 16.635-16.714 ms. Thus the retained kernel gain
is at most about 1%, while the apparent 7.5% change in one host-side elapsed
sample was synchronization skew. Application clocks were restored to the
default 715/1,189 MHz setting after the test.

Clock forcing is closed as a performance mechanism. It may still be useful as
a measurement-control setting, but no architectural replay credits it with a
speedup.

Evidence:

- `broadwave-ceiling-sustained-v1.out`
- `clock-default-v1-service-p2048-ub2048.out`
- `clock-default-v2-service-p2048-ub2048.out`
- `clock-locked1328-v1-service-p2048-ub2048.out`

## Exact balanced-attention dense boundary

The balanced attention epoch changes each GPU's Q/K/V token count from 2,032
to 1,984, 2,080, 2,016, or 2,048. Because cuBLAS may change its internal K
reduction when N changes, row independence alone was not accepted as an
accuracy argument.

A device oracle first computed four original 2,032-token Q projections with
production selector 99. It then packed the exact aligned chunk assignment and
tested every legal Pascal cuBLAS selector at the four balanced N values.
Algorithm 5 is the retained Q selector:

- 66,584,576 FP32 Q values checked;
- zero bitwise mismatches;
- 8.519 ms worst balanced call, versus 9.705 ms for selector 99.

The corresponding M512 K/V audit checked 4,161,536 values with zero
mismatches for every successful selector. Algorithm 3 measured 0.642 ms at
the worst balanced batch. This closes a previously untested accuracy hole in
the ScanFold-derived attention epoch without adding storage.

Evidence:

- `headfold-attention-q-exactness-probe.cu`
- `headfold-attention-q-exactness-v1.out`
- `headfold-attention-kv-exactness-v1.out`

## Exact selective-repair audit for HFMA2

The rejected K16 integer-HFMA2 path was reconsidered only as a speculative
predictor for an exact fast-path-plus-repair engine. A report-only form of the
existing BF16 oracle measured 888,832 mismatches out of 16,777,216 synthetic
owner-partial elements (5.30%) while the approximate stage ran in about
14.0 ms. K8 was worse at 1,859,584 mismatches (11.08%).

The decisive test captured the complete layer-0 owner partials from exact
BroadWave and K16 post-scale HFMA2 on identical live inputs:

- 16,777,216 BF16 values compared;
- 1,700,556 mismatches (10.136%);
- 4,446 of 8,192 token/owner rows contain at least one mismatch (54.27%);
- up to 620 of 2,048 columns differ in one row.

The errors are small in magnitude but not sparse in the dependency dimension
needed for repair. Any row-exact repair must recompute gate and up for at
least 54% of rows after already paying the 14.0 ms approximate pass. The
opposite construction, exact gate/up followed by approximate down and a
perfect 10.14%-density down repair, has an optimistic floor around 15.3 ms
before detection and compaction, versus about 16.6 ms for BroadWave. That is
at most an incremental gain and cannot justify the additional engine.

Selective HFMA2 repair is closed. The live captures remain useful evidence
if a future hardware backend offers a genuinely cheap exact residual.

Evidence:

- `fab-k16-qscale-exactcheck-report-v1-service-p2048-ub2048.out`
- `fab-k8-qscale-exactcheck-report-v1-service-p2048-ub2048.out`
- `repair-broadwave-dump-layer0-v1-bench-p2048-ub2048.out`
- `repair-fab-k16-dump-layer0-v1-bench-p2048-ub2048.out`
- `fab-k16-live-layer0-bf16-compare-v1.jsonl`
- `compare-bf16-dumps.cpp`

## CrestWave M128 exact-Q8 geometry

R44 produces enough pooled rows per expert that an M128 tile can often replace
two BroadWave M64 tiles. CrestWave tested this geometry directly:

- one M128xN128 CTA per SM;
- 512 threads and 16 resident warps;
- the same two ordered FP32 accumulator chains and final add as BroadWave;
- each exact T64 weight stage decoded once for up to 128 rows;
- 118 registers for F32 input or 120 for BF16 input;
- 49,152 bytes shared memory;
- no stack or local spills.

The edge-route service oracle passed all 16,777,216 BF16 owner-partial values
on every GPU. Performance lost decisively:

| stage | BroadWave | CrestWave |
| --- | ---: | ---: |
| gate | about 5.26 ms | 6.10 ms |
| up | about 5.26 ms | 6.10 ms |
| down | about 5.16 ms | 5.82 ms |
| complete instrumented service | about 16.65 ms | 18.98-19.01 ms |

Although both layouts retain 16 resident warps/SM, BroadWave has two
independent eight-warp CTAs. One can issue while the other is crossing a
K-stage barrier. CrestWave has a single 16-warp CTA at the 48 KiB
shared-memory limit, so every stage barrier drains the SM. Halving weight
decode does not compensate for losing inter-CTA latency hiding.

CrestWave is closed. M96/M80 variants would retain the one-CTA shared-memory
limit with fewer resident warps, and M128/N64 would duplicate the activation
tile while weight decode is already measured as a minor cost. Those
incremental geometries are not pursued.

Evidence:

- `crestwave-v1-service-p2048-ub2048.out`
- `ggml/src/ggml-cuda/affinity-wave.cu`

## Complete exact-Q8 weight-structure scan

A sequential GGUF scan inspected every byte of every expert gate, up, and
down tensor. This was not a sample:

- 123 tensors, 41 stored blocks, and 35,081,158,656 expert-weight bytes;
- 1,031,798,784 native Q8_0 blocks and 33,017,561,088 signed Q values;
- 1.8440% individual zero Q values;
- 0.7333% all-zero 32-value blocks;
- 0.7269% adjacent equal-value blocks, almost entirely zero runs.

The apparent structure is concentrated in stored block 0. Its gate and up
tensors each contain 41.3948% all-zero blocks, while its down tensor contains
no all-zero blocks. Across the 40-block prefill trunk, excluding block 0,
only 0.0631% of blocks are all zero and only 0.8461% of individual Q values
are zero. The down family contains no all-zero block anywhere.

An exact block-sparse kernel can therefore avoid at most 0.7514% of trunk
block work before metadata, branching, packing, and load-imbalance costs.
Even an impossible zero-cost per-value sparse representation has only a
1.8706% arithmetic ceiling. This is below the architectural gate by two
orders of magnitude. Weight sparsity, dictionary coding, and repeated-block
reuse are closed unless a different model has materially different measured
structure.

Evidence:

- `q8-weight-structure-probe.cpp`
- `q8-weight-structure-v1.jsonl`
- `q8-weight-structure-derived-v1.json`

## Stored block 40 is MTP, not a prefill trunk layer

The complete weight scan found expert tensors `blk.0` through `blk.40`, while
AffinityWave and the replay operate on 40 cells. A direct metadata probe
resolved the discrepancy:

```text
qwen35moe.block_count=41
qwen35moe.nextn_predict_layers=1
```

The runtime defines trunk depth as `n_layer_all - n_layer_nextn`, so ordinary
prefill executes 40 trunk layers and the appended block is selected only by
the separate MTP graph. The 30 recurrent plus 10 attention layer accounting
in HeadFold replay v4 is therefore complete and is not missing a production
layer.

Evidence:

- `gguf-layer-metadata-probe.cpp`
- `gguf-layer-metadata-v1.out`

## Complete recurrent dependency and replay v5

The initial QKV ring probe treated the 8,192 output rows as four contiguous
panels. The actual complete-head placement is not contiguous: Q contributes
four 128-channel heads, K contributes four 128-channel heads, and V contributes
eight 128-channel heads to each GPU. The probe was rebuilt with that exact
512-Q plus 512-K plus 1,024-V packing.

| quantity | complete-head result |
| --- | ---: |
| three-step hidden ring | 4.041596 ms |
| four exact packed-head GEMMs | 9.269259 ms |
| ring plus packed-head pipeline | 9.243291 ms |
| hidden values checked | 66,584,576 |
| QKV values checked | 66,584,576 |
| bitwise mismatches | 0 |

The separate GDN z projection is M4096/N2032/K2048. Four M1024
output-channel panels preserve the monolithic FP32 result with cuBLAS
algorithm 10, but take 4.373397 ms versus 4.174944 ms for exact monolithic
algorithm 5. The 0.198453 ms/layer panel penalty is retained rather than
hidden.

The alpha/g and beta projections are only M32. Splitting them into four M8
GEMMs is exact but raises each projection from about 0.108 ms to about
0.402 ms. HeadFold therefore computes each M32 projection sequence-locally
and redistributes the two 32-channel scalar arrays by complete head. The
direct peer-gather path moves two 520,192-byte arrays in 0.137219 ms and
passes all 520,192 FP32 comparisons bitwise.

Together these close the previously unmeasured recurrent dependency. Replay
v5 charges 0.335672 ms for every recurrent layer, or 10.070 ms over 30
layers:

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R16 central | 2,627.214 ms | 3,093.77 tok/s |
| HeadFold R16 conservative | 2,726.150 ms | 2,981.49 tok/s |
| HeadFold R44 central | 2,469.695 ms | 3,291.09 tok/s |
| HeadFold R44 conservative | 2,544.138 ms | 3,194.79 tok/s |
| R44 exact-Q8 at nominal FP32 peak | 2,185.589 ms | 3,718.91 tok/s |
| R44 zero-Q8 bound | 1,719.925 ms | 4,725.79 tok/s |

At R44, 3,500 tok/s requires 7.1895 effective exact-Q8 TFLOP/s, or a
1.2447x speedup over the current calibrated rate. Reaching 4,000 tok/s by
changing Q8 alone would require 13.877 TFLOP/s and is impossible on this
FP32 data path. The exact architecture must remove fixed work as well.

Evidence:

- `headfold-qkv-ring-headpacked-v1.out`
- `headfold-gdn-z-panel-selector-v1.jsonl`
- `headfold-gdn-alpha-beta-panel-selector-v1.jsonl`
- `headfold-gdn-scalar-transport-v1.out`
- `headfold-replay-v5.json`

## SplitRail cuBLAS exact-Q8 replacement

SplitRail tested whether vendor SGEMM could replace BroadWave without
changing the exact accumulation tree. Native Q8_0 weights were dequantized
to FP32, K offsets 0-15 and 16-31 from every K32 block were placed in two
separate matrices, two cuBLAS GEMMs produced the ordered rail partials, and
the original final FP32 add combined them.

Several Pascal cuBLAS algorithms are bitwise exact on a production-shaped
M256/N512/K2048 expert projection, but their arithmetic rate is too low:

| algorithm | time | rate | FP32 mismatches |
| ---: | ---: | ---: | ---: |
| 2 | 0.274734 ms | 1.954 TFLOP/s | 0 / 131,072 |
| 3 | 0.268222 ms | 2.002 TFLOP/s | 0 / 131,072 |
| 4 | 0.262358 ms | 2.046 TFLOP/s | 0 / 131,072 |
| 5 | 0.348397 ms | 1.541 TFLOP/s | 0 / 131,072 |
| 12 | 0.608301 ms | 0.883 TFLOP/s | 0 / 131,072 |

The fastest algorithms reach only 4.08-4.15 TFLOP/s and change 123,896 of
131,072 outputs. BroadWave reaches about 6.60 TFLOP/s on a uniform pooled
M64 service. SplitRail therefore loses before batching, conversion, or
weight-streaming costs are charged. The architecture is closed; a batched
version cannot reverse a single-projection arithmetic deficit of this size.

Evidence:

- `splitrail-cublas-exactness-probe.cu`
- `splitrail-cublas-m256-n512-k2048-v1.jsonl`

## FP64 AssistWave exact arithmetic

GP100 exposes unusually strong FP64 hardware next to its FP32 datapaths. An
FP64 fused multiply-add followed by round-to-FP32 after every term can
reproduce an FP32 fused multiply-add for finite FP32 operands. AssistWave
tested whether otherwise idle FP64 pipes could therefore process a fraction
of BroadWave output tiles without changing a single accumulator result.

Inline PTX forced `fma.rn.f64` and the two required conversions around every
term. Exactness passed on all four GPUs:

- 458,752 FP32 accumulator values checked per GPU;
- zero bitwise mismatches.

The conversion cost defeats the hardware opportunity. The pure emulation
path reaches only 0.83-0.90 equivalent TFLOP/s. A warp-partitioned kernel is
also invalid as a performance mechanism because its 108-register allocation
reduces occupancy for the FP32 warps. The decisive probe used separate
low-register FP32 and FP64-assist kernels in independent streams, then swept
the grid ratio:

| concurrent work | slowest-device equivalent rate |
| --- | ---: |
| 56 FP32 CTAs, no assist | 6.82 TFLOP/s |
| 56 FP32 plus 56 FP64 CTAs | 1.63 TFLOP/s |
| 224 FP32 plus 56 FP64 CTAs | 3.20 TFLOP/s |
| 448 FP32 plus 56 FP64 CTAs | 4.41 TFLOP/s |

The 448:56 ratio approximately duration-matches the isolated paths. It is
still much slower than FP32 alone, showing that conversion issue and
scheduler contention largely serialize the streams. FP64 AssistWave is
closed. The mathematical equivalence may be useful on hardware with a
vector FP64-to-FP32 rounding path, but not on P100.

Evidence:

- `fp64-assist-ceiling-probe.cu`
- `fp64-assist-ceiling-v1.out`
- `fp64-assist-ceiling-v2.out`
- `fp64-assist-ceiling-v3.out`

## Exact fused gate/up service control

The pre-existing 512-thread exact gate/up kernel was tested under four-cell
row pooling before committing to a persistent service superkernel. It shares
the activation stage across gate and up, uses a single CTA/SM resource shape,
and preserves the owner-partial oracle.

The control is flat:

| service | instrumented stage | bitwise mismatches |
| --- | ---: | ---: |
| separate gate/up | 20.04-20.10 ms | 0 / 16,777,216 |
| fused gate/up | 20.05-20.10 ms | 0 / 16,777,216 |

The shared activation load is not material relative to the ordered FP32 FMA
chains, and the fused CTA loses the latency hiding available to independent
CTAs. A persistent service kernel may still remove planner, SwiGLU, and
reduction launches, but this control bounds that work to roughly the existing
non-projection slice; it cannot change the arithmetic ceiling. A scheduler
rewrite based primarily on gate/up reuse is closed.

Evidence:

- `fused-gu-cuda-off-v1-service-p2048-ub2048.out`
- `fused-gu-cuda-on-v1-service-p2048-ub2048.out`

## DualRailCTA compact-weight 512-thread engine

DualRailCTA tested whether BroadWave's two ordered K16 accumulator rails
could be assigned to separate halves of a 512-thread CTA. Each CTA retained
the exact 32 KiB BroadWave A/expanded-B stage, dequantized every Q8 value
once, and used a shared-memory rail reduction only after all K32 stages.
The intended resource shape compiled exactly:

| resource | DualRailCTA |
| --- | ---: |
| threads/CTA | 512 |
| registers/thread | 64 |
| shared memory/CTA | 32,768 bytes |
| local allocation | 0 bytes |
| stack frame | 40 bytes |
| theoretical resident CTAs/SM | 2 |
| theoretical resident warps/SM | 32 |

The component oracle passed all 16,777,216 owner-output comparisons
bitwise. Despite twice BroadWave's resident warp count, captured four-cell
row pooling regressed every projection:

| engine | gate | up | down | instrumented service |
| --- | ---: | ---: | ---: | ---: |
| BroadWave control | 5.244-5.306 ms | 5.244-5.299 ms | 5.137-5.198 ms | 16.588-16.769 ms |
| DualRailCTA | 6.056-6.079 ms | 6.056-6.080 ms | 5.964-5.995 ms | 19.038-19.116 ms |

SASS explains why occupancy did not translate to issue throughput. One
thread executes 512 FFMA instructions instead of BroadWave's 1,024, but the
non-FFMA instruction stream only falls from 932 to 892 instructions.
Across 512 rather than 256 threads this nearly doubles address, control, and
staging instruction demand per output tile. The forced 64-register shape
also emits ten stack stores and ten stack loads, while BroadWave emits none.
The shared rail reduction is additional work that BroadWave performs in
registers. This is an instruction-issue regression, not a tail-distribution
artifact, so the geometry is closed.

The first result file reports the engine label as `legacy` because the
benchmark's integer-to-name table did not yet contain experimental engine
16; the stderr configuration line confirms `engine=dualrailwave`. The label
table was corrected after the measurement.

Evidence:

- `dualrailwave-v1-coalesced-service-p2048-ub2048.out`
- `dualrailwave-v1-coalesced-service-p2048-ub2048.err`
- `dualrailwave-v1-broad-control-coalesced-service-p2048-ub2048.out`
- `dualrailwave-sass-v1.out`

## RecurFill corridor overlap

The exact H8 HeadFold recurrence was paired with four production-shaped `z`
and four output SGEMMs per GPU using high- and low-priority nonblocking
streams. This is a favorable upper-bound probe: it omits normalization,
gating, head publication, copies, event costs, and slot backpressure.

The algorithm-10 SGEMM actually consumes 120 registers/thread, not the
suggested 53, and slows the GDN kernel while overlapping it. A complete exact
selector sweep found algorithm 8 best at 15.2401 ms for concurrent GDN plus
dense work. The fastest current sequential reference is 16.2056 ms, leaving
only 0.9655 ms/layer or 28.96 ms across 30 recurrent layers.

All 46,301,184 checked FP32 values are bitwise identical. RecurFill is closed
as an architectural contributor. See `RECURFILL-AUDIT.md`.
