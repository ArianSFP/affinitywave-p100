# TemporalQueue architectural audit

This is the live record for the no-replication temporal placement,
capacity-constrained token-home, four-layer queue, and materialization-free
MoE proposal. `NOTES-codex.md` and `HEADFOLD-AUDIT.md` remain the
historical record and the HeadFold dependency record respectively.

## Runtime facts established before modeling

- Production uses `GGML_CUDA_AW_GROUP_PATTERN=1111`, so each GPU allocates
  four independent `aw_live_state` instances for four 2,032-token cells.
- Every state materializes full route-sized gate, up, middle, and down-output
  tensors, plus a token-sized owner partial and a four-owner receive tensor.
  The major arrays alone occupy about 1.33 GiB/GPU across the four states.
- The placement manifest's `primary_owner` array is validated but does not
  drive weight lookup. Physical expert ownership is the contiguous expert
  slot group created by the load-time EPLB permutation.
- A second arbitrary expert permutation cannot preserve the current BF16
  logical-owner boundaries with the existing runtime. It needs an explicit
  physical-to-logical owner map and rank-ordered cross-device FP32 chains.
- The captured routes use expert IDs after `placement-primary.eplb`.
  Therefore a new placement can be modeled as an assignment of those 256
  existing slots to four physical GPUs, while immutable logical owner remains
  `captured_expert_id / 64`.

## Early falsification results

The current fixed placement has substantial layer-local Q8 imbalance on the
captured final route pass:

- current exact pooled Q8 model: about 937.46 ms;
- perfect per-layer physical balance lower bound: about 675.98 ms;
- available compute-imbalance span: about 261.48 ms.

This is much larger than the roughly 40 ms remaining after R44 replication,
so no-replication temporal placement is a valid architectural candidate.
However the route capture is from the synthetic pp8128 benchmark, while the
existing EPLB map was calibrated for code. A benchmark-specific map is only
a mechanism probe. Production retention requires training and held-out
captures from code, math, chat, and general text.

Capacity-constrained token homes are not a standalone mechanism. Under the
current fixed placement, a sequential exact-capacity flow reduced remote
owner groups from 881,433 to 861,984, but moved 19,704 rows. After charging
one FP32 hidden-vector move, total traffic fell by only 73.98 MiB over all 40
layers. Token migration remains closed unless a new expert placement creates
materially better locality.

## Replay under construction

`temporal-queue-replay.py` consumes the captured routes and measures:

- exact M64/M32/M16 three-projection tile cost per expert;
- one-copy, exactly-64-experts/GPU placement;
- FP32 input fanout;
- rank-ordered FP32 transitions inside immutable logical-owner chains;
- BF16 final logical-owner sends;
- per-device copy-engine byte load;
- exact-capacity token-home flow with and without one-layer oracle lookahead;
- four-layer attention-delimited resource bounds;
- explicit 128 MiB/GPU arena accounting.

The arithmetic placement uses CP-SAT, then communication-aware balanced swap
search from both the current and compute-only solutions. The replay reports
both a serialized bound and an optimistic compute/copy resource bound. No
performance credit is accepted until the N128 chain transport and queue
dependencies are measured on the P100s.

## Calibrated placement replay

The final captured route pass was solved with 0.5 seconds of CP-SAT time and
two 2,000-swap communication-aware searches per layer. The solver took
350.736 seconds. All points keep exactly one copy of each expert and exactly
64 experts/layer/GPU.

| placement | Q8 | copy critical | serialized | resource bound | FP32 chain rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| current logical groups | 937.457 ms | 294.644 ms | 1,232.101 ms | 937.457 ms | 0 |
| compute-only balance | 676.069 ms | 569.075 ms | 1,245.144 ms | 676.069 ms | 985,762 |
| coactivation refined | 726.579 ms | 430.795 ms | 1,157.374 ms | 726.579 ms | 573,556 |

Compute-only placement proves that the original 261 ms imbalance span is
real, but it loses under serialization because it fragments immutable
logical-owner chains. Coactivation refinement retains 50 ms more Q8 work to
remove roughly 412,000 chain transitions. Its conservative gain over the
current map is 74.727 ms.

Four-layer attention-delimited load smoothing changes the decision:

| placement | epoch Q8 | epoch copy | serialized | resource bound |
| --- | ---: | ---: | ---: | ---: |
| current logical groups | 800.438 ms | 290.165 ms | 1,090.603 ms | 800.438 ms |
| compute-only balance | 676.022 ms | 533.681 ms | 1,209.703 ms | 676.022 ms |
| coactivation refined | 713.219 ms | 424.156 ms | 1,137.375 ms | 713.219 ms |

The current placement already rotates naturally occurring hot-owner load
across the four GPUs. A four-layer queue can therefore recover a 137.019 ms
Q8 lower-bound gain without changing placement or introducing chain traffic.
After that smoothing, arbitrary coactivation has only another 87.219 ms of
compute span and adds 134 ms of modeled copy load.

### Token-home closure

On the refined coactivation placement, exact-capacity home flow was tested
over all 40 layers:

| policy | global copy critical | moved rows | total bytes |
| --- | ---: | ---: | ---: |
| fixed contiguous homes | 413.786 ms | 0 | 15,781,339,136 |
| sequential exact-capacity | 409.907 ms | 43,190 | 15,636,393,984 |
| one-layer oracle lookahead | 408.355 ms | 52,322 | 15,577,780,224 |

The admissible sequential flow saves only 3.879 ms after moving 43,190 hidden
rows. Even an oracle lookahead saves only 5.431 ms. Capacity-constrained
token homes are closed; their indexing, GDN reorder, and attention reorder
costs cannot be repaid.

Evidence:

- `temporal-queue-replay.py`
- `temporal-queue-replay-v1.json`
- `temporal-queue-replay-v1.md`
- `temporal-queue-replay-v1.out`

## Exact N128 peer-chain probe

An exact four-GPU probe implemented the physical-split logical-owner
mechanism rather than estimating it from bytes:

1. It processes all 8 route ranks in their captured order.
2. An N128 kernel directly reads the preceding FP32 accumulator from its peer
   GPU when a logical-owner chain changes physical device.
3. The current route weight multiply and FP32 add remain one `fmaf`.
4. The final owner accumulator is rounded to BF16 at the original owner
   boundary.
5. Logical owners are summed in the production token-dependent rotated
   order, including BF16 rounding after each add.
6. Sixteen panels cover all 2,048 output columns.

The full 40-layer result is exact:

- 2,663,383,040 owner values checked;
- 665,845,760 final values checked;
- zero owner mismatches;
- zero final mismatches.

| placement | full 40-layer chain | empty event schedule | net kernels |
| --- | ---: | ---: | ---: |
| current unsplit owners | 492.361 ms | 104.191 ms | 388.170 ms |
| coactivation split owners | 540.517 ms | 105.942 ms | 434.575 ms |

The physical split costs 48.156 ms across all 40 layers. On layer 36, where
Q8 falls from 30.642 to 17.090 ms, it costs only 1.968 ms and is highly
profitable. Across the four-layer queue, however, the placement's remaining
Q8 advantage is only 87.219 ms. Its measured net contribution is therefore
about 39 ms before plan construction and small-segment dependency costs.
Arbitrary physical splitting is closed as a primary architecture. It remains
a later optional refinement if the queue and bounded service are already
implemented.

Evidence:

- `temporal-chain-manifest.py`
- `temporal-chain-probe.cu`
- `run-temporal-chain-probe.sh`
- `temporal-chain-current-all40-v1.out`
- `temporal-chain-coactivation-all40-v1.out`

## Bounded arena

The explicit arena totals 97,725,440 bytes/GPU, or 93.20 MiB:

- eight ping-pong 508-token hidden slots: 33,292,288 bytes;
- four segment route/id/weight/index slots: 138,240 bytes;
- eight M64 middle slots: 1,048,576 bytes;
- eight N128 down panels: 262,144 bytes;
- compact tile records: 8,192 bytes;
- four N128 canonical-chain panels: 1,040,384 bytes;
- four BF16 final-owner panels: 520,192 bytes;
- global route plan and offsets: 1,056,768 bytes;
- phase-reused HeadFold QKV ring: 58,261,504 bytes;
- events and descriptor reserve: 2,097,152 bytes.

This leaves 36,492,288 bytes under the 128 MiB cap. The current four
`aw_live_state` instances materialize at least 1,332,211,712 bytes/GPU in
their major arrays. The modeled net reduction is 1.150 GiB/GPU. This memory
result survives even though arbitrary placement does not.

## End-to-end resource boundary

`temporal-headfold-replay.py` combines measured HeadFold lane work, exact
per-GPU Q8 tile loads, service overhead, post work, and calibrated copy
resources:

| scenario | wall bound | rate |
| --- | ---: | ---: |
| HeadFold R0 with layer barriers | 2,893.624 ms | 2,808.93 tok/s |
| current groups, queue, no copy overlap | 2,743.750 ms | 2,962.37 tok/s |
| current groups, queue resource bound | 2,440.342 ms | 3,330.68 tok/s |
| coactivation, queue, no copy overlap | 2,794.651 ms | 2,908.41 tok/s |
| coactivation queue resource bound | 2,351.135 ms | 3,457.05 tok/s |
| coactivation plus measured chain delta | 2,399.292 ms | 3,387.67 tok/s |
| coactivation at BroadWave's 7.011-TF/s staged ceiling | 2,273.700 ms | 3,574.79 tok/s |

These resource values assume ideal copy overlap and ignore the remaining
508-token dependency gaps. Even the measured BroadWave arithmetic ceiling is
241.700 ms short of 4,000 tok/s. Placement and scheduling cannot finish the
program alone. The 508-token queue was therefore held as a hypothesis rather
than accepted for implementation.

Evidence:

- `temporal-headfold-replay.py`
- `temporal-headfold-replay-v1.json`
- `temporal-headfold-replay-v1.out`

## Dense publication control

The complete cuBLAS algorithm range was swept for every live dense shape.
Each segmented result was compared bitwise with production selector 99.

| shape | best monolithic | best 4x508 | segmentation tax |
| --- | ---: | ---: | ---: |
| QKV, M8192 K2048 | alg 6, 8.140240 ms | alg 6, 8.707824 ms | 6.97% |
| output, M2048 K4096 | alg 10, 4.218656 ms | alg 10, 4.386864 ms | 3.99% |
| middle, M4096 K2048 | alg 5, 4.151168 ms | alg 10, 4.383520 ms | 5.60% |
| down, M2048 K512 | alg 6, 0.566016 ms | alg 113, 0.640928 ms | 13.24% |
| HeadFold head panel, M2048 K2048 | alg 10, 2.186688 ms | alg 11, 2.335221 ms | 6.79% |

All best candidates have zero bitwise mismatches. Dense publication is not
the queue's fatal cost: four segmented head-panel calls total 9.340884 ms,
only about 0.06 ms above the measured 9.281471 ms HeadFold QKV pipeline.

Evidence:

- `temporal-dense-selector-probe.cu`
- `run-temporal-dense-selector-probe.sh`
- `temporal-dense-selector-*-v1.jsonl`

## Expert-tail dependency falsification

The proposed rule issues complete M64 tiles immediately but holds incomplete
expert tails until the last of sixteen 508-token publications. Replaying the
actual route order proves that this does not expose a useful chronological
pipeline:

- 30.33% to 55.84% of tokens in a layer complete only in segment 15;
- the mean final-segment fraction is 41.92%;
- the chronological ready prefix is normally zero through segment 14;
- in layers 0, 1, and 2, even token 0 completes only in segment 15.

Gated DeltaNet therefore cannot begin the next layer on a chronological
prefix while full expert tiles accumulate. Flushing expert tails at every
publication restores causality but destroys exact-Q8 utilization:

| chronological panels | tokens/panel | Q8 critical | useful issued rows | cost over full layer |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 508 | 1,504.156 ms | 70.04% | +566.699 ms |
| 8 | 1,016 | 1,218.182 ms | 81.74% | +280.725 ms |
| 4 | 2,032 | 1,063.463 ms | 89.75% | +126.006 ms |
| 2 | 4,064 | 980.861 ms | 94.50% | +43.404 ms |
| 1 | 8,128 | 937.457 ms | 97.12% | baseline |

These figures use the final captured pass, current one-copy owner placement,
and calibrated M64/M32/M16 times. They supersede the preliminary hand
calculation. The final-tail 508-token queue is closed.

Evidence:

- `temporal-tail-readiness.py`
- `temporal-tail-readiness-v1.json`
- `temporal-tail-readiness-v1.out`

## Two-panel constrained schedule

Two chronological 4,064-token panels are the least costly causal adaptation.
An exact CP-SAT replay assigns every measured pre, exact panel-tiled Q8 plus
service-overhead, and post task to one P100 SM resource. It includes
cross-layer panel dependencies, attention barriers, and a cumulative live
panel cap. It deliberately omits copy-engine work and event overhead, so each
result is optimistic.

| live-panel cap | feasible wall | feasible rate | proven lower bound |
| ---: | ---: | ---: | ---: |
| 2 | 2,638.150 ms | 3,080.95 tok/s | 2,454.568 ms |
| 4 | 2,631.140 ms | 3,089.16 tok/s | 2,454.600 ms |
| unbounded | 2,575.298 ms | 3,156.14 tok/s | 2,454.841 ms |

Buffer depth is not the primary limit. The modeled workload contains
9,450.564 GPU-ms:

- 5,753.678 GPU-ms of pre work;
- 3,287.786 GPU-ms of exact-Q8 plus service overhead;
- 409.100 GPU-ms of post work.

GPU 2 alone carries 2,411.723 ms before precedence bubbles. Consequently the
2,454.6 ms lower bound is dominated by real measured device work, not solver
quality. It exceeds the 4,000 tok/s budget of 2,032 ms before any exposed
copy or runtime cost. The two-panel scheduler is closed as a standalone
rewrite at current arithmetic rates.

The surviving pieces are narrower:

- retain current physical/logical expert ownership;
- retain the exact bounded-memory service described below;
- do not build a cross-layer queue until faster exact arithmetic and
  fixed-floor fusion reduce the replay below 2.00 s;
- if that gate passes, use two chronological panels, not final-tail
  508-token publications.

Evidence:

- `temporal-panel-cpsat.py`
- `temporal-panel-current-cap2-v1.json`
- `temporal-panel-current-cap4-v1.json`
- `temporal-panel-current-unbounded-v1.json`

## Materialization-free down-panel probe

The exact down projection and owner reduction were measured with 2,048,
1,024, 512, 256, and 128 output columns per panel. Every variant is bitwise
identical over 16,777,216 BF16 values/GPU.

N128 is not viable on Pascal: repeatedly exposing only 16 M32 and 42 M16
tiles raises down plus reduction from 5.6815 to 8.1645 ms. N512 is the
retained boundary at 6.1417 ms. It removes 96 MiB from the isolated
route-output allocation and lifts the explicit arena to 131,017,728
bytes/GPU, still 3,200,000 bytes below the 128 MiB cap.

The synchronized copy-compute pipeline fails. Exact remote BF16 owner sends
expose 8.991 ms with simultaneous destination streams and 15.791 ms with
rotated single-stream sends, versus the proposal's 3 ms gate. Same-layer row
pooling is therefore closed. N512 remains only a low-memory primitive for
the existing diagonal wave.

Full evidence is in `NPANEL-SERVICE-AUDIT.md`.

## Complete gate/up/down materialization boundary

Two exact gate/up implementations were tested. A 512-thread CTA that fuses
gate, up, and SwiGLU is exact but raises gate/up/SwiGLU from 10.7107 to
12.6207 ms. It is closed. Leaving BroadWave unchanged and traversing gate/up
in output-column panels costs 11.5828 ms at N256 and 14.0576 ms at N128.

The best complete bounded composition is two N256 gate/up panels plus eight
N256 down/reduce panels:

- zero mismatches over 16,777,216 BF16 owner values/GPU;
- 18.4640 ms instrumented service versus 16.6480 ms control;
- 123,087,872 bytes/GPU after phase reuse;
- 11,129,856 bytes/GPU below the 128 MiB cap;
- about 1.126 GiB/GPU less than the current four live states.

The 1.8160 ms service tax is explicit. This mechanism is retained only as a
capacity option, not credited as a speedup.

Full evidence is in `MATERIALIZATION-FREE-SERVICE-AUDIT.md`.

## Route-plan fusion gate

The suggested fused route-plan path was tested before charging it to the
four-layer queue replay. A single-pass atomic scatter is not exact: all 16
layer-0 owner-partial files differ from the deterministic reference, with
1,366 differing bytes in total. Four stable O(routes) classifiers are
bitwise exact, but all are slower on P100.

Normalized pp2048 trace time for the existing count, single-thread plan, and
deterministic scatter is 14.352 ms on one critical GPU. Stable8 raises this
to 24.366 ms and segmented eight-lane scatter raises it to 27.486 ms.
Matched pp8128 wall time regresses by 37.947 ms with stable8. The apparent
64x route scan is cache-resident and exposes 64 independent CTAs; removing
the repeated loads removes useful Pascal parallelism.

The replay receives zero route-fusion credit. This fixed floor is too small
to alter the two-panel queue decision, and the route-scatter family is
closed. See `ROUTE-SCATTER-AUDIT.md`.
