# Exact P100 PairFold runtime checkpoint

## Status at a glance

PairWave is the one-copy two-pair expert placement, loader, and lane-exact
service primitive. PairFold is the complete pair-local recurrent, attention,
expert, and hidden-state scheduler built on PairWave.

This checkpoint contains a real end-to-end PairFold runtime. Its accurate
warm pp8128 result is 3,027.038 ms, or 2,685.133 prompt tokens/s. It is
default-off and experimental. It is not the production path.

| Item | Value |
| --- | --- |
| Source base | `619e7cc7c5425b57ad1f5350d2eaa85feecc4187` |
| Qualified arithmetic checkpoint | `36bad6bb3ad4c9b31edc4dae9fb6c3716b95704b` |
| Hardware | 4x Tesla P100 PCIe 16 GB |
| Model weights | Qwen3.6-35B-A3B Q8_0 |
| Runtime selector | `GGML_CUDA_AW_PAIRFOLD=1` |
| PairFold warm pp8128 | 3,027.038 ms / 2,685.133 tok/s |
| Same-build warm diagonal | 2,788.594 ms / 2,914.731 tok/s |
| Qualified production comparator | 2,790.956 ms / 2,912.264 tok/s |
| Matched PairFold gap | +238.444 ms / +8.551% wall |
| Exactness | byte-identical c512 service boundaries and saved logits |
| Saved-logits SHA-256 | `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5` |
| Measured runtime memory reclaimed | 0.817007 GiB/GPU |
| Production status | not promoted |

The older 2,921.533 tok/s PairWave number is a calibrated replay, not a
runtime result. The older approximately 2,828 tok/s AffinityWave checkpoint
is service-only and has no logits. Neither should be presented as an
end-to-end PairFold rate.

## Why the architecture exists

The production exact diagonal scheduler reaches 2,912.264 tok/s, but its
four-GPU execution and expert residency leave limited room for a different
memory topology. PairWave tests a second architecture:

- split the four GPUs into two fixed physical pairs;
- keep only one copy of each layer's 256 Q8_0 experts;
- execute two chronological prompt panels on the layer-owning pair;
- overlap the two pairs after pipeline fill;
- preserve every arithmetic and state-order boundary required by the
  accepted exact path.

The immediate result is slower than production, but it establishes a
working pair-local runtime. The longer-term motivation is host-resident
expert streaming: while one pair computes, the other pair can load a future
layer into bounded device slots. That streaming path is not implemented in
this checkpoint.

## Placement and prompt geometry

The physical pairs are:

```text
pair 0: GPU0, GPU1
pair 1: GPU2, GPU3
```

Layers 0 through 19 alternate between pair 0 and pair 1. Layer 20 repeats
pair 1 after layer 19, then layers 20 through 39 alternate pair 1 and pair
0. Each pair owns 20 layers, comprising 15 recurrent layers and five
attention layers.

At pp8128, the prompt is represented as two chronological 4,064-token
panels. Each panel contains two 2,032-token logical lanes:

```text
panel 0: logical lanes 0 and 1
panel 1: logical lanes 2 and 3
```

The active pair owns all 256 experts for its layer. Each active GPU owns two
complete canonical 64-expert groups. Expert weights remain single-copy.
The manifest records the layer pair, canonical group owners, and panel
orientation:

`evidence/frontier/pairwave-laneexact-v3.manifest`

Its SHA-256 is:

`316c55cb76f0663190c333680c1e6574afc63ad2f15f8c26621459345d7735da`

## PairFold scheduler

The static scheduler has 80 tasks:

```text
(layer, panel, pair, generation)
```

It enforces:

- `(layer, 0) -> (layer, 1)` for chronological state;
- `(layer, panel) -> (layer + 1, panel)` for hidden-state flow;
- one serialized compute resource per physical pair;
- the repeated pair-1 resource edge across layers 19 and 20.

The runtime uses:

- a serial scheduler as the arithmetic oracle;
- a pipelined topological wavefront for normal PairFold execution;
- two banks of task and terminal CUDA events;
- a bank-reuse wait before a generation is reused;
- two stable F32 hidden handoff slots per GPU;
- nonblocking copy/publication streams;
- one SM-compute stream per GPU;
- CUDA event dependencies rather than device-wide barriers or peer atomics.

Recurrent layers run pair-local dense and convolution preparation, then the
qualified exact two-lane GDN path. Panel 1 consumes panel 0's final recurrent
state before the canonical state is published.

Attention layers keep per-lane Q/K/V projections, chronological K/V
positions, masks, and output order. The current implementation is the
literal exact two-lane attention control. It does not yet implement the
planned packed pp8128 three-partition FlashAttention redistribution.

The lane-exact PairWave service runs only on the active pair. Owner BF16
partials are returned into home-local scratch by DMA and then summed in the
original canonical order. Final layer-39 outputs are restored to their four
logical lanes before the ordinary output tail.

## Runtime contract

PairFold is enabled only with:

```text
GGML_CUDA_AW_PAIRFOLD=1
```

The serial arithmetic control additionally uses:

```text
GGML_CUDA_AW_PAIRFOLD_SERIAL=1
```

The qualified path requires:

- exactly four Tesla P100 GPUs;
- an absolute `GGML_CUDA_AW_PAIRWAVE_MANIFEST` path;
- Q8_0 weights in the exact `t64k32` layout;
- CohortRail expert kernels;
- exact dense selectors;
- deterministic route scatter;
- F32 wire values;
- BF16 owner partials;
- canonical ordered owner sums;
- exact HeadFold recurrent and attention selectors;
- `GGML_CUDA_AW_P100_EXACT=1`;
- `GGML_CUDA_AW_P100_EXACT_SUM_WIDTH=4`.

The implemented qualified lane sizes are 128, 512, and 2,032 tokens,
corresponding to c512, pp2048, and pp8128. Unsupported or ragged shapes emit
a one-time diagnostic and use the existing path. Malformed selectors,
manifests, model shapes, or hardware are rejected rather than silently
changing arithmetic.

PairFold is default-off. The production diagonal and legacy PairWave paths
remain unchanged when it is disabled.

## Exactness evidence

The serial and pipelined c512 paths both report PPL 4.0783 and produce the
accepted saved-logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The final owner-return revision produces the same hash. Serial and
pipelined saved-logits files are byte-identical.

The checkpoint candidate binary was re-run after the trace-only terminal
marker was added. It again reported PPL 4.0783 and the accepted saved-logits
SHA-256. The curated output is named
`checkpoint-exact-pipeline-c512-v1.out`; the generated diagnostic log and
126 MB logits binary remain local.

All 640 common service artifacts match the same-build diagonal oracle
byte-for-byte. They cover all 160 layer/lane service boundaries and:

- F32 service inputs;
- router IDs;
- router weights;
- reduced F32 service outputs.

The CPU scheduler test covers:

- all 80 tasks emitted exactly once;
- every chronology, hidden, and pair-resource edge;
- the layer-19/20 repeated-pair boundary;
- randomized delayed completions;
- hidden handoff and slot-reader lifetimes;
- both generation banks;
- repeated generation wraparound.

A same-process runtime test completed generations 0, 1, and 2, including
bank-0 reuse. The two post-initialization c512 samples were 428.439 and
425.515 ms.

The current terminal marker is trace-only. It is disabled during ordinary
execution and does not alter the arithmetic path.

## Accurate warm pp8128 result

The first pp8128 campaign used fresh processes with `--no-warmup -r 1`.
Its 4,050.309 ms / 2,006.761 tok/s median was a cold-call result. It must
not be quoted as steady-state PairFold performance.

The corrected measurement used six repetitions in one process. Sample 0
was the full-shape initialization and CUDA graph-capture pass. Samples 1
through 5 were the warm set.

PairFold samples:

```text
cold: 4120.189 ms
warm: 3027.038, 3037.403, 3023.974, 3044.781, 3025.481 ms
```

Same-build diagonal samples:

```text
cold: 3850.423 ms
warm: 2815.923, 2797.259, 2783.959, 2788.594, 2786.640 ms
```

| Path | Warm median | Throughput | Warm peak-to-peak |
| --- | ---: | ---: | ---: |
| PairFold | 3,027.038 ms | 2,685.133 tok/s | 0.687% |
| Same-build diagonal | 2,788.594 ms | 2,914.731 tok/s | 1.146% |

The same-build diagonal result reproduces the qualified 2,790.956 ms
production comparator within 0.085% wall time. PairFold is:

- 238.444 ms and 8.551% slower in matched wall time;
- 7.877% lower in matched throughput;
- 236.082 ms above the qualified production wall gate;
- 7.799% below qualified production throughput.

Generations 0 through 5 completed, including repeated use of both event
banks. Neither arm emitted a fallback or error diagnostic.

## Trace and memory evidence

The corrected terminal-marked pp2048 trace has a 1,247.053 ms 80-task
window.

| GPU | SM busy | SM idle | Cross-stream SM overlap |
| ---: | ---: | ---: | ---: |
| 0 | 50.143% | 49.857% | 0 ms |
| 1 | 49.928% | 50.072% | 0 ms |
| 2 | 50.000% | 50.000% | 0 ms |
| 3 | 50.541% | 49.459% | 0 ms |

Both physical pairs have active SM work for 519.134 ms. The trace contains
8,224,997,856 copy bytes over a 261.663 ms copy union. Only 59.299 ms is
outside all SM work, so measured copy exposure is 22.662%. This replaces
the PairWave replay's assumed 25% exposure.

The trace contains:

- all 640 local ordered owner-sum kernels;
- no peer-load kernel;
- no same-GPU cross-stream SM overlap.

At pp8128:

- PairWave scratch is 199,793,408 bytes/GPU;
- two F32 handoff slots request 33,292,288 bytes/GPU;
- the measured PairFold runtime allocation delta is 289,406,976 bytes/GPU;
- the largest reported used-memory high-water is 13,785,432,064 bytes;
- 877,254,448 bytes/GPU, or 0.817007 GiB/GPU, are reclaimed relative to
  the documented legacy live state.

The runtime result does not claim the replay-only 131,017,728-byte arena.

## PairWave work already tried

The detailed history is in
[PAIRWAVE-AUDIT.md](archive/frontier/PAIRWAVE-AUDIT.md) and
[EXPERIMENT-LEDGER.md](EXPERIMENT-LEDGER.md). The important decisions are:

| Experiment | Result | Decision |
| --- | --- | --- |
| Two-lane route pooling | Changed 174 layer-0 F32 values, maximum absolute difference `6.103515625e-05` | rejected |
| Pooled replay v1 | Predicted 3,016.620 tok/s using invalid pooled tile classes | superseded |
| Lane-exact replay v3 | Predicted 2,921.533 tok/s at assumed 25% copy exposure | useful model, never a runtime rate |
| Strict N256 service | Added 2.312731 ms in the service probe | closed |
| Lane-exact N512 service | Added 0.235001 ms in the service probe | retained evidence |
| Exact `2xM32` shared staging | Byte-identical at layer 0 but slower at pp2048 | closed |
| Exact `2xM16` shared staging | Byte-identical but slower at both tested occupancy points | closed |
| Plain 256-descriptor service | Preserves lane M64/M32/M16 classes and passes c512 | retained |
| Serial PairFold | End-to-end arithmetic oracle, accepted c512 hash | retained test control |
| Pipelined PairFold | Exact, overlaps both pairs, warm 2,685.133 tok/s | retained experimental runtime |
| PairCache static R16 | Positive untouched-test replay but unresolved runtime boundaries | not implemented, out of scope |

M32 and M16 tails remain 66.3% of the lane-exact exact-Q8 device work in
the retained service trace. The exact `2xM32` and `2xM16` experiments are
closed without materially new evidence. Pooling lanes or promoting tile
classes is forbidden by the exactness contract.

Broader closed classes also remain closed:

- AsyncEP was 37% slower because filtering was also the row concentrator;
- same-GPU SM-kernel overlap increased busy time without reducing wall;
- atomic route scatter was not bitwise exact;
- stable route scatter variants were slower;
- sub-Q8 model weights are outside the project contract.

## Known limitations and next performance work

The checkpoint is complete enough to test the architecture, but not to
replace production.

1. Attention uses the literal exact two-lane control. The planned pp8128
   `c0+c3 / c1+c2` and `c4+c7 / c5+c6` packed mapping with the qualified
   three-partition FlashAttention signature is absent.
2. PairFold reports approximately 9.9 GiB of corridor/external traffic at
   pp8128, versus approximately 0.67 GiB for diagonal. Redundant staging,
   hidden handoff, and owner-return copies need a warm trace and audit.
3. The pp2048 cold campaign had material host/rig dispersion. Any
   optimization campaign must use same-process warm measurements.
4. Standalone recurrent, attention, and composed endpoint probes were not
   added. The end-to-end runtime and c512 dumps were used instead.
5. pp2048 and pp8128 do not yet have full intermediate byte dumps comparable
   to the c512 service-boundary campaign.
6. PairCache and predictive expert selection are not part of this
   checkpoint.

The shortest credible route toward the production wall is:

1. capture a warm PairFold trace;
2. implement and qualify the packed exact attention path;
3. remove redundant corridor/handoff/owner traffic;
4. revisit small-M service only with a mechanism different from the closed
   pooling and `2xM32`/`2xM16` experiments;
5. repeat c512 exactness, paired KLD/PPL, and randomized warm pp8128 gates.

## Host-resident expert streaming direction

No host-resident weight streaming code is included here. The checkpoint is
intended to be the arithmetic and scheduler oracle for that next
experiment.

For the current Q8_0 geometry:

- one 128-expert GPU shard for gate, up, and down is 427,819,008 bytes,
  exactly 408 MiB;
- one complete pair-local layer is 816 MiB;
- 20 resident shards occupy 7.96875 GiB/GPU;
- two 408 MiB device slots would reclaim 7.171875 GiB/GPU before any
  additional staging buffer;
- a full 40-layer prefill transfers 34,225,520,640 bytes, or 31.875 GiB,
  if every expert shard is streamed once.

Nearly every expert is selected at pp8128, so exact route sparsity does not
remove the full-layer transfer requirement. A measured capture averages
254.625 of 256 unique experts per layer.

Ignoring contention and runtime T64 repacking, perfect overlap requires:

| Target | Minimum aggregate host ingress |
| --- | ---: |
| Hide beneath current 3,027.038 ms PairFold wall | 11.307 GB/s |
| Stay within the 2,790.956 ms production wall | 12.263 GB/s |

These are mathematical floors, not performance predictions. Runtime
repacking, memory registration, shared PCIe contention, and prefetch
deadlines must be measured.

The proposed proof of concept is prefill-only:

- retain the current resident weights as the byte-exact oracle initially;
- snapshot selected exact T64 shards to host memory;
- use two rotating 408 MiB slots per GPU indexed by pair-local layer
  ordinal, not global `layer % 2`;
- preload two pair-local layers and refill a slot after panel 1 releases its
  last expert reader;
- cover layers 16 through 23 first so both pairs and the layer-19/20 repeat
  boundary are exercised;
- validate c512 bytes before expanding from 2 to 8 to 40 streamed layers;
- remove resident copies only after transfer and lifetime correctness is
  proven.

Whole-layer streaming is unsuitable for decode without a separate resident
or hot expert cache. Once weights are actually evicted, decode must not
silently fall back to a path whose weights are absent.

## Build and validation

The checkpoint was built with CUDA 12.8.61, Release mode, CUDA, Flash
Attention, NCCL, tests, and `sm_60`.

Representative targets:

```bash
cmake --build <build-dir> --parallel \
  --target test-pairfold-scheduler llama-bench llama-perplexity

ctest --test-dir <build-dir> \
  -R '^test-pairfold-scheduler$' \
  --output-on-failure
```

The rig-specific qualification helper contains the complete production
environment, lock, watchdog, CPU affinity, and selectors:

```text
bench/run-pairfold.sh
```

Its `ROOT`, `BUILD`, `REFERENCE`, `MODEL`, and `CORPUS` values are absolute
paths from the original rig and must be adapted elsewhere. The EPLB and
AffinityWave placement files referenced by the script are external
machine-specific runtime inputs.

Current-binary c512 oracle:

```bash
./bench/run-pairfold.sh pipeline-ppl 512 \
  checkpoint-exact-pipeline-c512-v1
sha256sum \
  results/qwen36-35b-moe-pp-20260721/pairfold-20260726/\
checkpoint-exact-pipeline-c512-v1-logits.bin
```

Warm pp8128 PairFold:

```bash
PAIRFOLD_REPS=6 ./bench/run-pairfold.sh pipeline 8128 \
  warm-pipeline-pp8128-v1
```

Matched warm diagonal:

```bash
PAIRFOLD_REPS=6 ./bench/run-pairfold.sh diagonal 8128 \
  warm-diagonal-pp8128-v1
```

For both warm commands, discard sample 0 and take the median of samples 1
through 5. Do not use the JSON aggregate because it includes cold
initialization.

The trace analyzer consumes an Nsight Systems SQLite export:

```bash
./bench/analyze-pairfold-trace.py pairfold-trace.sqlite
```

## Source map

| File | Role |
| --- | --- |
| `ggml/src/ggml-backend-meta.cpp` | selector validation, cell decomposition, serial control, wavefront submission, state/cache publication |
| `ggml/src/ggml-pairfold-scheduler.h` | static 80-task DAG and pair mapping |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | generation banks, events, handoff slots, copy/publication callbacks |
| `ggml/src/ggml-cuda/affinity-wave.cu` | pair-local GDN, PairWave service, owner returns and canonical local sums |
| `ggml/src/ggml-cuda/affinity-wave.cuh` | PairFold CUDA interfaces and handoff descriptor |
| `tests/test-pairfold-scheduler.cpp` | dependency, delay, slot, and generation-wrap tests |
| `bench/run-pairfold.sh` | original-rig guarded exactness/performance harness |
| `bench/analyze-pairfold-trace.py` | task-window, busy/idle, overlap, copy, and kernel trace analysis |

## Evidence policy

Small text outputs and the terminal trace analysis are retained with the
checkpoint. Bulky evidence remains local:

- generated logits binaries;
- 336 MB service-dump campaigns;
- Nsight `.nsys-rep` and `.sqlite` files;
- per-stage debug dumps;
- PairCache route captures;
- model weights and build products.

The detailed PairFold result record is:

`results/qwen36-35b-moe-pp-20260721/pairfold-20260726/RESULTS.md`

The broader historical archive remains under:

- `docs/p100-pairwave-project/archive/frontier/`;
- `docs/p100-pairwave-project/archive/stitchrail/`;
- `docs/p100-pairwave-project/archive/prehistory/`.

This checkpoint records an exact, working architectural alternative. Its
measured warm rate is 2,685.133 tok/s, not the replayed 2,921.533 tok/s, and
it remains experimental until it meets or beats the qualified production
comparator.
