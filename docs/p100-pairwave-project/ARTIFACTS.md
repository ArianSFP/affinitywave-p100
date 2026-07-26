# Artifact and provenance index

## Source and binary identity

| Artifact | SHA-256 |
| --- | --- |
| Qualified tracked source diff | `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f` |
| Qualified `libggml-cuda.so` | `44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7` |
| Saved logits | `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5` |
| PairWave lane-exact placement | `316c55cb76f0663190c333680c1e6574afc63ad2f15f8c26621459345d7735da` |
| PairCache report | `32257395cdd13ea40b0ab9632ad1ac28c6158c24b747ae384a788257a420a488` |
| PairCache manifest file | `7a1e0e90e1fb706f545170a2996dd6023ca4fb1a754d4ece543604f86e506387` |
| PairCache canonical manifest seal | `2a4d9600bc46e26ab20b6322c4d67f39cd1327999373b57127b82e3e507df7e3` |

The PairWave placement hash is the hash used by the replay and PairCache
requirements. Recompute all copied evidence hashes after checkout rather
than relying only on this table.

## Archive layout

### `archive/frontier`

Contains 63 ASCII-normalized Markdown files from the AffinityWave frontier
results:

- architectural audits for PairWave, HeadFold, TemporalQueue, FoldWave,
  RelayWave, ScanFold, InterferenceWave, and bounded diagonal service;
- kernel audits for HalfPipe, DoubleWave, VectorWave, RailWave,
  ExpandWave, route scatter, and materialization-free panels;
- all retained CohortRail trace analyses;
- live broadwave, tailwave, attention, and cache-relay analyses;
- the full frontier engineering notebook and pause handoff.

### `archive/stitchrail`

Contains 12 ASCII-normalized Markdown files from the final exact campaign:

- `FINAL-RESULTS.md`;
- the full final engineering notebook;
- baseline, candidate, and final pp8128 trace analyses;
- three retained CohortRail pp2048 traces;
- static resource audit;
- paired campaign summary;
- stacked-projection summary;
- balanced-attention pause record.

### `archive/prehistory`

Contains the relevant Qwen MoE execution history, round-3 research/results,
original execution plans, and the parked first AffinityWave record.

### `archive/paircache`

Contains the complete PairCache investigation notebook and its coordination
record. PairCache was not implemented.

## Machine-readable evidence

### Frontier

- `pairwave-replay-v3.json`
- `pairwave-laneexact-v3.manifest`
- `pairwave-promotion-audit-v1.json`
- `foldwave-replay-v1.json`
- `headfold-replay-v5.json`
- `relaywave-bound-v2.json`
- `temporal-queue-replay-v1.json`
- AtlasWave exact target models
- Q8 weight-structure summary

### Final exact campaign

- `stitchrail-replay-v1.json`
- `stitchrail-v1.manifest.json`
- randomized campaign manifest, metadata, events, and exactness TSVs
- selected static toolchain, selector, cubin-hash, and source-hash records

The StitchRail manifest is an archived replay artifact and is explicitly
non-deployable. Its presence does not indicate a runtime StitchRail
implementation.

### PairCache

- `paircache-replay-v2-selector-complete.json`
- `paircache-v1.selector-complete.manifest.json`

The manifest status `replay-candidate-runtime-pending` must be read together
with the authoritative report status
`replay-blocked-unresolved-model-boundaries`.

## Qualification harness

The `harness/` directory contains:

- the exact qualification runner;
- the randomized campaign driver;
- the final static archive script;
- the router selector probe;
- the stacked projection probe.

These scripts retain absolute paths from the original rig for provenance.

## Deliberately excluded bulky artifacts

The following classes remain on the original rig but are not appropriate for
a source repository:

- 333,025,280-byte canonical PairCache route capture;
- 353,839,360-byte raw PairCache route capture;
- Nsight `.nsys-rep` and `.sqlite` traces;
- per-cell binary service dumps;
- generated logits binaries;
- the 675 MB final static archive;
- 620 MB SASS concatenation;
- cubins, ptxas objects, and linked CUDA libraries;
- the Q8_0 model.

PairCache capture hashes:

| Capture | SHA-256 |
| --- | --- |
| raw 34-window capture | `c9003a8d2b165ff70139fcc1f970c51bba271cd9e886bc9d176558458266223c` |
| normalized 32-window capture | `9547982ef910dd884705c24571b601776d27206f1d7fe0122f69db0e8fea6135` |
| capture metadata | `f2b1a4989bbf7fa22bda8bffd3ba85ccfe209865c09df954b302b841e95fcc64` |

Static resource summaries and exactness results are included so a newcomer
can understand the conclusions without storing those bulky files in Git.
