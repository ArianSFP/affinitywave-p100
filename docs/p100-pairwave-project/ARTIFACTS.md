# Artifact and provenance index

## Source and binary identity

| Artifact | SHA-256 |
| --- | --- |
| Qualified tracked source diff | `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f` |
| Qualified `libggml-cuda.so` | `44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7` |
| Host-streaming source-only diff over `a7c878670` | `420fe139769a64c8ae7ac91844a52dcc39688e870de4ba17a977d03870b2fbef` |
| Host-streaming qualified `libggml-cuda.so` | `e7580770cbfa0cd7c7865295e74f8ebb884a6e51393d0e463905ad3cd3f548ed` |
| Host-streaming qualified `libggml.so` | `d28cee388c13ddd7ea4b7eeb1a12208d9c40db61358016f7be8f6ce74e30d080` |
| Host-streaming qualified `libllama.so` | `a027464a0a25f1d549003ab158b011bfdede127baac6a994add381f3b9851e02` |
| Saved logits | `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5` |
| PairWave lane-exact placement | `316c55cb76f0663190c333680c1e6574afc63ad2f15f8c26621459345d7735da` |
| PairCache report | `32257395cdd13ea40b0ab9632ad1ac28c6158c24b747ae384a788257a420a488` |
| PairCache manifest file | `7a1e0e90e1fb706f545170a2996dd6023ca4fb1a754d4ece543604f86e506387` |
| PairCache canonical manifest seal | `2a4d9600bc46e26ab20b6322c4d67f39cd1327999373b57127b82e3e507df7e3` |

The PairWave placement hash is the hash used by the replay and PairCache
requirements. Recompute all copied evidence hashes after checkout rather
than relying only on this table.

The host-streaming source hash covers the implementation, test, and
benchmark files listed in the source map in
[PAIRFOLD-HOST-STREAMING.md](PAIRFOLD-HOST-STREAMING.md). It excludes
documentation and result evidence so the hash is stable while this handoff
record is edited.

## PairFold runtime checkpoint

The PairFold checkpoint adds:

- the 80-task scheduler and CPU scheduler test;
- pair-local CUDA runtime and PairWave service integration;
- the guarded original-rig benchmark harness;
- the Nsight SQLite trace analyzer;
- [PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md);
- a detailed result record and selected small text evidence under
  `results/qwen36-35b-moe-pp-20260721/pairfold-20260726/`.

Selected warm evidence hashes:

| Artifact | SHA-256 |
| --- | --- |
| PairFold warm pp8128 JSON output | `8f706af95ec73d21833b0653733ec3edae9f990fb3648335ae2e9fc455911b4b` |
| PairFold warm pp8128 diagnostic log | `7fcd9cbe02f9e4e8234ea1c344081f87a9c184a2d5759655d430120141d33e29` |
| Diagonal warm pp8128 JSON output | `47579319eab7c6c26d3e6ece1dff28f0383426439c201a2669d50cb153193ef7` |
| Diagonal warm pp8128 diagnostic log | `6601e656085005e317a0547bbb442042bf16193254d5230e3a26c1c5f22e1717` |
| Current-binary c512 oracle output | `97d72c6ebf1eca4973956ea1d9b40b63fa335311838f108dac9b69672b193594` |

The PairFold JSON aggregate includes cold sample 0. The qualified warm
median is computed only from samples 1 through 5.

## PairFold host-streaming checkpoint

The host-streaming checkpoint adds:

- [PAIRFOLD-HOST-STREAMING.md](PAIRFOLD-HOST-STREAMING.md);
- the shadow-stream and loader-placement result narratives;
- final compact c512, pp8128, and trace-analysis text evidence under
  `results/qwen36-35b-moe-pp-20260721/pairfold-20260726/`.

Selected evidence hashes:

| Artifact | SHA-256 |
| --- | --- |
| `hoststream-final-default-c512-l16-23-v5.out` | `99465f604a393b0711e1591e35f365bb6227dd15d4f40a010b70b329a1d3f906` |
| `hoststream-final-default-c512-l16-23-v5.err` | `a5fb0e80d7236f6ddd9e013ee6a8529d89cbd89e665ea1ddf758367fef3f5b39` |
| `hoststream-final-default-warm-pp8128-l16-23-v5.out` | `50c9dcc08bceed343d758bcf3854b04f75bedf9f339517fb90795a8dc41e7655` |
| `hoststream-final-default-warm-pp8128-l16-23-v5.err` | `c2eae5d38d8748580b46dd9b0e0e08f12516030ef0e905ffb2d78a4254cd3436` |
| `hoststream-final-default-trace-pp2048-l16-23-v5-analysis.json` | `c709aad56f9670f061da73c79c5bed27c9acd1e9043c5bd5411d39f7d47d1046` |
| `loaderhost-c512-l16-23-v3.out` | `223cbe3204afc4516ec54ecc45092c0659d25247ccffd7ad36a7ee3e353a765b` |
| `loaderhost-c512-l16-23-v3.err` | `ac57c950bc9b6b267f49f2c977a67e20b1440a89e8ad04b626f368a58f9d71a7` |
| `loaderhost-warm-pp8128-l16-23-v1.out` | `53198451597d4b1866ef5845e3be25190b1775bc656c8539781b043b6cc7fff4` |
| `loaderhost-warm-pp8128-l16-23-v1.err` | `f3b1a820b615fe085fb5e9c599440ea65ee2423d5cc626ae7edaf955ecb01992` |
| `loaderhost-all40-c512-v1.out` | `1f305c526a00bd3057fc05ee66b0ea06b51bc04a908eac46793bb7243678a8de` |
| `loaderhost-all40-c512-v1.err` | `e4742164ccd7c1e2ee8fc41be72f2d8fa49e994b1a580fb8bfd7039ee16bcc5b` |
| `loaderhost-all40-warm-pp8128-v1.out` | `a776bbe7baee1ab025b6c54451749b5517931f85c64dadbdcc8c2e6963bab6a5` |
| `loaderhost-all40-warm-pp8128-v1.err` | `4029acb73a9417824eea995687a0285c38b9262439285d167a3b35c355f312a2` |
| `loaderhost-all40-host-memory-monitor.txt` | `c8f3c38cfa44a897de0683a459b4b912c0f31f3715559a82ba05b64a2f1e9164` |

The committed diagnostic logs have trailing spaces and tabs removed. This
matches the repository's existing ASCII-normalized evidence convention and
does not alter their numeric or diagnostic content.

Generated logits are not committed. The c512 resident, shadow, small-window
loader, and all-40 loader files all have SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

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
- PairFold per-layer and per-stage service dumps;
- PairFold Nsight `.nsys-rep` and `.sqlite` files;
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
