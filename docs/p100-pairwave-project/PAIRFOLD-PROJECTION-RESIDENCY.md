# PairFold projection residency design

## Status

The strict manifest path is implemented across the loader, CUDA runtime,
PairWave service, harness, and CPU tests. All four tracked policies pass the
c512 exactness gate. The three all-40 mixed policies also have frozen-binary
pp2048, pp8128, and VRAM measurements. Down-only now has an exactness-safe
pp8128 performance result with local-owner bypass disabled. The earlier
down-only, up+down, and high-capacity runs used a bypass that is
nondeterministic under the pp8128 pipeline, so they remain diagnostic
estimates. The policies remain default off and experimental because every
streamed policy is slower than resident PairFold and production. The
existing `FIRST:LAST` whole-layer path remains available.

This first integration keeps the two fixed 408 MiB device slots per GPU.
Every selected layer still acquires one complete slot as
`(generation, layer, slot)` and holds it through panel 1. Projection-sized
slots and independent gate/up/down leases are not implemented; they are a
separate resource-lifetime and event-graph change.

## Manifest

The selector is:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_RESIDENCY_MANIFEST=/absolute/path
```

It is valid only with loader host placement and PairFold host weights. It is
mutually exclusive with `GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS`. The path must be
absolute. The feature remains default off when the selector is absent.

The file has one exact row for every model layer:

```text
pairfold-residency-v1 40 gate up down
0 resident resident host
1 resident host host
2 resident resident host
...
19 resident resident resident
20 resident resident resident
...
39 resident host host
```

Only `resident` and `host` are accepted. Layers must appear exactly once in
the order 0 through 39. Blank lines and full-line comments are accepted.
Malformed, truncated, duplicate, reordered, relative-path, or extra-row
manifests fail closed.

The parser is in `ggml/src/ggml-pairfold-residency.h`. Projection identities
are the existing logical order gate=0, up=1, down=2. The physical slot layout
remains down/gate/up through `host_projection_offset()`.

Tracked policy examples are in
[`residency-policies/`](residency-policies/):

| Policy | Logical host projections | Physical host projections | Pinned host bytes |
| --- | ---: | ---: | ---: |
| [`layers16-23-all-host.manifest`](residency-policies/layers16-23-all-host.manifest) | 24 | 48 | 6,845,104,128 |
| [`all40-down-only.manifest`](residency-policies/all40-down-only.manifest) | 40 | 80 | 11,408,506,880 |
| [`all40-up-down.manifest`](residency-policies/all40-up-down.manifest) | 80 | 160 | 22,817,013,760 |
| [`high-capacity-keep-0-19-20.manifest`](residency-policies/high-capacity-keep-0-19-20.manifest) | 111 | 222 | 31,658,606,592 |

The byte counts cover the two physical tensors registered for each logical
projection. They are placement inputs, not qualification results.

## Legacy whole-layer compatibility

The range selector retains the established behavior:

1. Every projection in `FIRST:LAST` is hosted.
2. The loader keeps the physical down/gate/up slab layout.
3. Nonphased H2D can use the existing one-copy slab transfer.
4. Each selected layer retains one fixed-slot lease through panel 1.

The manifest path replaces only the placement and pointer choice. It does not
introduce projection-sized slots or projection-level leases.

## Coordinated fixed-slot integration

### Loader and model

- Add `host_residency_manifest` to `llama_model_loader`.
- When the manifest selector is present, require loader host placement,
  host weights, an absolute path, and no explicit `HOST_LAYERS`.
- Parse the file with the shared parser.
- Select `PairWaveHost` only when the layer/projection mask says `host`.
- Keep all unselected projections on their current CUDA Meta buffer type.
- Replace `(last-first+1)*3` with the manifest's logical host-projection
  count in the model completion check.
- Log the manifest path and the logical hosted layer/projection counts.

The logical loader count is one tensor per selected layer/projection. CUDA
registration sees two physical tensors because PairWave places the selected
projection on the two GPUs of the layer's active pair.

### CUDA configuration and catalogs

- Store the same `host_residency_manifest` in `aw_pairfold_host_config`.
- Define layer selection as `host_masks[layer] != 0`.
- Define projection selection from the corresponding mask bit.
- Iterate all 40 layers and skip unselected layers instead of iterating a
  first/last interval.
- For a hosted projection, require exactly one entry in
  `aw_pairfold_host_layout_entries` and no resident duplicate.
- For a resident projection in a partially hosted layer, require exactly one
  T64 entry in `aw_layout_entries` and no host duplicate.
- Validate exact Q8_0 shapes, byte counts, active pair, and logical projection
  identity in both cases.
- Expect `2 * n_host_projections` registered physical tensors and the
  corresponding byte count.

`aw_pairfold_host_source` holds individual source pointers. The manifest mask
is authoritative. A partial layer does not require a sparse three-projection
host slab; each selected host projection is copied into its canonical offset
in the fixed device slot.

### H2D and readiness

- Continue to allocate the existing 408 MiB slots so every hosted projection
  keeps its canonical offset.
- Copy only projections whose host bit is set.
- With `GGML_CUDA_AW_PAIRFOLD_HOST_PHASED=1`, record readiness after each
  gate, up, and down boundary. Each projection then waits only when its own
  host bit is set.
- Without `HOST_PHASED=1`, retain the conservative combined gate/up
  dependency. A resident gate can therefore wait for a hosted up projection.
  The legacy whole-layer range may use its single contiguous slab transfer.
- A layer with at least one hosted projection still owns one layer lease and
  releases it only after panel 1 and all current readers.
- Count actual copied projections in the H2D byte and bandwidth diagnostics.

Per-projection readiness and per-projection slot reuse are different
features. The first is implemented behind `HOST_PHASED=1`; the second is not.

Chunking and pair/global H2D serialization are diagnostic controls. Chunks
are queued consecutively, and serialization is whole-layer rather than a
scheduler-owned critical-path admission protocol.

### PairWave service

For each active device and projection:

```text
host bit set   -> slot base + canonical projection offset
host bit clear -> validated resident T64 catalog pointer
```

With `HOST_PHASED=1`, gate, up, and down wait only when their own host bit is
set. Nonphased execution keeps the combined gate/up dependency described
above. Descriptor order, expert mapping, panel order, FP32 accumulation,
BF16 owner boundaries, and canonical reductions do not change.

The complete 408 MiB slot remains live until panel 1 and all current output
consumers complete, even after gate or up has had its final reader. Safely
reusing those regions earlier requires projection-specific generation/layer
metadata, free events recorded after the final panel-1 reader, and a
prefetch schedule that cannot wait on a stale event from a prior lease. None
of that lifetime change is claimed here.

### Prefetch and harness

- Prime the first selected layer for each pair.
- Look ahead to the next layer with a nonzero host mask, not simply
  `layer + 1`.
- Preserve the explicit layer-19/20 same-pair boundary when either layer is
  selected.
- Add a harness variable for the absolute manifest and omit `HOST_LAYERS`
  from the clean environment when it is used.
- Print the file path, host masks, hosted projection count, copied bytes, and
  fixed slot allocation.

Sparse policies used with lookahead must preserve fixed-slot alternation for
each physical pair. An incompatible selection fails closed instead of
overwriting a live layer slot. Task-local streaming supports arbitrary
well-formed masks.

## Fail-closed cases

The process must reject:

- a relative, unreadable, malformed, or incomplete manifest;
- a manifest used without loader placement and host weights;
- simultaneous residency-manifest and explicit layer-range selectors;
- a selected host tensor that remains in the resident catalog;
- a resident tensor that is missing from the resident T64 catalog;
- any partial layer whose active pair, Q8_0 shape, or projection identity
  does not match the PairWave manifest;
- a prefetch or service lease for a layer with a zero host mask.

Loader host placement must continue to reject decode and unsupported graph
fallbacks because those paths require resident expert weights.

## Qualification and diagnostic results

The legacy-equivalent layers-16:23 policy, all-40 down-only policy, all-40
up+down policy, and high-capacity policy all report PPL 4.0783 and the
accepted c512 logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

At pp8128, repeated down-only pipeline runs with local-owner bypass enabled
produced four different logits SHA-256 values:

- `aabb2ba6feb49f8832a3eb887d7665ab7a444190a98e21d4ce4074a42449e214`;
- `9fdbf2f2f4a94356409cef289e85bc842a6653a332e0dca101a689fa347934de`;
- `1fbec941e0076e63211fa7554db4d031a372b09db5c43b080ca6ad185a5559fe`;
- `e734393e0dae263ba97f70e1851699437c66ed079e4389f5a8c818fa2304bb8b`
  after the direct-local wait was added.

The down-only pipeline with bypass disabled and the serial scheduler with
bypass enabled both report PPL 6.7466 and the resident literal SHA-256:

`8cf55039bf0107fee294dba9d3c4413d61c5e1cac4f170c68a3f66303520f86e`

This clears host packing, projection selection, H2D publication, and local
PairWave arithmetic. It localizes the failure to the direct-local pointer
under two-pair overlap. The runtime and harness now permit local-owner bypass
only with the serial scheduler. The final production-safe down-only pipeline
run uses bypass disabled.

After excluding the cold sample, its pp8128 walls are:

```text
3372.835585, 3353.908010, 3329.440313, 3368.096146, 3330.874014 ms
```

The 3,353.908010 ms median is 2,423.44 tok/s and reclaims 1.840 GiB/GPU. It
adds 33.526 ms, or 1.01%, to the unsafe bypass-on 3,320.382 ms result. The
resident PairFold comparator is 3,027.038 ms / 2,685.133 tok/s, and the
production comparator is 2,790.956 ms / 2,912.264 tok/s. Safe down-only
throughput is 16.79% below production.

The retained warm measurements discard sample 0 and take the median of
samples 1 through 5:

| Policy | pp2048 median ms | pp8128 median ms | pp8128 tok/s | VRAM recovered |
| --- | ---: | ---: | ---: | ---: |
| Resident PairFold | 897.343 | not rerun | not rerun | comparator |
| All-40 down-only, safe bypass off | not rerun | 3,353.908 | 2,423.44 | 1.840 GiB/GPU |
| All-40 down-only, prior unsafe bypass on | 1,430.897 | 3,320.382 | 2,447.91 | 1.840 GiB/GPU |
| All-40 up+down | 2,295.943 | 3,553.836 | 2,287.11 | 4.475 GiB/GPU |
| High capacity, keep 0/19/20 | 3,137.983 | 3,988.963 | 2,037.62 | 6.717 GiB on GPU0/1; 6.318 on GPU2/3 |
| All-40 all-host | not rerun | 4,131.291 | 1,967.42 | 7.109375 GiB/GPU |

Except for safe down-only, the streamed rows in this table used the
bypass-on pipeline configuration. They retain value as timing and capacity
estimates, but are not strict pp8128 exactness-qualified results.

The all-host row is the current rerun of the legacy range policy, not a
fifth manifest. Its previous measurement was 4,238.061 ms / 1,917.858
tok/s. The new controls improve that wall by 106.770 ms, but do not change
the placement or recovery result.

Safe down-only is the best exactness-qualified performance/capacity point:
it streams one third of the projection bytes and recovers 1.840 GiB/GPU.
The up+down and high-capacity timing rows still need bypass-disabled reruns
before production use.

Write-combined host allocation improved down-only pp2048 by 13.047 ms but
regressed pp8128 by 30.694 ms. That screen also used the bypass-on pipeline;
it is diagnostic and is not part of the recommended policy.

## Down-only pp2048 trace

The all-40 down-only trace contains 11,408,506,880 host-weight H2D bytes in
a 1,244.951171 ms all-device union, or 8.534 GiB/s aggregate. Time at H2D
concurrency widths 1, 2, 3, and 4 is 1.454, 793.573, 5.580, and 444.345 ms.
The H2D union overlaps at least one SM kernel for 72.706% of its duration.
Same-pair SM overlap is 34.914% for pair 0 and 36.127% for pair 1. There is
no cross-stream same-GPU SM-kernel overlap.

Estimated aggregate bandwidth at concurrency widths 1, 2, 3, and 4 is
3.819, 6.919, 9.349, and 11.424 GiB/s. The analyzer apportions bytes
uniformly within each CUPTI DMA interval; width 1 has only 1.454 ms of
support.

Native traffic stretches materially during host ingress. Examples from the
same trace:

| Native copy | Outside host H2D mean | During host H2D mean | Ratio |
| --- | ---: | ---: | ---: |
| P2P, 4 MiB | 352.756 us | 885.752 us | 2.511x |
| P2P, 256 KiB | 22.177 us | 64.552 us | 2.911x |
| D2D, 256 KiB | 3.224 us | 11.471 us | 3.558x |

When the native copy overlaps an H2D on the same device, the corresponding
ratios are 3.439x, 3.896x, and 3.363x. This is direct evidence that the
remaining cost is shared copy/HBM/PCIe contention, not merely a late
ready-event wait.

All current performance rows use literal exact attention. The default-off
packed pp8128 path now matches Q, K, V, and mask staging, but its
FlashAttention output and restored pregate still diverge from the literal
oracle. It remains untimed and is not part of these qualification results.
