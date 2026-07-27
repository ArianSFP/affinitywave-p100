# Epoch GroupWave audit

## Decision

The current exact diagonal candidate is:

- `GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048`
- `GGML_CUDA_AW_Q8_ENGINE=halfpipe_sync`
- `GGML_CUDA_AW_DENSE_SELECTORS=exact`

Its three-run pp8128 mean is 2927.554595 ms, or 2776.379 tok/s. Its
bounded service allocation reclaims 892,000,048 bytes, or about
0.831 GiB/GPU, relative to the legacy live-state allocation. The user
accepted this as satisfying the approximate 1 GiB/GPU capacity goal.

The remaining 3000 tok/s wall-time target is 2709.333 ms, so the measured
candidate needs another 218.221 ms.

## Helper-attention ceiling

The current production trace contains ten full-prefix FlashAttention calls
on GPU 3. An optimistic zero-copy allocator assigned measured two-head
attention work to every idle helper interval. Even this physically
unrealistic upper bound removes only 98.120 ms and produces 2804.84 ms,
or 2897.85 tok/s. A helper-only attention change cannot reach 3000 tok/s
and is retained only as a possible component.

## Corrected two-panel model

`temporal-panel-cpsat.py` now models each 4064-token panel as 1016 tokens
from each of the four 2032-token lanes. It separately accounts for exact
M64/M32/M16 Q8 work, F32 dispatch, canonical chain traffic, BF16 final
traffic, dense segmentation cost, and panel capacity.

Two earlier artifacts are invalid:

- `temporal-panel-current-n2048-halfpipe-denseexact-cap2-copy-v2.json`
  omitted combine traffic.
- The first directed-copy formulation created an unconstrained duplicate
  combine task.

They must not be used for performance decisions.

The corrected fixed-placement directed-duplex result is 2926.794 ms, or
2777.10 tok/s. Copy traffic under the current physical ownership consumes
essentially the entire scheduling benefit.

## Whole-owner-group placement

`temporal-group-placement.py` enumerates the 24 permutations of the four
complete 64-expert logical owner groups for every layer. It solves each
natural four-layer attention-delimited epoch. There are:

- no split logical owner groups;
- no expert replicas;
- no added persistent weights;
- no changed BF16 partial boundaries;
- no changed logical owner sum order.

The serial objective reduces the modeled critical exact-Q8 load from
825.745 ms to 715.004 ms and copy pressure from about 290 ms to
281.884 ms. The Q8 reduction is 110.741 ms.

When the selected group permutations are composed with balanced exact
attention and the corrected two-panel scheduler, every epoch solve is
optimal:

`temporal-panel-groupperm-epochserial-n2048-halfpipe-denseexact-epochs4-copy-v3.json`

- wall time: 2558.737 ms
- throughput: 3176.567 tok/s
- solver bound: 2558.737 ms
- margin to 3000 tok/s: 150.596 ms

This is the first current-trace architecture that passes the implementation
gate conservatively. It requires three exact components:

1. Compose the calibrated EPLB expert order with a per-layer permutation of
   complete logical owner groups.
2. Route physical compute to the new owner while placing each BF16 partial
   back into its original logical-owner slot before the fixed FP32 sum.
3. Add balanced exact attention epochs and two 1016-token-per-lane service
   panels.

Implementation is staged and default-off. Each stage must pass component
byte-identity before pp8128 timing.

## Stage 1: physical placement and canonical owner remap

The serial placement was composed with `placement-primary.eplb` by
`make-epoch-group-map.py`. The generated artifacts are:

- `placement-epoch-groupwave-serial-v1.eplb`
  - SHA-256 `43b678fa7f65478fc663ee32bf1e7d08d7f8529626d28343f32cd6ed060b5734`
- `placement-epoch-groupwave-serial-v1.groups`
  - SHA-256 `d70a032e7454e194f2bf1618d7465542c688ce7c58188a0c04e43520f467d7d0`

`GGML_CUDA_AW_GROUP_PLACEMENT=<file>` is default-off. For the bounded
diagonal service, route IDs select the new physical compute owner. Before
the existing owner-sum kernel, each physical partial is copied into the
slot for its original logical owner. The environment is rejected outside
the panelized diagonal path until later scheduler stages support it.

The pp512 component gate
`epoch-groupwave-diagonal-dump-v1-bench-p512-ub512` passed. All four layer-0
reduced F32 home outputs are byte-identical to
`diagonal-panel2048-halfpipe-dump-candidate-v1`:

- home 0: `3f19b36cb3d7004a49c4a68e48e84a0716e0e4eb3cdd0f718f34090a682dadee`
- home 1: `95eca717bc0967387a268aedae3de37410956ff22bbcdbcd25653d84691c11fa`
- home 2: `dd32d8d0d38f7d1c4320cc1313217ddcfaa4e8847eb8a4cc0d5f824dd3c7213c`
- home 3: `727f15358e0e1b958def0c9cbbc458e22f341b6f84cc9be151a85aadeaabbae4`

This validates the expert composition and canonical reduction boundary.

The stage-1 pp2048 timing gate is deliberately negative:

- current placement: 917.667416 ms, 2231.745 tok/s
- epoch placement on the unchanged diagonal: 924.597690 ms,
  2215.017 tok/s
- delta: +6.930274 ms

The epoch placement was optimized for same-layer two-panel resource balance,
not for the old cross-layer diagonal dependencies. It is not a standalone
diagonal optimization, and a pp8128 diagonal run is not justified. Retain
the exact loader/remap mechanism for the composed epoch scheduler.
