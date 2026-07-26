# PairFold host-resident expert-streaming proof of concept

Date: 2026-07-26

## Outcome

Exact host-to-device streaming of PairWave Q8 expert weights works. Streaming
layers 16 through 23 produces the accepted c512 logits and reaches a warm
pp8128 median of 3247.809 ms, or 2502.61 tok/s.

This is a shadow-resident proof of concept. It takes pinned host snapshots
from the existing device-resident raw T64 weights and redirects only the
active PairWave descriptors to two streaming slots per GPU. The original
device copies remain allocated as the exactness oracle, so this version
demonstrates data movement, scheduling, exactness, and performance but does
not yet reclaim VRAM.

## Runtime design

The default-off selector is:

`GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1`

The inclusive streamed-layer range is selected with:

`GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS=16:23`

The benchmark wrapper exposes these as `PAIRFOLD_HOST_WEIGHTS` and
`PAIRFOLD_HOST_LAYERS` and preserves them through its clean production
environment.

The selected default policy uses:

- pinned raw T64 host snapshots, with no pageable-memory fallback;
- two 427,819,008-byte device slots per GPU;
- one low-priority nonblocking H2D stream per GPU;
- task-local, contiguous, parallel pair copies;
- compute-stream waits immediately before the expert-weight readers;
- a slot-free event after the last down-projection reader;
- generation, layer, pair, device, byte, bandwidth, and exposed-wait
  diagnostics.

Slot leases retain the expected generation, layer, and device identity. The
CPU scheduler test covers ranges `0:0`, `19:20`, `16:23`, and `0:39` over
five generations with delayed completions. It checks the repeated-pair
layer-19/20 boundary and rejects reuse before both chronological panels have
released the slot.

Three experimental policies remain available only as default-off
diagnostics:

- `GGML_CUDA_AW_PAIRFOLD_HOST_LOOKAHEAD=1`;
- `GGML_CUDA_AW_PAIRFOLD_HOST_PHASED=1`;
- `GGML_CUDA_AW_PAIRFOLD_HOST_SERIAL_H2D=1`.

Task-local contiguous parallel ingress was the fastest pp8128 policy.

## Exactness and default-off regression

The post-audit hardened streamed c512 run reports PPL 4.0783 and
saved-logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The final same-binary resident c512 run, with host streaming disabled,
reports the same PPL and SHA-256. Earlier lookahead, phased, and serialized
ingress variants also produced the accepted hash.

The final hardening records a slot as free only after chronological panel 1
has enqueued its last down projection and requires that panel-1 release on
every reuse, including across generations. This does not change the selected
two-slot copy schedule or the pp8128 performance result.

Artifacts:

- `hoststream-final-hardening-c512-v6.out` / `.err`;
- `hoststream-final-hardening-c512-v6-logits.bin`;
- `hoststream-final-default-c512-l16-23-v5.out` / `.err`;
- `hoststream-final-default-c512-l16-23-v5-logits.bin`;
- `hoststream-final-resident-c512-v5.out` / `.err`;
- `hoststream-final-resident-c512-v5-logits.bin`.

## Warm pp8128 performance

Sample 0 is initialization and is excluded from each warm median.

| Path | Median wall ms | tok/s |
| --- | ---: | ---: |
| Qualified production diagonal | 2790.956 | 2912.264 |
| Previously qualified resident PairFold | 3027.038 | 2685.133 |
| Same-binary resident PairFold | 3040.948 | 2672.85 |
| Host-stream layers 16:23 | 3247.809 | 2502.61 |

The streamed warm samples are:

`3289.481, 3247.809, 3278.445, 3243.917, 3244.737 ms`

Their peak-to-peak range is 45.564 ms, or 1.403% of the median. Relative to
the same-binary resident control, eight streamed layers add 206.861 ms or
6.802% wall time and reduce throughput by 6.37%. Relative to the production
diagonal comparator, they add 456.853 ms or 16.369% wall time and reduce
throughput by about 14.1%.

Across warm generations 1 through 5, the serialized pair transfer rate is
5.773-5.797 GiB/s. The median is 5.781 GiB/s. Total gate/up and down
ready-event waits are only about 0.24 ms per generation. At pp8128, weight
readiness is therefore hidden; the remaining wall penalty is PCIe, copy
engine, memory-system, and kernel contention.

Artifact:

`hoststream-final-default-warm-pp8128-l16-23-v5.out` / `.err`

## pp2048 trace

The generation-1 task window is 1328.216 ms. It contains 16 exact
427,819,008-byte host-weight transfers, for 6,845,104,128 bytes total.
Their all-device union is 689.645 ms, equivalent to 9.244 GiB/s over the
union:

- 376.316 ms, or 54.567%, overlaps at least one SM kernel;
- 313.329 ms, or 45.433%, is outside all SM work;
- four copies are active concurrently for 580.852 ms;
- each GPU receives 1,711,276,032 bytes in 635.009-637.471 ms.

The full trace has no peer-named kernel and no cross-stream same-GPU SM
overlap. Per-GPU SM busy union is 46.266-46.631%. Both physical pairs are
active for 469.039 ms.

Artifacts:

- `hoststream-final-default-trace-pp2048-l16-23-v5.nsys-rep`;
- `hoststream-final-default-trace-pp2048-l16-23-v5.sqlite`;
- `hoststream-final-default-trace-pp2048-l16-23-v5-analysis.json`.

## Memory

Eight selected layers require a 6,845,104,128-byte pinned host snapshot.
The two streaming slots allocate 855,638,016 bytes per GPU. Snapshot
creation took 4900.543 ms once during process initialization and is excluded
from the warm samples.

The largest resident PairFold used-memory high-water was 13,785,432,064
bytes. The largest shadow-streaming high-water was 14,653,652,992 bytes, an
increase of 868,220,928 bytes per worst-case GPU. This is expected because
the proof of concept keeps both the resident originals and the two new
slots.

If loader-time placement removes the original device copies:

- this eight-layer window would net about 816 MiB reclaimed per GPU after
  retaining two slots;
- all 40 Q8 layers would require 34,225,520,640 bytes, or 31.875 GiB, of
  host snapshots;
- each GPU currently owns 20 408 MiB layer partitions, so two slots would
  net 7344 MiB, or 7.171875 GiB, reclaimed per GPU.

The full-model host snapshot is close to the rig's available host-memory
headroom and should not be attempted by extending this snapshot-only
implementation blindly. Loader-time host placement and a bounded pinned
staging source are the appropriate next phase.

## Full-model estimate

At the measured pp8128 pair rate, perfect overlap of both pair ingress paths
would copy all 31.875 GiB in about 2.76 seconds. The raw transfer-only
ceiling is therefore about 2948 tok/s. Hiding the full transfer volume under
the same-binary resident PairFold wall requires 10.48 GiB/s across both
pairs; the idealized measured pair aggregate is 11.56 GiB/s. This leaves
only about 10% raw bandwidth headroom and does not include the measured
compute-contention penalty.

A first-order bytes-linear extrapolation of the measured eight-layer wall
tax is:

`3040.948 ms + 5 * 206.861 ms = 4075.253 ms`

That is about 1994.5 tok/s for all 40 Q8 layers on this P100 PCIe rig. It is
an engineering estimate, not a bound: layer mix, endpoint contention, host
memory pressure, and a loader-native source path can move it in either
direction. The direct measurements nevertheless make a production-rate
full-Q8 result on this host path unlikely without improving PairFold itself
and reducing transfer contention or volume.

No sub-Q8 weights were tested or evaluated. That remains outside the
project's accuracy contract.

## Decision

The architecture is viable as an exact experimental path: host-resident
expert streaming works, both pairs can transfer concurrently, and pp8128
readiness stalls are fully hidden. It remains default-off and is not
production-qualified because even the eight-layer proof is 14.1% below the
production token rate.

The next implementation boundary is loader-time host placement for a small
layer window, replacing the retained originals and demonstrating actual
VRAM recovery. Only after that memory proof should the range be expanded or
the ingress path be tuned. The current snapshot registry is process-lifetime
state intended for the single-model benchmark process; model unload/reload
support is not part of this proof of concept.

## Loader-time follow-up

That implementation boundary has now been completed for layers 16 through
23. The selected tensors remain in CUDA-pinned loader buffers, are packed to
T64 during model load, and do not retain device-resident originals.

The path remains byte-exact at c512 and recovers 843,055,104 bytes (804 MiB)
per GPU after both streaming slots are live. Its warm pp8128 median is
3289.595 ms / 2470.821 tok/s. Full implementation, memory, exactness, and
performance details are in `LOADER-HOST-PLACEMENT.md`.

The range was subsequently expanded to all 40 PairFold inference layers.
It remains byte-exact, recovers 7.109375 GiB/GPU, and measures
4238.061 ms / 1917.858 tok/s at warm pp8128. The measured wall is 3.995%
slower than this document's bytes-linear estimate. The full host consumes
31.875 GiB of pinned memory and leaves little operating margin on this
machine.
