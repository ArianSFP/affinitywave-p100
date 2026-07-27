# FoldWave dependency replay

Date: 2026-07-24

## Decision

The measured FoldWave composition does not justify a scheduler
implementation. Its central exact estimate is 2,971.971 ms, or 2,734.89
tok/s. This misses 3,000 tok/s by 262.637 ms, 3,500 tok/s by 649.685 ms,
and 4,000 tok/s by 939.971 ms.

An intentionally invalid upper bound, which combines mechanisms that cannot
coexist and omits required rendezvous cost, reaches only 2,707.216 ms, or
3,002.35 tok/s. It misses 3,500 tok/s by 384.930 ms and 4,000 tok/s by
675.216 ms. Therefore no retained placement mechanism can make the current
arithmetic backend a 3,500 or 4,000 tok/s system.

The exact balanced-attention epoch remains useful as a supporting mechanism.
The StreamCacheRelay placement remains useful only as a low-memory boundary
and as evidence about expert aggregation. Neither is a current integration
candidate.

## Replay calibration

The replay uses the static AffinityWave dependency graph and preserves the
base critical-path scheduling order across scenarios. This avoids treating a
different greedy scheduling tie-break as a performance change.

| quantity | measured/replayed value |
| --- | ---: |
| BroadWave measured profile window | 3,187.830587 ms |
| BroadWave replay | 3,188.557522 ms |
| absolute error | 0.726935 ms |
| relative error | 0.0228% |

The model is calibrated by measured mechanism-level quantities:

- diagonal R0 exact-Q8 work: 866.920 ms;
- diagonal R16 exact-Q8 work: 852.241 ms;
- diagonal R0 communication: 322.646 ms;
- diagonal R16 communication: 180.086 ms;
- exact dense selector factor: 0.906;
- current attention critical slice: 44.013 ms/layer;
- exact balanced-attention epoch: 34.664 ms/layer;
- long-corridor R16 exposed copy: 0.026688 ms/layer;
- short-corridor R16 exposed copy: 4.189514 ms/layer.

## The important aggregation result

The earlier 278.4 ms R16 projection benefit was a same-layer pooling result.
It assumed that all four token shards rendezvous at the same layer, so rows
for an expert can be merged before tiling. The live AffinityWave diagonal
contains four cells from different layers. Moving complete owner groups to
the token home in that schedule does not merge those rows.

At the actual diagonal service unit, R16 reduces modeled exact-Q8 work only
from 866.920 to 852.241 ms, a 14.679 ms or 1.69% reduction. It does reduce
modeled owner communication from 322.646 to 180.086 ms. The resulting
barrier-free DAG improvement is 52.698 ms, not 278.4 ms.

At the individual-cell level, local caching also destroys the aggregation
that BroadWave obtains at the owner:

- R0 issued rows: 994,544;
- R16 issued rows: 1,551,760;
- increase: 56.0%.

Pooling the four different-layer cells on each diagonal restores most of the
occupancy, but still leaves only the 14.679 ms exact-Q8 saving above. Recovering
the large same-layer benefit requires a layer rendezvous. The rejected CP4 GDN
scan cannot carry that rendezvous, so the cost is not composable with the live
chronological recurrence.

## Scenario results

| scenario | wall | rate | status |
| --- | ---: | ---: | --- |
| diagonal R16 | 3,135.860 ms | 2,591.95 tok/s | exact placement model |
| plus exact dense selectors | 3,064.393 ms | 2,652.40 tok/s | exact mechanisms |
| central exact composition | 2,971.971 ms | 2,734.89 tok/s | retained ceiling |
| short-overlap conservative case | 3,231.974 ms | 2,514.87 tok/s | regression |
| same-layer pooling with no barrier | 2,915.638 ms | 2,787.73 tok/s | invalid control |
| invalid pooling plus attention | 2,823.216 ms | 2,878.99 tok/s | invalid control |
| all measured ceilings combined | 2,707.216 ms | 3,002.35 tok/s | noncomposable bound |

The central composition includes barrier-free diagonal R16, exact dense
selectors, the full measured balanced-attention ceiling, and long-corridor
weight streaming. The noncomposable bound additionally grants same-layer
pooling without its rendezvous and subtracts the RelayWave panel ceiling even
though RelayWave requires a full hidden all-gather and exhibits measured
copy/Q8 contention.

## Consequence

ScanFold supplied one valid new placement primitive, but it closes as a
complete architecture on this rig. Further scheduler work is gated on a
fundamentally faster exact expert backend. The next experiment isolates the
cost of native Q8 decode and scale conversion by pre-expanding a small
ephemeral T64 weight panel to FP32 and running the same ordered FP32 FMA
chains. It must predict at least 300 ms of pp8128 wall removal before a
system integration is considered.

Evidence:

- `foldwave-replay.py`
- `foldwave-replay-v1.json`
- `crosswave-model.py`
- `broadwave-live-pp8128.analysis.json`
- `cache-relay-model-frequency-v1.json`
- `SCANFOLD-AUDIT.md`
