# Experiment ledger

This ledger is a navigation layer over the ASCII-normalized audit archive.
"Closed" means the measured mechanism should not be retried without
materially new evidence. "Replay-only" means no production performance or
exactness claim.

## Placement and scheduling

| Experiment | Question | Outcome | Detailed record |
| --- | --- | --- | --- |
| Original AffinityWave | Can cross-layer owner service reach the original gate? | Parked. The optimistic model missed its gate and the 2828 tok/s checkpoint was service-only. | [parked record](archive/prehistory/p100-affinitywave-parked-20260723.md) |
| TemporalQueue | Can one-copy temporal placement and a bounded four-layer queue remove imbalance? | Standalone two-panel rewrite closed at current arithmetic rates. Useful bounded-memory pieces retained. | [audit](archive/frontier/TEMPORAL-QUEUE-AUDIT.md) |
| Epoch GroupWave | Can diagonal scheduling preserve exact owner groups with bounded memory? | Produced the exact 2776.379 tok/s diagonal candidate and the 0.831 GiB/GPU reclamation boundary. | [audit](archive/frontier/EPOCH-GROUPWAVE-AUDIT.md) |
| HeadFold | Can recurrent/attention heads be split exactly across GPUs? | Exact central replay crossed 3000; runtime was implemented and banked, but final pp8128 was about 2299 tok/s. | [audit](archive/frontier/HEADFOLD-AUDIT.md), [production log](archive/frontier/HEADFOLD-R44-N256-PRODUCTION.md) |
| PairWave/PairFold | Can two fixed pairs own alternating layers with one expert copy? | One-copy loader, lane-exact service, and full 80-task runtime are exact at c512. Warm pp8128 is 2685.133 tok/s, 7.877% below the matched diagonal control, so it remains default-off experimental. | [checkpoint](PAIRFOLD-CHECKPOINT.md), [audit](archive/frontier/PAIRWAVE-AUDIT.md) |
| FoldWave | Can exact balanced attention and StreamCacheRelay compose into a large gain? | Central replay 2734.89 tok/s. Scheduler implementation rejected. | [replay](archive/frontier/FOLDWAVE-REPLAY.md) |
| ScanFold | Can prefix-scan recurrence and balanced attention remove the corridor? | Full ScanFold rejected: timing gate failed and affine composition changed FP32 order. | [audit](archive/frontier/SCANFOLD-AUDIT.md) |
| InterferenceWave/TileFrontier | Can a two-diagonal ready queue use natural P2 work and bounded tile slots? | Positive 3545.585 tok/s offline replay; isolated proof-of-concept only, not production. | [audit](archive/frontier/INTERFERENCEWAVE-TILEFRONTIER-AUDIT.md) |
| StitchRail | Can exact ready-tile publication remove whole-diagonal idle? | Replay/trace study only. Generated manifest marked non-deployable; final campaign retained smaller exact static improvements instead. | [notebook](archive/stitchrail/NOTES-codex.md) |
| Balanced attention | Can attention work be split while preserving exact order? | Explicitly paused for a later effort. | [paused record](archive/stitchrail/BALANCED-ATTENTION-PAUSED.md) |

## Exact-Q8 kernel and tile experiments

| Experiment | Outcome | Detailed record |
| --- | --- | --- |
| BroadWave/HalfPipe | `halfpipe_sync` was exact and recovered a small staging cost. Split-barrier variants closed. | [HalfPipe audit](archive/frontier/HALFPIPE-AUDIT.md) |
| DoubleWave | About 20% slower because shared-load fanout grew 66.7%. Closed. | [audit](archive/frontier/DOUBLEWAVE-AUDIT.md) |
| VectorWave | Exact but 8.30% slower due shared-memory transaction behavior. Closed. | [audit](archive/frontier/VECTORWAVE-AUDIT.md) |
| RailWave | Exact but 18.77% slower; split rails lost before reduction. Closed. | [audit](archive/frontier/RAILWAVE-AUDIT.md) |
| ExpandWave | Weight conversion was not the bottleneck; large expanded caches unjustified. Closed. | [audit](archive/frontier/EXPANDWAVE-AUDIT.md) |
| PairWave 2xM32 | Byte-identical at layer 0 but slower at pp2048. Closed. | [PairWave audit](archive/frontier/PAIRWAVE-AUDIT.md) |
| PairWave 2xM16 | Byte-identical at layer 0 but slower at both tested occupancy points. Closed. | [PairWave audit](archive/frontier/PAIRWAVE-AUDIT.md) |
| CohortRail variants | M64/M32/M16/P2 launch, cursor, grid, prefetch, rail, and barrier variants were traced. The final static source, not every trace candidate, is retained. | [frontier trace archive](archive/frontier/) |
| M512+M512 stacking | Exact but flat. Closed. | [stacked projection summary](archive/stitchrail/p100-stacked-projection-summary.md) |
| M32+M32 stacking | 1.672x isolated but less than 5 ms realistic pp8128 ceiling. Not integrated. | [stacked projection summary](archive/stitchrail/p100-stacked-projection-summary.md) |
| Router GemmEx selector | Exact projected gain about 1.63 ms pp8128. Below continuation threshold. | [final notebook](archive/stitchrail/NOTES-codex.md) |

Every individual CohortRail trace analysis is retained. Relevant file
prefixes include:

```text
cohortrail-grid*
cohortrail-m32-*
cohortrail-m64-*
cohortrail-p2-*
cohortrail-pairwave-*
cohortrail-prefetch-*
cohortrail-rail-*
cohortrail-single-*
```

## Memory and transport experiments

| Experiment | Outcome | Detailed record |
| --- | --- | --- |
| N-panel down projection | N512 was the best down-only memory boundary. Same-layer copy pipeline exposed too much PCIe time. | [audit](archive/frontier/NPANEL-SERVICE-AUDIT.md) |
| Materialization-free gate/up | Fused gate/up/SwiGLU was exact but slower. N256 panels retained only as a memory option. | [audit](archive/frontier/MATERIALIZATION-FREE-SERVICE-AUDIT.md) |
| Direct owner gather | Exact and saves memory, but raises the perfect-balance work floor. Diagnostic only. | [audit](archive/frontier/DIRECT-OWNER-AUDIT.md) |
| RelayWave | Projection-only helpers cannot meet the 3500 tok/s replay gate at measured rates. Closed as an architecture. | [audit](archive/frontier/RELAYWAVE-AUDIT.md) |
| RecurFill | Exact and memory-safe, but under 1% wall-time ceiling. Closed. | [audit](archive/frontier/RECURFILL-AUDIT.md) |
| CacheRelay models | Frequency, wide-frequency, and hypergraph R40 variants were modeled. No persistent helper cache was promoted. | [model records](archive/frontier/) |
| PairCache | Static R16 exact-weight prefetch was positive on untouched replay, but runtime/model boundaries remained unresolved. No implementation. | [PairCache notes](archive/paircache/PAIRCache-NOTES-codex.md) |

## Routing and reduction

| Experiment | Outcome | Detailed record |
| --- | --- | --- |
| Route pooling | Changed lane tile classes and 174 layer-0 F32 values. Rejected. | [PairWave audit](archive/frontier/PAIRWAVE-AUDIT.md) |
| Atomic route scatter | Fast-looking but not bitwise exact. Rejected. | [route-scatter audit](archive/frontier/ROUTE-SCATTER-AUDIT.md) |
| Stable route scatters | Exact but slower because they removed useful Pascal CTA parallelism. Closed. | [route-scatter audit](archive/frontier/ROUTE-SCATTER-AUDIT.md) |
| W2/W4 owner reduction | W4 exact owner reduction and canonical sum retained; W2 remains a control. | [final result](archive/stitchrail/FINAL-RESULTS.md) |

## Dense, recurrent, and attention work

| Experiment | Outcome | Detailed record |
| --- | --- | --- |
| Exact dense selectors | Eight complete shape/signature selectors retained. | [final result](archive/stitchrail/FINAL-RESULTS.md) |
| GDN precomputed decay | Retained without changing recurrence order. | [final notebook](archive/stitchrail/NOTES-codex.md) |
| Balanced attention | Not implemented in the final campaign. | [paused record](archive/stitchrail/BALANCED-ATTENTION-PAUSED.md) |
| FA query tile probes | Query-tile variants measured; no headline production claim. | [frontier archive](archive/frontier/) |

## Pre-AffinityWave closed axes

The prehistory matters because several tempting ideas had already failed:

- AsyncEP was 37% slower because expert filtering was also the
  row-concentration mechanism.
- Two-batch SM-kernel overlap increased busy percentage without reducing
  wall time.
- NCCL tuning was not the main prefill lever on this uniform topology.
- FP16/half2 substitution and sub-Q8 model weights were outside the accuracy
  contract.
- Persistent grouped GEMM, device-side routing plans, and submit workers
  were retained because they removed launches and host drains.

See [prehistory](archive/prehistory/) before reopening any of those classes.
