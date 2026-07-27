# Exact route-scatter audit

## Hypothesis

The live service originally launches one 256-thread block for every
cell/expert descriptor. Each block scans every top-8 route in its cell and
retains only one expert. This rereads the route IDs 64 times per owner.

Four default-off replacements were tested:

- `atomic`: scan every route once and allocate its expert row with
  `atomicAdd`;
- `stable8`: use eight warps, each responsible for eight experts, and
  preserve slot order with warp ballots;
- `stable256`: count 256-slot segments, prefix their exact row ranges, and
  classify each segment with 64 warp ballots;
- `radix256`: replace the 64 comparisons with a stable seven-bit shared
  radix partition;
- `segment8`: prefix 256-slot segments, then let eight lanes own eight
  experts each and traverse their segment monotonically.

The last four preserve the original slot order by construction. The atomic
variant does not.

## Exactness

A matched pp2048 run dumped every layer-0 BF16 owner partial for four token
homes and four logical owners. Each file contains 1,048,576 BF16 values.

| mode | files checked | result |
| --- | ---: | --- |
| `stable8` | 16 | byte-identical |
| `stable256` | 16 | byte-identical |
| `radix256` | 16 | byte-identical |
| `segment8` | 16 | byte-identical |
| `atomic` | 16 | all files differ; 1,366 differing bytes |

The atomic row allocation changes which routes share an M64/M32/M16 tile.
That changes a small number of final BF16 roundings even though token-route
bookkeeping follows the allocated rows. It therefore fails the exact
component gate and is not a candidate for production.

The deterministic reference was rerun after all source changes. Its 16
files are byte-identical to the earlier repaired BroadWave dump, so the
oracle itself is stable.

## Timeline measurement

Nsight Systems pp2048 traces contain two graph-pre-capture waves and one
measured wave on four GPUs. Kernel sums below are divided by 12 to give one
critical-GPU-equivalent full wave. Every listed kernel has 1,920 invocations
in the raw trace.

| route planner | count | build plan | prefix | scatter | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic | 0.794 ms | 11.065 ms | - | 2.492 ms | 14.352 ms |
| `stable8` | 0.802 ms | 11.053 ms | - | 12.511 ms | 24.366 ms |
| `segment8` | 0.801 ms | 11.048 ms | 0.917 ms | 14.720 ms | 27.486 ms |

The deterministic implementation's redundant route loads are served from
cache while its 64 independent CTAs expose enough parallelism to P100.
`stable8` reduces loads but leaves only eight warps active for each cell.
The segmented implementations restore CTA count but serialize expert
classification or slot traversal inside each CTA. Both lose.

The full pp8128 confirmation agrees with the timeline:

| mode | time | rate |
| --- | ---: | ---: |
| deterministic | 3,156.324 ms | 2,575.147 tok/s |
| `stable8` | 3,194.271 ms | 2,544.556 tok/s |

The exact alternative regresses by 37.947 ms. The pp2048 cold/capture wall
samples varied enough to suggest a false gain before tracing; they are not
used for the decision.

## Decision

The route-scatter family is closed.

- The only fast-looking O(routes) form fails bitwise owner-partial
  correctness.
- Stable O(routes) forms lose the CTA parallelism that makes the apparent
  64-pass implementation cheap on P100.
- Even deleting count, build, and scatter entirely has only a 14.35 ms
  pp2048 critical-GPU ceiling in the measured trace. Parallelizing the
  single-thread plan cannot make this an architectural-scale contributor.

The production default remains `deterministic`. All alternatives remain
default-off research selectors only.

Evidence:

- `route-scatter-deterministic-dump-v1-bench-p2048-ub2048.out`
- `route-scatter-atomic-dump-v1-bench-p2048-ub2048.out`
- `route-scatter-stable8-dump-v1-bench-p2048-ub2048.out`
- `route-scatter-stable256-dump-v1-bench-p2048-ub2048.out`
- `route-scatter-radix256-dump-v1-bench-p2048-ub2048.out`
- `route-scatter-segment8-dump-v1-bench-p2048-ub2048.out`
- `dumps/route-scatter-*-layer0-v1/`
- `route-scatter-det-live-p2048-v1.nsys-rep`
- `route-scatter-stable8-live-p2048-v1.nsys-rep`
- `route-scatter-segment8-live-p2048-v1.nsys-rep`
- `route-scatter-det-pp8128-v1-bench-p8128-ub8128.out`
- `route-scatter-stable8-pp8128-v1-bench-p8128-ub8128.out`
