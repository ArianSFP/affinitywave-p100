# Direct peer owner-gather audit

Date: 2026-07-24

Status: exact low-memory diagnostic retained default-off; closed as a
performance architecture.

## Question

The existing live service first copies every owner's BF16 partial into a
four-owner receive tensor on the logical home GPU, then executes the original
rank-ordered FP32 sum locally. The synchronized N-panel copy experiment showed
that panel readiness alone did not prevent PCIe copy-engine exposure.

This experiment tested a materially different transport:

- leave each BF16 owner partial on the GPU that produced it;
- let the home GPU read the four partials directly through peer UVA;
- reproduce the existing rotated owner order;
- retain the BF16 rounding boundary after every ordered addition;
- write the final FP32 home output directly; and
- omit the four-owner receive allocation.

The implementation is selected by
`GGML_CUDA_AW_DIRECT_OWNER=1`. The independent in-run oracle is selected by
`GGML_CUDA_AW_DIRECT_OWNER_CHECK=1`. Both are default-off.

## Standalone primitive

`direct-owner-gather-probe.cu` measures one 2,032-token home slice per GPU.
Each GPU reads 23,812.5 KiB remotely, consumes its local owner contribution,
and writes 15,875 KiB of FP32 output. The staged control uses the original
peer-copy-then-local-sum structure.

Over 100 repetitions:

| mechanism | launch geometry | time (ms) |
| --- | --- | ---: |
| empty event chain | - | 0.007491 |
| staged owner copies | - | 3.991807 |
| staged local sum | - | 0.140549 |
| complete staged path | - | 4.137333 |
| direct peer gather | width 1, grid 56 | 2.674601 |
| direct peer gather | width 1, grid 112 | 2.961539 |
| direct peer gather | width 1, grid 224 | 2.975040 |
| direct peer gather | width 1, grid 448 | 3.171376 |
| direct peer gather | width 2, grid 56 | 2.508967 |
| direct peer gather | width 2, grid 112 | 2.664816 |
| direct peer gather | width 2, grid 224 | 2.549602 |
| direct peer gather | width 2, grid 448 | 2.471268 |
| direct peer gather | width 4, grid 56 | 2.605593 |
| direct peer gather | width 4, grid 112 | 2.503595 |
| direct peer gather | width 4, grid 224 | 2.469728 |
| direct peer gather | width 4, grid 448 | 2.551815 |

All variants reproduce the staged FP32 result bit-for-bit. The probe checked
199,753,728 FP32 values in total with zero mismatches. Width 4/grid 224 was
integrated because it is the fastest measured geometry, not because of an
assumed bandwidth model.

## Live dependency implementation

For each service group and home:

1. The home copy stream waits for all four owner `compute_done` events.
2. `aw_live_sum_owners_peer` reads the four owner partials in the original
   rotated order and writes the cell output.
3. The home records a consumed event and publishes it to the normal graph
   stream.
4. Every owner's release stream waits for every home-consumed event before
   allowing that owner's partial buffer to be reused.

The check mode also stages the same live partials through the original
receive layout, runs the original `aw_live_sum_owners`, and compares every
FP32 output bit inside the same process. A pp2048 live run completed all
layer-0 checks without an error.

An earlier comparison of dumps produced by separate benchmark processes was
discarded. The benchmark regenerates its input between processes, so those
dumps do not share an input oracle. The in-run staged comparison is the
valid exactness result.

## pp8128 result

A clean HalfPipe control was run in the same measurement window:

| mode | pp8128 wall (ms) | rate (tok/s) |
| --- | ---: | ---: |
| HalfPipe staged control | 3,143.375892 | 2,585.755 |
| HalfPipe direct owner | 3,264.180001 | 2,490.059 |

Direct owner-gather regresses wall time by 120.804109 ms and rate by
95.696 tok/s. This is a 3.70% throughput regression.

The paired Nsight Systems traces isolate the mechanism:

| critical-GPU metric | staged | direct | direct - staged |
| --- | ---: | ---: | ---: |
| trace window | 3,186.562 | 3,294.855 | +108.293 |
| busy union | 2,737.353 | 2,815.922 | +78.569 |
| idle in window | 449.209 | 478.933 | +29.724 |
| perfect-balance work floor | 2,563.441 | 2,633.494 | +70.053 |
| memcpy | 290.396 | 184.029 | -106.367 |
| owner sum | 5.574 | 115.405 | +109.831 |

Exact-Q8, dense SGEMM, FlashAttention, and GDN time differ by less than
0.5 ms each between the traces. The result is therefore not a workload
change. Direct peer reads replace copy-engine work with almost the same
amount of SM-side work, and the all-owner-ready dependency exposes another
idle interval. The isolated primitive looked favorable because its staged
copies were serialized in the standalone dependency; the production
diagonal wave already overlaps much of those copies with useful work.

## Memory result

With the production `GGML_CUDA_AW_GROUP_PATTERN=1111`, the skipped receive
allocation is exactly:

`4 owners * 8,128 tokens * 2,048 columns * 4 bytes = 266,338,304 bytes`

This is 254 MiB/GPU. It is a real capacity saving, but it does not remove the
larger gate, up, middle, route-output, and owner-partial materializations.
The complete N256 materialization-free service remains the stronger memory
boundary at 123,087,872 bytes/GPU total scratch and about 1.126 GiB/GPU less
than the current four live states.

## Decision

Retain direct owner-gather only as a default-off exact diagnostic or a
254 MiB/GPU emergency capacity option. Do not use it for the throughput
configuration.

Do not implement panel-ready direct peer gathering as the next performance
experiment. Earlier publication could attack the 29.724 ms idle increase,
but it cannot remove the measured 70.053 ms increase in the
perfect-balance device-work floor. The complete transport slice is also far
below the campaign's 300 ms architecture gate.

Reopen only with a materially new primitive that keeps remote movement off
the critical SM resource, removes the receive tensor, and measures below the
staged path while other production kernels are active. More event
partitioning around the same peer-load kernel is not new evidence.

## Evidence

- `direct-owner-gather-probe.cu`
- `run-direct-owner-gather-probe.sh`
- `direct-owner-gather-v1.out`
- `direct-owner-live-check-v1-bench-p2048-ub2048.out`
- `direct-owner-halfpipe-live-v1-bench-p8128-ub8128.out`
- `halfpipe-control-live-v3-bench-p8128-ub8128.out`
- `direct-owner-halfpipe-trace-v1.nsys-rep`
- `direct-owner-halfpipe-trace-v1.analysis.md`
- `halfpipe-control-trace-v1.nsys-rep`
- `halfpipe-control-trace-v1.analysis.md`
