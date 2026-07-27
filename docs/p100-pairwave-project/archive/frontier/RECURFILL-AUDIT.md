# RecurFill exact recurrent-corridor audit

Date: 2026-07-24

Status: closed as an architectural contributor.

## Question

RecurFill proposed running already-required dense `z` and output projections
in a low-priority stream while the exact head-sharded Gated DeltaNet
recurrence runs in a high-priority stream. The proposal is attractive because
it adds no persistent weight copy and bounds transient `z` storage to between
one and four 8,323,072-byte slots.

The gate tested here is deliberately more favorable than a production
implementation. It places all eight independent dense GEMMs next to the exact
16-segment recurrence with no normalization, gating, head publication, peer
copy, slot backpressure, or downstream dependency cost. Therefore its measured
gain is an upper bound on the useful overlap available to the full design.

## Probe

`recurfill-overlap-probe.cu` runs on all four P100s and reproduces:

- one exact HeadFold H8 recurrence over 8,128 tokens as sixteen chronological
  508-token publications;
- four `M1024 x N2032 x K2048` FP16-weight/F32-output `z` GEMMs per GPU;
- four `M2048 x N508 x K4096` FP16-weight/F32-output projection GEMMs per GPU;
- high-priority recurrence and low-priority nonblocking dense streams;
- isolated recurrence, isolated dense work, and concurrent makespans; and
- bitwise comparison of the recurrence and all selected dense outputs against
  isolated references.

Every run checks 46,301,184 FP32 values and reports zero bitwise errors.
`run-recurfill-overlap-probe.sh` preserves the four-GPU lock and watchdog and
uses plain `env` when collecting Nsight Systems traces.

## The resource premise was false

The proposal assumed algorithm 10 was a 53-register, 256-thread, 16-KiB
kernel that could coexist cheaply with the recurrence. The production-shaped
trace instead resolves algorithm 10 to:

| resource | measured implementation |
| --- | ---: |
| kernel | `maxwell_sgemm_fp16_128x64_tn` |
| threads/CTA | 128 |
| registers/thread | 120 |
| static shared memory | about 13 KiB |
| exact H8 GDN registers/thread | 40 |
| exact H8 GDN shared memory | 0 |

The algorithm-10 trace contains 544 SGEMM calls averaging 1.295 ms. The 768
GDN publications average 0.743 ms under contention, with a 1.841 ms maximum,
while isolated GDN publications cluster near 0.495 ms. The two streams do
execute concurrently, but they slow each other enough that little wall time
is removed.

## Exact selector sweep

Because cuBLAS algorithms can map to different resource shapes, all locally
available selectors that completed correctly were tested before closing the
mechanism:

| algorithm | dense isolated (ms) | GDN+dense concurrent (ms) |
| ---: | ---: | ---: |
| -1 | 10.092 | 16.149 |
| 2 | 11.595 | 16.847 |
| 3 | 9.452 | 15.590 |
| 4 | 11.381 | 16.850 |
| 5 | 10.569 | 15.776 |
| 6 | 12.236 | 15.634 |
| 7 | 9.926 | 16.443 |
| 8 | 9.033 | 15.251 |
| 9 | 9.922 | 16.625 |
| 11 | 9.211 | 15.649 |
| 99 | 10.025 | 16.130 |
| 100 | 10.008 | 16.131 |

Algorithm 8 has the best absolute concurrent time. Its trace resolves the
selected calls to `maxwell_sgemm_fp16_64x64_tn`: 64 threads, 124
registers/thread, and about 9 KiB shared memory. It co-schedules somewhat
better by reducing CTA size, not by exposing a large unused execution
corridor.

The 100-repeat confirmations are:

| configuration | GDN (ms) | dense (ms) | concurrent (ms) | bitwise errors |
| --- | ---: | ---: | ---: | ---: |
| algorithm 10 | 7.3831 | 8.8225 | 15.8443 | 0 |
| algorithm 8 | 7.3965 | 9.0358 | 15.2401 | 0 |

The correct comparison for an implementation is the fastest current
sequential schedule, algorithm-10 dense plus GDN:

`7.383086 + 8.822480 = 16.205566 ms/layer`

The best concurrent result is 15.240111 ms/layer, so the independent-work
upper bound is only:

`16.205566 - 15.240111 = 0.965455 ms/layer`

Across all 30 recurrent layers this is 28.96 ms before adding the production
normalization, gate, event, peer-copy, slot, and backpressure costs.

## Decision

RecurFill is exact and memory-safe, but it is not architectural-scale on
GP100. The measured upper bound is under 1% of current pp8128 wall time and
cannot justify production integration or the one/two/four-slot sweep. Stream
priority cannot repair issue and register contention between these kernels.

Reopen only if the dense projection kernel is replaced by a materially lower
register implementation whose isolated speed is at least algorithm 10 and a
new full-corridor probe removes at least 300 ms over 30 layers.

## Evidence

- `recurfill-overlap-probe.cu`
- `run-recurfill-overlap-probe.sh`
- `recurfill-overlap-v2.out`
- `recurfill-overlap-trace-v1.nsys-rep`
- `recurfill-selector-a-1-v1.out`
- `recurfill-selector-a2-v1.out` through
  `recurfill-selector-a11-v1.out`
- `recurfill-selector-a99-v1.out`
- `recurfill-selector-a100-v1.out`
- `recurfill-selector-a8-v2.out`
- `recurfill-selector-a10-v2.out`
- `recurfill-selector-a8-trace-v1.nsys-rep`
