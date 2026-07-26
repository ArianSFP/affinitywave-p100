# RailWave exact split-rail audit

Date: 2026-07-24

## Architecture

BroadWave keeps two independent FP32 accumulator chains for every output:
K offsets 0-15 and 16-31 within each Q8_0 block. RailWave assigns those
already-independent chains to separate CTAs. The two CTAs consume disjoint
activation and weight values, so total FMA and weight traffic are unchanged.
Rail 0 writes the normal output, rail 1 writes an FP32 partial, and a separate
kernel performs BroadWave's final rail0 + rail1 addition.

Each rail uses an M64xN128 tile with a K16 stage:

- 80 registers/thread;
- 16 KiB shared memory;
- three 256-thread CTAs/SM;
- 512 FFMA instructions/rail, or the same 1,024 per pair as BroadWave;
- zero local-memory allocation;
- 128 MiB reusable worst-case partial scratch.

The service oracle passed bitwise on all four GPUs.

## Measured result

Four-cell coalesced edge-route service:

| quantity | BroadWave | RailWave | change |
| --- | ---: | ---: | ---: |
| instrumented stage | 16.814186 ms | 19.970924 ms | +18.77% |
| gate | 5.283967 ms | 5.908104 ms | +11.81% |
| up | 5.279710 ms | 5.903179 ms | +11.81% |
| down | 5.179555 ms | 7.090537 ms | +36.89% |
| device allocation | 688,088,708 B | 822,306,436 B | +128 MiB |
| mismatches/GPU | 0 | 0 | bitwise |

The matched Nsight traces separate compute from the materialized partial:

| kernel | average call |
| --- | ---: |
| BroadWave M64 N128 compute | 4.773927 ms |
| RailWave paired-rail compute | 4.958832 ms |
| RailWave gate/up final add | 0.434648 ms |
| RailWave down final add | 1.514674 ms |

The three-CTA compute kernel is already 3.87% slower before the final add.
Two CTAs duplicate descriptor, address, scale, loop, and control instructions;
the extra resident warps do not compensate on P100. Down then pays the
expected fourfold wider partial traffic.

## Decision

RailWave is closed. A counter/flag handoff could remove the materialized
partial, but it cannot rescue a compute kernel that already loses. Atomics
would also jeopardize the required rail0-then-rail1 order. No counter handoff,
grid-size sweep, or tail port is justified.

This result combines with Compact-Q8 and ExpandWave to establish a harder
BroadWave boundary:

- more occupancy loses to synchronization/control work;
- removing dequantization is worth only 2.48%;
- splitting the exact accumulator rails loses before reduction.

Evidence:

- `railwave-v1-coalesced-service-p2048-ub2048.out`
- `railwave-v1-coalesced-trace.nsys-rep`
- `railwave-v1-coalesced-trace.sqlite`
- `broadwave-v2-coalesced-trace.nsys-rep`
- `broadwave-v2-coalesced-trace.sqlite`
- `railwave-sass-summary-v1.txt`
