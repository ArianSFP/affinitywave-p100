# Exact materialization-free MoE service audit

## Scope

This audit measures two exact ways to remove the full gate and up
materializations, then composes the surviving mechanism with the down-panel
implementation from `NPANEL-SERVICE-AUDIT.md`. The control is four-cell
coalesced BroadWave on the edge-route mix. Every result checks the complete
8,192-token by 2,048-column BF16 owner output on all four GPUs.

## Fused gate/up CTA

The first implementation extends the existing 512-thread gate/up CTA. Its
M64 epilogue evaluates the production SwiGLU expression and writes middle
directly; the existing M32 and M16 kernels are followed by tile-local
SwiGLU. It preserves the BroadWave split-K2 accumulation sequence.

The result is exact but slower:

| implementation | gate/up/SwiGLU | service stage |
| --- | ---: | ---: |
| separate BroadWave control | 10.7107 ms | 16.6480 ms |
| fused 512-thread CTA | 12.6207 ms | 18.5572 ms |

The fused CTA compiles at 108 registers/thread and 48 KiB shared memory.
Combining gate and up increases the resident CTA's instruction and data
working set enough to lose 1.9100 ms per service. This geometry is closed.

## Output-panel gate/up

The second implementation does not change BroadWave arithmetic. It traverses
the exact T64 gate and up weights in output-column panels, uses two compact
panel buffers, immediately evaluates SwiGLU into the full middle tensor, and
then reuses those panel buffers.

| gate/up panels | columns/panel | gate/up/SwiGLU | service stage | bytes removed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 10.7107 ms | 16.6480 ms | 0 |
| 2 | 256 | 11.5828 ms | 17.5186 ms | 33,542,144 |
| 4 | 128 | 14.0576 ms | 19.9925 ms | 50,307,072 |

Both variants have zero mismatches across 16,777,216 checked values/GPU.
N128 loses output-column parallelism just as it did for the down projection.
N256 is the retained gate/up boundary.

## Bounded composition

Two compositions remove the same 144 MiB from the isolated service:

| gate/up | down | instrumented service | device bytes | mismatches |
| --- | --- | ---: | ---: | ---: |
| N128 x 4 | N512 x 4 | 20.3966 ms | 537,130,628 | 0 |
| N256 x 2 | N256 x 8 | 18.4640 ms | 537,130,628 | 0 |

N256 gate/up plus N256 down is the retained bounded-memory composition. Its
1.8160 ms service tax is much lower than the 3.7485 ms tax of the N128/N512
composition.

The liveness calculation is:

- prior N512 arena: 131,017,728 bytes/GPU;
- replace the 33,554,432-byte N512 route output with a
  16,777,216-byte N256 route output;
- gate, up, and middle need 67,108,864 bytes during MoE;
- the phase-reused HeadFold ring supplies 58,261,504 of those bytes;
- final arena: 123,087,872 bytes/GPU;
- headroom under 128 MiB: 11,129,856 bytes/GPU.

No allocation grows with total context at fixed ubatch. Relative to the
1,332,211,712 bytes/GPU in the current four `aw_live_state` materializations,
the bounded design removes 1,209,123,840 bytes/GPU, or about 1.126 GiB/GPU.

This is a memory boundary, not a token-rate optimization. At 40 service
invocations its isolated arithmetic tax is about 72.6 ms. It can be retained
only when that memory is needed for KV capacity or when a production trace
shows the panel launches hidden under diagonal work.

Evidence:

- `fused-swiglu-v1-service-p2048-ub2048.out`
- `gupanel2-v1-service-p2048-ub2048.out`
- `gupanel4-v1-service-p2048-ub2048.out`
- `gupanel4-npanel4-v1-service-p2048-ub2048.out`
- `gupanel2-npanel8-v1-service-p2048-ub2048.out`
