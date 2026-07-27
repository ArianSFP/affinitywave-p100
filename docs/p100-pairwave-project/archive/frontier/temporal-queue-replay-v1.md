# TemporalQueue placement and bounded-memory replay

- Capture pass: 3
- Capture SHA-256: `0dd89bf538822716dbed425a6aadbe1b90f452de0a867859e9ca3ad4a1c618e4`
- Placement solver seconds: 350.736
- Local swap steps/layer/start: 2000

All placements retain one copy of every expert and exactly 64 experts
per layer/GPU. Logical owner is the pre-existing expert_id/64 group;
the communication model preserves its rank-ordered FP32 accumulation
and BF16 boundary even when physical placement splits the group.

| placement | Q8 ms | copy critical ms | serialized ms | resource bound ms | input vectors | FP32 chain vectors | BF16 final vectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current EPLB groups | 937.457 | 294.644 | 1232.101 | 937.457 | 881,433 | 0 | 881,433 |
| compute-only balanced | 676.069 | 569.075 | 1245.144 | 676.069 | 927,647 | 985,762 | 872,421 |
| coactivation refined | 726.579 | 430.795 | 1157.374 | 726.579 | 912,586 | 573,556 | 880,582 |

Four-layer queue lower bounds:

| placement | epoch Q8 ms | epoch copy ms | epoch serialized ms | epoch resource bound ms |
| --- | ---: | ---: | ---: | ---: |
| current | 800.438 | 290.165 | 1090.603 | 800.438 |
| compute_only | 676.022 | 533.681 | 1209.703 | 676.022 |
| coactivation | 713.219 | 424.156 | 1137.375 | 713.219 |

Capacity-constrained token homes:

| policy | communication critical ms | moved rows |
| --- | ---: | ---: |
| fixed contiguous | 413.786 | 0 |
| sequential exact-capacity | 409.907 | 43,190 |
| one-layer oracle lookahead | 408.355 | 52,322 |

Memory accounting:

- Bounded arena: 97,725,440 bytes (93.20 MiB), fits 128 MiB: true.
- Major current aw_live_state materializations: 1,332,211,712 bytes/GPU (1.241 GiB).
- Net modeled reduction: 1.150 GiB/GPU.

The epoch resource number is an optimistic dependency lower bound,
not a wall-time prediction. The serialized number is the conservative
no-overlap bound. A retained mechanism still requires measured N128
chain transport, small-segment dense selectors, and a dependency replay
with measured event/copy-engine contention.
