# TemporalQueue placement and bounded-memory replay

- Capture pass: 3
- Capture SHA-256: `0dd89bf538822716dbed425a6aadbe1b90f452de0a867859e9ca3ad4a1c618e4`
- Placement solver seconds: 47.260
- Local swap steps/layer/start: 0

All placements retain one copy of every expert and exactly 64 experts
per layer/GPU. Logical owner is the pre-existing expert_id/64 group;
the communication model preserves its rank-ordered FP32 accumulation
and BF16 boundary even when physical placement splits the group.

| placement | Q8 ms | copy critical ms | serialized ms | resource bound ms | input vectors | FP32 chain vectors | BF16 final vectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current EPLB groups | 937.457 | 294.644 | 1232.101 | 937.457 | 881,433 | 0 | 881,433 |
| compute-only balanced | 676.065 | 563.664 | 1239.729 | 676.197 | 928,720 | 947,193 | 873,828 |
| coactivation refined | 793.013 | 403.791 | 1196.805 | 793.013 | 908,695 | 398,508 | 878,852 |

Four-layer queue lower bounds:

| placement | epoch Q8 ms | epoch copy ms | epoch serialized ms | epoch resource bound ms |
| --- | ---: | ---: | ---: | ---: |
| current | 800.438 | 290.165 | 1090.603 | 800.438 |
| compute_only | 676.031 | 536.241 | 1212.272 | 676.031 |
| coactivation | 760.767 | 391.952 | 1152.719 | 760.767 |

Capacity-constrained token homes:

| policy | communication critical ms | moved rows |
| --- | ---: | ---: |
| fixed contiguous | 379.258 | 0 |
| sequential exact-capacity | 377.097 | 32,011 |
| one-layer oracle lookahead | 376.047 | 42,483 |

Memory accounting:

- Bounded arena: 97,725,440 bytes (93.20 MiB), fits 128 MiB: true.
- Major current aw_live_state materializations: 1,332,211,712 bytes/GPU (1.241 GiB).
- Net modeled reduction: 1.150 GiB/GPU.

The epoch resource number is an optimistic dependency lower bound,
not a wall-time prediction. The serialized number is the conservative
no-overlap bound. A retained mechanism still requires measured N128
chain transport, small-segment dense selectors, and a dependency replay
with measured event/copy-engine contention.
