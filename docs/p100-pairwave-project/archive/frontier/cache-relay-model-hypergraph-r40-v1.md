# CacheRelay token-level placement model

- Capture passes: 4
- Passes bitwise identical: false
- Selected pass: 3 (final measured evaluation)
- Tokens per lane: 2032
- Capture SHA-256: `0dd89bf538822716dbed425a6aadbe1b90f452de0a867859e9ca3ad4a1c618e4`
- Exact Q8 bytes per expert/layer: 3,342,336

The model keeps the immutable logical owner groups. A group may move to a
different physical GPU, but its per-route rank order, BF16 owner rounding,
and final logical-owner reduction order remain unchanged. A token/owner
group moves to its lane GPU only when that GPU has every expert weight used
by the group. Mixed groups remain wholly on the primary, avoiding any
reassociation of the owner partial.

Cache selector: hypergraph.
The placement enumerates all 24 logical-group permutations per layer and
minimizes the measured-row/PCIe calibrated stage model.

| extra experts/layer/GPU | replica GiB/GPU | cached routes | local owner groups | remote groups/token | padding | Q8 model ms | comm model ms | combined ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 4.980 | 54.098% | 81.317% | 0.676 | 95.358% | 711.2 | 90.0 | 801.2 |

Calibration constants:

- Q8 service: 0.998314 us per issued row
- Bidirectional input/partial pair: 1.273402 us

The combined number is a mechanism gate, not an end-to-end prediction.
It deliberately charges the measured Q8/PCIe serialization seen in the
feature-panel probe. Any retained point still requires a live sparse
scatter/owner-partial microbenchmark and bitwise comparison.
