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

Cache selector: frequency.
The placement enumerates all 24 logical-group permutations per layer and
minimizes the measured-row/PCIe calibrated stage model.

| extra experts/layer/GPU | replica GiB/GPU | cached routes | local owner groups | remote groups/token | padding | Q8 model ms | comm model ms | combined ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000% | 25.187% | 2.706 | 97.117% | 937.5 | 308.1 | 1245.5 |
| 16 | 1.992 | 31.062% | 61.401% | 1.396 | 96.414% | 803.6 | 163.5 | 967.2 |
| 32 | 3.984 | 47.489% | 76.454% | 0.852 | 95.733% | 744.8 | 99.6 | 844.4 |
| 44 | 5.479 | 55.565% | 83.558% | 0.595 | 95.193% | 723.2 | 70.1 | 793.3 |
| 48 | 5.977 | 57.693% | 85.374% | 0.529 | 94.998% | 719.6 | 61.5 | 781.1 |
| 56 | 6.973 | 61.371% | 88.450% | 0.418 | 94.679% | 713.6 | 48.9 | 762.5 |
| 64 | 7.969 | 64.238% | 90.902% | 0.329 | 94.347% | 708.2 | 38.8 | 747.0 |
| 80 | 9.961 | 68.406% | 94.413% | 0.202 | 93.762% | 703.1 | 24.1 | 727.1 |
| 96 | 11.953 | 71.018% | 96.707% | 0.119 | 92.990% | 704.7 | 14.2 | 718.9 |

Calibration constants:

- Q8 service: 0.998314 us per issued row
- Bidirectional input/partial pair: 1.273402 us

The combined number is a mechanism gate, not an end-to-end prediction.
It deliberately charges the measured Q8/PCIe serialization seen in the
feature-panel probe. Any retained point still requires a live sparse
scatter/owner-partial microbenchmark and bitwise comparison.
