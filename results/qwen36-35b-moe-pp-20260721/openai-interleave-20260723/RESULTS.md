# AffinityWave production result

Checkpoint: `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`

## Outcome

The OpenAI-style Q8 software pipeline is integrated into the AffinityWave
production graph. Real output is enabled, the final logits path runs, and the
matched-graph accuracy gate passes.

| Check | Result |
| --- | --- |
| Real-logit PPL delta vs normal `c512/ub128` | -0.002294 (PASS) |
| KLD | 0.000880 |
| Same top token | 99.608% |
| Focused CUDA backend tests | 2,172/2,172 passed |
| Final allowed-precision CUDA tests | 221/221 passed |
| Uniform synthetic arithmetic | 0/16,777,216 mismatches |
| Edge-mix synthetic arithmetic | 0/16,777,216 mismatches |
| Final AffinityWave pp8128/ub8128 mean | 2517.169 tok/s |
| Normal checkpoint pp8128/ub4096 mean | 1386.743 tok/s |
| Prompt-length throughput peak | 2575.968 tok/s at pp4096 |

The retained 2827.6 tok/s pp8128 result is service-only with output withheld
and remains the primary service baseline. It is not directly comparable with
the production-output measurements above.

See `NOTES-codex.md` for implementation details, differential probes,
rejected reduction variants, and the clean-`ub512` secondary comparison.

## Final production matrix

All results below include real output and use the accuracy-passed production
configuration. The 8128-token entry is the mean of two final-binary samples.

| Prompt/ubatch | End-to-end tok/s |
| ---: | ---: |
| 512 | 1185.465 |
| 1024 | 1715.424 |
| 2048 | 2253.077 |
| 4096 | 2575.968 |
| 8128 | 2517.169 |

The optimized mask setup is used only for the guarded AffinityWave,
single-sequence, monotonic causal-prompt case. Other sequence layouts, SWA,
and Alibi retain the general mask implementation.
