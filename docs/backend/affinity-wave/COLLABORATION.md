# Private collaboration guide

## Repository purpose

This repository is a private research copy for trusted collaborators. It is
not a fork intended for an upstream llama.cpp pull request.

The AffinityWave code was substantially AI-assisted. Anyone changing it should
first understand the relevant CUDA kernels, meta-backend graph partitioning,
event dependencies, and benchmark semantics independently.

## Ground rules

- Keep the repository private unless the owner makes a separate decision.
- Do not open an upstream llama.cpp pull request from this work.
- Preserve the llama.cpp MIT license and history.
- Never upload model weights or private prompts.
- Never use or evaluate sub-Q8 weights for this project.
- Accuracy comes before throughput.
- One four-GPU job at a time, always with the disk watchdog.
- New implementation paths must be opt-in and default off.
- Do not mutate checkpoint `9d3983b8`; branch from it.

## Performance materiality

Backend complexity must earn its cost.

- Under 2%: document and remove.
- 2% to 5%: retain only as a simple operational knob or isolated research
  patch.
- Above 5%: consider retaining after repeated accuracy-gated A/B results.
- Large scheduler or backend machinery should have credible double-digit
  headroom before implementation.

This rule follows the measured record: several sophisticated scheduling ideas
lost badly, while early state corridors returned only about 1.3%.

## Suggested workflow

1. Create a personal branch from `affinitywave-private-collab`.
2. Write a short hypothesis with a quantitative upper bound.
3. Check [EXPERIMENTS.md](EXPERIMENTS.md) for closed paths.
4. Implement one opt-in variable.
5. Build in a new directory.
6. Run exactness before performance.
7. Run p1024 before pp8128 where tail behavior can change.
8. Use fresh-process interleaved controls and the watchdog.
9. Commit results and raw small logs with the source change.
10. Explain whether the change should be retained, removed, or used only to
    update the model.

## Highest-value starting points

### 1. Expert service compute

The complete exact service needs roughly 7.0 effective TFLOP/s/GPU to justify
reopening the original plan. The best exact result is about 5.24 TFLOP/s/GPU.
Packing and owner reduction are already small, so focus on the native-Q8
gate/up/down path.

Any proposal must explain why it differs from the rejected:

- 32x128 shared-A tile;
- split-K1;
- N128 down projection;
- direct-grid dispatch;
- M64-for-all-tails;
- FP16/HFMA2 substitution.

### 2. Token-level route calibration

The current placement uses aggregate histograms and cannot estimate exact
token-owner co-occurrence. The archived trace patch is the right starting
point. A useful dataset should preserve:

- layer;
- token;
- all top-8 expert IDs;
- route weights if needed for later accuracy analysis;
- workload domain and held-out split.

Do not upload private user prompts with the trace.

### 3. Production harvest of T64/K32

Test the exact layout and M64 split-K2 service against the ordinary production
MoE plan. This is less ambitious than the wavefront and may recover value from
the prototype without building state normalization.

### 4. Chunked GDN validation

The checkpoint contains an experimental chunked GDN kernel. It contributed to
the retained service benchmark but has not passed a full numerical gate.
Validate its recurrence against the reference before performance work.

## Reopening the full wavefront

Do not resume production integration merely because one microbenchmark
improves. Reopen only after:

- token-level placement data exists;
- complete expert service reaches about 7.0 TFLOP/s/GPU;
- peer transport remains at least 10 GB/s under compute;
- a new simulation passes with measured, not idealized, costs;
- enough margin remains for logits, append-prefill, and decode normalization.

## Reporting checklist

Every result should state:

- commit and source diff;
- build directory and binary hashes;
- complete environment;
- model hash;
- prompt length and benchmark semantics;
- whether output is withheld;
- exactness, byte-identity, perplexity, or KLD evidence;
- run order and individual samples;
- watchdog and GPU coordination state;
- limitations and disposition.

## AI-assistance disclosure

AffinityWave was developed with substantial Codex assistance and with
Claude Code project memories supplying prior measurements and dead ends.
Collaborators should keep that disclosure in any private presentation or later
publication and should not represent the code as an upstream-reviewed llama.cpp
feature.
