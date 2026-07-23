# AffinityWave private research prototype

AffinityWave explores a four-lane diagonal prefill schedule and owner-routed
Mixture-of-Experts service for Qwen3.6-35B-A3B Q8_0 on four Tesla P100 PCIe
GPUs. This branch is a private collaboration snapshot rooted at the locally
banked checkpoint.

## Read this first

The headline result has narrow semantics:

- pp8128 service benchmark mean: 2874.531 ms, or 2827.6 tok/s;
- the benchmark intentionally withholds model output and returns an aborted
  status after timing;
- the 40-layer by 4-lane path does not return validated logits;
- the exactness check covers the isolated expert service and deterministic
  BF16 owner partials, not the complete wavefront;
- append-prefill, decode-state normalization, and production request handling
  are not implemented.

Do not compare 2827.6 tok/s with an ordinary end-to-end llama.cpp result. It is
a projected service rate from a benchmark-only execution path.

## Checkpoint identity

| Item | Value |
|---|---|
| Parent commit | `05dcabf9a97b8a91bec5621d2b4c5ce1ce9b2ca8` |
| AffinityWave checkpoint | `9d3983b89952c7fc1c6aa38fc1a7bd3182992382` |
| Checkpoint tree | `acbecf68f512212b95015540e3774864bf935aed` |
| Source delta | 14 files, 4,979 insertions, 71 deletions |
| Model format | exact Q8_0 weights only |
| Target hardware | exactly four CUDA compute-capability 6.0 GPUs |

## Documentation

- [Architecture](affinity-wave/ARCHITECTURE.md)
- [Current status and measured evidence](affinity-wave/STATUS.md)
- [Complete experiment and dead-end log](affinity-wave/EXPERIMENTS.md)
- [Build and reproduction guide](affinity-wave/REPRODUCTION.md)
- [Private collaboration guide](affinity-wave/COLLABORATION.md)
- [Curated artifact index](affinity-wave/artifacts/README.md)
- [Raw experiment archive](affinity-wave/archive/README.md)

## What the checkpoint contains

- strict opt-in hardware and placement-manifest validation;
- native-Q8 T64/K32 expert-weight repacking;
- M64 split-K2, M32, and M16 expert projection paths;
- deterministic F32 or BF16 request packing and BF16 owner partials;
- four-GPU isolated expert-service benchmark with exact synthetic checks;
- benchmark-only dense token lanes, a 40-layer diagonal schedule, state
  corridors, and chunked gated-delta-net execution;
- graph inspection, timing, and placement hooks needed for continued research.

Ordinary model execution still falls back to the existing EP4 path outside the
benchmark-specific controls.

## Research verdict

The original feasibility gate was a NO-GO. The optimistic model required about
7.0 effective TFLOP/s/GPU from the complete expert service to have useful
margin. The exact retained service measured about 5.13 to 5.29 TFLOP/s/GPU,
and the service-only wave benchmark reached about 2828 tok/s rather than the
3500 to 4000 tok/s target.

The checkpoint is published privately so collaborators can inspect the design,
reproduce the measurements, and seek a material breakthrough. It is not a
claim that the architecture is ready to integrate.

## Provenance and AI assistance

This repository preserves the llama.cpp history and MIT license. AffinityWave
was developed as a user-directed private experiment with substantial AI
assistance from Codex and with measurements and project context shared from
Claude Code sessions. Human collaborators must independently understand and
review the implementation before relying on it.

No upstream pull request is intended.
