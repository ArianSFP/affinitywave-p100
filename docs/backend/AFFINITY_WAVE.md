# AffinityWave Phase-1 prototype

AffinityWave is an experimental Qwen3.6-35B-A3B prefill backend for four
Tesla P100 PCIe GPUs. The current implementation is a Phase-1 springboard, not
an end-to-end wavefront scheduler. Ordinary model graphs continue to use the
existing EP4 path.

The prototype adds:

- strict opt-in configuration and placement-manifest validation;
- a backend proc-address entry for an isolated cross-layer expert service;
- one grouped native-Q8 service per gate, up, and down projection;
- descriptors carrying layer, expert, input, weight, output, and row offsets;
- F32 owner requests by default, optional BF16 requests, and fixed-order BF16
  owner partials;
- request packing, SwiGLU, owner aggregation, stage timing, a synthetic
  reference check, and a four-GPU benchmark.

It does not yet implement dense/hot-expert loading, the 40x4 diagonal scheduler,
state corridors, append-prefill, or decode-state normalization. Enabling the
prototype therefore logs that normal model execution remains on EP4.

## Build

Use an isolated sm_60 build. The project build used for the registered result
was configured with CUDA, Flash Attention, and NCCL and produced both
`llama-affinity-wave-bench` and `llama-bench`.

## Controls

The benchmark requires:

```text
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_MAP=/absolute/path/to/placement-hot16.json
GGML_CUDA_AW_WIRE=f32|bf16
GGML_CUDA_AW_CHECK=1
```

`GGML_CUDA_AW_WIRE` defaults to `f32`. Explicit AffinityWave mode rejects a
missing or malformed manifest, a non-four-GPU visible set, non-sm_60 devices,
and unsupported wire formats. The v1 manifest validator checks all 40 layers,
all 256 primary owners per layer, exactly 64 primaries per GPU, 16 unique hot
experts, architecture metadata, and the GGUF digest format.

## Benchmark semantics

The registered shape uses four active cells, 2048 tokens per cell, two routes
per owner token (top-8 divided across four owners), and 256 cross-layer work
descriptors per GPU. Each timed iteration includes:

1. packing 8192 owner requests in the selected wire format;
2. native-Q8 gate and up projections (`K=2048`, `N=512`);
3. SwiGLU;
4. native-Q8 down projection (`K=512`, `N=2048`);
5. deterministic two-route accumulation and BF16 response rounding.

All four devices run concurrently. The reported effective TFLOP/s divides the
fixed expert arithmetic by the complete timed service, not just GEMM time.

## Current result

The F32 path is bit-exact against the deterministic BF16 owner-partial check on
all four GPUs. The best retained kernel reaches 4.81 TFLOP/s on the slowest GPU
for the registered four-cell shape, below the 5.5 Phase-1 gate. The measured
PCIe copy-engine gate on the same rig remains 11.96 GB/s under full GEMM load.

The result is intentionally reported as a failed performance gate. It is useful
as a buildable service boundary and stage-timed baseline, but it is not evidence
that the full backend reaches 3200 or 4000 prompt tokens/s.
