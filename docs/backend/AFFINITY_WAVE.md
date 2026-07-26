# AffinityWave Phase-1 prototype

AffinityWave is an experimental Qwen3.6-35B-A3B prefill backend for four
Tesla P100 PCIe GPUs. Ordinary model graphs continue to use the existing EP4
path unless one of the experimental schedulers is explicitly selected.

The prototype adds:

- strict opt-in configuration and placement-manifest validation;
- a backend proc-address entry for an isolated cross-layer expert service;
- one grouped native-Q8 service per gate, up, and down projection;
- descriptors carrying layer, expert, input, weight, output, and row offsets;
- F32 owner requests by default, optional BF16 requests, and fixed-order BF16
  owner partials;
- request packing, SwiGLU, owner aggregation, stage timing, a synthetic
  reference check, and a four-GPU benchmark.

The frontier build also contains a default-off HeadFold scheduler, exact R44
expert-weight streaming, and an N256 bounded service. Append-prefill and
decode-state normalization remain on the ordinary runtime.

## Build

Use an isolated sm_60 build. The project build used for the registered result
was configured with CUDA, Flash Attention, and NCCL and produced both
`llama-affinity-wave-bench` and `llama-bench`.

## Controls

The benchmark requires:

```text
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_MAP=/absolute/path/to/placement-hot16.json
GGML_CUDA_AW_WIRE=f32|bf16|f16
GGML_CUDA_AW_CHECK=1
```

`GGML_CUDA_AW_WIRE` defaults to `f32`. Explicit AffinityWave mode rejects a
missing or malformed manifest, a non-four-GPU visible set, non-sm_60 devices,
and unsupported wire formats. The v1 manifest validator checks all 40 layers,
all 256 primary owners per layer, exactly 64 primaries per GPU, 16 unique hot
experts, architecture metadata, and the GGUF digest format.

### HeadFold R44 plus N256

The banked HeadFold configuration is:

```text
GGML_CUDA_AW_HEADFOLD=1
GGML_CUDA_AW_HEADFOLD_SPLIT_PRE=1
GGML_CUDA_AW_R44=service
GGML_CUDA_AW_R44_MAP=/absolute/path/to/r44-placement.bin
GGML_CUDA_AW_R44_SLOTS=2
GGML_CUDA_AW_SERVICE=n256
GGML_CUDA_AW_Q8_ENGINE=broadwave
```

`GGML_CUDA_AW_R44_SLOTS=1` is diagnostic only; it is exact but exposes the
next-layer weight copy and substantially reduces throughput. The R44 service
requires F32 request transport, BF16 logical-owner partials, no shared-service
or direct-owner mode, and four same-layer HeadFold cells.

`GGML_CUDA_AW_SERVICE=n256` traverses gate and up in two 256-column panels and
down in eight 256-column panels. The down route-output panel reuses the
completed gate panel. BF16 owner partials use a bounded seven-token-shard
layout and remain in ordinary device memory for P2P reads.

Use `GGML_CUDA_AW_Q8_ENGINE=halfpipe_sync` to select the exact HalfPipe
BroadWave variant. `halfpipe_bar` remains diagnostic and is slower on P100.

The following controls are default-off diagnostics:

```text
GGML_CUDA_AW_R44_STATS=1
GGML_CUDA_AW_R44_MEMORY=1
GGML_CUDA_AW_SERVICE_DUMP=/absolute/output/directory
GGML_CUDA_AW_SERVICE_DUMP_LAYER=0
```

The memory diagnostic synchronizes every layer and must not be used for
production timing.

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

The isolated Phase-1 result remains a failed performance gate. The separately
qualified HeadFold R44/N256 runtime reaches about 2294 prompt tokens/s with
BroadWave and 2299 prompt tokens/s with HalfPipe at pp8128. It is banked for
continued scheduler work, but remains default-off because the current diagonal
production scheduler is faster.
