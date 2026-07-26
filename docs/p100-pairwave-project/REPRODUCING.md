# Reproducing the qualified result

## Hardware and software

The result is specific to:

- four Tesla P100 PCIe 16 GB GPUs;
- compute capability 6.0;
- CUDA 12.8.61;
- cuBLAS 12.8.3;
- Release build with CUDA, Flash Attention, and NCCL enabled;
- Qwen3.6-35B-A3B Q8_0 weights;
- the exact source checkpoint described in [RESULTS.md](RESULTS.md).

The dense selector path validates its complete compile/runtime signature and
falls back when it does not match. A result on another CUDA/cuBLAS version is
a new qualification, not a reproduction.

## Build outline

Use an isolated build directory and target `sm_60`. The exact local compiler
paths depend on the installed CUDA toolkit. A representative configuration
is:

```bash
cmake -S . -B ../build-p100-pairwave \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60

cmake --build ../build-p100-pairwave -j \
  --target llama-bench llama-perplexity \
           llama-affinity-wave-bench test-backend-ops
```

The archived [toolchain record](evidence/stitchrail/static/toolchain.txt) is
the source of truth for the qualified binary.

## Production environment

The full environment is preserved in
[run-qualification.sh](harness/run-qualification.sh). The important
selectors are:

```text
GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1
GGML_CUDA_MOE_CSORT=1
GGML_CUDA_MOE_EP=1
GGML_CUDA_MOE_GROUPED=1
GGML_CUDA_MOE_PLAN=1
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_WIRE=f32
GGML_CUDA_AW_PARTIAL=bf16
GGML_CUDA_AW_Q8_LAYOUT=t64k32
GGML_CUDA_AW_Q8_ENGINE=cohortrail
GGML_CUDA_AW_ROUTE_SCATTER=deterministic
GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
GGML_CUDA_AW_DENSE_SELECTORS=exact
GGML_CUDA_AW_P100_EXACT=1
GGML_CUDA_AW_P100_EXACT_SUM_WIDTH=4
```

The qualification also needs the code-calibrated EPLB map and AffinityWave
placement manifest referenced by the harness. Those inputs are
machine-specific and were not folded into the source.

## Benchmark shape

The retained pp8128 benchmark uses:

```text
llama-bench
--model Qwen3.6-35B-A3B-Q8_0.gguf
-ngl 99
-sm tensor
-fa 1
-b 8128
-ub 8128
-mmp 0
-p 8128
-n 0
-r 1
-o json
```

Run under a clean environment and the same CPU affinity:

```text
env -i <production environment> taskset --cpu-list 0-11 ...
```

Under Nsight Systems, use plain `env`; `env -i` removes the profiler
injection variables.

## Four-GPU safety protocol

Every four-GPU job must:

1. acquire `/tmp/affinitywave-4gpu.lock`;
2. reject a second llama or Nsight process;
3. start the `.xsession-errors` and free-disk watchdog;
4. use the production environment and CPU taskset;
5. release the lock and verify no project GPU process remains.

The P100 desktop Vulkan loop can grow `.xsession-errors` at roughly
12 MB/s and fill the root filesystem. Do not run the campaign without the
watchdog. The archived harness implements the lock and watchdog.

At archive time `nvidia-smi` could not initialize because NVML 580.173 did
not match the loaded 580.159.03 kernel driver. Process and device-handle
inspection was therefore used as a fallback; only desktop handles remained.

## Qualification sequence

The final campaign order was:

1. two c512 service-dump arms, umbrella OFF and ON;
2. two c512 saved-logits arms;
3. five randomized pp2048 OFF/ON pairs;
4. three randomized pp8128 OFF/ON pairs;
5. static retained-kernel resource audit.

Acceptance required:

- byte identity for all 640 service-boundary comparisons;
- byte-identical saved logits with the expected SHA-256;
- every paired performance comparison positive;
- no retained target stack, local memory, or spills;
- production memory reclamation approximately 0.8 GiB/GPU or greater.

The campaign definition and results are preserved in
[evidence/stitchrail/campaign/](evidence/stitchrail/campaign/) and
[the campaign summary](archive/stitchrail/p100-exact-campaigns/stitchrail-final-prefill-v2/summary.md).

## Useful harness modes

The archived harness supports:

- `service`
- `bench`
- `diagonal-bench`
- `diagonal-bench-dump`
- `diagonal-ppl-save`
- `diagonal-trace`
- `bench-dump`
- `ppl-save`
- `trace`

Its paths are the original rig paths. Copy and edit it for another machine;
do not run it blindly.
