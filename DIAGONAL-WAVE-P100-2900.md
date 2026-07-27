# Exact diagonal wave on four P100 GPUs

This branch preserves the production-qualified Qwen3.6-35B-A3B Q8_0
prefill checkpoint for four Tesla P100 PCIe 16 GB GPUs.

The qualified source checkpoint is:

```text
36bad6bb3ad4c9b31edc4dae9fb6c3716b95704b
```

The retained pp8128 median is 2790.956 ms, or 2912.264 prompt tokens/s.
The result uses the exact diagonal N2048 scheduler and CohortRail Q8_0
service. It is not a PairWave-only or PairFold result.

Start with:

1. [Project overview](docs/p100-pairwave-project/README.md)
2. [Qualified results](docs/p100-pairwave-project/RESULTS.md)
3. [Architecture](docs/p100-pairwave-project/ARCHITECTURE.md)
4. [Reproduction procedure](docs/p100-pairwave-project/REPRODUCING.md)
5. [Experiment ledger](docs/p100-pairwave-project/EXPERIMENT-LEDGER.md)
6. [Artifact and hash index](docs/p100-pairwave-project/ARTIFACTS.md)

The primary production selectors are:

```text
GGML_CUDA_AW_P100_EXACT=1
GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
GGML_CUDA_AW_Q8_ENGINE=cohortrail
GGML_CUDA_AW_DENSE_SELECTORS=exact
```

The qualified `libggml-cuda.so` SHA-256 is:

```text
44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7
```

The documentation includes later PairFold, host-streaming, PairCache, and
other investigations so their conclusions and rejected paths are not lost.
Those records do not change the source checkpoint or production status of
this branch. Large traces, generated logits, model weights, binaries, and
service dumps are deliberately excluded.
