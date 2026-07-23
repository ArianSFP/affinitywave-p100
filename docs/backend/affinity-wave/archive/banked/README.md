# AffinityWave Phase-0 tools

`affinitywave_phase0.py` is the implementation gate for the proposed backend.
It produces a deterministic placement manifest and a 43-diagonal resource
simulation. It intentionally returns exit status 2 for a valid NO-GO result.

## Compact token-routing format

`AWTRV001` is little-endian:

1. Eight-byte file magic `AWTRV001`.
2. Repeated record header `<uint16 layer, uint32 n_tokens, uint16 top_k>`.
3. `n_tokens * top_k` row-major `uint16` global expert IDs.

The capture patch writes one record per MoE layer and evaluation chunk. Capture
with `GGML_CUDA_MOE_PLAN` disabled because the device-plan path intentionally
avoids the host IDs readback where this low-risk instrument lives.

## Reproduce the current analysis

```bash
python3 affinitywave_phase0.py \
  --routes ../round3-phase0/hist-code.txt \
  --nsys ../grouped-20260722/plan-pp8192.sqlite \
  --gguf /home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf \
  --gguf-sha256 "$(cat GGUF.sha256)" \
  --output .
```

The existing histogram contains exact per-expert marginals but not token-level
co-occurrence, so its transport result is explicitly a lower bound. It is still
sufficient for the compute feasibility decision. A future token trace can be
passed to the same `--routes` option without changing the manifest or report
schema.

Run the unit tests with:

```bash
python3 -m unittest -v test_affinitywave_phase0.py
```

