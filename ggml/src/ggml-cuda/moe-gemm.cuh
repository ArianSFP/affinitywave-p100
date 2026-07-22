#pragma once

#include "common.cuh"

// [TAG_MOE_GROUPED] one-launch grouped GEMM over all local experts for the
// mul_mat_id cuBLAS fallback (env GGML_CUDA_MOE_GROUPED). Consumes native
// Q8_0 weights directly (no per-call dequant pass) and the f16-gathered
// src1_sorted; writes f32 dst_sorted in the same row order as the
// per-expert loop. Reduction differs from cuBLAS (fixed 32x64 tile,
// 4-way split-K; dequant d*q in fp32 = exact, vs the loop's fp16-rounded
// dequant) -> not byte-identical, KLD/ppl-gated like the other GEMM paths.
bool ggml_cuda_moe_gemm_q8_grouped_supported(int64_t ne00, int64_t ne0, ggml_type type_src0);

void ggml_cuda_moe_gemm_q8_grouped(
    ggml_backend_cuda_context & ctx,
    const char    * w_q8,               // base of this device's expert weights (Q8_0)
    size_t          nb01,               // row stride in bytes
    size_t          nb02,               // expert stride in bytes
    const half    * src1_sorted,        // [n_local][ne00] f16, expert-contiguous
    float         * dst_sorted,         // [n_local][ne0] f32, same order
    const int32_t * tokens_per_expert,  // host, per LOCAL expert
    int64_t         n_experts,
    int64_t         ne0,                // output features (N)
    int64_t         ne00,               // contraction (K)
    cudaStream_t    stream);
