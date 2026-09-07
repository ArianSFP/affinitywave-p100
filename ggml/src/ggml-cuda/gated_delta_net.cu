#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>
#include <cstdio>
#include <vector>

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const float * beta_pre,
                                     float *       beta_sig,
                                     const float * s_base,
                                     const int32_t * s_rows,
                                     int64_t         s_row_stride,
                                     float *         s_gather_dst) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    if (s_rows != nullptr) {
        const float * src = s_base + (int64_t) s_rows[sequence] * s_row_stride +
                            h_idx * S_v * S_v + col * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = src[i];
        }
        if (s_gather_dst != nullptr) {
            float * gd = s_gather_dst + state_in_offset + col * S_v;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                gd[i] = s_shard[r];
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r]  = curr_state[i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const unsigned int warp_mask = __activemask();
        float beta_val;
        if (beta_sig != nullptr) {
            beta_val = 1.0f / (1.0f + expf(-beta_pre[gb_offset]));
            if (lane == 0 && threadIdx.y == 0 && blockIdx.z == 0) {
                beta_sig[gb_offset] = beta_val;
            }
        } else {
            beta_val = __shfl_sync(warp_mask, lane == 0 ? *beta_t : 0.0f, 0, warp_size);
        }
        const float v_col = __shfl_sync(warp_mask, lane == 0 ? v_t[col] : 0.0f, 0, warp_size);

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = __shfl_sync(warp_mask, lane == 0 ? expf(*g_t) : 0.0f, 0, warp_size);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_col - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_col - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp, int warps_per_block, bool preexp = false>
__global__ void __launch_bounds__(32*warps_per_block, warps_per_block == 16 ? 1 : 2)
gated_delta_net_chunked_cuda(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             float *       state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_seqs,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ?
            ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    constexpr int cols_per_block = warps_per_block*cols_per_warp;

    const uint32_t h_idx = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int col0 = blockIdx.z*cols_per_block + warp*cols_per_warp;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const int64_t state_in_offset = sequence*H*S_v*S_v + h_idx*S_v*S_v;
    const int64_t state_out_offset = (sequence*H + h_idx)*S_v*S_v;

    curr_state += state_in_offset;
    state += state_out_offset;
    float * attn_data = dst + (sequence*n_tokens*H + h_idx)*S_v;

    float s_shard[cols_per_warp][rows_per_lane];
    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r*warp_size + lane;
            s_shard[c][r] = curr_state[(col0 + c)*S_v + i];
        }
    }

    ggml_cuda_pdl_sync();
    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * k_t = k + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * v_t = v + sequence*sv3 + t*sv2 + h_idx*sv1;
        const int64_t gb_offset = sequence*sb3 + t*sb2 + h_idx*sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t = g + gb_offset*(KDA ? S_v : 1);
        const unsigned int warp_mask = __activemask();
        const float beta_val = __shfl_sync(warp_mask, lane == 0 ? *beta_t : 0.0f, 0, warp_size);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float g_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r*warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
            if constexpr (KDA) {
                g_reg[r] = expf(g_t[i]);
            }
        }
        float g_val = 0.0f;
        if constexpr (!KDA) {
            g_val = __shfl_sync(warp_mask, lane == 0 ? (preexp ? *g_t : expf(*g_t)) : 0.0f, 0, warp_size);
        }

        #pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            const int col = col0 + c;
            const float v_col = __shfl_sync(warp_mask, lane == 0 ? v_t[col] : 0.0f, 0, warp_size);
            float kv_shard = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                if constexpr (KDA) {
                    kv_shard += g_reg[r]*s_shard[c][r]*k_reg[r];
                } else {
                    kv_shard += s_shard[c][r]*k_reg[r];
                }
            }
            const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_col - (KDA ? kv_col : g_val*kv_col))*beta_val;
            float attn_partial = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const float decay = KDA ? g_reg[r] : g_val;
                s_shard[c][r] = decay*s_shard[c][r] + k_reg[r]*delta_col;
                attn_partial += s_shard[c][r]*q_reg[r];
            }
            const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
            if (lane == 0) {
                attn_data[col] = attn_col*scale;
            }
        }
        attn_data += S_v*H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * target_state = state + target_slot*state_slot_stride;
                #pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; ++r) {
                        const int i = r*warp_size + lane;
                        target_state[(col0 + c)*S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        #pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            #pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r*warp_size + lane;
                state[(col0 + c)*S_v + i] = s_shard[c][r];
            }
        }
    }
}

__global__ void gated_delta_net_preexp_cuda(const float * src, float * dst, int64_t ne) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < ne) {
        dst[i] = expf(src[i]);
    }
}

static void launch_gated_delta_net_preexp(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, float * g_exp_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale,
        int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int64_t ne = H*n_tokens*n_seqs;
    const ggml_cuda_kernel_launch_params exp_params(
            dim3((ne + 255)/256, 1, 1), dim3(256, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_preexp_cuda, exp_params, g_d, g_exp_d, ne);

    constexpr int S_v = 128;
    constexpr int cols_per_warp = 2;
    constexpr int warps_per_block = 4;
    constexpr int cols_per_block = cols_per_warp*warps_per_block;
    const dim3 grid_dims(H, n_seqs, S_v/cols_per_block);
    const dim3 block_dims(32, warps_per_block, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(
            gated_delta_net_chunked_cuda<S_v, false, false, cols_per_warp, warps_per_block, true>,
            launch_params, q_d, k_d, v_d, g_exp_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
            scale, state_slot_stride, K);
}

__global__ void __launch_bounds__(128, 2)
gated_delta_net_subwarp16_cuda(const float * q,
                               const float * k,
                               const float * v,
                               const float * g,
                               const float * beta,
                               const float * curr_state,
                               float *       dst,
                               float *       state,
                               int64_t       H,
                               int64_t       n_tokens,
                               int64_t       n_seqs,
                               int64_t       sq1,
                               int64_t       sq2,
                               int64_t       sq3,
                               int64_t       sv1,
                               int64_t       sv2,
                               int64_t       sv3,
                               int64_t       sb1,
                               int64_t       sb2,
                               int64_t       sb3,
                               const uint3   neqk1_magic,
                               const uint3   rq3_magic,
                               float         scale) {
    constexpr int S_v = 128;
    constexpr int rows_per_strand = 4;
    constexpr int warps_per_block = 4;
    constexpr int cols_per_warp = 2;
    constexpr int cols_per_block = warps_per_block*cols_per_warp;

    const uint32_t h_idx = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int sub_lane = lane & 15;
    const int col = blockIdx.z*cols_per_block + warp*cols_per_warp + (lane >> 4);
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const int64_t state_offset = (sequence*H + h_idx)*S_v*S_v;

    curr_state += state_offset;
    state += state_offset;
    float * attn_data = dst + (sequence*n_tokens*H + h_idx)*S_v;

    float s_lo[rows_per_strand];
    float s_hi[rows_per_strand];
    #pragma unroll
    for (int r = 0; r < rows_per_strand; ++r) {
        const int i_lo = r*32 + sub_lane;
        s_lo[r] = curr_state[col*S_v + i_lo];
        s_hi[r] = curr_state[col*S_v + i_lo + 16];
    }

    ggml_cuda_pdl_sync();
    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * k_t = k + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * v_t = v + sequence*sv3 + t*sv2 + h_idx*sv1;
        const int64_t gb_offset = sequence*sb3 + t*sb2 + h_idx*sb1;
        const unsigned int warp_mask = __activemask();
        const float beta_val = __shfl_sync(warp_mask, lane == 0 ? beta[gb_offset] : 0.0f, 0, 32);
        const float g_val = __shfl_sync(warp_mask, lane == 0 ? expf(g[gb_offset]) : 0.0f, 0, 32);
        const float v_col = __shfl_sync(warp_mask, sub_lane == 0 ? v_t[col] : 0.0f, 0, 16);

        float k_lo[rows_per_strand];
        float k_hi[rows_per_strand];
        float q_lo[rows_per_strand];
        float q_hi[rows_per_strand];
        #pragma unroll
        for (int r = 0; r < rows_per_strand; ++r) {
            const int i_lo = r*32 + sub_lane;
            k_lo[r] = k_t[i_lo];
            k_hi[r] = k_t[i_lo + 16];
            q_lo[r] = q_t[i_lo];
            q_hi[r] = q_t[i_lo + 16];
        }

        float kv_lo = 0.0f;
        float kv_hi = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_strand; ++r) {
            kv_lo += s_lo[r]*k_lo[r];
            kv_hi += s_hi[r]*k_hi[r];
        }
        const float kv_col = warp_reduce_sum<16>(kv_lo + kv_hi);
        const float delta_col = (v_col - g_val*kv_col)*beta_val;

        float attn_lo = 0.0f;
        float attn_hi = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_strand; ++r) {
            s_lo[r] = g_val*s_lo[r] + k_lo[r]*delta_col;
            s_hi[r] = g_val*s_hi[r] + k_hi[r]*delta_col;
            attn_lo += s_lo[r]*q_lo[r];
            attn_hi += s_hi[r]*q_hi[r];
        }
        const float attn_col = warp_reduce_sum<16>(attn_lo + attn_hi);
        if (sub_lane == 0) {
            attn_data[col] = attn_col*scale;
        }
        attn_data += S_v*H;
    }

    #pragma unroll
    for (int r = 0; r < rows_per_strand; ++r) {
        const int i_lo = r*32 + sub_lane;
        state[col*S_v + i_lo] = s_lo[r];
        state[col*S_v + i_lo + 16] = s_hi[r];
    }
}

static void launch_gated_delta_net_subwarp16(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    constexpr int S_v = 128;
    constexpr int warps_per_block = 4;
    constexpr int cols_per_block = 8;
    const dim3 grid_dims(H, n_seqs, S_v/cols_per_block);
    const dim3 block_dims(32, warps_per_block, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_subwarp16_cuda, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
}

__global__ void __launch_bounds__(128, 2)
gated_delta_net_pair_cuda(const float * q,
                          const float * k,
                          const float * v,
                          const float * g,
                          const float * beta,
                          const float * curr_state,
                          float *       dst,
                          float *       state,
                          int64_t       H,
                          int64_t       n_tokens,
                          int64_t       n_seqs,
                          int64_t       sq1,
                          int64_t       sq2,
                          int64_t       sq3,
                          int64_t       sv1,
                          int64_t       sv2,
                          int64_t       sv3,
                          int64_t       sb1,
                          int64_t       sb2,
                          int64_t       sb3,
                          const uint3   neqk1_magic,
                          const uint3   rq3_magic,
                          float         scale) {
    constexpr int S_v = 128;
    constexpr int rows_per_lane = 4;
    constexpr int warps_per_block = 4;
    constexpr int cols_per_warp = 2;
    constexpr int cols_per_block = warps_per_block*cols_per_warp;

    const uint32_t h_idx = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int col0 = blockIdx.z*cols_per_block + warp*cols_per_warp;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const int64_t state_offset = (sequence*H + h_idx)*S_v*S_v;

    curr_state += state_offset;
    state += state_offset;
    float * attn_data = dst + (sequence*n_tokens*H + h_idx)*S_v;

    float s_shard[cols_per_warp][rows_per_lane];
    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r*32 + lane;
            s_shard[c][r] = curr_state[(col0 + c)*S_v + i];
        }
    }

    ggml_cuda_pdl_sync();
    for (int t = 0; t < n_tokens; ++t) {
        const float * q_t = q + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * k_t = k + iq3*sq3 + t*sq2 + iq1*sq1;
        const float * v_t = v + sequence*sv3 + t*sv2 + h_idx*sv1;
        const int64_t gb_offset = sequence*sb3 + t*sb2 + h_idx*sb1;
        const unsigned int warp_mask = __activemask();
        const float beta_val = __shfl_sync(warp_mask, lane == 0 ? beta[gb_offset] : 0.0f, 0, 32);
        const float g_val = __shfl_sync(warp_mask, lane == 0 ? expf(g[gb_offset]) : 0.0f, 0, 32);
        const float v0 = __shfl_sync(warp_mask, lane == 0 ? v_t[col0] : 0.0f, 0, 32);
        const float v1 = __shfl_sync(warp_mask, lane == 0 ? v_t[col0 + 1] : 0.0f, 0, 32);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r*32 + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        float2 kv = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            kv.x += s_shard[0][r]*k_reg[r];
            kv.y += s_shard[1][r]*k_reg[r];
        }
        kv = warp_reduce_sum(kv);
        const float2 delta = make_float2(
                (v0 - g_val*kv.x)*beta_val,
                (v1 - g_val*kv.y)*beta_val);

        float2 attn = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            s_shard[0][r] = g_val*s_shard[0][r] + k_reg[r]*delta.x;
            attn.x += s_shard[0][r]*q_reg[r];
            s_shard[1][r] = g_val*s_shard[1][r] + k_reg[r]*delta.y;
            attn.y += s_shard[1][r]*q_reg[r];
        }
        attn = warp_reduce_sum(attn);
        if (lane == 0) {
            attn_data[col0] = attn.x*scale;
            attn_data[col0 + 1] = attn.y*scale;
        }
        attn_data += S_v*H;
    }

    #pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r*32 + lane;
            state[(col0 + c)*S_v + i] = s_shard[c][r];
        }
    }
}

static void launch_gated_delta_net_pair(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    constexpr int S_v = 128;
    constexpr int warps_per_block = 4;
    constexpr int cols_per_block = 8;
    const dim3 grid_dims(H, n_seqs, S_v/cols_per_block);
    const dim3 block_dims(32, warps_per_block, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_pair_cuda, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
}

template <bool KDA, bool keep_rs_t, int cols_per_warp, int warps_per_block>
static void launch_gated_delta_net_chunked_variant(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale,
        int64_t state_slot_stride, int K, cudaStream_t stream) {
    constexpr int S_v = 128;
    constexpr int cols_per_block = warps_per_block*cols_per_warp;
    dim3 grid_dims(H, n_seqs, (S_v + cols_per_block - 1)/cols_per_block);
    dim3 block_dims(32, warps_per_block, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<S_v, KDA, keep_rs_t, cols_per_warp, warps_per_block>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
            scale, state_slot_stride, K);
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale,
        int64_t state_slot_stride, int K, int cols_per_warp, int warps_per_block, cudaStream_t stream) {
#define LAUNCH_GDN_CHUNKED(COLS, WARPS) \
    launch_gated_delta_net_chunked_variant<KDA, keep_rs_t, COLS, WARPS>(q_d, k_d, v_d, g_d, b_d, s_d, \
            dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream)

    if (cols_per_warp == 2 && warps_per_block == 4) {
        LAUNCH_GDN_CHUNKED(2, 4);
    } else if (cols_per_warp == 2 && warps_per_block == 16) {
        LAUNCH_GDN_CHUNKED(2, 16);
    } else if (cols_per_warp == 2) {
        LAUNCH_GDN_CHUNKED(2, 8);
    } else if (cols_per_warp == 8) {
        launch_gated_delta_net_chunked_variant<KDA, keep_rs_t, 8, 8>(q_d, k_d, v_d, g_d, b_d, s_d,
                dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
    } else {
        launch_gated_delta_net_chunked_variant<KDA, keep_rs_t, 4, 8>(q_d, k_d, v_d, g_d, b_d, s_d,
                dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
    }
#undef LAUNCH_GDN_CHUNKED
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const float * beta_pre_d, float * beta_sig_d,
        const float * s_base_d, const int32_t * s_rows_d, int64_t s_row_stride,
        float * s_gather_dst_d, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_gated_delta_net_fused_cache * cache, bool fuse_beta_sigmoid = false,
        const ggml_cuda_gated_delta_net_fused_gather * gather = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    const float * beta_pre_d = nullptr;
    float *       beta_sig_d = nullptr;
    if (fuse_beta_sigmoid) {
        GGML_ASSERT(src_beta->src[0] != nullptr);
        GGML_ASSERT(ggml_are_same_shape(src_beta, src_beta->src[0]));
        GGML_ASSERT(ggml_are_same_stride(src_beta, src_beta->src[0]));
        GGML_ASSERT(ggml_is_contiguous(src_beta->src[0]));
        beta_pre_d = (const float *) src_beta->src[0]->data;
        beta_sig_d = (float *) src_beta->data;
        GGML_ASSERT(beta_pre_d != nullptr && beta_sig_d != nullptr);
    }

    const float * s_base_d = nullptr;
    const int32_t * s_rows_d = nullptr;
    int64_t s_row_stride = 0;
    float * s_gather_dst_d = nullptr;
    if (gather != nullptr) {
        GGML_ASSERT(gather->base != nullptr && gather->rows != nullptr && gather->row_stride > 0);
        s_base_d = gather->base;
        s_rows_d = gather->rows;
        s_row_stride = gather->row_stride;
        s_gather_dst_d = gather->gather_dst;
    }

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int chunk_cols_env = getenv("GGML_CUDA_AW_GDN_CHUNKED") != nullptr ?
            atoi(getenv("GGML_CUDA_AW_GDN_CHUNKED")) : 0;
    const int chunk_cols = chunk_cols_env == 1 ? 4 : chunk_cols_env;
    const int chunk_warps_env = getenv("GGML_CUDA_AW_GDN_WARPS") != nullptr ?
            atoi(getenv("GGML_CUDA_AW_GDN_WARPS")) : 8;
    const int chunk_warps = chunk_warps_env == 4 || chunk_warps_env == 16 ? chunk_warps_env : 8;
    const bool chunked = (chunk_cols == 2 || chunk_cols == 4 || chunk_cols == 8) &&
            S_v == 128 && n_tokens >= 512;
    auto dump_once = [&]() {
        static bool dumped = false;
        const char * path = getenv("GGML_CUDA_AW_GDN_DUMP");
        if (dumped || path == nullptr || path[0] == '\0') {
            return;
        }
        dumped = true;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> host(ggml_nbytes(dst));
        CUDA_CHECK(cudaMemcpy(host.data(), dst_d, host.size(), cudaMemcpyDeviceToHost));
        FILE * file = fopen(path, "wb");
        GGML_ASSERT(file != nullptr);
        GGML_ASSERT(fwrite(host.data(), 1, host.size(), file) == host.size());
        GGML_ASSERT(fclose(file) == 0);
        fprintf(stderr, "AffinityWave GDN dump: %s (%zu bytes)\n", path, host.size());
    };

    const int segment_tokens = getenv("GGML_CUDA_AW_GDN_SEGMENT_TOKENS") != nullptr ?
            atoi(getenv("GGML_CUDA_AW_GDN_SEGMENT_TOKENS")) : 0;
    if (segment_tokens > 0 && !fuse_beta_sigmoid && gather == nullptr && !chunked && !kda && !keep_rs &&
            S_v == 128 && n_seqs == 1 && n_tokens % segment_tokens == 0) {
        const float * segment_state = s_d;
        for (int64_t token = 0; token < n_tokens; token += segment_tokens) {
            launch_gated_delta_net<false, false>(
                    q_d + token*sq2,
                    k_d + token*sq2,
                    v_d + token*sv2,
                    g_d + token*sb2,
                    b_d + token*sb2,
                    segment_state,
                    dst_d + token*H*S_v,
                    state_d,
                    S_v, H, segment_tokens, n_seqs,
                    sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale,
                    state_slot_stride, K, nullptr, nullptr, nullptr, nullptr, 0, nullptr, stream);
            segment_state = state_d;
        }
        dump_once();
        return;
    }

    const char * preexp_env = getenv("GGML_CUDA_AW_GDN_PREEXP");
    const char * p100_exact_env = getenv("GGML_CUDA_AW_P100_EXACT");
    const bool p100_exact =
            ggml_cuda_info().devices[ctx.device].cc ==
                    GGML_CUDA_CC_PASCAL &&
            p100_exact_env != nullptr &&
            atoi(p100_exact_env) != 0 &&
            n_tokens >= 512 &&
            n_seqs == 1;
    const bool preexp = preexp_env != nullptr ? atoi(preexp_env) != 0 : p100_exact;
    if (!fuse_beta_sigmoid && gather == nullptr && preexp && !kda && !keep_rs && S_v == 128) {
        ggml_cuda_pool_alloc<float> g_exp(ctx.pool(), H*n_tokens*n_seqs);
        launch_gated_delta_net_preexp(q_d, k_d, v_d, g_d, g_exp.get(), b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1, rq3, scale, state_slot_stride, K, stream);
        return;
    }
    const bool subwarp16 = getenv("GGML_CUDA_AW_GDN_SUBWARP") != nullptr &&
            atoi(getenv("GGML_CUDA_AW_GDN_SUBWARP")) != 0;
    if (subwarp16 && !kda && !keep_rs && S_v == 128) {
        launch_gated_delta_net_subwarp16(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1, rq3, scale, stream);
        return;
    }
    const bool pair = getenv("GGML_CUDA_AW_GDN_PAIR") != nullptr &&
            atoi(getenv("GGML_CUDA_AW_GDN_PAIR")) != 0;
    if (pair && !kda && !keep_rs && S_v == 128) {
        launch_gated_delta_net_pair(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1, rq3, scale, stream);
        return;
    }
    if (!fuse_beta_sigmoid && gather == nullptr && chunked) {
        if (kda) {
            if (keep_rs) {
                launch_gated_delta_net_chunked<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                        neqk1, rq3, scale, state_slot_stride, K, chunk_cols, chunk_warps, stream);
            } else {
                launch_gated_delta_net_chunked<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                        neqk1, rq3, scale, state_slot_stride, K, chunk_cols, chunk_warps, stream);
            }
        } else if (keep_rs) {
            launch_gated_delta_net_chunked<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                    neqk1, rq3, scale, state_slot_stride, K, chunk_cols, chunk_warps, stream);
        } else {
            launch_gated_delta_net_chunked<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                    neqk1, rq3, scale, state_slot_stride, K, chunk_cols, chunk_warps, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                beta_pre_d, beta_sig_d, s_base_d, s_rows_d, s_row_stride, s_gather_dst_d, stream);
        }
    }
    dump_once();
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_cuda_op_gated_delta_net_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_gated_delta_net_fused_cache * cache, bool fuse_beta_sigmoid,
        const ggml_cuda_gated_delta_net_fused_gather * gather) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, fuse_beta_sigmoid, gather);
}
