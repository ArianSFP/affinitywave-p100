// [TAG_MOE_GROUPED] grouped MoE expert GEMM, one launch per matrix.
// Design and microbenchmarks: bench/a1-gemm/ (m32v5 mainloop + a4b native-Q8
// staging), results/qwen36-35b-moe-pp-20260721/round3-phase0/RESULTS.md.
// 32x64 output tile per CTA, 128 threads = 4 K-groups x 32 lanes, 8x8 f32
// register microtile, 4-way split-K, KSTAGE 32 double-buffered smem.
// Smem layout is interleaved and XOR-swizzled by (k>>3)&3 so compute reads
// are LDS.128 AND conflict-free (the XOR term is constant per thread inside
// a K-stage). B is read in the native ggml Q8_0 block layout (34B packed
// blocks) with 2-byte loads; one K-stage = exactly one block per row, and
// dequant d*q runs in fp32 during staging (exact - int8 x half fits fp32).

#include "moe-gemm.cuh"

#define MOE_GEMM_KSTAGE 32

union moe_gemm_smem {
    struct { float4 A[2][MOE_GEMM_KSTAGE][8]; float4 B[2][MOE_GEMM_KSTAGE][16]; } stage;
    float red[8][32 * 8];
};

__global__ __launch_bounds__(128, 2)
static void moe_gemm_q8_grouped(const half * __restrict__ A, const char * __restrict__ W,
                                float * __restrict__ C, const int32_t * __restrict__ tiles,
                                int n_tiles, size_t nb01, size_t nb02, int n, int k) {
    __shared__ moe_gemm_smem sm;

    const int tid  = threadIdx.x;
    const int kgrp = tid >> 5;
    const int lane = tid & 31;
    const int mt   = lane >> 3;
    const int nt   = lane & 7;
    const int texp = tiles[blockIdx.x];
    const int row0 = tiles[n_tiles + blockIdx.x];
    const int rows = tiles[2 * n_tiles + blockIdx.x];
    const int col0 = blockIdx.y * 64;

    const char * bq = W + (size_t) texp * nb02;

    const int a_r = tid >> 2, a_k = (tid & 3) * 8;
    const int b_r0 = tid >> 2;
    const int b_r1 = 32 + (tid >> 2);
    const int xk   = tid & 3;
    const int b_k  = xk * 8;

    // src1_sorted is expert-contiguous: A row = global sorted row (clamped for
    // the tail tile's padding lanes; their results are masked at the store)
    const int arow = row0 + (a_r < rows ? a_r : rows - 1);

    float4 pa;
    unsigned short pq0[4], pq1[4];
    float pd0, pd1;
    const float4 * gA = (const float4 *) (A + (size_t) arow * k + a_k);
    const char * gQ0 = bq + (size_t) (col0 + b_r0) * nb01;
    const char * gQ1 = bq + (size_t) (col0 + b_r1) * nb01;

    const int aslot = (((a_r >> 2) & 1) * 4 + (a_r >> 3)) ^ xk;
    const int aoff  = aslot * 4 + (a_r & 3);
    const int bslot0 = ((((b_r0 >> 2) & 1) * 8 + (b_r0 >> 3)) ^ xk);
    const int boff0  = bslot0 * 4 + (b_r0 & 3);
    const int bslot1 = ((((b_r1 >> 2) & 1) * 8 + (b_r1 >> 3)) ^ xk);
    const int boff1  = bslot1 * 4 + (b_r1 & 3);

    auto fetchb = [&](int s) {
        const char * p0 = gQ0 + (size_t) s * sizeof(block_q8_0);
        const char * p1 = gQ1 + (size_t) s * sizeof(block_q8_0);
        pd0 = __half2float(*(const half *) p0);
        pd1 = __half2float(*(const half *) p1);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            pq0[t] = *(const unsigned short *) (p0 + 2 + b_k + 2 * t);
            pq1[t] = *(const unsigned short *) (p1 + 2 + b_k + 2 * t);
        }
    };

    auto stage = [&](int buf, float4 va) {
        float * fA = (float *) sm.stage.A[buf];
        float * fB = (float *) sm.stage.B[buf];
        const half2 * ha = (const half2 *) &va;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __half22float2(ha[i]);
            fA[(a_k + 2*i    ) * 32 + aoff] = f.x;
            fA[(a_k + 2*i + 1) * 32 + aoff] = f.y;
            fB[(b_k + 2*i    ) * 64 + boff0] = (float) (signed char) (pq0[i] & 0xff) * pd0;
            fB[(b_k + 2*i + 1) * 64 + boff0] = (float) (signed char) (pq0[i] >> 8)   * pd0;
            fB[(b_k + 2*i    ) * 64 + boff1] = (float) (signed char) (pq1[i] & 0xff) * pd1;
            fB[(b_k + 2*i + 1) * 64 + boff1] = (float) (signed char) (pq1[i] >> 8)   * pd1;
        }
    };

    const int sa0 = (0 + mt) ^ kgrp, sa1 = (4 + mt) ^ kgrp;
    const int sb0 = (0 + nt) ^ kgrp, sb1 = (8 + nt) ^ kgrp;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
    }

    pa = gA[0]; fetchb(0);
    stage(0, pa);
    __syncthreads();

    const int kv = MOE_GEMM_KSTAGE / 8;
    int buf = 0;
    for (int kb = 0; kb < k; kb += MOE_GEMM_KSTAGE) {
        const bool last = kb + MOE_GEMM_KSTAGE >= k;
        if (!last) {
            const int s = (kb + MOE_GEMM_KSTAGE) / MOE_GEMM_KSTAGE;
            pa = gA[s * kv];
            fetchb(s);
        }
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int ks = kgrp * 8 + kk;
            const float4 va0 = sm.stage.A[buf][ks][sa0];
            const float4 va1 = sm.stage.A[buf][ks][sa1];
            const float4 vb0 = sm.stage.B[buf][ks][sb0];
            const float4 vb1 = sm.stage.B[buf][ks][sb1];
            const float a[8] = {va0.x, va0.y, va0.z, va0.w, va1.x, va1.y, va1.z, va1.w};
            const float b[8] = {vb0.x, vb0.y, vb0.z, vb0.w, vb1.x, vb1.y, vb1.z, vb1.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) { acc[i][j] += a[i] * b[j]; }
            }
        }
        if (!last) {
            stage(buf ^ 1, pa);
        }
        __syncthreads();
        buf ^= 1;
    }

    // 4-way split-K reduction through smem (aliases the dead staging buffers)
    for (int rnd = 0; rnd < 8; ++rnd) {
        if (kgrp != 0) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) { sm.red[j][(kgrp - 1) * 32 + lane] = acc[rnd][j]; }
        }
        __syncthreads();
        if (kgrp == 0) {
            const int r = mt * 8 + rnd;
            if (r < rows) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float v = acc[rnd][j] + sm.red[j][0 * 32 + lane] + sm.red[j][1 * 32 + lane] + sm.red[j][2 * 32 + lane];
                    C[(size_t) (row0 + r) * n + col0 + nt * 8 + j] = v;
                }
            }
        }
        __syncthreads();
    }
}

bool ggml_cuda_moe_gemm_q8_grouped_supported(int64_t ne00, int64_t ne0, ggml_type type_src0) {
    return type_src0 == GGML_TYPE_Q8_0 && ne00 % MOE_GEMM_KSTAGE == 0 && ne0 % 64 == 0;
}

void ggml_cuda_moe_gemm_q8_grouped(
        ggml_backend_cuda_context & ctx, const char * w_q8, size_t nb01, size_t nb02,
        const half * src1_sorted, float * dst_sorted, const int32_t * tokens_per_expert,
        int64_t n_experts, int64_t ne0, int64_t ne00, cudaStream_t stream) {
    static_assert(sizeof(block_q8_0) == 34, "moe-gemm assumes packed Q8_0 blocks");

    // host tile table [expert | row0 | rows], one entry per 32-row tile
    int n_tiles = 0;
    for (int64_t e = 0; e < n_experts; ++e) {
        n_tiles += (tokens_per_expert[e] + 31) / 32;
    }
    if (n_tiles == 0) {
        return;
    }
    std::vector<int32_t> table((size_t) 3 * n_tiles);
    int t = 0, row = 0;
    for (int64_t e = 0; e < n_experts; ++e) {
        const int m = tokens_per_expert[e];
        for (int o = 0; o < m; o += 32, ++t) {
            table[t]               = (int32_t) e;
            table[n_tiles + t]     = row + o;
            table[2 * n_tiles + t] = m - o < 32 ? m - o : 32;
        }
        row += m;
    }

    ggml_cuda_pool_alloc<int32_t> table_dev(ctx.pool(), table.size());
    CUDA_CHECK(cudaMemcpyAsync(table_dev.ptr, table.data(), table.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

    const dim3 grid(n_tiles, ne0 / 64);
    moe_gemm_q8_grouped<<<grid, 128, 0, stream>>>(src1_sorted, w_q8, dst_sorted,
        table_dev.ptr, n_tiles, nb01, nb02, (int) ne0, (int) ne00);
    CUDA_CHECK(cudaGetLastError());
}
