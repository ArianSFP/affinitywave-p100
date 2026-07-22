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

// ---------------------------------------------------------------------------
// [TAG_MOE_PLAN] A6: device-built routing plan + persistent grouped GEMM.
// Design + review notes: results/qwen36-35b-moe-pp-20260721/round3-phase0/
// RESULTS.md (A6/A7 section). The whole path is asynchronous and every
// launch is shape-static, so it is CUDA-graph-capturable (A7).
// ---------------------------------------------------------------------------

#include "getrows.cuh"

#define MOE_PLAN_MAX_EXPERTS 512

// kernel 1: single-CTA align/build - histogram over local experts, exclusive
// scans (rows and tiles), tile-descriptor emit, cursor init, scalars.
// One CTA reading ne12*neu ids (~32k) is ~us-scale next to the GEMMs.
__global__ static void moe_plan_build(
        const char * __restrict__ ids, int64_t ne12, int64_t neu,
        size_t ids_nb0, size_t ids_nb1, int n_experts, int expert_base,
        int32_t * __restrict__ counts, int32_t * __restrict__ cursors,
        int32_t * __restrict__ tiles, int tile_cap,
        int32_t * __restrict__ scalars) {  // scalars: [0]=n_local [1]=n_tiles
    __shared__ int hist[MOE_PLAN_MAX_EXPERTS];
    __shared__ int row0[MOE_PLAN_MAX_EXPERTS];
    __shared__ int tbase[MOE_PLAN_MAX_EXPERTS];
    const int tid = threadIdx.x;
    for (int e = tid; e < n_experts; e += blockDim.x) { hist[e] = 0; }
    __syncthreads();
    const int64_t n_assign = ne12 * neu;
    for (int64_t i = tid; i < n_assign; i += blockDim.x) {
        const int64_t i12 = i / neu, iex = i % neu;
        const int32_t e = *(const int32_t *) (ids + i12*ids_nb1 + iex*ids_nb0);
        const int32_t le = e - expert_base;
        if (le >= 0 && le < n_experts) {
            atomicAdd(&hist[le], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int rrun = 0, trun = 0;
        for (int e = 0; e < n_experts; ++e) {
            row0[e]  = rrun;
            tbase[e] = trun;
            rrun += hist[e];
            trun += (hist[e] + 31) / 32;
        }
        scalars[0] = rrun;
        scalars[1] = trun;
    }
    __syncthreads();
    for (int e = tid; e < n_experts; e += blockDim.x) {
        counts[e]  = hist[e];
        cursors[e] = row0[e];
        const int nt = (hist[e] + 31) / 32;
        for (int t = 0; t < nt; ++t) {
            const int idx = tbase[e] + t;
            tiles[idx]                = e;
            tiles[tile_cap + idx]     = row0[e] + t*32;
            tiles[2*tile_cap + idx]   = hist[e] - t*32 < 32 ? hist[e] - t*32 : 32;
        }
    }
}

// kernel 2: scatter - consumes the cursors, fills to_sorted (token index per
// sorted row; order within an expert is unstable, which is safe: each output
// row depends only on (token, expert) and from_sorted preserves the exact
// inverse mapping) and from_sorted (position per assignment; non-local ->
// zero_row = the CAPACITY slot, shape-static).
__global__ static void moe_plan_scatter(
        const char * __restrict__ ids, int64_t ne12, int64_t neu, int64_t ne11,
        size_t ids_nb0, size_t ids_nb1, int n_experts, int expert_base,
        int zero_row, int32_t * __restrict__ cursors,
        int32_t * __restrict__ to_sorted, int32_t * __restrict__ from_sorted) {
    const int64_t n_assign = ne12 * neu;
    for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n_assign; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t i12 = i / neu, iex = i % neu;
        const int32_t e = *(const int32_t *) (ids + i12*ids_nb1 + iex*ids_nb0);
        const int32_t le = e - expert_base;
        if (le < 0 || le >= n_experts) {
            from_sorted[i] = zero_row;
            continue;
        }
        const int pos = atomicAdd(&cursors[le], 1);
        // src1 row, same formula as the host sort: token-major, slot within
        to_sorted[pos]  = (int32_t) (i12*ne11 + iex % ne11);
        from_sorted[i]  = pos;
    }
}

// persistent fixed-grid grouped GEMM: same 32x64/128thr/splitK4 mainloop as
// moe_gemm_q8_grouped, but (a) grid is a fixed occupancy-sized launch that
// grid-strides over m_tile x n_tile work read from the DEVICE plan, and
// (b) A is gathered in-kernel from the ORIGINAL f32 src1 through to_sorted
// (A3) - no f16 gather pass, activations enter the FFMA path unrounded.
__global__ __launch_bounds__(128, 2)
static void moe_gemm_q8_plan(const float * __restrict__ A, size_t a_stride,
                             const char * __restrict__ W, float * __restrict__ C,
                             const int32_t * __restrict__ to_sorted,
                             const int32_t * __restrict__ tiles, int tile_cap,
                             const int32_t * __restrict__ scalars,
                             size_t nb01, size_t nb02, int n, int k) {
    __shared__ moe_gemm_smem sm;

    const int n_ntiles = n / 64;
    const int n_work   = scalars[1] * n_ntiles;

    const int tid  = threadIdx.x;
    const int kgrp = tid >> 5;
    const int lane = tid & 31;
    const int mt   = lane >> 3;
    const int nt   = lane & 7;

    const int a_r = tid >> 2, a_k = (tid & 3) * 8;
    const int b_r0 = tid >> 2;
    const int b_r1 = 32 + (tid >> 2);
    const int xk   = tid & 3;
    const int b_k  = xk * 8;

    const int aslot = (((a_r >> 2) & 1) * 4 + (a_r >> 3)) ^ xk;
    const int aoff  = aslot * 4 + (a_r & 3);
    const int bslot0 = ((((b_r0 >> 2) & 1) * 8 + (b_r0 >> 3)) ^ xk);
    const int boff0  = bslot0 * 4 + (b_r0 & 3);
    const int bslot1 = ((((b_r1 >> 2) & 1) * 8 + (b_r1 >> 3)) ^ xk);
    const int boff1  = bslot1 * 4 + (b_r1 & 3);
    const int sa0 = (0 + mt) ^ kgrp, sa1 = (4 + mt) ^ kgrp;
    const int sb0 = (0 + nt) ^ kgrp, sb1 = (8 + nt) ^ kgrp;

    for (int work = blockIdx.x; work < n_work; work += gridDim.x) {
        const int mt_id = work / n_ntiles;
        const int col0  = (work % n_ntiles) * 64;
        const int texp = tiles[mt_id];
        const int row0 = tiles[tile_cap + mt_id];
        const int rows = tiles[2*tile_cap + mt_id];

        const char * bq = W + (size_t) texp * nb02;
        const int arow = to_sorted[row0 + (a_r < rows ? a_r : rows - 1)];

        float4 pa0, pa1;
        unsigned short pq0[4], pq1[4];
        float pd0, pd1;
        const float4 * gA = (const float4 *) (A + (size_t) arow * a_stride + a_k);
        const char * gQ0 = bq + (size_t) (col0 + b_r0) * nb01;
        const char * gQ1 = bq + (size_t) (col0 + b_r1) * nb01;

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
        auto stage = [&](int buf, float4 va0, float4 va1) {
            float * fA = (float *) sm.stage.A[buf];
            float * fB = (float *) sm.stage.B[buf];
            const float * af = (const float *) &va0;
            const float * ag = (const float *) &va1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                fA[(a_k + i    ) * 32 + aoff] = af[i];
                fA[(a_k + i + 4) * 32 + aoff] = ag[i];
                fB[(b_k + 2*i    ) * 64 + boff0] = (float) (signed char) (pq0[i] & 0xff) * pd0;
                fB[(b_k + 2*i + 1) * 64 + boff0] = (float) (signed char) (pq0[i] >> 8)   * pd0;
                fB[(b_k + 2*i    ) * 64 + boff1] = (float) (signed char) (pq1[i] & 0xff) * pd1;
                fB[(b_k + 2*i + 1) * 64 + boff1] = (float) (signed char) (pq1[i] >> 8)   * pd1;
            }
        };

        float acc[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
        }

        pa0 = gA[0]; pa1 = gA[1]; fetchb(0);
        stage(0, pa0, pa1);
        __syncthreads();

        int buf = 0;
        for (int kb = 0; kb < k; kb += MOE_GEMM_KSTAGE) {
            const bool last = kb + MOE_GEMM_KSTAGE >= k;
            if (!last) {
                const int s = (kb + MOE_GEMM_KSTAGE) / MOE_GEMM_KSTAGE;
                pa0 = gA[s * 8]; pa1 = gA[s * 8 + 1];   // f32: 8 float4 per 32-k stage row
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
                stage(buf ^ 1, pa0, pa1);
            }
            __syncthreads();
            buf ^= 1;
        }

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
}

// [TAG_MOE_PLAN] persistent per-(device,stream-slot) plan workspace. Stable
// cudaMalloc addresses that survive graph capture/replay (pool allocations
// give no such guarantee - the Q8_1_DEDUP stale-cache lesson). The build/
// scatter kernels re-fill it every call; this struct only owns memory.
struct moe_plan_ws {
    int32_t * buf = nullptr;   // [counts E | cursors E | scalars 2 | tiles 3*tile_cap | to_sorted A | from_sorted A]
    size_t    a_cap = 0;       // assignment capacity
    size_t    e_cap = 0;       // expert capacity
    int       device = -1;
    int32_t * counts()      { return buf; }
    int32_t * cursors()     { return buf + e_cap; }
    int32_t * scalars()     { return buf + 2*e_cap; }
    int32_t * tiles()       { return buf + 2*e_cap + 2; }
    size_t    tile_cap()    { return a_cap/32 + e_cap + 1; }
    int32_t * to_sorted()   { return tiles() + 3*tile_cap(); }
    int32_t * from_sorted() { return to_sorted() + a_cap; }
    size_t    total()       { return 2*e_cap + 2 + 3*tile_cap() + 2*a_cap; }
};

bool ggml_cuda_moe_mul_mat_id_plan(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, int32_t expert_base,
        bool moe_ep, const char * src0_data, cudaStream_t stream) {
    static const bool moe_plan = getenv("GGML_CUDA_MOE_PLAN") != nullptr;
    // host-inspection debug paths need the host sort - fall through to it
    static const bool host_dbg = getenv("GGML_CUDA_MOE_HIST") != nullptr
                              || getenv("GGML_CUDA_MOE_DEBUG_STATS") != nullptr;
    if (!moe_plan || host_dbg) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int64_t ne00 = src0->ne[0], ne0 = src0->ne[1], n_experts = src0->ne[2];
    const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
    const int64_t neu  = ids->ne[0];
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ne00 % MOE_GEMM_KSTAGE != 0 || ne0 % 64 != 0 ||
        n_experts > MOE_PLAN_MAX_EXPERTS || src1->nb[0] != sizeof(float) ||
        src1->nb[2] != src1->nb[1]*ne11 ||   // flat row indexing needs contiguous dim 1/2
        ne00 % 8 != 0) {
        return false;
    }

    const int64_t n_assign = ne12 * neu;   // == ne_get_rows (shape-static)
    const size_t  a_stride = src1->nb[1] / sizeof(float);

    // persistent workspace (per device + stream slot, like the sort cache)
    static thread_local moe_plan_ws ws_arr[GGML_CUDA_MAX_DEVICES][2];
    const int dev  = ggml_cuda_get_device();
    const int slot = ctx.curr_stream_no < 2 ? ctx.curr_stream_no : 0;
    moe_plan_ws & ws = ws_arr[dev][slot];
    if (ws.device != dev || ws.a_cap < (size_t) n_assign || ws.e_cap < (size_t) n_experts) {
        if (ws.buf != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));   // fence in-flight users; realloc is rare (growth only)
            CUDA_CHECK(cudaFree(ws.buf));
            ws.buf = nullptr;
        }
        ws.a_cap  = std::max<size_t>(n_assign, ws.a_cap);
        ws.e_cap  = std::max<size_t>(n_experts, ws.e_cap);
        ws.device = dev;
        CUDA_CHECK(cudaMalloc((void **) &ws.buf, ws.total()*sizeof(int32_t)));
    }
    const int tile_cap = (int) ws.tile_cap();
    const int zero_row = (int) n_assign;   // CAPACITY slot, shape-static

    // plan build + scatter (device-only, no syncs)
    moe_plan_build<<<1, 256, 0, stream>>>((const char *) ids->data, ne12, neu,
        ids->nb[0], ids->nb[1], (int) n_experts, expert_base,
        ws.counts(), ws.cursors(), ws.tiles(), tile_cap, ws.scalars());
    const int scatter_blocks = (int) std::min<int64_t>((n_assign + 255) / 256, 256);
    moe_plan_scatter<<<scatter_blocks, 256, 0, stream>>>((const char *) ids->data, ne12, neu, ne11,
        ids->nb[0], ids->nb[1], (int) n_experts, expert_base, zero_row,
        ws.cursors(), ws.to_sorted(), ws.from_sorted());
    CUDA_CHECK(cudaGetLastError());

    // optional host-reference validation (env GGML_CUDA_MOE_PLAN_CHECK) - syncs, debug only
    static const bool plan_check = getenv("GGML_CUDA_MOE_PLAN_CHECK") != nullptr;
    if (plan_check) {
        static int checks_left = 64;
        if (checks_left > 0) {
            --checks_left;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<int32_t> h(ws.total());
            CUDA_CHECK(cudaMemcpy(h.data(), ws.buf, ws.total()*sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::vector<char> hids(ggml_nbytes(ids));
            CUDA_CHECK(cudaMemcpy(hids.data(), ids->data, hids.size(), cudaMemcpyDeviceToHost));
            const int32_t * counts = h.data(), * scal = h.data() + 2*ws.e_cap;
            const int32_t * tls = scal + 2;
            const int32_t * tos = tls + 3*tile_cap, * frs = tos + ws.a_cap;
            std::vector<int32_t> ref_cnt(n_experts, 0);
            for (int64_t i = 0; i < n_assign; ++i) {
                const int32_t e = *(const int32_t *)(hids.data() + (i/neu)*ids->nb[1] + (i%neu)*ids->nb[0]);
                const int32_t le = e - expert_base;
                if (le >= 0 && le < n_experts) { ref_cnt[le]++; }
            }
            int32_t ref_local = 0, ref_tiles = 0;
            std::vector<int32_t> ref_row0(n_experts);
            for (int64_t e = 0; e < n_experts; ++e) {
                ref_row0[e] = ref_local;
                ref_local += ref_cnt[e]; ref_tiles += (ref_cnt[e] + 31)/32;
            }
            int bad = 0;
            for (int64_t e = 0; e < n_experts && bad < 5; ++e) {
                if (counts[e] != ref_cnt[e]) { fprintf(stderr, "[moe-plan] BAD count e=%lld %d!=%d\n", (long long) e, counts[e], ref_cnt[e]); ++bad; }
            }
            if (scal[0] != ref_local || scal[1] != ref_tiles) { fprintf(stderr, "[moe-plan] BAD scalars %d/%d != %d/%d\n", scal[0], scal[1], ref_local, ref_tiles); ++bad; }
            int t = 0;
            for (int64_t e = 0; e < n_experts && bad < 5; ++e) {
                for (int o = 0; o < ref_cnt[e]; o += 32, ++t) {
                    if (tls[t] != e || tls[tile_cap + t] != ref_row0[e] + o ||
                        tls[2*tile_cap + t] != std::min<int32_t>(32, ref_cnt[e] - o)) {
                        fprintf(stderr, "[moe-plan] BAD tile %d\n", t); ++bad; break;
                    }
                }
            }
            std::vector<char> seen(ref_local, 0);
            for (int64_t i = 0; i < n_assign && bad < 5; ++i) {
                const int32_t e = *(const int32_t *)(hids.data() + (i/neu)*ids->nb[1] + (i%neu)*ids->nb[0]);
                const int32_t le = e - expert_base;
                const int32_t pos = frs[i];
                if (le < 0 || le >= n_experts) {
                    if (pos != zero_row) { fprintf(stderr, "[moe-plan] BAD nonlocal from[%lld]=%d\n", (long long) i, pos); ++bad; }
                    continue;
                }
                if (pos < ref_row0[le] || pos >= ref_row0[le] + ref_cnt[le] || seen[pos] ||
                    tos[pos] != (int32_t)((i/neu)*ne11 + (i%neu) % ne11)) {
                    fprintf(stderr, "[moe-plan] BAD from[%lld]=%d (e=%d)\n", (long long) i, pos, le); ++bad;
                } else { seen[pos] = 1; }
            }
            fprintf(stderr, "[moe-plan] check dev%d %s n_local=%d n_tiles=%d %s\n",
                dev, dst->name, scal[0], scal[1], bad ? "FAIL" : "PASS");
        }
    }

    // dst_sorted at CAPACITY (+1 zero row at the capacity slot)
    ggml_cuda_pool_alloc<float> dst_sorted(ctx.pool(), ((size_t) n_assign + 1) * ne0);
    CUDA_CHECK(cudaMemsetAsync(dst_sorted.ptr + (size_t) zero_row * ne0, 0, ne0*sizeof(float), stream));

    // persistent fixed-grid grouped GEMM, in-kernel f32 gather (A3)
    const int nsm = ggml_cuda_info().devices[dev].nsm;
    moe_gemm_q8_plan<<<nsm * 2, 128, 0, stream>>>((const float *) src1->data, a_stride,
        src0_data, dst_sorted.ptr, ws.to_sorted(), ws.tiles(), tile_cap, ws.scalars(),
        src0->nb[1], src0->nb[2], (int) ne0, (int) ne00);
    CUDA_CHECK(cudaGetLastError());

    // capacity-sized inverse scatter (unchanged machinery; launch is shape-static)
    get_rows_cuda(dst_sorted.ptr, GGML_TYPE_F32, ws.from_sorted(), dst->data, dst->type,
        ne0, ne0*sizeof(float), n_assign*ne0*sizeof(float), n_assign*ne0*sizeof(float),
        n_assign, 1, 1, sizeof(int32_t), n_assign*sizeof(int32_t), n_assign*sizeof(int32_t),
        dst->nb[1], dst->nb[2], dst->nb[3], stream);
    CUDA_CHECK(cudaGetLastError());

    GGML_UNUSED(moe_ep);
    return true;
}
