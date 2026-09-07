#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#define CUDA_TOP_K_BLOCK      256
#define CUDA_TOP_K_WARPS      (CUDA_TOP_K_BLOCK/32)
#define CUDA_TOP_K_MAX_K      16
#define CUDA_TOP_K_MAX_BLOCKS 128
#define CUDA_TOP_K_MIN_NCOLS  4096
#define CUDA_TOP_K_MAX_ELEMS  16
#define CUDA_TOP_K_MERGE      ((CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K + 31)/32)

static __device__ __forceinline__ uint32_t top_k_key(const float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

static __device__ __forceinline__ uint64_t top_k_pack(const uint32_t key, const int index) {
    return ((uint64_t) key << 32) | (uint32_t) ~index;
}

static __device__ __forceinline__ void top_k_warp_max(uint64_t & value) {
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
        const uint64_t other = __shfl_xor_sync(0xffffffffu, value, shift);
        value = other > value ? other : value;
    }
}

template <int ELEMS>
static __global__ void __launch_bounds__(CUDA_TOP_K_BLOCK)
    k_top_k_stage1(const float * __restrict__ src, uint64_t * __restrict__ out,
                   const int ncols, const int k) {
    __shared__ uint64_t shared[CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float * row = src + (int64_t) blockIdx.y*ncols;
    const int base = blockIdx.x*(CUDA_TOP_K_BLOCK*ELEMS) + warp*(32*ELEMS);
    uint64_t candidates[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        const int index = base + e*32 + lane;
        candidates[e] = index < ncols ? top_k_pack(top_k_key(row[index]), index) : 0ull;
    }
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { best = candidates[e] > best ? candidates[e] : best; }
        top_k_warp_max(best);
        if (lane == 0) {
            shared[warp*k + r] = best;
        }
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { if (candidates[e] == best) { candidates[e] = 0ull; } }
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    const int count = CUDA_TOP_K_WARPS*k;
    uint64_t merged[CUDA_TOP_K_MERGE];
#pragma unroll
    for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) {
        const int index = e*32 + lane;
        merged[e] = index < count ? shared[index] : 0ull;
    }
    uint64_t * row_out = out + ((int64_t) blockIdx.y*gridDim.x + blockIdx.x)*k;
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { best = merged[e] > best ? merged[e] : best; }
        top_k_warp_max(best);
        if (lane == 0) {
            row_out[r] = best;
        }
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { if (merged[e] == best) { merged[e] = 0ull; } }
    }
}

template <int ELEMS>
static __global__ void __launch_bounds__(CUDA_TOP_K_BLOCK)
    k_top_k_stage2(const uint64_t * __restrict__ in, int * __restrict__ dst,
                   const int ncand, const int k) {
    __shared__ uint64_t shared[CUDA_TOP_K_WARPS*CUDA_TOP_K_MAX_K];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const uint64_t * row = in + (int64_t) blockIdx.y*ncand;
    uint64_t candidates[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; ++e) {
        const int index = warp*(32*ELEMS) + e*32 + lane;
        candidates[e] = index < ncand ? row[index] : 0ull;
    }
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { best = candidates[e] > best ? candidates[e] : best; }
        top_k_warp_max(best);
        if (lane == 0) {
            shared[warp*k + r] = best;
        }
#pragma unroll
        for (int e = 0; e < ELEMS; ++e) { if (candidates[e] == best) { candidates[e] = 0ull; } }
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    const int count = CUDA_TOP_K_WARPS*k;
    uint64_t merged[CUDA_TOP_K_MERGE];
#pragma unroll
    for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) {
        const int index = e*32 + lane;
        merged[e] = index < count ? shared[index] : 0ull;
    }
    int * row_out = dst + (int64_t) blockIdx.y*k;
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { best = merged[e] > best ? merged[e] : best; }
        top_k_warp_max(best);
        if (lane == 0) {
            row_out[r] = (int) ~(uint32_t) best;
        }
#pragma unroll
        for (int e = 0; e < CUDA_TOP_K_MERGE; ++e) { if (merged[e] == best) { merged[e] = 0ull; } }
    }
}

static bool top_k_partial_cuda(ggml_cuda_pool & pool, const float * src, int * dst,
                               const int ncols, const int nrows, const int k, cudaStream_t stream) {
    static const bool disable = getenv("GGML_CUDA_DISABLE_TOP_K_PARTIAL") != nullptr &&
                                std::atoi(getenv("GGML_CUDA_DISABLE_TOP_K_PARTIAL"));
    if (disable || k < 1 || k > CUDA_TOP_K_MAX_K || ncols < CUDA_TOP_K_MIN_NCOLS || nrows < 1) {
        return false;
    }
    int elems = 1;
    while (elems < CUDA_TOP_K_MAX_ELEMS &&
           (int64_t) ncols > (int64_t) CUDA_TOP_K_BLOCK*elems*CUDA_TOP_K_MAX_BLOCKS) {
        elems *= 2;
    }
    if ((int64_t) ncols > (int64_t) CUDA_TOP_K_BLOCK*elems*CUDA_TOP_K_MAX_BLOCKS) {
        return false;
    }
    const int stripe = CUDA_TOP_K_BLOCK*elems;
    const int nblocks = (ncols + stripe - 1) / stripe;
    const int ncand = nblocks*k;
    int elems2 = 1;
    while (elems2 < CUDA_TOP_K_MAX_ELEMS && ncand > CUDA_TOP_K_BLOCK*elems2) {
        elems2 *= 2;
    }
    if (ncand > CUDA_TOP_K_BLOCK*elems2) {
        return false;
    }
    ggml_cuda_pool_alloc<uint64_t> candidates(pool, (size_t) ncand*nrows);
    const dim3 grid1(nblocks, nrows, 1);
    const dim3 grid2(1, nrows, 1);
#define CUDA_TOP_K_LAUNCH1(E) \
    case E: k_top_k_stage1<E><<<grid1, CUDA_TOP_K_BLOCK, 0, stream>>>(src, candidates.get(), ncols, k); break
    switch (elems) {
        CUDA_TOP_K_LAUNCH1(1);
        CUDA_TOP_K_LAUNCH1(2);
        CUDA_TOP_K_LAUNCH1(4);
        CUDA_TOP_K_LAUNCH1(8);
        CUDA_TOP_K_LAUNCH1(16);
        default: return false;
    }
#undef CUDA_TOP_K_LAUNCH1
#define CUDA_TOP_K_LAUNCH2(E) \
    case E: k_top_k_stage2<E><<<grid2, CUDA_TOP_K_BLOCK, 0, stream>>>(candidates.get(), dst, ncand, k); break
    switch (elems2) {
        CUDA_TOP_K_LAUNCH2(1);
        CUDA_TOP_K_LAUNCH2(2);
        CUDA_TOP_K_LAUNCH2(4);
        CUDA_TOP_K_LAUNCH2(8);
        CUDA_TOP_K_LAUNCH2(16);
        default: return false;
    }
#undef CUDA_TOP_K_LAUNCH2
    return true;
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

#ifndef CUB_TOP_K_AVAILABLE
    if (ncols <= INT_MAX && nrows <= INT_MAX && k <= INT_MAX &&
        top_k_partial_cuda(pool, src0_d, dst_d, (int) ncols, (int) nrows, (int) k, stream)) {
        return;
    }
#endif
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
