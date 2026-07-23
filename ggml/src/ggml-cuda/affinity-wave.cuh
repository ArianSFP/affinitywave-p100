#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// [TAG_AFFINITY_WAVE] Phase-1 service prototype.  This is intentionally an
// internal CUDA-backend interface, obtained by the benchmark through the
// backend proc-address table; it does not change the public llama API.
struct ggml_cuda_aw_bench_params {
    int32_t device;
    int32_t active_cells;
    int32_t tokens_per_cell;
    int32_t repeats;
    int32_t route_pattern;
};

struct ggml_cuda_aw_bench_result {
    double  elapsed_ms;
    double  service_ms;
    double  instrumented_ms;
    double  stage_sum_ms;
    double  effective_tflops;
    double  issued_tflops;
    double  useful_fraction;
    double  request_gib_s;
    double  pack_ms;
    double  gate_ms;
    double  up_ms;
    double  swiglu_ms;
    double  down_ms;
    double  reduce_ms;
    double  layout_ms;
    double  down_expand_ms;
    uint64_t device_bytes;
    uint64_t free_bytes_after;
    int32_t descriptors;
    int32_t route_rows;
    int32_t tiles_m64;
    int32_t tiles_m32;
    int32_t tiles_m16;
    int32_t check_ran;
    int32_t check_passed;
    uint64_t check_count;
    uint64_t check_mismatches;
    uint64_t check_first_mismatch;
    uint16_t check_expected;
    uint16_t check_observed;
};

struct ggml_cuda_aw_live_cell {
    int32_t       layer;
    int32_t       home_device;
    int32_t       tokens;
    int32_t       reserved;
    const float * input;
    const int32_t * ids;
    const float * weights;
    float *       output;
    float *       shared_output;
};

// Validate the four opt-in environment controls and the placement manifest.
// When the mode is disabled this is a no-op.  Invalid explicit configurations
// abort at backend initialization rather than silently selecting another path.
void ggml_cuda_affinity_wave_validate_env_or_abort();

// Internal same-byte Q8_0 layout integration. Expert tensors are repacked once
// after their final upload and tagged by device pointer. The native tensor
// shape and byte count remain unchanged.
bool ggml_cuda_affinity_wave_t64_enabled();
bool ggml_cuda_affinity_wave_is_t64(const void * data);
void ggml_cuda_affinity_wave_forget_tensor(const ggml_tensor * tensor);
void ggml_cuda_affinity_wave_forget_range(const void * data, size_t size);
void ggml_cuda_affinity_wave_repack_uploaded_tensor(const ggml_tensor * tensor, bool upload_complete);
void ggml_cuda_affinity_wave_repack_uploaded_tensor_async(
        const ggml_tensor * tensor, bool upload_complete, void * stream);
void ggml_cuda_affinity_wave_normalize_decode(void * stream);
bool ggml_cuda_affinity_wave_get_tensor_native(
        const ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_cuda_affinity_wave_get_tensor_native_2d(
        const ggml_tensor * tensor, void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data);

extern "C" int ggml_cuda_affinity_wave_service_bench(
        const ggml_cuda_aw_bench_params * params,
        ggml_cuda_aw_bench_result       * result,
        char                            * error,
        size_t                            error_capacity);

int ggml_cuda_affinity_wave_live_service(
        void * const * streams,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity);
