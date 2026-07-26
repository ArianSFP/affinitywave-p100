#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// [TAG_AFFINITY_WAVE] Phase-1 service prototype.  This is intentionally an
// internal CUDA-backend interface, obtained by the benchmark through the
// backend proc-address table; it does not change the public llama API.
struct ggml_cuda_aw_bench_params {
    int32_t device;
    int32_t participants;
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
    int32_t tiles_warp;
    int32_t tiles_m8;
    int32_t tiles_m4;
    int32_t q8_kernel;
    int32_t q8_engine;
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
    size_t        ids_stride;
    size_t        weights_stride;
    const float * input;
    const int32_t * ids;
    const float * weights;
    float *       output;
    float *       shared_output;
    const float * reference;
};

struct ggml_cuda_aw_headfold_gdn_lane {
    int32_t       tokens;
    int32_t       qk_stride;
    int32_t       v_stride;
    int32_t       scalar_stride;
    const float * q;
    const float * k;
    const float * v;
    const float * gate;
    const float * beta;
    const float * state;
    float *       output;
    float *       state_output;
};

struct ggml_cuda_aw_pairfold_handoff {
    int32_t       generation;
    int32_t       task;
    int32_t       panel;
    int32_t       tokens;
    int32_t       source[2];
    int32_t       destination[2];
    const float * output[2];
    float *       input[2];
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

int ggml_cuda_affinity_wave_r44_prefetch(
        int32_t layer,
        char * error,
        size_t error_capacity);

int ggml_cuda_affinity_wave_headfold_gdn(
        void * const * streams,
        const ggml_cuda_aw_headfold_gdn_lane * lanes,
        int32_t n_lanes,
        char * error,
        size_t error_capacity);

int ggml_cuda_affinity_wave_pairfold_gdn(
        void * const * streams,
        const int32_t * devices,
        const int32_t * groups,
        const ggml_cuda_aw_headfold_gdn_lane * lanes,
        int32_t n_lanes,
        char * error,
        size_t error_capacity);

int ggml_cuda_affinity_wave_pairfold_service(
        void * const * streams,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity);

bool ggml_cuda_affinity_wave_trace_enabled();
bool ggml_cuda_affinity_wave_gemm_map_enabled();
void ggml_cuda_affinity_wave_trace_push(const char * name, uint32_t category, uint64_t payload);
void ggml_cuda_affinity_wave_trace_pop();
void ggml_cuda_affinity_wave_trace_mark(const char * name, uint32_t category, uint64_t payload);
void ggml_cuda_affinity_wave_trace_name_thread(const char * name);
void ggml_cuda_affinity_wave_trace_name_stream(void * stream, const char * name);
