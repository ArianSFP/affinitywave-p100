#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_OK(call) do { \
    const cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_OK(call) do { \
    const cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS failure at %s:%d: %d\n", \
                __FILE__, __LINE__, static_cast<int>(status_)); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kASeed = 0x5a17c9e3U;
constexpr uint32_t kBSeed = 0xc31d7f49U;

struct options {
    int base_m = 512;
    int n = 2032;
    int k = 2048;
    int baseline_algorithm = 3;
    int warmup = 5;
    int repeats = 30;
    int device = 0;
};

struct timing_result {
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    double ms = -1.0;
};

struct mismatch_result {
    unsigned long long first = 0;
    unsigned long long second = 0;
};

static void usage(const char * program) {
    std::fprintf(
            stderr,
            "usage: %s [--base-m M] [--n N] [--k K] "
            "[--baseline-algorithm A] [--warmup W] "
            "[--repeats R] [--device D]\n",
            program);
}

static bool parse_int(const char * text, int minimum, int * value) {
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
            parsed < minimum || parsed > INT_MAX) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

static bool is_swept_algorithm(int algorithm) {
    return algorithm == static_cast<int>(CUBLAS_GEMM_DEFAULT) ||
            (algorithm >= 0 && algorithm <= 23) ||
            (algorithm >= 99 && algorithm <= 115);
}

static bool parse_options(int argc, char ** argv, options * result) {
    for (int index = 1; index < argc; ++index) {
        const char * name = argv[index];
        if (std::strcmp(name, "--help") == 0 ||
                std::strcmp(name, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", name);
            return false;
        }

        const char * text = argv[++index];
        int * destination = nullptr;
        int minimum = 1;
        if (std::strcmp(name, "--base-m") == 0) {
            destination = &result->base_m;
        } else if (std::strcmp(name, "--n") == 0) {
            destination = &result->n;
        } else if (std::strcmp(name, "--k") == 0) {
            destination = &result->k;
        } else if (std::strcmp(name, "--baseline-algorithm") == 0) {
            destination = &result->baseline_algorithm;
            minimum = -1;
        } else if (std::strcmp(name, "--warmup") == 0) {
            destination = &result->warmup;
            minimum = 0;
        } else if (std::strcmp(name, "--repeats") == 0) {
            destination = &result->repeats;
        } else if (std::strcmp(name, "--device") == 0) {
            destination = &result->device;
            minimum = 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", name);
            return false;
        }

        if (!parse_int(text, minimum, destination)) {
            std::fprintf(stderr, "invalid value for %s: %s\n", name, text);
            return false;
        }
    }

    if (!is_swept_algorithm(result->baseline_algorithm)) {
        std::fprintf(
                stderr,
                "baseline algorithm must be -1, 0-23, or 99-115\n");
        return false;
    }
    if (result->base_m > INT_MAX/2) {
        std::fprintf(stderr, "2*base-m exceeds the cuBLAS int range\n");
        return false;
    }
    return true;
}

static bool checked_product(
        size_t left, size_t right, size_t element_size, size_t * result) {
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    if (right != 0 && left > maximum/right) {
        return false;
    }
    const size_t elements = left*right;
    if (element_size != 0 && elements > maximum/element_size) {
        return false;
    }
    *result = elements;
    return true;
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__global__ void init_f16_cuda(
        half * destination, size_t elements, uint32_t seed) {
    for (size_t index =
                    static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
            index < elements;
            index += static_cast<size_t>(gridDim.x)*blockDim.x) {
        const uint32_t folded =
                static_cast<uint32_t>(index) ^
                static_cast<uint32_t>(index >> 32);
        const uint32_t bits = mix32(folded ^ seed);
        const int value = static_cast<int>(bits & 0x7ffU) - 1024;
        float input = static_cast<float>(value)*(1.0f/1024.0f);
        switch (index & 0x3fffU) {
            case 0:
                input = 0.0f;
                break;
            case 1:
                input = -0.0f;
                break;
            case 2:
                input = 0x1.0p-24f;
                break;
            case 3:
                input = -0x1.0p-24f;
                break;
            default:
                break;
        }
        destination[index] = __float2half_rn(input);
    }
}

__global__ void count_stacked_mismatches_cuda(
        const float * reference,
        const float * stacked,
        size_t region_elements,
        int base_m,
        unsigned long long * errors) {
    unsigned long long first = 0;
    unsigned long long second = 0;
    const size_t stacked_m = static_cast<size_t>(2)*base_m;
    for (size_t index =
                    static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
            index < region_elements;
            index += static_cast<size_t>(gridDim.x)*blockDim.x) {
        const size_t column = index/static_cast<size_t>(base_m);
        const size_t row = index -
                column*static_cast<size_t>(base_m);
        const size_t stacked_base = column*stacked_m + row;
        first += __float_as_uint(reference[index]) !=
                __float_as_uint(stacked[stacked_base]);
        second += __float_as_uint(reference[region_elements + index]) !=
                __float_as_uint(
                        stacked[stacked_base +
                                static_cast<size_t>(base_m)]);
    }
    if (first != 0) {
        atomicAdd(errors, first);
    }
    if (second != 0) {
        atomicAdd(errors + 1, second);
    }
}

static cublasStatus_t run_projection(
        cublasHandle_t handle,
        const half * a,
        const half * b,
        float * c,
        int m,
        const options & shape,
        cublasGemmAlgo_t algorithm) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            shape.n,
            shape.k,
            &alpha,
            a,
            CUDA_R_16F,
            shape.k,
            b,
            CUDA_R_16F,
            shape.k,
            &beta,
            c,
            CUDA_R_32F,
            m,
            CUBLAS_COMPUTE_32F,
            algorithm);
}

static cublasStatus_t run_separate_pair(
        cublasHandle_t handle,
        const half * a,
        const half * b,
        float * c,
        size_t a_region_elements,
        size_t c_region_elements,
        const options & shape,
        cublasGemmAlgo_t algorithm) {
    cublasStatus_t status = run_projection(
            handle,
            a,
            b,
            c,
            shape.base_m,
            shape,
            algorithm);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return run_projection(
            handle,
            a + a_region_elements,
            b,
            c + c_region_elements,
            shape.base_m,
            shape,
            algorithm);
}

template<typename Enqueue>
static timing_result time_call(
        cudaStream_t stream,
        cudaEvent_t begin,
        cudaEvent_t end,
        int warmup,
        int repeats,
        const Enqueue & enqueue) {
    timing_result result;
    for (int iteration = 0; iteration < warmup; ++iteration) {
        result.cublas_status = enqueue();
        if (result.cublas_status != CUBLAS_STATUS_SUCCESS) {
            result.cuda_status = cudaStreamSynchronize(stream);
            return result;
        }
    }

    result.cuda_status = cudaStreamSynchronize(stream);
    if (result.cuda_status != cudaSuccess) {
        return result;
    }

    CUDA_OK(cudaEventRecord(begin, stream));
    for (int iteration = 0; iteration < repeats; ++iteration) {
        result.cublas_status = enqueue();
        if (result.cublas_status != CUBLAS_STATUS_SUCCESS) {
            result.cuda_status = cudaStreamSynchronize(stream);
            return result;
        }
    }
    CUDA_OK(cudaEventRecord(end, stream));
    result.cuda_status = cudaEventSynchronize(end);
    if (result.cuda_status != cudaSuccess) {
        return result;
    }

    float elapsed = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));
    result.ms = static_cast<double>(elapsed)/repeats;
    return result;
}

static mismatch_result count_mismatches(
        const float * reference,
        const float * candidate,
        size_t region_elements,
        int base_m,
        unsigned long long * errors,
        cudaStream_t stream) {
    const int blocks = static_cast<int>(
            std::min<size_t>(
                    (region_elements + kThreads - 1)/kThreads,
                    4096));
    CUDA_OK(cudaMemsetAsync(
            errors, 0, 2*sizeof(*errors), stream));
    count_stacked_mismatches_cuda<<<blocks, kThreads, 0, stream>>>(
            reference,
            candidate,
            region_elements,
            base_m,
            errors);
    CUDA_OK(cudaGetLastError());

    mismatch_result result;
    CUDA_OK(cudaMemcpyAsync(
            &result,
            errors,
            sizeof(result),
            cudaMemcpyDeviceToHost,
            stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return result;
}

static const char * algorithm_name(int algorithm) {
    if (algorithm == static_cast<int>(CUBLAS_GEMM_DEFAULT)) {
        return "DEFAULT";
    }
    if (algorithm == static_cast<int>(CUBLAS_GEMM_DEFAULT_TENSOR_OP)) {
        return "DEFAULT_TENSOR_OP";
    }
    if (algorithm >= 0 && algorithm <= 23) {
        return "ALGO";
    }
    if (algorithm >= 100 && algorithm <= 115) {
        return "ALGO_TENSOR_OP";
    }
    return "UNKNOWN";
}

static void print_json_string(const char * text) {
    std::putchar('"');
    for (const unsigned char * cursor =
                    reinterpret_cast<const unsigned char *>(text);
            *cursor != '\0';
            ++cursor) {
        switch (*cursor) {
            case '"':
                std::fputs("\\\"", stdout);
                break;
            case '\\':
                std::fputs("\\\\", stdout);
                break;
            case '\b':
                std::fputs("\\b", stdout);
                break;
            case '\f':
                std::fputs("\\f", stdout);
                break;
            case '\n':
                std::fputs("\\n", stdout);
                break;
            case '\r':
                std::fputs("\\r", stdout);
                break;
            case '\t':
                std::fputs("\\t", stdout);
                break;
            default:
                if (*cursor < 0x20) {
                    std::printf("\\u%04x", static_cast<unsigned>(*cursor));
                } else {
                    std::putchar(*cursor);
                }
                break;
        }
    }
    std::putchar('"');
}

static void print_signature(const options & shape) {
    std::printf(
            "\"base_m\":%d,\"stacked_m\":%d,\"n\":%d,\"k\":%d,"
            "\"op_a\":\"CUBLAS_OP_T\",\"op_b\":\"CUBLAS_OP_N\","
            "\"lda\":%d,\"ldb\":%d,"
            "\"separate_ldc\":%d,\"stacked_ldc\":%d,"
            "\"a_type\":\"CUDA_R_16F\","
            "\"b_type\":\"CUDA_R_16F\","
            "\"c_type\":\"CUDA_R_32F\","
            "\"compute_type\":\"CUBLAS_COMPUTE_32F\","
            "\"alpha\":1.0,\"beta\":0.0,"
            "\"baseline_algorithm\":%d,"
            "\"baseline_algorithm_name\":\"%s\","
            "\"warmup\":%d,\"repeats\":%d",
            shape.base_m,
            2*shape.base_m,
            shape.n,
            shape.k,
            shape.k,
            shape.k,
            shape.base_m,
            2*shape.base_m,
            shape.baseline_algorithm,
            algorithm_name(shape.baseline_algorithm),
            shape.warmup,
            shape.repeats);
}

static bool timing_completed(const timing_result & timing) {
    return timing.cublas_status == CUBLAS_STATUS_SUCCESS &&
            timing.cuda_status == cudaSuccess;
}

static void print_baseline(
        const options & shape, const timing_result & timing) {
    std::fputs(
            "{\"kind\":\"baseline_pair\","
            "\"api\":\"two_cublasGemmEx\",",
            stdout);
    print_signature(shape);
    std::printf(
            ",\"algorithm\":%d,\"algorithm_name\":\"%s\","
            "\"gemm_calls_per_repeat\":2,"
            "\"cublas_status\":%d,\"cuda_status\":%d,"
            "\"ms_per_pair\":%.9f,"
            "\"bitwise_mismatches_first\":0,"
            "\"bitwise_mismatches_second\":0,"
            "\"bitwise_mismatches_total\":0,"
            "\"exact\":true}\n",
            shape.baseline_algorithm,
            algorithm_name(shape.baseline_algorithm),
            static_cast<int>(timing.cublas_status),
            static_cast<int>(timing.cuda_status),
            timing.ms);
    std::fflush(stdout);
}

static void print_candidate(
        const options & shape,
        int algorithm,
        const timing_result & timing,
        const mismatch_result & mismatches,
        bool compared,
        double baseline_ms) {
    std::fputs(
            "{\"kind\":\"stacked_candidate\","
            "\"api\":\"one_cublasGemmEx\",",
            stdout);
    print_signature(shape);
    std::printf(
            ",\"algorithm\":%d,\"algorithm_name\":\"%s\","
            "\"gemm_calls_per_repeat\":1,"
            "\"cublas_status\":%d,\"cuda_status\":%d,"
            "\"ms_per_stacked_call\":%.9f,"
            "\"bitwise_mismatches_first\":",
            algorithm,
            algorithm_name(algorithm),
            static_cast<int>(timing.cublas_status),
            static_cast<int>(timing.cuda_status),
            timing.ms);
    if (compared) {
        std::printf("%llu", mismatches.first);
    } else {
        std::fputs("null", stdout);
    }
    std::fputs(",\"bitwise_mismatches_second\":", stdout);
    if (compared) {
        std::printf("%llu", mismatches.second);
    } else {
        std::fputs("null", stdout);
    }
    std::fputs(",\"bitwise_mismatches_total\":", stdout);
    if (compared) {
        std::printf("%llu", mismatches.first + mismatches.second);
    } else {
        std::fputs("null", stdout);
    }

    const bool exact = compared &&
            mismatches.first == 0 &&
            mismatches.second == 0;
    std::printf(",\"exact\":%s,\"speedup_vs_separate_pair\":",
            exact ? "true" : "false");
    if (timing.ms > 0.0 && baseline_ms > 0.0) {
        std::printf("%.9f", baseline_ms/timing.ms);
    } else {
        std::fputs("null", stdout);
    }
    std::fputs(",\"saved_ms_vs_separate_pair\":", stdout);
    if (timing.ms >= 0.0 && baseline_ms >= 0.0) {
        std::printf("%.9f", baseline_ms - timing.ms);
    } else {
        std::fputs("null", stdout);
    }
    std::fputs("}\n", stdout);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char ** argv) {
    options shape;
    if (!parse_options(argc, argv, &shape)) {
        usage(argv[0]);
        return 2;
    }

    const size_t base_m = static_cast<size_t>(shape.base_m);
    const size_t n = static_cast<size_t>(shape.n);
    const size_t k = static_cast<size_t>(shape.k);
    size_t a_region_elements = 0;
    size_t a_elements = 0;
    size_t b_elements = 0;
    size_t c_region_elements = 0;
    size_t c_elements = 0;
    if (!checked_product(base_m, k, sizeof(half), &a_region_elements) ||
            !checked_product(2, a_region_elements, sizeof(half), &a_elements) ||
            !checked_product(n, k, sizeof(half), &b_elements) ||
            !checked_product(base_m, n, sizeof(float), &c_region_elements) ||
            !checked_product(
                    2, c_region_elements, sizeof(float), &c_elements)) {
        std::fprintf(stderr, "matrix allocation size overflow\n");
        return 2;
    }

    CUDA_OK(cudaSetDevice(shape.device));
    cudaDeviceProp device_properties{};
    CUDA_OK(cudaGetDeviceProperties(&device_properties, shape.device));
    if (device_properties.major != 6 || device_properties.minor != 0) {
        std::fprintf(
                stderr,
                "P100 sm_60 required, found compute capability %d.%d\n",
                device_properties.major,
                device_properties.minor);
        return 2;
    }

    int cuda_runtime_version = 0;
    int cuda_driver_version = 0;
    CUDA_OK(cudaRuntimeGetVersion(&cuda_runtime_version));
    CUDA_OK(cudaDriverGetVersion(&cuda_driver_version));

    cudaStream_t stream = nullptr;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t handle = nullptr;
    CUBLAS_OK(cublasCreate(&handle));
    CUBLAS_OK(cublasSetStream(handle, stream));
    CUBLAS_OK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    int cublas_runtime_version = 0;
    CUBLAS_OK(cublasGetVersion(handle, &cublas_runtime_version));

    half * a = nullptr;
    half * b = nullptr;
    float * reference = nullptr;
    float * candidate = nullptr;
    unsigned long long * errors = nullptr;
    CUDA_OK(cudaMalloc(&a, a_elements*sizeof(*a)));
    CUDA_OK(cudaMalloc(&b, b_elements*sizeof(*b)));
    CUDA_OK(cudaMalloc(&reference, c_elements*sizeof(*reference)));
    CUDA_OK(cudaMalloc(&candidate, c_elements*sizeof(*candidate)));
    CUDA_OK(cudaMalloc(&errors, 2*sizeof(*errors)));

    char pci_bus_id[32] = {};
    CUDA_OK(cudaDeviceGetPCIBusId(
            pci_bus_id, sizeof(pci_bus_id), shape.device));
    std::fputs(
            "{\"kind\":\"metadata\","
            "\"probe\":\"p100-stacked-projection-exact\",",
            stdout);
    print_signature(shape);
    std::printf(",\"device\":%d,\"device_name\":", shape.device);
    print_json_string(device_properties.name);
    std::fputs(",\"pci_bus_id\":", stdout);
    print_json_string(pci_bus_id);
    std::printf(
            ",\"compute_capability\":\"%d.%d\","
            "\"multiprocessor_count\":%d,"
            "\"total_global_memory_bytes\":%llu,"
            "\"cuda_toolkit_compile\":%d,"
            "\"cuda_runtime\":%d,\"cuda_driver\":%d,"
            "\"cublas_compile\":%d,\"cublas_compile_build\":%d,"
            "\"cublas_runtime\":%d,"
            "\"math_mode\":\"CUBLAS_TF32_TENSOR_OP_MATH\","
            "\"stream\":\"nonblocking_single_stream\","
            "\"workspace_bytes\":0,"
            "\"timing\":\"cuda_event_batch_average\","
            "\"a_layout\":\"two_contiguous_KxM_column_major_regions\","
            "\"b_layout\":\"shared_KxN_column_major\","
            "\"reference_c_layout\":\"two_contiguous_MxN_regions\","
            "\"candidate_c_layout\":\"one_2MxN_column_major_region\","
            "\"a_region_offset_elements\":%llu,"
            "\"reference_region_offset_elements\":%llu,"
            "\"a_bytes\":%llu,\"b_bytes\":%llu,"
            "\"reference_bytes\":%llu,\"candidate_bytes\":%llu,"
            "\"a_alignment_mod_256\":%llu,"
            "\"b_alignment_mod_256\":%llu,"
            "\"reference_alignment_mod_256\":%llu,"
            "\"candidate_alignment_mod_256\":%llu,"
            "\"a_seed\":%u,\"b_seed\":%u,"
            "\"sweep\":[-1,\"0-23\",\"99-115\"]}\n",
            device_properties.major,
            device_properties.minor,
            device_properties.multiProcessorCount,
            static_cast<unsigned long long>(
                    device_properties.totalGlobalMem),
            CUDART_VERSION,
            cuda_runtime_version,
            cuda_driver_version,
            CUBLAS_VERSION,
            CUBLAS_VER_BUILD,
            cublas_runtime_version,
            static_cast<unsigned long long>(a_region_elements),
            static_cast<unsigned long long>(c_region_elements),
            static_cast<unsigned long long>(a_elements*sizeof(*a)),
            static_cast<unsigned long long>(b_elements*sizeof(*b)),
            static_cast<unsigned long long>(
                    c_elements*sizeof(*reference)),
            static_cast<unsigned long long>(
                    c_elements*sizeof(*candidate)),
            static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(a) & 255U),
            static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(b) & 255U),
            static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(reference) & 255U),
            static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(candidate) & 255U),
            kASeed,
            kBSeed);
    std::fflush(stdout);

    const int a_blocks = static_cast<int>(
            std::min<size_t>((a_elements + kThreads - 1)/kThreads, 4096));
    const int b_blocks = static_cast<int>(
            std::min<size_t>((b_elements + kThreads - 1)/kThreads, 4096));
    init_f16_cuda<<<a_blocks, kThreads, 0, stream>>>(
            a, a_elements, kASeed);
    init_f16_cuda<<<b_blocks, kThreads, 0, stream>>>(
            b, b_elements, kBSeed);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamSynchronize(stream));

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));

    const cublasGemmAlgo_t baseline_algorithm =
            static_cast<cublasGemmAlgo_t>(shape.baseline_algorithm);
    const timing_result baseline_timing = time_call(
            stream,
            begin,
            end,
            shape.warmup,
            shape.repeats,
            [&]() {
                return run_separate_pair(
                        handle,
                        a,
                        b,
                        reference,
                        a_region_elements,
                        c_region_elements,
                        shape,
                        baseline_algorithm);
            });
    if (!timing_completed(baseline_timing)) {
        std::fputs(
                "{\"kind\":\"baseline_pair_failure\","
                "\"api\":\"two_cublasGemmEx\",",
                stdout);
        print_signature(shape);
        std::printf(
                ",\"algorithm\":%d,\"algorithm_name\":\"%s\","
                "\"cublas_status\":%d,\"cuda_status\":%d,"
                "\"ms_per_pair\":%.9f}\n",
                shape.baseline_algorithm,
                algorithm_name(shape.baseline_algorithm),
                static_cast<int>(baseline_timing.cublas_status),
                static_cast<int>(baseline_timing.cuda_status),
                baseline_timing.ms);
        return 1;
    }
    print_baseline(shape, baseline_timing);

    std::vector<int> algorithms;
    algorithms.reserve(42);
    algorithms.push_back(static_cast<int>(CUBLAS_GEMM_DEFAULT));
    for (int algorithm = 0; algorithm <= 23; ++algorithm) {
        algorithms.push_back(algorithm);
    }
    for (int algorithm = 99; algorithm <= 115; ++algorithm) {
        algorithms.push_back(algorithm);
    }

    int supported_candidates = 0;
    int exact_candidates = 0;
    int fastest_exact_algorithm = 0;
    double fastest_exact_ms = std::numeric_limits<double>::infinity();
    for (const int algorithm_value : algorithms) {
        const cublasGemmAlgo_t algorithm =
                static_cast<cublasGemmAlgo_t>(algorithm_value);
        const timing_result timing = time_call(
                stream,
                begin,
                end,
                shape.warmup,
                shape.repeats,
                [&]() {
                    return run_projection(
                            handle,
                            a,
                            b,
                            candidate,
                            2*shape.base_m,
                            shape,
                            algorithm);
                });

        const bool completed = timing_completed(timing);
        mismatch_result mismatches;
        if (completed) {
            ++supported_candidates;
            mismatches = count_mismatches(
                    reference,
                    candidate,
                    c_region_elements,
                    shape.base_m,
                    errors,
                    stream);
            if (mismatches.first == 0 && mismatches.second == 0) {
                ++exact_candidates;
                if (timing.ms < fastest_exact_ms) {
                    fastest_exact_ms = timing.ms;
                    fastest_exact_algorithm = algorithm_value;
                }
            }
        }
        print_candidate(
                shape,
                algorithm_value,
                timing,
                mismatches,
                completed,
                baseline_timing.ms);
    }

    std::fputs("{\"kind\":\"summary\",", stdout);
    print_signature(shape);
    std::printf(
            ",\"algorithms_tested\":%llu,"
            "\"supported_candidates\":%d,"
            "\"exact_candidates\":%d,"
            "\"baseline_pair_ms\":%.9f,"
            "\"fastest_exact_algorithm\":",
            static_cast<unsigned long long>(algorithms.size()),
            supported_candidates,
            exact_candidates,
            baseline_timing.ms);
    if (exact_candidates != 0) {
        std::printf(
                "%d,\"fastest_exact_algorithm_name\":\"%s\","
                "\"fastest_exact_ms\":%.9f,"
                "\"fastest_exact_speedup\":%.9f,"
                "\"fastest_exact_saved_ms\":%.9f",
                fastest_exact_algorithm,
                algorithm_name(fastest_exact_algorithm),
                fastest_exact_ms,
                baseline_timing.ms/fastest_exact_ms,
                baseline_timing.ms - fastest_exact_ms);
    } else {
        std::fputs(
                "null,\"fastest_exact_algorithm_name\":null,"
                "\"fastest_exact_ms\":null,"
                "\"fastest_exact_speedup\":null,"
                "\"fastest_exact_saved_ms\":null",
                stdout);
    }
    std::fputs("}\n", stdout);
    std::fflush(stdout);

    CUDA_OK(cudaEventDestroy(end));
    CUDA_OK(cudaEventDestroy(begin));
    CUDA_OK(cudaFree(errors));
    CUDA_OK(cudaFree(candidate));
    CUDA_OK(cudaFree(reference));
    CUDA_OK(cudaFree(b));
    CUDA_OK(cudaFree(a));
    CUBLAS_OK(cublasDestroy(handle));
    CUDA_OK(cudaStreamDestroy(stream));
    return 0;
}
