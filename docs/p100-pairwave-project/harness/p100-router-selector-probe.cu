#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
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

#ifndef PROBE_M
#define PROBE_M 256
#endif

#ifndef PROBE_N
#define PROBE_N 2032
#endif

#ifndef PROBE_K
#define PROBE_K 2048
#endif

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kASeed = 0x13579bdfU;
constexpr uint32_t kBSeed = 0x2468ace0U;

struct options {
    int m = PROBE_M;
    int n = PROBE_N;
    int k = PROBE_K;
    int warmup = 5;
    int repeats = 30;
    int device = 0;
};

struct timing_result {
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;
    double ms = -1.0;
};

static void usage(const char * program) {
    std::fprintf(
            stderr,
            "usage: %s [--m M] [--n N] [--k K] [--warmup W] "
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
        if (std::strcmp(name, "--m") == 0) {
            destination = &result->m;
        } else if (std::strcmp(name, "--n") == 0) {
            destination = &result->n;
        } else if (std::strcmp(name, "--k") == 0) {
            destination = &result->k;
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
    return true;
}

static bool checked_elements(int rows, int columns, size_t * result) {
    const size_t left = static_cast<size_t>(rows);
    const size_t right = static_cast<size_t>(columns);
    constexpr size_t maximum = std::numeric_limits<size_t>::max();
    if (right != 0 && left > maximum/right) {
        return false;
    }
    *result = left*right;
    return *result <= maximum/sizeof(float);
}

__device__ __forceinline__ uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__global__ void init_f32_cuda(
        float * destination, size_t elements, uint32_t seed) {
    for (size_t index =
                    static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
            index < elements;
            index += static_cast<size_t>(gridDim.x)*blockDim.x) {
        const uint32_t folded =
                static_cast<uint32_t>(index) ^
                static_cast<uint32_t>(index >> 32);
        const uint32_t bits = mix32(folded ^ seed);
        const int value = static_cast<int>(bits & 0xffffU) - 32768;
        destination[index] =
                static_cast<float>(value)*(1.0f/32768.0f);
    }
}

__global__ void count_mismatches_cuda(
        const float * reference,
        const float * candidate,
        size_t elements,
        unsigned long long * errors) {
    unsigned long long local = 0;
    for (size_t index =
                    static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
            index < elements;
            index += static_cast<size_t>(gridDim.x)*blockDim.x) {
        local += __float_as_uint(reference[index]) !=
                __float_as_uint(candidate[index]);
    }
    if (local != 0) {
        atomicAdd(errors, local);
    }
}

static cublasStatus_t run_sgemm(
        cublasHandle_t handle,
        const float * a,
        const float * b,
        float * c,
        const options & shape) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            shape.m,
            shape.n,
            shape.k,
            &alpha,
            a,
            shape.k,
            b,
            shape.k,
            &beta,
            c,
            shape.m);
}

static cublasStatus_t run_gemm_ex(
        cublasHandle_t handle,
        const float * a,
        const float * b,
        float * c,
        const options & shape,
        cublasGemmAlgo_t algorithm) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            shape.m,
            shape.n,
            shape.k,
            &alpha,
            a,
            CUDA_R_32F,
            shape.k,
            b,
            CUDA_R_32F,
            shape.k,
            &beta,
            c,
            CUDA_R_32F,
            shape.m,
            CUBLAS_COMPUTE_32F,
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

static unsigned long long count_mismatches(
        const float * reference,
        const float * candidate,
        size_t elements,
        unsigned long long * errors,
        cudaStream_t stream) {
    const int blocks = static_cast<int>(
            std::min<size_t>((elements + kThreads - 1)/kThreads, 4096));
    CUDA_OK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
    count_mismatches_cuda<<<blocks, kThreads, 0, stream>>>(
            reference, candidate, elements, errors);
    CUDA_OK(cudaGetLastError());

    unsigned long long host_errors = 0;
    CUDA_OK(cudaMemcpyAsync(
            &host_errors,
            errors,
            sizeof(host_errors),
            cudaMemcpyDeviceToHost,
            stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return host_errors;
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

static void print_shape(const options & shape) {
    std::printf(
            "\"m\":%d,\"n\":%d,\"k\":%d,"
            "\"op_a\":\"T\",\"op_b\":\"N\","
            "\"lda\":%d,\"ldb\":%d,\"ldc\":%d,"
            "\"a_type\":\"CUDA_R_32F\","
            "\"b_type\":\"CUDA_R_32F\","
            "\"c_type\":\"CUDA_R_32F\","
            "\"compute_type\":\"CUBLAS_COMPUTE_32F\","
            "\"warmup\":%d,\"repeats\":%d",
            shape.m,
            shape.n,
            shape.k,
            shape.k,
            shape.k,
            shape.m,
            shape.warmup,
            shape.repeats);
}

static void print_measurement(
        const char * kind,
        const char * api,
        const options & shape,
        const timing_result & timing,
        int algorithm,
        bool has_algorithm,
        unsigned long long mismatches,
        bool has_mismatches) {
    std::printf("{\"kind\":\"%s\",\"api\":\"%s\",", kind, api);
    print_shape(shape);
    if (has_algorithm) {
        std::printf(
                ",\"algorithm\":%d,\"algorithm_name\":\"%s\"",
                algorithm,
                algorithm_name(algorithm));
    } else {
        std::fputs(",\"algorithm\":null,\"algorithm_name\":null", stdout);
    }
    std::printf(
            ",\"cublas_status\":%d,\"cuda_status\":%d,"
            "\"ms\":%.9f,\"bitwise_mismatches\":",
            static_cast<int>(timing.cublas_status),
            static_cast<int>(timing.cuda_status),
            timing.ms);
    if (has_mismatches) {
        std::printf("%llu", mismatches);
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

    size_t a_elements = 0;
    size_t b_elements = 0;
    size_t c_elements = 0;
    if (!checked_elements(shape.m, shape.k, &a_elements) ||
            !checked_elements(shape.n, shape.k, &b_elements) ||
            !checked_elements(shape.m, shape.n, &c_elements)) {
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

    std::fputs("{\"kind\":\"metadata\",\"probe\":\"p100-router-selector\",", stdout);
    print_shape(shape);
    std::printf(
            ",\"device\":%d,\"device_name\":",
            shape.device);
    print_json_string(device_properties.name);
    std::printf(
            ",\"compute_capability\":\"%d.%d\","
            "\"cuda_toolkit_compile\":%d,"
            "\"cuda_runtime\":%d,\"cuda_driver\":%d,"
            "\"cublas_compile\":%d,\"cublas_compile_build\":%d,"
            "\"cublas_runtime\":%d,"
            "\"math_mode\":\"CUBLAS_TF32_TENSOR_OP_MATH\","
            "\"reference\":\"cublasSgemm\","
            "\"a_seed\":%u,\"b_seed\":%u,"
            "\"sweep\":[-1,\"0-23\",\"99-115\"]}\n",
            device_properties.major,
            device_properties.minor,
            CUDART_VERSION,
            cuda_runtime_version,
            cuda_driver_version,
            CUBLAS_VERSION,
            CUBLAS_VER_BUILD,
            cublas_runtime_version,
            kASeed,
            kBSeed);
    std::fflush(stdout);

    float * a = nullptr;
    float * b = nullptr;
    float * reference = nullptr;
    float * candidate = nullptr;
    unsigned long long * errors = nullptr;
    CUDA_OK(cudaMalloc(&a, a_elements*sizeof(*a)));
    CUDA_OK(cudaMalloc(&b, b_elements*sizeof(*b)));
    CUDA_OK(cudaMalloc(&reference, c_elements*sizeof(*reference)));
    CUDA_OK(cudaMalloc(&candidate, c_elements*sizeof(*candidate)));
    CUDA_OK(cudaMalloc(&errors, sizeof(*errors)));

    const int a_blocks = static_cast<int>(
            std::min<size_t>((a_elements + kThreads - 1)/kThreads, 4096));
    const int b_blocks = static_cast<int>(
            std::min<size_t>((b_elements + kThreads - 1)/kThreads, 4096));
    init_f32_cuda<<<a_blocks, kThreads, 0, stream>>>(
            a, a_elements, kASeed);
    init_f32_cuda<<<b_blocks, kThreads, 0, stream>>>(
            b, b_elements, kBSeed);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamSynchronize(stream));

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));

    const timing_result reference_timing = time_call(
            stream,
            begin,
            end,
            shape.warmup,
            shape.repeats,
            [&]() {
                return run_sgemm(handle, a, b, reference, shape);
            });
    if (reference_timing.cublas_status != CUBLAS_STATUS_SUCCESS ||
            reference_timing.cuda_status != cudaSuccess) {
        print_measurement(
                "reference",
                "cublasSgemm",
                shape,
                reference_timing,
                0,
                false,
                0,
                false);
        return 1;
    }
    print_measurement(
            "reference",
            "cublasSgemm",
            shape,
            reference_timing,
            0,
            false,
            0,
            true);

    std::vector<int> algorithms;
    algorithms.reserve(42);
    algorithms.push_back(static_cast<int>(CUBLAS_GEMM_DEFAULT));
    for (int algorithm = 0; algorithm <= 23; ++algorithm) {
        algorithms.push_back(algorithm);
    }
    for (int algorithm = 99; algorithm <= 115; ++algorithm) {
        algorithms.push_back(algorithm);
    }

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
                    return run_gemm_ex(
                            handle,
                            a,
                            b,
                            candidate,
                            shape,
                            algorithm);
                });

        const bool completed =
                timing.cublas_status == CUBLAS_STATUS_SUCCESS &&
                timing.cuda_status == cudaSuccess;
        unsigned long long mismatches = 0;
        if (completed) {
            mismatches = count_mismatches(
                    reference,
                    candidate,
                    c_elements,
                    errors,
                    stream);
        }
        print_measurement(
                "candidate",
                "cublasGemmEx",
                shape,
                timing,
                algorithm_value,
                true,
                mismatches,
                completed);
    }

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
