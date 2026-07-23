#include "affinity-wave.cuh"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using aw_bench_fn = int (*)(
        const ggml_cuda_aw_bench_params *, ggml_cuda_aw_bench_result *, char *, size_t);

static int parse_positive(const char * text, const char * name) {
    char * end = nullptr;
    const long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1 || value > 1000000) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return (int) value;
}

int main(int argc, char ** argv) {
    int tokens_per_cell = 2048;
    int repeats = 12;
    int route_pattern = 0;
    int devices = 4;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--tokens-per-cell") == 0 && i + 1 < argc) {
            tokens_per_cell = parse_positive(argv[++i], "tokens-per-cell");
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = parse_positive(argv[++i], "repeats");
        } else if (strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            devices = parse_positive(argv[++i], "devices");
            if (devices > 4) {
                fprintf(stderr, "devices must be between 1 and 4\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--route-pattern") == 0 && i + 1 < argc) {
            const char * pattern = argv[++i];
            if (strcmp(pattern, "uniform") == 0) {
                route_pattern = 0;
            } else if (strcmp(pattern, "tail-mix") == 0) {
                route_pattern = 1;
            } else if (strcmp(pattern, "edge-mix") == 0) {
                route_pattern = 2;
            } else {
                fprintf(stderr, "invalid route pattern: %s\n", pattern);
                return 2;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: %s [--devices N] [--tokens-per-cell N] [--repeats N] "
                   "[--route-pattern uniform|tail-mix|edge-mix]\n", argv[0]);
            printf("requires GGML_CUDA_AFFINITY_WAVE=1 and GGML_CUDA_AW_MAP=<manifest>\n");
            return 0;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            return 2;
        }
    }

    ggml_backend_reg_t reg = ggml_backend_cuda_reg();
    auto bench = reinterpret_cast<aw_bench_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_affinity_wave_service_bench"));
    if (bench == nullptr) {
        fprintf(stderr, "AffinityWave service proc is unavailable\n");
        return 1;
    }

    constexpr int max_devices = 4;
    std::array<ggml_cuda_aw_bench_result, max_devices> results{};
    std::array<std::array<char, 512>, max_devices> errors{};
    std::array<int, max_devices> status{};
    std::array<std::thread, max_devices> workers;
    for (int device = 0; device < devices; ++device) {
        workers[device] = std::thread([&, device]() {
            const ggml_cuda_aw_bench_params params = {
                device, devices, 4, tokens_per_cell, repeats, route_pattern
            };
            status[device] = bench(&params, &results[device], errors[device].data(), errors[device].size());
        });
    }
    for (int device = 0; device < devices; ++device) {
        workers[device].join();
    }

    double min_tflops = 1e30;
    bool passed = true;
    for (int device = 0; device < devices; ++device) {
        if (status[device] != 0) {
            fprintf(stderr, "GPU%d: %s\n", device, errors[device].data());
            passed = false;
            continue;
        }
        const auto & r = results[device];
        min_tflops = std::min(min_tflops, r.effective_tflops);
        printf("{\"device\":%d,\"elapsed_ms\":%.6f,\"service_ms\":%.6f,"
               "\"instrumented_ms\":%.6f,\"stage_sum_ms\":%.6f,"
               "\"effective_tflops\":%.6f,\"issued_tflops\":%.6f,\"useful_fraction\":%.6f,"
               "\"request_gib_s\":%.6f,\"pack_ms\":%.6f,\"gate_ms\":%.6f,\"up_ms\":%.6f,"
               "\"swiglu_ms\":%.6f,\"down_ms\":%.6f,\"reduce_ms\":%.6f,"
               "\"layout_ms\":%.6f,\"down_expand_ms\":%.6f,"
               "\"device_bytes\":%llu,\"free_bytes_after\":%llu,"
               "\"descriptors\":%d,\"route_rows\":%d,\"tiles_m64\":%d,\"tiles_m32\":%d,"
               "\"tiles_m16\":%d,\"q8_kernel\":\"%s\","
               "\"check_ran\":%d,\"check_passed\":%d,\"check_count\":%llu,"
               "\"check_mismatches\":%llu,\"check_first_mismatch\":%llu,"
               "\"check_expected\":%u,\"check_observed\":%u}\n",
                device, r.elapsed_ms, r.service_ms, r.instrumented_ms, r.stage_sum_ms,
                r.effective_tflops, r.issued_tflops, r.useful_fraction, r.request_gib_s,
                r.pack_ms, r.gate_ms, r.up_ms, r.swiglu_ms, r.down_ms, r.reduce_ms,
                r.layout_ms, r.down_expand_ms,
                (unsigned long long) r.device_bytes, (unsigned long long) r.free_bytes_after,
                r.descriptors, r.route_rows, r.tiles_m64, r.tiles_m32, r.tiles_m16,
                r.q8_kernel == 1 ? "interleave" : "cuda",
                r.check_ran, r.check_passed,
                (unsigned long long) r.check_count, (unsigned long long) r.check_mismatches,
                (unsigned long long) r.check_first_mismatch,
                (unsigned) r.check_expected, (unsigned) r.check_observed);
        if (r.effective_tflops < 5.5 || (r.check_ran && !r.check_passed)) {
            passed = false;
        }
    }
    if (min_tflops < 1e29) {
        printf("{\"summary\":\"AffinityWave Phase 1\",\"min_effective_tflops\":%.6f,"
               "\"gate_tflops\":5.5,\"passed\":%s}\n", min_tflops, passed ? "true" : "false");
    }
    return passed ? 0 : 1;
}
