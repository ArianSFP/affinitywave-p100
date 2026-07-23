// [TAG_AFFINITY_WAVE] Phase-1 cross-layer expert service for PCIe Pascal.
//
// This file deliberately stops at the isolated service boundary.  It proves
// the native-Q8 compute and owner-wire dataflow before the much more invasive
// chunk/layer scheduler is allowed to replace the production EP4 graph.

#include "affinity-wave.cuh"
#include "common.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr int AW_GPU_COUNT             = 4;
constexpr int AW_LAYERS                = 40;
constexpr int AW_EXPERTS               = 256;
constexpr int AW_HOT_EXPERTS           = 16;
constexpr int AW_PRIMARY_PER_GPU       = 64;
constexpr int AW_ROUTES_PER_OWNER_TOKEN = 2; // top-8 distributed over 4 owners
constexpr int AW_KSTAGE                = 32;
constexpr int AW_Q8_BLOCK_BYTES        = 34;
constexpr int AW_T64_ROWS              = 64;
constexpr int AW_T64_STAGE_BYTES       = AW_T64_ROWS*2 + AW_T64_ROWS*AW_KSTAGE;
constexpr int AW_EMBD                  = 2048;
constexpr int AW_EXPERT_FF             = 512;

struct aw_config {
    std::string map;
    std::string wire;
    std::string q8_layout;
    std::string home;
    std::string gguf_sha256;
    int down_cache = 0;
    int m64_split = 2;
    bool check = false;
};

class aw_process_barrier {
public:
    bool arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (broken_) {
            return false;
        }
        const int generation = generation_;
        if (++arrived_ == AW_GPU_COUNT) {
            arrived_ = 0;
            ++generation_;
            condition_.notify_all();
            return true;
        }
        condition_.wait(lock, [&]() { return broken_ || generation_ != generation; });
        return !broken_;
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        broken_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    int arrived_ = 0;
    int generation_ = 0;
    bool broken_ = false;
};

static aw_process_barrier aw_bench_barrier;
static std::mutex aw_layout_mutex;
static std::unordered_set<const void *> aw_t64_tensors;
static std::array<std::atomic<int>, AW_GPU_COUNT> aw_t64_tensor_counts{};

struct aw_layout_entry {
    void * data;
    size_t bytes;
    int n;
    int k;
    int device;
    int layer;
    int projection;
};

static std::vector<aw_layout_entry> aw_layout_entries;

enum aw_projection {
    AW_PROJECTION_GATE        = 0,
    AW_PROJECTION_UP          = 1,
    AW_PROJECTION_DOWN        = 2,
    AW_PROJECTION_SHARED_GATE = 3,
    AW_PROJECTION_SHARED_UP   = 4,
    AW_PROJECTION_SHARED_DOWN = 5,
};

static bool aw_env_on(const char * value) {
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool aw_find_value(const std::string & text, const char * key, size_t & value_pos, std::string & error) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        error = std::string("manifest is missing key '") + key + "'";
        return false;
    }
    const size_t colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        error = std::string("manifest key '") + key + "' has no value";
        return false;
    }
    value_pos = text.find_first_not_of(" \t\r\n", colon + 1);
    if (value_pos == std::string::npos) {
        error = std::string("manifest key '") + key + "' has an empty value";
        return false;
    }
    return true;
}

static bool aw_parse_int_at(const std::string & text, size_t pos, int & value, std::string & error) {
    errno = 0;
    char * end = nullptr;
    const long parsed = strtol(text.c_str() + pos, &end, 10);
    if (errno != 0 || end == text.c_str() + pos || parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
        error = "manifest contains an invalid integer";
        return false;
    }
    value = (int) parsed;
    return true;
}

static bool aw_get_int(const std::string & text, const char * key, int & value, std::string & error) {
    size_t pos;
    return aw_find_value(text, key, pos, error) && aw_parse_int_at(text, pos, value, error);
}

static bool aw_get_string(const std::string & text, const char * key, std::string & value, std::string & error) {
    size_t pos;
    if (!aw_find_value(text, key, pos, error) || text[pos] != '"') {
        error = std::string("manifest key '") + key + "' is not a string";
        return false;
    }
    const size_t end = text.find('"', pos + 1);
    if (end == std::string::npos) {
        error = std::string("manifest key '") + key + "' has an unterminated string";
        return false;
    }
    value.assign(text, pos + 1, end - pos - 1);
    return true;
}

static bool aw_match_delimited(
        const std::string & text, size_t begin, char open, char close, size_t & end, std::string & error) {
    if (begin >= text.size() || text[begin] != open) {
        error = "manifest has an invalid container";
        return false;
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == open) {
            ++depth;
        } else if (c == close && --depth == 0) {
            end = i;
            return true;
        }
    }
    error = "manifest has an unterminated container";
    return false;
}

static bool aw_get_int_array(
        const std::string & object, const char * key, std::vector<int> & values, std::string & error) {
    size_t begin;
    if (!aw_find_value(object, key, begin, error) || object[begin] != '[') {
        error = std::string("manifest key '") + key + "' is not an array";
        return false;
    }
    size_t end;
    if (!aw_match_delimited(object, begin, '[', ']', end, error)) {
        return false;
    }
    values.clear();
    size_t pos = begin + 1;
    while (pos < end) {
        pos = object.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= end) {
            break;
        }
        int value;
        if (!aw_parse_int_at(object, pos, value, error)) {
            return false;
        }
        values.push_back(value);
        pos = object.find_first_of(",]", pos);
        if (pos == std::string::npos) {
            error = std::string("manifest array '") + key + "' is unterminated";
            return false;
        }
    }
    return true;
}

static bool aw_validate_manifest(const std::string & path, aw_config & config, std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open placement manifest '" + path + "'";
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string text = buffer.str();
    if (text.empty() || text.size() > 8u*1024u*1024u) {
        error = "placement manifest is empty or unexpectedly large";
        return false;
    }

    int layers, experts, top_k, gpu_count, primary_per_gpu;
    std::string schema, architecture, wire_default;
    if (!aw_get_int(text, "layers", layers, error) ||
            !aw_get_int(text, "experts_per_layer", experts, error) ||
            !aw_get_int(text, "experts_per_token", top_k, error) ||
            !aw_get_int(text, "gpu_count", gpu_count, error) ||
            !aw_get_int(text, "primary_experts_per_gpu", primary_per_gpu, error) ||
            !aw_get_string(text, "schema", schema, error) ||
            !aw_get_string(text, "architecture", architecture, error) ||
            !aw_get_string(text, "gguf_sha256", config.gguf_sha256, error) ||
            !aw_get_string(text, "wire_default", wire_default, error)) {
        return false;
    }
    if (schema != "ggml.cuda.affinity_wave.placement.v1" || architecture != "qwen35moe" ||
            layers != AW_LAYERS || experts != AW_EXPERTS || top_k != 8 ||
            gpu_count != AW_GPU_COUNT || primary_per_gpu != AW_PRIMARY_PER_GPU || wire_default != "f32") {
        error = "placement manifest metadata does not describe Qwen3.6-35B-A3B hot-16 on four GPUs";
        return false;
    }
    if (config.gguf_sha256.size() != 64 || !std::all_of(config.gguf_sha256.begin(), config.gguf_sha256.end(),
            [](unsigned char c) { return std::isxdigit(c) != 0; })) {
        error = "placement manifest gguf_sha256 is not a 64-digit hexadecimal digest";
        return false;
    }

    size_t placement_begin;
    if (!aw_find_value(text, "placement", placement_begin, error) || text[placement_begin] != '[') {
        error = "manifest placement is not an array";
        return false;
    }
    size_t placement_end;
    if (!aw_match_delimited(text, placement_begin, '[', ']', placement_end, error)) {
        return false;
    }

    std::array<bool, AW_LAYERS> seen_layers{};
    int object_count = 0;
    size_t pos = placement_begin + 1;
    while (pos < placement_end) {
        pos = text.find('{', pos);
        if (pos == std::string::npos || pos >= placement_end) {
            break;
        }
        size_t object_end;
        if (!aw_match_delimited(text, pos, '{', '}', object_end, error) || object_end > placement_end) {
            return false;
        }
        const std::string object = text.substr(pos, object_end - pos + 1);
        int layer;
        std::vector<int> owners, hot;
        if (!aw_get_int(object, "layer", layer, error) ||
                !aw_get_int_array(object, "primary_owner", owners, error) ||
                !aw_get_int_array(object, "replicated_experts", hot, error)) {
            return false;
        }
        if (layer < 0 || layer >= AW_LAYERS || seen_layers[layer]) {
            error = "placement manifest has a duplicate or out-of-range layer";
            return false;
        }
        seen_layers[layer] = true;
        if ((int) owners.size() != AW_EXPERTS || (int) hot.size() != AW_HOT_EXPERTS) {
            error = "placement manifest has an incorrect owner or hot-expert count";
            return false;
        }
        std::array<int, AW_GPU_COUNT> owner_counts{};
        for (int owner : owners) {
            if (owner < 0 || owner >= AW_GPU_COUNT) {
                error = "placement manifest contains an out-of-range primary owner";
                return false;
            }
            ++owner_counts[owner];
        }
        if (!std::all_of(owner_counts.begin(), owner_counts.end(),
                [](int count) { return count == AW_PRIMARY_PER_GPU; })) {
            error = "placement manifest does not assign exactly 64 primary experts per GPU";
            return false;
        }
        std::set<int> hot_unique;
        for (int expert : hot) {
            if (expert < 0 || expert >= AW_EXPERTS || !hot_unique.insert(expert).second) {
                error = "placement manifest contains an invalid replicated expert";
                return false;
            }
        }
        ++object_count;
        pos = object_end + 1;
    }
    if (object_count != AW_LAYERS || !std::all_of(seen_layers.begin(), seen_layers.end(), [](bool v) { return v; })) {
        error = "placement manifest does not contain exactly 40 distinct layers";
        return false;
    }
    return true;
}

static bool aw_load_config(aw_config & config, std::string & error) {
    const char * enabled = getenv("GGML_CUDA_AFFINITY_WAVE");
    if (!aw_env_on(enabled)) {
        error = "GGML_CUDA_AFFINITY_WAVE is not enabled";
        return false;
    }
    const char * map = getenv("GGML_CUDA_AW_MAP");
    if (map == nullptr || map[0] == '\0') {
        error = "GGML_CUDA_AW_MAP is required when AffinityWave is enabled";
        return false;
    }
    config.map = map;
    config.wire = getenv("GGML_CUDA_AW_WIRE") != nullptr ? getenv("GGML_CUDA_AW_WIRE") : "f32";
    config.q8_layout = getenv("GGML_CUDA_AW_Q8_LAYOUT") != nullptr ?
            getenv("GGML_CUDA_AW_Q8_LAYOUT") : "native";
    config.home = getenv("GGML_CUDA_AW_HOME") != nullptr ? getenv("GGML_CUDA_AW_HOME") : "chunk";
    config.check = aw_env_on(getenv("GGML_CUDA_AW_CHECK"));
    if (config.wire != "f32" && config.wire != "bf16") {
        error = "GGML_CUDA_AW_WIRE must be 'f32' or 'bf16'";
        return false;
    }
    if (config.q8_layout != "native" && config.q8_layout != "t64k32") {
        error = "GGML_CUDA_AW_Q8_LAYOUT must be 'native' or 't64k32'";
        return false;
    }
    if (config.home != "chunk" && config.home != "layer") {
        error = "GGML_CUDA_AW_HOME must be 'chunk' or 'layer'";
        return false;
    }
    const char * down_cache = getenv("GGML_CUDA_AW_DOWN_CACHE");
    if (down_cache != nullptr && down_cache[0] != '\0') {
        char * end = nullptr;
        const long value = strtol(down_cache, &end, 10);
        if (*end != '\0' || (value != 0 && value != 4)) {
            error = "GGML_CUDA_AW_DOWN_CACHE must be 0 or 4";
            return false;
        }
        config.down_cache = (int) value;
    }
    if (config.down_cache == 4 && config.q8_layout != "t64k32") {
        error = "GGML_CUDA_AW_DOWN_CACHE=4 requires GGML_CUDA_AW_Q8_LAYOUT=t64k32";
        return false;
    }
    const char * m64_split = getenv("GGML_CUDA_AW_M64_SPLIT");
    if (m64_split != nullptr && m64_split[0] != '\0') {
        char * end = nullptr;
        const long value = strtol(m64_split, &end, 10);
        if (*end != '\0' || (value != 1 && value != 2)) {
            error = "GGML_CUDA_AW_M64_SPLIT must be 1 or 2";
            return false;
        }
        config.m64_split = (int) value;
    }
    if (!aw_validate_manifest(config.map, config, error)) {
        return false;
    }

    int ndev = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&ndev);
    if (count_error != cudaSuccess) {
        error = std::string("cannot enumerate CUDA devices: ") + cudaGetErrorString(count_error);
        return false;
    }
    if (ndev != AW_GPU_COUNT) {
        error = "AffinityWave requires exactly four visible CUDA devices";
        return false;
    }
    for (int device = 0; device < AW_GPU_COUNT; ++device) {
        cudaDeviceProp prop;
        const cudaError_t prop_error = cudaGetDeviceProperties(&prop, device);
        if (prop_error != cudaSuccess) {
            error = std::string("cannot query CUDA device: ") + cudaGetErrorString(prop_error);
            return false;
        }
        if (prop.major != 6 || prop.minor != 0) {
            error = "AffinityWave Phase 1 is restricted to four compute-capability 6.0 GPUs";
            return false;
        }
    }
    return true;
}

struct aw_work_desc {
    const void    * input;
    const char    * weight;
    float         * output;
    const int32_t * input_rows;
    int32_t         row_offset;
    int32_t         rows;
    int32_t         layer;
    int32_t         expert;
};

struct aw_tile_desc {
    int32_t work;
    int32_t row;
    int32_t rows;
};

union aw_smem {
    struct {
        float4 A[2][AW_KSTAGE][8];
        float4 B[2][AW_KSTAGE][16];
    } stage;
    float red[8][32*8];
};

__device__ __forceinline__ float aw_bf16_to_float(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

__device__ __forceinline__ uint16_t aw_float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

template<bool BF16_INPUT>
__device__ __forceinline__ float aw_load_input(const void * input, size_t index) {
    if constexpr (BF16_INPUT) {
        return aw_bf16_to_float(((const uint16_t *) input)[index]);
    } else {
        return ((const float *) input)[index];
    }
}

__device__ __forceinline__ void aw_load_bf16x8(
        const void * input, size_t index, float4 & lo, float4 & hi) {
    const uint4 packed = *(const uint4 *) ((const uint16_t *) input + index);
    const uint32_t words[4] = { packed.x, packed.y, packed.z, packed.w };
    float * lo_values = (float *) &lo;
    float * hi_values = (float *) &hi;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        lo_values[2*i] = aw_bf16_to_float((uint16_t) words[i]);
        lo_values[2*i + 1] = aw_bf16_to_float((uint16_t) (words[i] >> 16));
        hi_values[2*i] = aw_bf16_to_float((uint16_t) words[i + 2]);
        hi_values[2*i + 1] = aw_bf16_to_float((uint16_t) (words[i + 2] >> 16));
    }
}

// Cross-layer grouped service.  Every work descriptor owns its
// input/weight/output pointers and (layer, expert, row-offset) identity; a
// projection launch therefore pools all active diagonal cells without making
// tensor addresses layer-global.
template<bool BF16_INPUT, bool T64_LAYOUT>
__global__ __launch_bounds__(128, 2)
static void aw_q8_service_m32(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem sm;

    const int tid  = threadIdx.x;
    const int kgrp = tid >> 5;
    const int lane = tid & 31;
    const int mt   = lane >> 3;
    const int nt   = lane & 7;
    const int a_r  = tid >> 2;
    const int a_k  = (tid & 3)*8;
    const int b_r0 = tid >> 2;
    const int b_r1 = 32 + (tid >> 2);
    const int xk   = tid & 3;
    const int b_k  = xk*8;

    const int aslot  = (((a_r >> 2) & 1)*4 + (a_r >> 3)) ^ xk;
    const int aoff   = aslot*4 + (a_r & 3);
    const int bslot0 = ((((b_r0 >> 2) & 1)*8 + (b_r0 >> 3)) ^ xk);
    const int boff0  = bslot0*4 + (b_r0 & 3);
    const int bslot1 = ((((b_r1 >> 2) & 1)*8 + (b_r1 >> 3)) ^ xk);
    const int boff1  = bslot1*4 + (b_r1 & 3);
    const int sa0 = (0 + mt) ^ kgrp;
    const int sa1 = (4 + mt) ^ kgrp;
    const int sb0 = (0 + nt) ^ kgrp;
    const int sb1 = (8 + nt) ^ kgrp;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/64;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*64;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const char * gq0 = nullptr;
        const char * gq1 = nullptr;
        if constexpr (!T64_LAYOUT) {
            gq0 = desc.weight + (size_t) (col0 + b_r0)*(k/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
            gq1 = desc.weight + (size_t) (col0 + b_r1)*(k/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
        }

        float4 pa0, pa1;
        unsigned short pq0[4], pq1[4];
        float pd0, pd1;
        auto fetcha = [&](int stage) {
            const size_t base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, pa0, pa1);
            } else {
                // Match the proven [TAG_MOE_PLAN] path: two aligned LDG.128s
                // instead of eight scalar loads through the generic descriptor.
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            const char * p0;
            const char * p1;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks = k/AW_KSTAGE;
                const char * tile_stage = desc.weight +
                        ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
                p0 = tile_stage + AW_T64_ROWS*2 + b_r0*AW_KSTAGE - 2;
                p1 = tile_stage + AW_T64_ROWS*2 + b_r1*AW_KSTAGE - 2;
                pd0 = __half2float(*(const half *) (tile_stage + b_r0*2));
                pd1 = __half2float(*(const half *) (tile_stage + b_r1*2));
            } else {
                p0 = gq0 + (size_t) stage*AW_Q8_BLOCK_BYTES;
                p1 = gq1 + (size_t) stage*AW_Q8_BLOCK_BYTES;
                pd0 = __half2float(*(const half *) p0);
                pd1 = __half2float(*(const half *) p1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq0[i] = *(const unsigned short *) (p0 + 2 + b_k + 2*i);
                pq1[i] = *(const unsigned short *) (p1 + 2 + b_k + 2*i);
            }
        };
        auto stage_data = [&](int buffer) {
            float * fA = (float *) sm.stage.A[buffer];
            float * fB = (float *) sm.stage.B[buffer];
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                fA[(a_k + i    )*32 + aoff] = a0[i];
                fA[(a_k + i + 4)*32 + aoff] = a1[i];
                fB[(b_k + 2*i    )*64 + boff0] = (float) (signed char) (pq0[i] & 0xff)*pd0;
                fB[(b_k + 2*i + 1)*64 + boff0] = (float) (signed char) (pq0[i] >> 8  )*pd0;
                fB[(b_k + 2*i    )*64 + boff1] = (float) (signed char) (pq1[i] & 0xff)*pd1;
                fB[(b_k + 2*i + 1)*64 + boff1] = (float) (signed char) (pq1[i] >> 8  )*pd1;
            }
        };

        float acc[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[i][j] = 0.0f;
            }
        }

        fetcha(0);
        fetchb(0);
        stage_data(0);
        __syncthreads();
        int buffer = 0;
        for (int kb = 0; kb < k; kb += AW_KSTAGE) {
            const bool last = kb + AW_KSTAGE >= k;
            if (!last) {
                const int next = (kb + AW_KSTAGE)/AW_KSTAGE;
                fetcha(next);
                fetchb(next);
            }
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int ks = kgrp*8 + kk;
                const float4 va0 = sm.stage.A[buffer][ks][sa0];
                const float4 va1 = sm.stage.A[buffer][ks][sa1];
                const float4 vb0 = sm.stage.B[buffer][ks][sb0];
                const float4 vb1 = sm.stage.B[buffer][ks][sb1];
                const float av[8] = { va0.x, va0.y, va0.z, va0.w, va1.x, va1.y, va1.z, va1.w };
                const float bv[8] = { vb0.x, vb0.y, vb0.z, vb0.w, vb1.x, vb1.y, vb1.z, vb1.w };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] += av[i]*bv[j];
                    }
                }
            }
            if (!last) {
                stage_data(buffer ^ 1);
            }
            __syncthreads();
            buffer ^= 1;
        }

        for (int round = 0; round < 8; ++round) {
            if (kgrp != 0) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sm.red[j][(kgrp - 1)*32 + lane] = acc[round][j];
                }
            }
            __syncthreads();
            if (kgrp == 0) {
                const int row = mt*8 + round;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float value = acc[round][j] + sm.red[j][lane] +
                                sm.red[j][32 + lane] + sm.red[j][64 + lane];
                        desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] = value;
                    }
                }
            }
            __syncthreads();
        }
    }
}

union aw_smem_m64 {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[AW_T64_ROWS][AW_T64_ROWS];
};

template<bool BF16_INPUT, bool T64_LAYOUT, bool F32_WEIGHT = false, int SPLIT_K = 2>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kgrp = warp >> 2;
    const int wm = warp & 3;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 2;
    const int b_k = (tid & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float4 pa0, pa1;
        float4 pb0, pb1;
        unsigned short pq[4];
        float pd;
        auto fetcha = [&](int stage) {
            const size_t base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            if constexpr (F32_WEIGHT) {
                const float4 * weight4 = (const float4 *) ((const float *) desc.weight +
                        (size_t) (col0 + b_r)*k + stage*AW_KSTAGE + b_k);
                pb0 = weight4[0];
                pb1 = weight4[1];
                return;
            }
            const char * p;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks = k/AW_KSTAGE;
                const char * tile_stage = desc.weight +
                        ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
                pd = __half2float(*(const half *) (tile_stage + b_r*2));
                p = tile_stage + AW_T64_ROWS*2 + b_r*AW_KSTAGE;
            } else {
                p = desc.weight + ((size_t) (col0 + b_r)*(k/AW_KSTAGE) + stage)*AW_Q8_BLOCK_BYTES;
                pd = __half2float(*(const half *) p);
                p += 2;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq[i] = *(const unsigned short *) (p + b_k + 2*i);
            }
        };
        auto stage_data = [&](int buffer) {
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            const float * b0 = (const float *) &pb0;
            const float * b1 = (const float *) &pb1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[buffer][a_k + i][a_r] = a0[i];
                sm.stage.A[buffer][a_k + i + 4][a_r] = a1[i];
                if constexpr (F32_WEIGHT) {
                    sm.stage.B[buffer][b_k + i][b_r] = b0[i];
                    sm.stage.B[buffer][b_k + i + 4][b_r] = b1[i];
                } else {
                    sm.stage.B[buffer][b_k + 2*i][b_r] = (float) (signed char) (pq[i] & 0xff)*pd;
                    sm.stage.B[buffer][b_k + 2*i + 1][b_r] = (float) (signed char) (pq[i] >> 8)*pd;
                }
            }
        };

        float acc[4][8] = {};
        fetcha(0);
        fetchb(0);
        stage_data(0);
        __syncthreads();
        int buffer = 0;
        for (int kb = 0; kb < k; kb += AW_KSTAGE) {
            const bool last = kb + AW_KSTAGE >= k;
            if (!last) {
                const int next = (kb + AW_KSTAGE)/AW_KSTAGE;
                fetcha(next);
                fetchb(next);
            }
            if constexpr (SPLIT_K == 1) {
                #pragma unroll
                for (int kk = 0; kk < AW_KSTAGE; ++kk) {
                    const int row0 = warp*8 + rm*2;
                    const float2 av = *(const float2 *) &sm.stage.A[buffer][kk][row0];
                    const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][kk][nt*8];
                    const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][kk][nt*8 + 4];
                    const float a[2] = { av.x, av.y };
                    const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[i][j] += a[i]*b[j];
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                    const int ks = kgrp*(AW_KSTAGE/2) + kk;
                    const int row0 = wm*16 + rm*4;
                    const float4 av = *(const float4 *) &sm.stage.A[buffer][ks][row0];
                    const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][ks][nt*8];
                    const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][ks][nt*8 + 4];
                    const float a[4] = { av.x, av.y, av.z, av.w };
                    const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[i][j] += a[i]*b[j];
                        }
                    }
                }
            }
            if (!last) {
                stage_data(buffer ^ 1);
            }
            __syncthreads();
            buffer ^= 1;
        }

        if constexpr (SPLIT_K == 1) {
            const int output_row0 = warp*8 + rm*2;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = output_row0 + i;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] = acc[i][j];
                    }
                }
            }
        } else {
            const int output_row0 = wm*16 + rm*4;
            if (kgrp == 1) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm.red[output_row0 + i][nt*8 + j] = acc[i][j];
                    }
                }
            }
            __syncthreads();
            if (kgrp == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = output_row0 + i;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] =
                                    acc[i][j] + sm.red[row][nt*8 + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

union aw_smem_m64_gate_up {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[2][2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[2][AW_T64_ROWS][AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(512, 1)
static void aw_q8_service_m64_gate_up(
        const aw_work_desc * __restrict__ gate_descs,
        const aw_work_desc * __restrict__ up_descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_gate_up sm;

    const int projection = threadIdx.x >> 8;
    const int tid = threadIdx.x & 255;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kgrp = warp >> 2;
    const int wm = warp & 3;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 2;
    const int b_k = (tid & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = projection == 0 ? gate_descs[tile.work] : up_descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float4 pa0, pa1;
        unsigned short pq[4];
        float pd;
        auto fetcha = [&](int stage) {
            if (projection != 0) {
                return;
            }
            const size_t base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            const int n_kblocks = k/AW_KSTAGE;
            const char * tile_stage = desc.weight +
                    ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
            pd = __half2float(*(const half *) (tile_stage + b_r*2));
            const char * p = tile_stage + AW_T64_ROWS*2 + b_r*AW_KSTAGE;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq[i] = *(const unsigned short *) (p + b_k + 2*i);
            }
        };
        auto stage_data = [&](int buffer) {
            if (projection == 0) {
                const float * a0 = (const float *) &pa0;
                const float * a1 = (const float *) &pa1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.stage.A[buffer][a_k + i][a_r] = a0[i];
                    sm.stage.A[buffer][a_k + i + 4][a_r] = a1[i];
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.B[buffer][projection][b_k + 2*i][b_r] =
                        (float) (signed char) (pq[i] & 0xff)*pd;
                sm.stage.B[buffer][projection][b_k + 2*i + 1][b_r] =
                        (float) (signed char) (pq[i] >> 8)*pd;
            }
        };

        float acc[4][8] = {};
        fetcha(0);
        fetchb(0);
        stage_data(0);
        __syncthreads();
        int buffer = 0;
        for (int kb = 0; kb < k; kb += AW_KSTAGE) {
            const bool last = kb + AW_KSTAGE >= k;
            if (!last) {
                const int next = (kb + AW_KSTAGE)/AW_KSTAGE;
                fetcha(next);
                fetchb(next);
            }
            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const int ks = kgrp*(AW_KSTAGE/2) + kk;
                const int row0 = wm*16 + rm*4;
                const float4 av = *(const float4 *) &sm.stage.A[buffer][ks][row0];
                const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][projection][ks][nt*8];
                const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][projection][ks][nt*8 + 4];
                const float a[4] = { av.x, av.y, av.z, av.w };
                const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] += a[i]*b[j];
                    }
                }
            }
            if (!last) {
                stage_data(buffer ^ 1);
            }
            __syncthreads();
            buffer ^= 1;
        }

        const int output_row0 = wm*16 + rm*4;
        if (kgrp == 1) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sm.red[projection][output_row0 + i][nt*8 + j] = acc[i][j];
                }
            }
        }
        __syncthreads();
        if (kgrp == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = output_row0 + i;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] =
                                acc[i][j] + sm.red[projection][row][nt*8 + j];
                    }
                }
            }
        }
        __syncthreads();
    }
}

union aw_smem_m16_pair {
    struct {
        float A[2][AW_KSTAGE][16];
        float B[2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[2][16][AW_T64_ROWS];
};

template<bool BF16_INPUT, bool T64_LAYOUT>
__global__ __launch_bounds__(128, 3)
static void aw_q8_service_m16_pair(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m16_pair sm;

    const int group = threadIdx.x >> 6;
    const int local_tid = threadIdx.x & 63;
    const int kgrp = local_tid >> 5;
    const int lane = local_tid & 31;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = local_tid >> 2;
    const int a_k = (local_tid & 3)*8;
    const int b_r = local_tid;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_pairs = (n_mtiles + 1)/2;
    const int n_work = n_pairs*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int pair = work_id/n_ntiles;
        const int col0 = (work_id % n_ntiles)*AW_T64_ROWS;
        const int tile_index = pair*2 + group;
        const bool active = tile_index < n_mtiles;
        const aw_tile_desc tile = tiles[active ? tile_index : n_mtiles - 1];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float acc[4][8] = {};
        for (int stage = 0; stage < k/AW_KSTAGE; ++stage) {
            float4 pa0, pa1;
            const size_t input_base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, input_base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + input_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }

            float scale;
            const char * values;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks = k/AW_KSTAGE;
                const char * tile_stage = desc.weight +
                        ((size_t) (col0/AW_T64_ROWS)*n_kblocks + stage)*AW_T64_STAGE_BYTES;
                scale = __half2float(*(const half *) (tile_stage + b_r*2));
                values = tile_stage + AW_T64_ROWS*2 + b_r*AW_KSTAGE;
            } else {
                const char * block = desc.weight +
                        ((size_t) (col0 + b_r)*(k/AW_KSTAGE) + stage)*AW_Q8_BLOCK_BYTES;
                scale = __half2float(*(const half *) block);
                values = block + 2;
            }

            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[group][a_k + i][a_r] = a0[i];
                sm.stage.A[group][a_k + i + 4][a_r] = a1[i];
            }
            #pragma unroll
            for (int i = 0; i < AW_KSTAGE/2; ++i) {
                const unsigned short pair_values = *(const unsigned short *) (values + 2*i);
                sm.stage.B[group][2*i][b_r] = (float) (signed char) (pair_values & 0xff)*scale;
                sm.stage.B[group][2*i + 1][b_r] = (float) (signed char) (pair_values >> 8)*scale;
            }
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const int ks = kgrp*(AW_KSTAGE/2) + kk;
                const float4 av = *(const float4 *) &sm.stage.A[group][ks][rm*4];
                const float4 bv0 = *(const float4 *) &sm.stage.B[group][ks][nt*8];
                const float4 bv1 = *(const float4 *) &sm.stage.B[group][ks][nt*8 + 4];
                const float a[4] = { av.x, av.y, av.z, av.w };
                const float b[8] = { bv0.x, bv0.y, bv0.z, bv0.w, bv1.x, bv1.y, bv1.z, bv1.w };
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] += a[i]*b[j];
                    }
                }
            }
            __syncthreads();
        }

        if (kgrp == 1) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sm.red[group][rm*4 + i][nt*8 + j] = acc[i][j];
                }
            }
        }
        __syncthreads();
        if (active && kgrp == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = rm*4 + i;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        desc.output[(size_t) (tile.row + row)*n + col0 + nt*8 + j] =
                                acc[i][j] + sm.red[group][row][nt*8 + j];
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ static void aw_repack_q8_t64(
        const char * src, char * dst, size_t blocks, int n, int k) {
    const int n_kblocks = k/AW_KSTAGE;
    const size_t blocks_per_matrix = (size_t) n*n_kblocks;
    for (size_t block = (size_t) blockIdx.x*blockDim.x + threadIdx.x; block < blocks;
            block += (size_t) gridDim.x*blockDim.x) {
        const size_t matrix = block/blocks_per_matrix;
        const size_t within = block - matrix*blocks_per_matrix;
        const int row = (int) (within/n_kblocks);
        const int kblock = (int) (within - (size_t) row*n_kblocks);
        const char * input = src + block*AW_Q8_BLOCK_BYTES;
        char * tile_stage = dst +
                ((matrix*(n/AW_T64_ROWS) + row/AW_T64_ROWS)*n_kblocks + kblock)*AW_T64_STAGE_BYTES;
        *(uint16_t *) (tile_stage + (row % AW_T64_ROWS)*2) = *(const uint16_t *) input;
        char * output = tile_stage + AW_T64_ROWS*2 + (row % AW_T64_ROWS)*AW_KSTAGE;
        #pragma unroll
        for (int i = 0; i < AW_KSTAGE/2; ++i) {
            *(uint16_t *) (output + 2*i) = *(const uint16_t *) (input + 2 + 2*i);
        }
    }
}

__global__ static void aw_unpack_q8_t64(
        const char * src, char * dst, size_t blocks, int n, int k) {
    const int n_kblocks = k/AW_KSTAGE;
    const size_t blocks_per_matrix = (size_t) n*n_kblocks;
    for (size_t block = (size_t) blockIdx.x*blockDim.x + threadIdx.x; block < blocks;
            block += (size_t) gridDim.x*blockDim.x) {
        const size_t matrix = block/blocks_per_matrix;
        const size_t within = block - matrix*blocks_per_matrix;
        const int row = (int) (within/n_kblocks);
        const int kblock = (int) (within - (size_t) row*n_kblocks);
        const char * tile_stage = src +
                ((matrix*(n/AW_T64_ROWS) + row/AW_T64_ROWS)*n_kblocks + kblock)*AW_T64_STAGE_BYTES;
        char * output = dst + block*AW_Q8_BLOCK_BYTES;
        *(uint16_t *) output = *(const uint16_t *) (tile_stage + (row % AW_T64_ROWS)*2);
        const char * input = tile_stage + AW_T64_ROWS*2 + (row % AW_T64_ROWS)*AW_KSTAGE;
        #pragma unroll
        for (int i = 0; i < AW_KSTAGE/2; ++i) {
            *(uint16_t *) (output + 2 + 2*i) = *(const uint16_t *) (input + 2*i);
        }
    }
}

__global__ static void aw_expand_q8_t64_f32(
        const char * src, float * dst, size_t tile_stages, int n, int k) {
    const int n_kblocks = k/AW_KSTAGE;
    const int n_ntiles = n/AW_T64_ROWS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (size_t tile = blockIdx.x; tile < tile_stages; tile += gridDim.x) {
        const size_t matrix = tile/((size_t) n_ntiles*n_kblocks);
        const size_t within = tile - matrix*n_ntiles*n_kblocks;
        const int ntile = (int) (within/n_kblocks);
        const int kblock = (int) (within - (size_t) ntile*n_kblocks);
        const char * tile_stage = src + tile*AW_T64_STAGE_BYTES;
        for (int row = warp; row < AW_T64_ROWS; row += blockDim.x/32) {
            const float scale = __half2float(*(const half *) (tile_stage + row*2));
            const signed char * values = (const signed char *)
                    (tile_stage + AW_T64_ROWS*2 + row*AW_KSTAGE);
            float * output = dst + (matrix*n + ntile*AW_T64_ROWS + row)*k + kblock*AW_KSTAGE;
            output[lane] = (float) values[lane]*scale;
        }
    }
}

__global__ static void aw_init_input(float * data, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        data[i] = (float) ((int) (i % 17) - 8)*(1.0f/32.0f);
    }
}

__global__ static void aw_init_q8(char * data, size_t blocks, float scale) {
    for (size_t block = (size_t) blockIdx.x*blockDim.x + threadIdx.x; block < blocks;
            block += (size_t) gridDim.x*blockDim.x) {
        char * ptr = data + block*AW_Q8_BLOCK_BYTES;
        *(half *) ptr = __float2half(scale);
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            ptr[2 + i] = (char) ((i % 13) - 6);
        }
    }
}

template<bool BF16_WIRE>
__global__ static void aw_pack_requests(const float * src, void * dst, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        if constexpr (BF16_WIRE) {
            ((uint16_t *) dst)[i] = aw_float_to_bf16(src[i]);
        } else {
            ((float *) dst)[i] = src[i];
        }
    }
}

__global__ static void aw_swiglu(const float * gate, const float * up, float * output, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const float g = gate[i];
        output[i] = (g/(1.0f + expf(-g)))*up[i];
    }
}

__global__ static void aw_owner_reduce_bf16(
        const float * routes, const int32_t * token_routes, const float * route_weights,
        uint16_t * output, int tokens, int n) {
    const size_t count = (size_t) tokens*n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/n);
        const int col = (int) (i - (size_t) token*n);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < AW_ROUTES_PER_OWNER_TOKEN; ++rank) {
            const int slot = token*AW_ROUTES_PER_OWNER_TOKEN + rank;
            sum += route_weights[slot]*routes[(size_t) token_routes[slot]*n + col];
        }
        output[i] = aw_float_to_bf16(sum);
    }
}

__global__ static void aw_live_count_routes(
        const int32_t * ids, int32_t * counts, int total_slots, int owner, int tokens) {
    for (int slot = blockIdx.x*blockDim.x + threadIdx.x; slot < total_slots;
            slot += gridDim.x*blockDim.x) {
        const int cell = slot/(tokens*8);
        const int expert = ids[slot];
        if (expert >= owner*AW_PRIMARY_PER_GPU && expert < (owner + 1)*AW_PRIMARY_PER_GPU) {
            atomicAdd(&counts[cell*AW_PRIMARY_PER_GPU + expert % AW_PRIMARY_PER_GPU], 1);
        }
    }
}

__global__ static void aw_live_build_plan(
        const int32_t * counts,
        int32_t * offsets,
        int32_t * cursors,
        int32_t * tile_counts,
        aw_tile_desc * tiles_m64,
        aw_tile_desc * tiles_m32,
        aw_tile_desc * tiles_m16,
        aw_work_desc * gate_desc,
        aw_work_desc * up_desc,
        aw_work_desc * down_desc,
        const char * const * weight_ptrs,
        const void * input,
        int32_t * route_input,
        float * gate,
        float * up,
        float * middle,
        float * route_output,
        int descriptors) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    tile_counts[0] = 0;
    tile_counts[1] = 0;
    tile_counts[2] = 0;
    int row0 = 0;
    for (int desc = 0; desc < descriptors; ++desc) {
        const int rows = counts[desc];
        offsets[desc] = row0;
        cursors[desc] = row0;
        const int cell = desc/AW_PRIMARY_PER_GPU;
        const int expert = desc % AW_PRIMARY_PER_GPU;
        const size_t q8_gate_bytes = (size_t) AW_EXPERT_FF*(AW_EMBD/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
        const size_t q8_down_bytes = (size_t) AW_EMBD*(AW_EXPERT_FF/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
        gate_desc[desc] = {
            input, weight_ptrs[cell*3 + AW_PROJECTION_GATE] + (size_t) expert*q8_gate_bytes,
            gate + (size_t) row0*AW_EXPERT_FF,
            route_input + row0, row0, rows, cell, expert
        };
        up_desc[desc] = {
            input, weight_ptrs[cell*3 + AW_PROJECTION_UP] + (size_t) expert*q8_gate_bytes,
            up + (size_t) row0*AW_EXPERT_FF,
            route_input + row0, row0, rows, cell, expert
        };
        down_desc[desc] = {
            middle + (size_t) row0*AW_EXPERT_FF,
            weight_ptrs[cell*3 + AW_PROJECTION_DOWN] + (size_t) expert*q8_down_bytes,
            route_output + (size_t) row0*AW_EMBD,
            nullptr, row0, rows, cell, expert
        };
        int row = 0;
        while (rows - row >= 48) {
            const int tile_rows = min(64, rows - row);
            tiles_m64[tile_counts[0]++] = { desc, row, tile_rows };
            row += tile_rows;
        }
        if (rows - row > 16) {
            const int tile_rows = min(32, rows - row);
            tiles_m32[tile_counts[1]++] = { desc, row, tile_rows };
            row += tile_rows;
        }
        if (row < rows) {
            tiles_m16[tile_counts[2]++] = { desc, row, rows - row };
        }
        row0 += rows;
    }
    offsets[descriptors] = row0;
}

__global__ static void aw_live_build_shared_plan(
        int32_t * tile_counts,
        aw_tile_desc * tiles_m64,
        aw_tile_desc * tiles_m32,
        aw_tile_desc * tiles_m16,
        aw_work_desc * gate_desc,
        aw_work_desc * up_desc,
        aw_work_desc * down_desc,
        const char * const * weight_ptrs,
        const void * input,
        float * gate,
        float * up,
        float * middle,
        float * output,
        int rows) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    gate_desc[0] = { input, weight_ptrs[AW_PROJECTION_GATE], gate,
        nullptr, 0, rows, 0, 0 };
    up_desc[0] = { input, weight_ptrs[AW_PROJECTION_UP], up,
        nullptr, 0, rows, 0, 0 };
    down_desc[0] = { middle, weight_ptrs[AW_PROJECTION_DOWN], output,
        nullptr, 0, rows, 0, 0 };

    tile_counts[0] = 0;
    tile_counts[1] = 0;
    tile_counts[2] = 0;
    int row = 0;
    while (rows - row >= 48) {
        const int tile_rows = min(64, rows - row);
        tiles_m64[tile_counts[0]++] = { 0, row, tile_rows };
        row += tile_rows;
    }
    if (rows - row > 16) {
        const int tile_rows = min(32, rows - row);
        tiles_m32[tile_counts[1]++] = { 0, row, tile_rows };
        row += tile_rows;
    }
    if (row < rows) {
        tiles_m16[tile_counts[2]++] = { 0, row, rows - row };
    }
}

__global__ static void aw_live_fill_routes(
        const int32_t * ids,
        int32_t * cursors,
        int32_t * route_input,
        int32_t * token_routes,
        int total_slots,
        int owner,
        int tokens) {
    for (int slot = blockIdx.x*blockDim.x + threadIdx.x; slot < total_slots;
            slot += gridDim.x*blockDim.x) {
        const int cell = slot/(tokens*8);
        const int token = (slot/8) % tokens;
        const int expert = ids[slot];
        if (expert >= owner*AW_PRIMARY_PER_GPU && expert < (owner + 1)*AW_PRIMARY_PER_GPU) {
            const int desc = cell*AW_PRIMARY_PER_GPU + expert % AW_PRIMARY_PER_GPU;
            const int row = atomicAdd(&cursors[desc], 1);
            route_input[row] = cell*tokens + token;
            token_routes[slot] = row;
        }
    }
}

__global__ static void aw_live_owner_reduce(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * partial,
        int total_tokens,
        int owner) {
    const size_t count = (size_t) total_tokens*AW_EMBD;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/AW_EMBD);
        const int col = (int) (i - (size_t) token*AW_EMBD);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert >= owner*AW_PRIMARY_PER_GPU && expert < (owner + 1)*AW_PRIMARY_PER_GPU) {
                sum += route_weights[slot]*routes[(size_t) token_routes[slot]*AW_EMBD + col];
            }
        }
        partial[i] = aw_float_to_bf16(sum);
    }
}

__global__ static void aw_live_swiglu(
        const float * gate,
        const float * up,
        float * output,
        const int32_t * total_routes) {
    const size_t count = (size_t) *total_routes*AW_EXPERT_FF;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const float g = gate[i];
        output[i] = (g/(1.0f + expf(-g)))*up[i];
    }
}

__global__ static void aw_live_sum_owners(
        const uint16_t * recv,
        float * output,
        int total_tokens,
        int token_offset,
        int tokens) {
    const size_t count = (size_t) tokens*AW_EMBD;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const size_t global = (size_t) token_offset*AW_EMBD + i;
        float sum = aw_bf16_to_float(recv[global]);
        #pragma unroll
        for (int owner = 1; owner < AW_GPU_COUNT; ++owner) {
            sum += aw_bf16_to_float(recv[(size_t) owner*total_tokens*AW_EMBD + global]);
        }
        output[i] = sum;
    }
}

class aw_device_buffer {
public:
    aw_device_buffer() = default;
    explicit aw_device_buffer(size_t size) : size_(size) {
        if (size == 0) {
            return;
        }
        const cudaError_t status = cudaMalloc(&ptr_, size);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc(") + std::to_string(size) + ") failed: " +
                    cudaGetErrorString(status));
        }
    }
    ~aw_device_buffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    aw_device_buffer(const aw_device_buffer &) = delete;
    aw_device_buffer & operator=(const aw_device_buffer &) = delete;
    void * get() const { return ptr_; }
    size_t size() const { return size_; }
    void ensure(size_t size) {
        if (size <= size_) {
            return;
        }
        reset();
        const cudaError_t status = cudaMalloc(&ptr_, size);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc(") + std::to_string(size) + ") failed: " +
                    cudaGetErrorString(status));
        }
        size_ = size;
    }
    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }
private:
    void * ptr_ = nullptr;
    size_t size_ = 0;
};

static void aw_cuda_throw(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

static uint16_t aw_float_to_bf16_host(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

static float aw_bf16_to_float_host(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float aw_dot_reference(const std::vector<float> & input, int k, float scale, int split_groups) {
    std::array<float, 4> partial{};
    for (int kb = 0; kb < k; kb += AW_KSTAGE) {
        const int group_width = AW_KSTAGE/split_groups;
        for (int group = 0; group < split_groups; ++group) {
            for (int i = 0; i < group_width; ++i) {
                const int kk = kb + group*group_width + i;
                partial[group] += input[kk]*(float) ((kk % 32) % 13 - 6)*scale;
            }
        }
    }
    float result = partial[0];
    for (int group = 1; group < split_groups; ++group) {
        result += partial[group];
    }
    return result;
}

static float aw_expected_route(int token, bool bf16_wire, int gate_split_groups, int down_split_groups) {
    std::vector<float> input(AW_EMBD);
    for (int k = 0; k < AW_EMBD; ++k) {
        const size_t index = (size_t) token*AW_EMBD + k;
        input[k] = (float) ((int) (index % 17) - 8)*(1.0f/32.0f);
        if (bf16_wire) {
            input[k] = aw_bf16_to_float_host(aw_float_to_bf16_host(input[k]));
        }
    }
    const float gate = aw_dot_reference(input, AW_EMBD, 1.0f/128.0f, gate_split_groups);
    const float up   = aw_dot_reference(input, AW_EMBD, 1.0f/256.0f, gate_split_groups);
    const float middle_value = (gate/(1.0f + std::exp(-gate)))*up;
    std::vector<float> middle(AW_EXPERT_FF, middle_value);
    return aw_dot_reference(middle, AW_EXPERT_FF, 1.0f/512.0f, down_split_groups);
}

static int aw_grid(size_t count, int threads = 256) {
    return (int) std::min<size_t>((count + threads - 1)/threads, 65535);
}

struct aw_live_state {
    int device = -1;
    int cells = 0;
    int tokens = 0;
    aw_device_buffer input;
    aw_device_buffer ids;
    aw_device_buffer weights;
    aw_device_buffer counts;
    aw_device_buffer offsets;
    aw_device_buffer cursors;
    aw_device_buffer tile_counts;
    aw_device_buffer tiles_m64;
    aw_device_buffer tiles_m32;
    aw_device_buffer tiles_m16;
    aw_device_buffer gate_desc;
    aw_device_buffer up_desc;
    aw_device_buffer down_desc;
    aw_device_buffer weight_ptrs;
    aw_device_buffer route_input;
    aw_device_buffer token_routes;
    aw_device_buffer gate;
    aw_device_buffer up;
    aw_device_buffer middle;
    aw_device_buffer route_output;
    aw_device_buffer partial;
    aw_device_buffer recv;
    cudaEvent_t source_ready = nullptr;
    cudaEvent_t input_ready = nullptr;
    cudaEvent_t compute_done = nullptr;
    cudaEvent_t scratch_free = nullptr;
    cudaEvent_t output_ready = nullptr;
    cudaEvent_t recv_free = nullptr;

    void ensure(int new_device, int new_cells, int new_tokens) {
        device = new_device;
        cells = std::max(cells, new_cells);
        tokens = std::max(tokens, new_tokens);
        const cudaError_t set_status = cudaSetDevice(device);
        if (set_status != cudaSuccess) {
            throw std::runtime_error(std::string("live cudaSetDevice: ") + cudaGetErrorString(set_status));
        }
        const size_t total_tokens = (size_t) cells*tokens;
        const size_t max_routes = total_tokens*8;
        const size_t descriptors = (size_t) cells*AW_PRIMARY_PER_GPU;
        const size_t max_tiles = (max_routes + 15)/16 + descriptors;
        input.ensure(total_tokens*AW_EMBD*sizeof(float));
        ids.ensure(max_routes*sizeof(int32_t));
        weights.ensure(max_routes*sizeof(float));
        counts.ensure(descriptors*sizeof(int32_t));
        offsets.ensure((descriptors + 1)*sizeof(int32_t));
        cursors.ensure(descriptors*sizeof(int32_t));
        tile_counts.ensure(3*sizeof(int32_t));
        tiles_m64.ensure(max_tiles*sizeof(aw_tile_desc));
        tiles_m32.ensure(max_tiles*sizeof(aw_tile_desc));
        tiles_m16.ensure(max_tiles*sizeof(aw_tile_desc));
        gate_desc.ensure(descriptors*sizeof(aw_work_desc));
        up_desc.ensure(descriptors*sizeof(aw_work_desc));
        down_desc.ensure(descriptors*sizeof(aw_work_desc));
        weight_ptrs.ensure((size_t) cells*6*sizeof(const char *));
        route_input.ensure(max_routes*sizeof(int32_t));
        token_routes.ensure(max_routes*sizeof(int32_t));
        gate.ensure(max_routes*AW_EXPERT_FF*sizeof(float));
        up.ensure(max_routes*AW_EXPERT_FF*sizeof(float));
        middle.ensure(max_routes*AW_EXPERT_FF*sizeof(float));
        route_output.ensure(max_routes*AW_EMBD*sizeof(float));
        partial.ensure(total_tokens*AW_EMBD*sizeof(uint16_t));
        recv.ensure((size_t) AW_GPU_COUNT*total_tokens*AW_EMBD*sizeof(uint16_t));
        if (source_ready == nullptr) {
            auto create_event = [](cudaEvent_t * event) {
                const cudaError_t status = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
                if (status != cudaSuccess) {
                    throw std::runtime_error(std::string("live cudaEventCreate: ") + cudaGetErrorString(status));
                }
            };
            create_event(&source_ready);
            create_event(&input_ready);
            create_event(&compute_done);
            create_event(&scratch_free);
            create_event(&output_ready);
            create_event(&recv_free);
        }
    }

    ~aw_live_state() {
        if (device >= 0) {
            (void) cudaSetDevice(device);
        }
        if (source_ready != nullptr) {
            (void) cudaEventDestroy(source_ready);
            (void) cudaEventDestroy(input_ready);
            (void) cudaEventDestroy(compute_done);
            (void) cudaEventDestroy(scratch_free);
            (void) cudaEventDestroy(output_ready);
            (void) cudaEventDestroy(recv_free);
        }
    }
};

static std::array<std::array<aw_live_state, AW_GPU_COUNT>, AW_GPU_COUNT> aw_live_states;

struct aw_live_copy_streams {
    std::array<cudaStream_t, AW_GPU_COUNT> streams{};

    cudaStream_t get(int device) {
        if (streams[device] == nullptr) {
            aw_cuda_throw(cudaSetDevice(device), "set live copy-stream device");
            aw_cuda_throw(cudaStreamCreateWithFlags(&streams[device], cudaStreamNonBlocking),
                    "create live copy stream");
        }
        return streams[device];
    }

    ~aw_live_copy_streams() {
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            if (streams[device] != nullptr) {
                (void) cudaSetDevice(device);
                (void) cudaStreamDestroy(streams[device]);
            }
        }
    }
};

static aw_live_copy_streams aw_copy_streams;

static void aw_live_launch_projection(
        aw_live_state & state,
        const aw_device_buffer & desc,
        int n,
        int k,
        cudaStream_t stream,
        int persistent_blocks,
        int pair_blocks,
        bool bf16_input) {
    const auto * desc_ptr = (const aw_work_desc *) desc.get();
    const auto * tile_count = (const int32_t *) state.tile_counts.get();
    if (bf16_input) {
        aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
        aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(), 0, n, k, tile_count + 2);
    } else {
        aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
        aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(), 0, n, k, tile_count + 2);
    }
}

static void aw_live_launch_gate_up(
        aw_live_state & state,
        cudaStream_t stream,
        int fused_blocks,
        int persistent_blocks,
        int pair_blocks,
        bool bf16_input) {
    const auto * gate_desc = (const aw_work_desc *) state.gate_desc.get();
    const auto * up_desc = (const aw_work_desc *) state.up_desc.get();
    const auto * tile_count = (const int32_t *) state.tile_counts.get();
    if (bf16_input) {
        aw_q8_service_m64_gate_up<true><<<fused_blocks, 512, 0, stream>>>(
                gate_desc, up_desc, (const aw_tile_desc *) state.tiles_m64.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 0);
        aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
        aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
    } else {
        aw_q8_service_m64_gate_up<false><<<fused_blocks, 512, 0, stream>>>(
                gate_desc, up_desc, (const aw_tile_desc *) state.tiles_m64.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 0);
        aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
        aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
    }
}

static void aw_set_error(char * error, size_t capacity, const std::string & message) {
    if (error != nullptr && capacity != 0) {
        snprintf(error, capacity, "%s", message.c_str());
    }
}

} // namespace

int ggml_cuda_affinity_wave_live_service(
        void * const * streams_ptr,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity) {
    try {
        static int live_calls = 0;
        const int live_call = live_calls++;
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service entered cells=%d\n", n_cells);
        }
        if (!ggml_cuda_affinity_wave_t64_enabled() || streams_ptr == nullptr || cells == nullptr ||
                n_cells < 1 || n_cells > AW_GPU_COUNT) {
            throw std::runtime_error("invalid live AffinityWave service request");
        }
        const bool shared_service = aw_env_on(getenv("GGML_CUDA_AW_SHARED_SERVICE"));
        const bool fused_gate_up = aw_env_on(getenv("GGML_CUDA_AW_FUSED_GATE_UP"));
        const bool bf16_wire = getenv("GGML_CUDA_AW_WIRE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_WIRE"), "bf16") == 0;
        const size_t wire_element_size = bf16_wire ? sizeof(uint16_t) : sizeof(float);
        int max_tokens = 0;
        std::array<int, AW_GPU_COUNT> cell_for_home;
        cell_for_home.fill(-1);
        for (int cell = 0; cell < n_cells; ++cell) {
            if (cells[cell].tokens < 1 || cells[cell].layer < 0 || cells[cell].layer >= AW_LAYERS ||
                    cells[cell].home_device < 0 || cells[cell].home_device >= AW_GPU_COUNT ||
                    cells[cell].input == nullptr || cells[cell].ids == nullptr ||
                    cells[cell].weights == nullptr || cells[cell].output == nullptr ||
                    (shared_service && cells[cell].shared_output == nullptr) ||
                    cell_for_home[cells[cell].home_device] != -1) {
                throw std::runtime_error("invalid live AffinityWave cell descriptor");
            }
            cell_for_home[cells[cell].home_device] = cell;
            max_tokens = std::max(max_tokens, cells[cell].tokens);
        }
        constexpr int MAX_GROUP_CELLS = AW_GPU_COUNT;
        const char * group_cells_env = getenv("GGML_CUDA_AW_GROUP_CELLS");
        const int group_cells_limit = group_cells_env != nullptr ? atoi(group_cells_env) : 2;
        if (group_cells_limit < 1 || group_cells_limit > MAX_GROUP_CELLS) {
            throw std::runtime_error("GGML_CUDA_AW_GROUP_CELLS must be between 1 and 4");
        }
        std::array<int, AW_GPU_COUNT + 1> group_offsets{};
        int n_groups = 0;
        int grouped_cells = 0;
        const char * group_pattern = getenv("GGML_CUDA_AW_GROUP_PATTERN");
        if (group_pattern != nullptr && group_pattern[0] != '\0') {
            for (const char * p = group_pattern; *p != '\0' && grouped_cells < n_cells; ++p) {
                if (*p < '1' || *p > '4' || n_groups >= AW_GPU_COUNT) {
                    throw std::runtime_error("GGML_CUDA_AW_GROUP_PATTERN must contain one to four digits from 1 to 4");
                }
                const int width = std::min(*p - '0', n_cells - grouped_cells);
                grouped_cells += width;
                group_offsets[++n_groups] = grouped_cells;
            }
        }
        while (grouped_cells < n_cells) {
            if (n_groups >= AW_GPU_COUNT) {
                throw std::runtime_error("GGML_CUDA_AW_GROUP_PATTERN does not cover the active cells");
            }
            grouped_cells += std::min(group_cells_limit, n_cells - grouped_cells);
            group_offsets[++n_groups] = grouped_cells;
        }
        auto group_begin = [&](int group) { return group_offsets[group]; };
        auto group_size = [&](int group) { return group_offsets[group + 1] - group_offsets[group]; };
        std::array<int, AW_GPU_COUNT> group_max_tokens{};
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                group_max_tokens[group] = std::max(group_max_tokens[group], cells[cell_begin + group_cell].tokens);
            }
        }
        std::array<cudaStream_t, AW_GPU_COUNT> streams;
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            streams[device] = (cudaStream_t) streams_ptr[device];
            (void) aw_copy_streams.get(device);
            for (int group = 0; group < n_groups; ++group) {
                aw_live_states[group][device].ensure(device, group_size(group), max_tokens);
            }
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service scratch ready max_tokens=%d\n", max_tokens);
        }

        std::array<std::array<std::array<const char *, MAX_GROUP_CELLS*6>, AW_GPU_COUNT>, AW_GPU_COUNT> weight_ptrs{};
        {
            std::lock_guard<std::mutex> lock(aw_layout_mutex);
            for (int group = 0; group < n_groups; ++group) {
                const int cell_begin = group_begin(group);
                const int group_cells = group_size(group);
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                        const int cell = cell_begin + group_cell;
                        for (int projection = 0; projection < 3; ++projection) {
                            const auto it = std::find_if(aw_layout_entries.begin(), aw_layout_entries.end(),
                                    [&](const aw_layout_entry & entry) {
                                        return entry.device == owner && entry.layer == cells[cell].layer &&
                                                entry.projection == projection && aw_t64_tensors.count(entry.data) != 0;
                                    });
                            if (it == aw_layout_entries.end()) {
                                throw std::runtime_error("live AffinityWave weight catalog is incomplete");
                            }
                            weight_ptrs[group][owner][group_cell*3 + projection] = (const char *) it->data;
                        }
                        if (shared_service) {
                            for (int projection = 0; projection < 3; ++projection) {
                                const auto it = std::find_if(aw_layout_entries.begin(), aw_layout_entries.end(),
                                        [&](const aw_layout_entry & entry) {
                                            return entry.device == owner && entry.layer == cells[cell].layer &&
                                                    entry.projection == AW_PROJECTION_SHARED_GATE + projection &&
                                                    aw_t64_tensors.count(entry.data) != 0;
                                        });
                                if (it == aw_layout_entries.end()) {
                                    throw std::runtime_error("live AffinityWave shared-weight catalog is incomplete");
                                }
                                weight_ptrs[group][owner][group_cells*3 + group_cell*3 + projection] =
                                        (const char *) it->data;
                            }
                        }
                    }
                }
            }
        }

        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const size_t input_cell_stride_bytes = (size_t) stride_tokens*AW_EMBD*wire_element_size;
            const size_t route_cell_stride_bytes = (size_t) stride_tokens*8*sizeof(int32_t);
            for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                const int cell = cell_begin + group_cell;
                const int home = cells[cell].home_device;
                const size_t input_elements = (size_t) cells[cell].tokens*AW_EMBD;
                const size_t input_cell_bytes = input_elements*wire_element_size;
                const size_t route_cell_bytes = (size_t) cells[cell].tokens*8*sizeof(int32_t);
                aw_live_state & home_state = aw_live_states[group][home];
                aw_cuda_throw(cudaSetDevice(home), "set live input home device");
                aw_cuda_throw(cudaEventRecord(home_state.source_ready, streams[home]),
                        "record live input source ready");
                cudaStream_t copy_stream = aw_copy_streams.get(home);
                aw_cuda_throw(cudaStreamWaitEvent(copy_stream, home_state.source_ready, 0),
                        "wait for live input source");
                aw_cuda_throw(cudaStreamWaitEvent(copy_stream, home_state.output_ready, 0),
                        "wait for live home input scratch");
                aw_cuda_throw(cudaStreamWaitEvent(copy_stream, home_state.scratch_free, 0),
                        "wait for live home compute scratch");
                char * home_input = (char *) home_state.input.get() +
                        (size_t) group_cell*input_cell_stride_bytes;
                if (bf16_wire) {
                    aw_pack_requests<true><<<aw_grid(input_elements), 256, 0, copy_stream>>>(
                            cells[cell].input, home_input, input_elements);
                    aw_cuda_throw(cudaGetLastError(), "pack BF16 live input");
                } else {
                    aw_cuda_throw(cudaMemcpyAsync(home_input, cells[cell].input, input_cell_bytes,
                                cudaMemcpyDeviceToDevice, copy_stream), "copy local live input");
                }
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_live_state & dst = aw_live_states[group][owner];
                    if (owner != home) {
                        aw_cuda_throw(cudaStreamWaitEvent(copy_stream, dst.output_ready, 0),
                                "wait for live owner input scratch");
                        aw_cuda_throw(cudaStreamWaitEvent(copy_stream, dst.scratch_free, 0),
                                "wait for live owner compute scratch");
                    }
                    char * input_dst = (char *) dst.input.get() + (size_t) group_cell*input_cell_stride_bytes;
                    char * ids_dst = (char *) dst.ids.get() + (size_t) group_cell*route_cell_stride_bytes;
                    char * weights_dst = (char *) dst.weights.get() + (size_t) group_cell*route_cell_stride_bytes;
                    if (owner == home) {
                        aw_cuda_throw(cudaMemcpyAsync(ids_dst, cells[cell].ids, route_cell_bytes,
                                    cudaMemcpyDeviceToDevice, copy_stream), "copy local live ids");
                        aw_cuda_throw(cudaMemcpyAsync(weights_dst, cells[cell].weights, route_cell_bytes,
                                    cudaMemcpyDeviceToDevice, copy_stream), "copy local live weights");
                    } else {
                        aw_cuda_throw(cudaMemcpyPeerAsync(input_dst, owner, home_input, home,
                                    input_cell_bytes, copy_stream), "copy peer live input");
                        aw_cuda_throw(cudaMemcpyPeerAsync(ids_dst, owner, cells[cell].ids, home,
                                    route_cell_bytes, copy_stream), "copy peer live ids");
                        aw_cuda_throw(cudaMemcpyPeerAsync(weights_dst, owner, cells[cell].weights, home,
                                    route_cell_bytes, copy_stream), "copy peer live weights");
                    }
                }
                aw_cuda_throw(cudaEventRecord(home_state.input_ready, copy_stream),
                        "record live input ready");
            }
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service inputs queued\n");
        }

        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const size_t input_cell_stride_bytes = (size_t) stride_tokens*AW_EMBD*wire_element_size;
            const int total_tokens = group_cells*stride_tokens;
            const int total_slots = total_tokens*8;
            const int descriptors = group_cells*AW_PRIMARY_PER_GPU;
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaSetDevice(owner), "set live owner device");
                aw_live_state & state = aw_live_states[group][owner];
                cudaStream_t compute_stream = streams[owner];
                for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                    const int home = cells[cell_begin + group_cell].home_device;
                    aw_cuda_throw(cudaStreamWaitEvent(compute_stream,
                                aw_live_states[group][home].input_ready, 0), "wait for live input");
                    const int tail_tokens = stride_tokens - cells[cell_begin + group_cell].tokens;
                    if (tail_tokens > 0) {
                        char * ids_tail = (char *) state.ids.get() +
                                ((size_t) group_cell*stride_tokens + cells[cell_begin + group_cell].tokens)*8*sizeof(int32_t);
                        aw_cuda_throw(cudaMemsetAsync(ids_tail, 0xff,
                                    (size_t) tail_tokens*8*sizeof(int32_t), compute_stream),
                                "clear live padded ids");
                    }
                }
                aw_cuda_throw(cudaMemcpyAsync(state.weight_ptrs.get(), weight_ptrs[group][owner].data(),
                            (size_t) group_cells*(shared_service ? 6 : 3)*sizeof(const char *),
                            cudaMemcpyHostToDevice, compute_stream),
                        "copy live weight pointers");
                aw_cuda_throw(cudaMemsetAsync(state.counts.get(), 0,
                            (size_t) descriptors*sizeof(int32_t), compute_stream), "clear live route counts");
                aw_live_count_routes<<<aw_grid(total_slots), 256, 0, compute_stream>>>(
                        (const int32_t *) state.ids.get(), (int32_t *) state.counts.get(),
                        total_slots, owner, stride_tokens);
                aw_live_build_plan<<<1, 1, 0, compute_stream>>>(
                        (const int32_t *) state.counts.get(),
                        (int32_t *) state.offsets.get(), (int32_t *) state.cursors.get(),
                        (int32_t *) state.tile_counts.get(),
                        (aw_tile_desc *) state.tiles_m64.get(),
                        (aw_tile_desc *) state.tiles_m32.get(),
                        (aw_tile_desc *) state.tiles_m16.get(),
                        (aw_work_desc *) state.gate_desc.get(),
                        (aw_work_desc *) state.up_desc.get(),
                        (aw_work_desc *) state.down_desc.get(),
                        (const char * const *) state.weight_ptrs.get(),
                        state.input.get(),
                        (int32_t *) state.route_input.get(),
                        (float *) state.gate.get(), (float *) state.up.get(),
                        (float *) state.middle.get(), (float *) state.route_output.get(), descriptors);
                aw_live_fill_routes<<<aw_grid(total_slots), 256, 0, compute_stream>>>(
                        (const int32_t *) state.ids.get(), (int32_t *) state.cursors.get(),
                        (int32_t *) state.route_input.get(), (int32_t *) state.token_routes.get(),
                        total_slots, owner, stride_tokens);

                cudaDeviceProp prop;
                aw_cuda_throw(cudaGetDeviceProperties(&prop, owner), "get live owner properties");
                const int persistent_blocks = prop.multiProcessorCount*2;
                const int pair_blocks = prop.multiProcessorCount*3;
                if (fused_gate_up) {
                    aw_live_launch_gate_up(state, compute_stream, prop.multiProcessorCount,
                            persistent_blocks, pair_blocks, bf16_wire);
                } else {
                    aw_live_launch_projection(state, state.gate_desc, AW_EXPERT_FF, AW_EMBD,
                            compute_stream, persistent_blocks, pair_blocks, bf16_wire);
                    aw_live_launch_projection(state, state.up_desc, AW_EXPERT_FF, AW_EMBD,
                            compute_stream, persistent_blocks, pair_blocks, bf16_wire);
                }
                aw_live_swiglu<<<aw_grid((size_t) total_slots*AW_EXPERT_FF), 256, 0, compute_stream>>>(
                        (const float *) state.gate.get(), (const float *) state.up.get(),
                        (float *) state.middle.get(), (const int32_t *) state.offsets.get() + descriptors);
                aw_live_launch_projection(state, state.down_desc, AW_EMBD, AW_EXPERT_FF,
                        compute_stream, persistent_blocks, pair_blocks, false);
                aw_live_owner_reduce<<<aw_grid((size_t) total_tokens*AW_EMBD), 256, 0, compute_stream>>>(
                        (const float *) state.route_output.get(), (const int32_t *) state.ids.get(),
                        (const int32_t *) state.token_routes.get(), (const float *) state.weights.get(),
                        (uint16_t *) state.partial.get(), total_tokens, owner);
                aw_cuda_throw(cudaGetLastError(), "launch live owner service");
                aw_cuda_throw(cudaEventRecord(state.compute_done, compute_stream),
                        "record live owner compute done");
                if (shared_service) {
                    for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                        const int cell = cell_begin + group_cell;
                        if (cells[cell].home_device != owner) {
                            continue;
                        }
                        const char * const * shared_weights =
                                (const char * const *) state.weight_ptrs.get() + group_cells*3 + group_cell*3;
                        const char * shared_input = (const char *) state.input.get() +
                                (size_t) group_cell*input_cell_stride_bytes;
                        aw_live_build_shared_plan<<<1, 1, 0, compute_stream>>>(
                                (int32_t *) state.tile_counts.get(),
                                (aw_tile_desc *) state.tiles_m64.get(),
                                (aw_tile_desc *) state.tiles_m32.get(),
                                (aw_tile_desc *) state.tiles_m16.get(),
                                (aw_work_desc *) state.gate_desc.get(),
                                (aw_work_desc *) state.up_desc.get(),
                                (aw_work_desc *) state.down_desc.get(),
                                shared_weights, shared_input,
                                (float *) state.gate.get(), (float *) state.up.get(),
                                (float *) state.middle.get(), cells[cell].shared_output, cells[cell].tokens);
                        aw_live_launch_projection(state, state.gate_desc, AW_EXPERT_FF, AW_EMBD,
                                compute_stream, persistent_blocks, pair_blocks, bf16_wire);
                        aw_live_launch_projection(state, state.up_desc, AW_EXPERT_FF, AW_EMBD,
                                compute_stream, persistent_blocks, pair_blocks, bf16_wire);
                        aw_swiglu<<<aw_grid((size_t) cells[cell].tokens*AW_EXPERT_FF), 256, 0, compute_stream>>>(
                                (const float *) state.gate.get(), (const float *) state.up.get(),
                                (float *) state.middle.get(), (size_t) cells[cell].tokens*AW_EXPERT_FF);
                        aw_live_launch_projection(state, state.down_desc, AW_EMBD, AW_EXPERT_FF,
                                compute_stream, persistent_blocks, pair_blocks, false);
                        aw_cuda_throw(cudaGetLastError(), "launch live shared expert");
                    }
                }
                aw_cuda_throw(cudaEventRecord(state.scratch_free, compute_stream),
                        "record live compute scratch free");
            }
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service owners queued\n");
        }

        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const size_t partial_cell_stride_bytes = (size_t) stride_tokens*AW_EMBD*sizeof(uint16_t);
            const int total_tokens = group_cells*stride_tokens;
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaSetDevice(owner), "set live output owner device");
                aw_live_state & src = aw_live_states[group][owner];
                cudaStream_t copy_stream = aw_copy_streams.get(owner);
                aw_cuda_throw(cudaStreamWaitEvent(copy_stream, src.compute_done, 0),
                        "wait for live owner compute");
                for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                    const int cell = cell_begin + group_cell;
                    const int home = cells[cell].home_device;
                    aw_cuda_throw(cudaStreamWaitEvent(copy_stream,
                                aw_live_states[group][home].recv_free, 0), "wait for live receive scratch");
                    const char * partial_src = (const char *) src.partial.get() +
                            (size_t) group_cell*partial_cell_stride_bytes;
                    char * recv_dst = (char *) aw_live_states[group][home].recv.get() +
                            ((size_t) owner*total_tokens + (size_t) group_cell*stride_tokens)*AW_EMBD*sizeof(uint16_t);
                    const size_t partial_cell_bytes =
                            (size_t) cells[cell].tokens*AW_EMBD*sizeof(uint16_t);
                    if (owner == home) {
                        aw_cuda_throw(cudaMemcpyAsync(recv_dst, partial_src, partial_cell_bytes,
                                    cudaMemcpyDeviceToDevice, copy_stream), "copy local owner partial");
                    } else {
                        aw_cuda_throw(cudaMemcpyPeerAsync(recv_dst, home, partial_src, owner,
                                    partial_cell_bytes, copy_stream), "copy peer owner partial");
                    }
                }
                aw_cuda_throw(cudaEventRecord(src.output_ready, copy_stream), "record live output ready");
            }
        }

        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const int total_tokens = group_cells*stride_tokens;
            for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                const int cell = cell_begin + group_cell;
                const int home = cells[cell].home_device;
                aw_cuda_throw(cudaSetDevice(home), "set live output home device");
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_cuda_throw(cudaStreamWaitEvent(streams[home],
                                aw_live_states[group][owner].output_ready, 0), "wait for live owner output");
                }
                aw_live_sum_owners<<<aw_grid((size_t) cells[cell].tokens*AW_EMBD), 256, 0, streams[home]>>>(
                        (const uint16_t *) aw_live_states[group][home].recv.get(), cells[cell].output,
                        total_tokens, group_cell*stride_tokens, cells[cell].tokens);
                aw_cuda_throw(cudaGetLastError(), "launch live owner sum");
                aw_cuda_throw(cudaEventRecord(aw_live_states[group][home].recv_free, streams[home]),
                        "record live receive scratch free");
            }
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service first call complete\n");
        }
        aw_cuda_throw(cudaSetDevice(0), "restore live scheduler device");
        return 0;
    } catch (const std::exception & exception) {
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}

bool ggml_cuda_affinity_wave_t64_enabled() {
    static const bool enabled = aw_env_on(getenv("GGML_CUDA_AFFINITY_WAVE")) &&
            getenv("GGML_CUDA_AW_Q8_LAYOUT") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_Q8_LAYOUT"), "t64k32") == 0 &&
            aw_env_on(getenv("GGML_CUDA_MOE_EP")) &&
            aw_env_on(getenv("GGML_CUDA_MOE_PLAN")) &&
            getenv("GGML_CUDA_MOE_ASYNCEP") == nullptr &&
            getenv("GGML_CUDA_MOE_HIST") == nullptr &&
            getenv("GGML_CUDA_MOE_DEBUG_STATS") == nullptr;
    return enabled;
}

static bool aw_is_expert_q8_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr || tensor->type != GGML_TYPE_Q8_0 ||
            tensor->ne[0] % AW_KSTAGE != 0 || tensor->ne[1] % AW_T64_ROWS != 0 || tensor->ne[2] < 1 ||
            tensor->ne[3] != 1 || !ggml_is_contiguous(tensor)) {
        return false;
    }
    const bool gate = strstr(tensor->name, "ffn_gate_exps") != nullptr;
    const bool up = strstr(tensor->name, "ffn_up_exps") != nullptr;
    const bool gate_up = strstr(tensor->name, "ffn_gate_up_exps") != nullptr;
    const bool down = strstr(tensor->name, "ffn_down_exps") != nullptr;
    const bool shared_gate = strstr(tensor->name, "ffn_gate_shexp.weight") != nullptr;
    const bool shared_up = strstr(tensor->name, "ffn_up_shexp.weight") != nullptr;
    const bool shared_down = strstr(tensor->name, "ffn_down_shexp.weight") != nullptr;
    const bool shared = aw_env_on(getenv("GGML_CUDA_AW_SHARED_SERVICE")) &&
            (shared_gate || shared_up || shared_down);
    if (!gate && !up && !gate_up && !down && !shared) {
        return false;
    }
    if (shared) {
        return true;
    }
    const char * scope = getenv("GGML_CUDA_AW_T64_SCOPE");
    if (scope == nullptr || strcmp(scope, "all") == 0) {
        return true;
    }
    if (strcmp(scope, "gate-up") == 0) {
        return gate || up || gate_up;
    }
    if (strcmp(scope, "down") == 0) {
        return down;
    }
    if (strcmp(scope, "gate") == 0) {
        return gate || gate_up;
    }
    if (strcmp(scope, "up") == 0) {
        return up || gate_up;
    }
    return false;
}

static int aw_tensor_projection(const ggml_tensor * tensor) {
    if (strstr(tensor->name, "ffn_gate_exps") != nullptr) {
        return AW_PROJECTION_GATE;
    }
    if (strstr(tensor->name, "ffn_up_exps") != nullptr) {
        return AW_PROJECTION_UP;
    }
    if (strstr(tensor->name, "ffn_down_exps") != nullptr) {
        return AW_PROJECTION_DOWN;
    }
    if (strstr(tensor->name, "ffn_gate_shexp.weight") != nullptr) {
        return AW_PROJECTION_SHARED_GATE;
    }
    if (strstr(tensor->name, "ffn_up_shexp.weight") != nullptr) {
        return AW_PROJECTION_SHARED_UP;
    }
    if (strstr(tensor->name, "ffn_down_shexp.weight") != nullptr) {
        return AW_PROJECTION_SHARED_DOWN;
    }
    return -1;
}

bool ggml_cuda_affinity_wave_is_t64(const void * data) {
    if (!ggml_cuda_affinity_wave_t64_enabled() || data == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(aw_layout_mutex);
    return aw_t64_tensors.count(data) != 0;
}

void ggml_cuda_affinity_wave_forget_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(aw_layout_mutex);
    aw_t64_tensors.erase(tensor->data);
    aw_layout_entries.erase(std::remove_if(aw_layout_entries.begin(), aw_layout_entries.end(),
            [&](const aw_layout_entry & entry) { return entry.data == tensor->data; }), aw_layout_entries.end());
}

void ggml_cuda_affinity_wave_forget_range(const void * data, size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    const uintptr_t begin = (uintptr_t) data;
    const uintptr_t end = begin + size;
    std::lock_guard<std::mutex> lock(aw_layout_mutex);
    for (auto it = aw_t64_tensors.begin(); it != aw_t64_tensors.end();) {
        const uintptr_t ptr = (uintptr_t) *it;
        it = ptr >= begin && ptr < end ? aw_t64_tensors.erase(it) : std::next(it);
    }
    aw_layout_entries.erase(std::remove_if(aw_layout_entries.begin(), aw_layout_entries.end(),
            [&](const aw_layout_entry & entry) {
                const uintptr_t ptr = (uintptr_t) entry.data;
                return ptr >= begin && ptr < end;
            }), aw_layout_entries.end());
}

static void aw_repack_uploaded_tensor(
        const ggml_tensor * tensor, bool upload_complete, cudaStream_t stream) {
    const bool eligible = aw_is_expert_q8_tensor(tensor);
    if (ggml_cuda_affinity_wave_t64_enabled() && tensor != nullptr && strstr(tensor->name, "exps") != nullptr) {
        static std::atomic<int> probes{0};
        const int probe = probes.fetch_add(1);
        if (probe < 8) {
            fprintf(stderr, "AffinityWave: upload probe name=%s type=%s ne=%lld,%lld,%lld,%lld complete=%d contiguous=%d enabled=%d eligible=%d\n",
                    tensor->name, ggml_type_name(tensor->type),
                    (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
                    (long long) tensor->ne[3],
                    upload_complete ? 1 : 0, ggml_is_contiguous(tensor) ? 1 : 0,
                    ggml_cuda_affinity_wave_t64_enabled() ? 1 : 0, eligible ? 1 : 0);
        }
    }
    if (!upload_complete || !ggml_cuda_affinity_wave_t64_enabled() || !eligible) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(aw_layout_mutex);
        if (aw_t64_tensors.count(tensor->data) != 0) {
            return;
        }
    }

    const size_t bytes = ggml_nbytes(tensor);
    const size_t blocks = bytes/AW_Q8_BLOCK_BYTES;
    void * scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, bytes));
    aw_repack_q8_t64<<<aw_grid(blocks), 256, 0, stream>>>(
            (const char *) tensor->data, (char *) scratch, blocks, (int) tensor->ne[1], (int) tensor->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, scratch, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(scratch));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int layer = -1;
    (void) sscanf(tensor->name, "blk.%d.", &layer);
    const int projection = aw_tensor_projection(tensor);
    {
        std::lock_guard<std::mutex> lock(aw_layout_mutex);
        aw_t64_tensors.insert(tensor->data);
        if (layer >= 0 && layer < AW_LAYERS && projection >= 0) {
            aw_layout_entries.push_back({
                    tensor->data, bytes, (int) tensor->ne[1], (int) tensor->ne[0], device, layer, projection });
        }
    }
    const int count = ++aw_t64_tensor_counts[device];
    if (count == 1 || count % (AW_LAYERS*3) == 0) {
        fprintf(stderr, "AffinityWave: GPU%d T64 expert tensors=%d (latest %s)\n",
                device, count, tensor->name);
    }
}

void ggml_cuda_affinity_wave_repack_uploaded_tensor(const ggml_tensor * tensor, bool upload_complete) {
    aw_repack_uploaded_tensor(tensor, upload_complete, cudaStreamPerThread);
}

void ggml_cuda_affinity_wave_repack_uploaded_tensor_async(
        const ggml_tensor * tensor, bool upload_complete, void * stream) {
    aw_repack_uploaded_tensor(tensor, upload_complete, (cudaStream_t) stream);
}

void ggml_cuda_affinity_wave_normalize_decode(void * stream_ptr) {
    const char * mode = getenv("GGML_CUDA_AW_DECODE");
    if (mode == nullptr || strcmp(mode, "native") != 0) {
        return;
    }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::vector<aw_layout_entry> entries;
    {
        std::lock_guard<std::mutex> lock(aw_layout_mutex);
        for (const aw_layout_entry & entry : aw_layout_entries) {
            if (entry.device == device && aw_t64_tensors.count(entry.data) != 0) {
                entries.push_back(entry);
            }
        }
    }
    if (entries.empty()) {
        return;
    }

    size_t scratch_bytes = 0;
    for (const aw_layout_entry & entry : entries) {
        scratch_bytes = std::max(scratch_bytes, entry.bytes);
    }
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    void * scratch = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(begin, stream));
    for (const aw_layout_entry & entry : entries) {
        const size_t blocks = entry.bytes/AW_Q8_BLOCK_BYTES;
        aw_unpack_q8_t64<<<aw_grid(blocks), 256, 0, stream>>>(
                (const char *) entry.data, (char *) scratch, blocks, entry.n, entry.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(entry.data, scratch, entry.bytes, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(scratch));

    {
        std::lock_guard<std::mutex> lock(aw_layout_mutex);
        for (const aw_layout_entry & entry : entries) {
            aw_t64_tensors.erase(entry.data);
        }
    }
    fprintf(stderr, "AffinityWave: GPU%d normalized %zu expert tensors to native Q8 in %.3f ms\n",
            device, entries.size(), elapsed_ms);
}

static void * aw_unpack_native_scratch(const ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    const size_t blocks = bytes/AW_Q8_BLOCK_BYTES;
    void * scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, bytes));
    aw_unpack_q8_t64<<<aw_grid(blocks), 256, 0, cudaStreamPerThread>>>(
            (const char *) tensor->data, (char *) scratch, blocks, (int) tensor->ne[1], (int) tensor->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    return scratch;
}

bool ggml_cuda_affinity_wave_get_tensor_native(
        const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!ggml_cuda_affinity_wave_is_t64(tensor != nullptr ? tensor->data : nullptr)) {
        return false;
    }
    GGML_ASSERT(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset);
    void * scratch = aw_unpack_native_scratch(tensor);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) scratch + offset, size,
            cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK(cudaFree(scratch));
    return true;
}

bool ggml_cuda_affinity_wave_get_tensor_native_2d(
        const ggml_tensor * tensor, void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (!ggml_cuda_affinity_wave_is_t64(tensor != nullptr ? tensor->data : nullptr)) {
        return false;
    }
    void * scratch = aw_unpack_native_scratch(tensor);
    CUDA_CHECK(cudaMemcpy2DAsync(data, stride_data, (const char *) scratch + offset, stride_tensor,
            size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK(cudaFree(scratch));
    return true;
}

void ggml_cuda_affinity_wave_validate_env_or_abort() {
    if (!aw_env_on(getenv("GGML_CUDA_AFFINITY_WAVE"))) {
        return;
    }
    aw_config config;
    std::string error;
    if (!aw_load_config(config, error)) {
        GGML_ABORT("AffinityWave configuration error: %s", error.c_str());
    }
    GGML_LOG_INFO("AffinityWave: expert service enabled; map=%s wire=%s layout=%s "
            "down-cache=%d m64-split=%d home=%s check=%d\n", config.map.c_str(), config.wire.c_str(),
            config.q8_layout.c_str(), config.down_cache, config.m64_split, config.home.c_str(),
            config.check ? 1 : 0);
    if (ggml_cuda_affinity_wave_t64_enabled()) {
        GGML_LOG_INFO("AffinityWave: in-place T64 expert loading, M64 plan prefill, and T64 MMVQ decode enabled\n");
    } else {
        GGML_LOG_WARN("AffinityWave: model integration requires GGML_CUDA_MOE_EP=1 and GGML_CUDA_MOE_PLAN=1; "
                "the isolated service remains available\n");
    }
}

extern "C" int ggml_cuda_affinity_wave_service_bench(
        const ggml_cuda_aw_bench_params * params,
        ggml_cuda_aw_bench_result       * result,
        char                            * error,
        size_t                            error_capacity) {
    if (params == nullptr || result == nullptr) {
        aw_set_error(error, error_capacity, "null benchmark parameter or result");
        return 1;
    }
    memset(result, 0, sizeof(*result));

    try {
        aw_config config;
        std::string config_error;
        if (!aw_load_config(config, config_error)) {
            throw std::runtime_error(config_error);
        }
        if (params->device < 0 || params->device >= AW_GPU_COUNT || params->active_cells < 1 ||
                params->active_cells > AW_GPU_COUNT || params->tokens_per_cell < 32 ||
                params->tokens_per_cell % 32 != 0 || params->repeats < 1 ||
                params->route_pattern < 0 || params->route_pattern > 1) {
            throw std::runtime_error("invalid service benchmark dimensions");
        }
        if (params->route_pattern == 1 && params->tokens_per_cell != 2048) {
            throw std::runtime_error("tail-mix routing currently requires 2048 tokens per cell");
        }
        aw_cuda_throw(cudaSetDevice(params->device), "cudaSetDevice");

        const int descriptors = params->active_cells*AW_PRIMARY_PER_GPU;
        const int tokens = params->active_cells*params->tokens_per_cell;
        const int route_rows = tokens*AW_ROUTES_PER_OWNER_TOKEN;
        const int uniform_rows_per_desc = route_rows/descriptors;
        if (route_rows % descriptors != 0 || uniform_rows_per_desc < 1) {
            throw std::runtime_error("synthetic routing does not divide evenly over descriptors");
        }

        const size_t q8_gate_bytes = (size_t) AW_EXPERT_FF*(AW_EMBD/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
        const size_t q8_down_bytes = (size_t) AW_EMBD*(AW_EXPERT_FF/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
        const size_t source_bytes = (size_t) tokens*AW_EMBD*sizeof(float);
        const bool bf16_wire = config.wire == "bf16";
        const size_t wire_bytes = (size_t) tokens*AW_EMBD*(bf16_wire ? sizeof(uint16_t) : sizeof(float));
        const size_t narrow_bytes = (size_t) route_rows*AW_EXPERT_FF*sizeof(float);
        const size_t wide_bytes = (size_t) route_rows*AW_EMBD*sizeof(float);
        const size_t partial_bytes = (size_t) tokens*AW_EMBD*sizeof(uint16_t);

        std::vector<int32_t> route_input(route_rows);
        std::vector<int32_t> token_routes((size_t) tokens*AW_ROUTES_PER_OWNER_TOKEN);
        std::vector<float> route_weights(token_routes.size(), 0.5f);
        std::vector<int32_t> route_groups(route_rows);
        std::vector<int32_t> desc_row0(descriptors), desc_rows(descriptors);
        std::vector<aw_tile_desc> tiles_m64, tiles_m32, tiles_m16;
        std::vector<aw_work_desc> gate_desc(descriptors), up_desc(descriptors), down_desc(descriptors);
        tiles_m64.reserve((route_rows + 63)/64);
        tiles_m32.reserve((route_rows + 31)/32);
        tiles_m16.reserve((route_rows + 15)/16);

        std::vector<int> route_expert((size_t) params->tokens_per_cell*AW_ROUTES_PER_OWNER_TOKEN);
        if (params->route_pattern == 0) {
            for (size_t route = 0; route < route_expert.size(); ++route) {
                route_expert[route] = (int) (route % AW_PRIMARY_PER_GPU);
            }
        } else {
            constexpr std::array<int, 8> tail_counts = { 1, 15, 17, 31, 33, 63, 95, 257 };
            size_t route = 0;
            for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
                const int count = tail_counts[expert % (int) tail_counts.size()];
                for (int i = 0; i < count; ++i) {
                    route_expert[route++] = expert;
                }
            }
            if (route != route_expert.size()) {
                throw std::runtime_error("internal tail-mix routing count mismatch");
            }
        }

        // Expert-contiguous route rows, but one owner request per token.  The
        // descriptor-local input_rows vector is the sparse expert gather.
        int route_cursor = 0;
        for (int cell = 0; cell < params->active_cells; ++cell) {
            for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
                const int desc = cell*AW_PRIMARY_PER_GPU + expert;
                const int row0 = route_cursor;
                for (int token = 0; token < params->tokens_per_cell; ++token) {
                    for (int rank = 0; rank < AW_ROUTES_PER_OWNER_TOKEN; ++rank) {
                        if (route_expert[token*AW_ROUTES_PER_OWNER_TOKEN + rank] != expert) {
                            continue;
                        }
                        const int owner_token = cell*params->tokens_per_cell + token;
                        route_input[route_cursor] = owner_token;
                        token_routes[owner_token*AW_ROUTES_PER_OWNER_TOKEN + rank] = route_cursor;
                        ++route_cursor;
                    }
                }
                const int rows = route_cursor - row0;
                desc_row0[desc] = row0;
                desc_rows[desc] = rows;
                int row = 0;
                while (rows - row >= 48) {
                    const int tile_rows = std::min(64, rows - row);
                    tiles_m64.push_back({ desc, row, tile_rows });
                    std::fill(route_groups.begin() + row0 + row,
                            route_groups.begin() + row0 + row + tile_rows, config.m64_split);
                    row += tile_rows;
                }
                if (rows - row > 16) {
                    const int tile_rows = std::min(32, rows - row);
                    tiles_m32.push_back({ desc, row, tile_rows });
                    std::fill(route_groups.begin() + row0 + row,
                            route_groups.begin() + row0 + row + tile_rows, 4);
                    row += tile_rows;
                }
                if (row < rows) {
                    const int tile_rows = rows - row;
                    tiles_m16.push_back({ desc, row, tile_rows });
                    std::fill(route_groups.begin() + row0 + row,
                            route_groups.begin() + row0 + rows, 2);
                }
            }
        }
        if (route_cursor != route_rows) {
            throw std::runtime_error("internal route count mismatch");
        }
        std::vector<int32_t> route_down_groups = route_groups;
        if (config.down_cache == 4) {
            std::fill(route_down_groups.begin(), route_down_groups.end(), config.m64_split);
        }

        size_t free_before = 0, total_bytes = 0;
        aw_cuda_throw(cudaMemGetInfo(&free_before, &total_bytes), "cudaMemGetInfo(before)");
        const size_t expected_alloc = source_bytes + wire_bytes + 3*narrow_bytes + wide_bytes + partial_bytes +
                (size_t) descriptors*(2*q8_gate_bytes + q8_down_bytes) +
                (config.down_cache == 4 ? (size_t) descriptors*AW_EMBD*AW_EXPERT_FF*sizeof(float) : 0) +
                (config.q8_layout == "t64k32" ? (size_t) descriptors*q8_gate_bytes : 0) +
                route_input.size()*sizeof(int32_t) + token_routes.size()*sizeof(int32_t) +
                route_weights.size()*sizeof(float) +
                (tiles_m64.size() + tiles_m32.size() + tiles_m16.size())*sizeof(aw_tile_desc) +
                (size_t) descriptors*3*sizeof(aw_work_desc);
        const size_t required_headroom = 768ull*1024ull*1024ull;
        if (free_before < expected_alloc + required_headroom) {
            throw std::runtime_error("insufficient device memory for service prototype plus 768 MiB headroom");
        }

        aw_device_buffer source(source_bytes);
        aw_device_buffer wire(wire_bytes);
        aw_device_buffer gate(narrow_bytes);
        aw_device_buffer up(narrow_bytes);
        aw_device_buffer middle(narrow_bytes);
        aw_device_buffer route_output(wide_bytes);
        aw_device_buffer partial(partial_bytes);
        aw_device_buffer weights_gate((size_t) descriptors*q8_gate_bytes);
        aw_device_buffer weights_up((size_t) descriptors*q8_gate_bytes);
        aw_device_buffer weights_down((size_t) descriptors*q8_down_bytes);
        aw_device_buffer weights_down_f32(config.down_cache == 4 ?
                (size_t) descriptors*AW_EMBD*AW_EXPERT_FF*sizeof(float) : 0);
        aw_device_buffer layout_scratch(config.q8_layout == "t64k32" ?
                (size_t) descriptors*q8_gate_bytes : 0);
        aw_device_buffer route_input_dev(route_input.size()*sizeof(int32_t));
        aw_device_buffer token_routes_dev(token_routes.size()*sizeof(int32_t));
        aw_device_buffer route_weights_dev(route_weights.size()*sizeof(float));
        aw_device_buffer tiles_m64_dev(tiles_m64.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m32_dev(tiles_m32.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m16_dev(tiles_m16.size()*sizeof(aw_tile_desc));
        aw_device_buffer gate_desc_dev((size_t) descriptors*sizeof(aw_work_desc));
        aw_device_buffer up_desc_dev((size_t) descriptors*sizeof(aw_work_desc));
        aw_device_buffer down_desc_dev((size_t) descriptors*sizeof(aw_work_desc));

        for (int desc = 0; desc < descriptors; ++desc) {
            const int row0 = desc_row0[desc];
            const int rows = desc_rows[desc];
            const int layer = desc/AW_PRIMARY_PER_GPU;
            const int expert = desc % AW_PRIMARY_PER_GPU;
            gate_desc[desc] = {
                wire.get(), (const char *) weights_gate.get() + (size_t) desc*q8_gate_bytes,
                (float *) gate.get() + (size_t) row0*AW_EXPERT_FF,
                (const int32_t *) route_input_dev.get() + row0, row0, rows, layer, expert
            };
            up_desc[desc] = {
                wire.get(), (const char *) weights_up.get() + (size_t) desc*q8_gate_bytes,
                (float *) up.get() + (size_t) row0*AW_EXPERT_FF,
                (const int32_t *) route_input_dev.get() + row0, row0, rows, layer, expert
            };
            down_desc[desc] = {
                (const float *) middle.get() + (size_t) row0*AW_EXPERT_FF,
                config.down_cache == 4 ?
                        (const char *) weights_down_f32.get() +
                                (size_t) desc*AW_EMBD*AW_EXPERT_FF*sizeof(float) :
                        (const char *) weights_down.get() + (size_t) desc*q8_down_bytes,
                (float *) route_output.get() + (size_t) row0*AW_EMBD,
                nullptr, row0, rows, layer, expert
            };
        }

        cudaStream_t stream = nullptr;
        cudaEvent_t begin = nullptr, end = nullptr;
        aw_cuda_throw(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        aw_cuda_throw(cudaEventCreate(&begin), "cudaEventCreate(begin)");
        aw_cuda_throw(cudaEventCreate(&end), "cudaEventCreate(end)");

        aw_cuda_throw(cudaMemcpyAsync(route_input_dev.get(), route_input.data(), route_input_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy route input");
        aw_cuda_throw(cudaMemcpyAsync(token_routes_dev.get(), token_routes.data(), token_routes_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy token routes");
        aw_cuda_throw(cudaMemcpyAsync(route_weights_dev.get(), route_weights.data(), route_weights_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy route weights");
        if (!tiles_m64.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_m64_dev.get(), tiles_m64.data(), tiles_m64_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M64 tiles");
        }
        if (!tiles_m32.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_m32_dev.get(), tiles_m32.data(), tiles_m32_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M32 tiles");
        }
        if (!tiles_m16.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_m16_dev.get(), tiles_m16.data(), tiles_m16_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M16 tiles");
        }
        aw_cuda_throw(cudaMemcpyAsync(gate_desc_dev.get(), gate_desc.data(), gate_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy gate descriptors");
        aw_cuda_throw(cudaMemcpyAsync(up_desc_dev.get(), up_desc.data(), up_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy up descriptors");
        aw_cuda_throw(cudaMemcpyAsync(down_desc_dev.get(), down_desc.data(), down_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy down descriptors");

        aw_init_input<<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                (float *) source.get(), (size_t) tokens*AW_EMBD);
        aw_init_q8<<<65535, 256, 0, stream>>>((char *) weights_gate.get(), weights_gate.size()/AW_Q8_BLOCK_BYTES,
                1.0f/128.0f);
        aw_init_q8<<<65535, 256, 0, stream>>>((char *) weights_up.get(), weights_up.size()/AW_Q8_BLOCK_BYTES,
                1.0f/256.0f);
        aw_init_q8<<<65535, 256, 0, stream>>>((char *) weights_down.get(), weights_down.size()/AW_Q8_BLOCK_BYTES,
                1.0f/512.0f);
        aw_cuda_throw(cudaGetLastError(), "initialization kernels");
        aw_cuda_throw(cudaStreamSynchronize(stream), "initialize benchmark data");

        float layout_ms = 0.0f;
        if (config.q8_layout == "t64k32") {
            aw_cuda_throw(cudaEventRecord(begin, stream), "record layout begin");
            auto repack = [&](aw_device_buffer & weights, int n, int k) {
                const size_t blocks = weights.size()/AW_Q8_BLOCK_BYTES;
                aw_repack_q8_t64<<<aw_grid(blocks), 256, 0, stream>>>(
                        (const char *) weights.get(), (char *) layout_scratch.get(), blocks, n, k);
                aw_cuda_throw(cudaMemcpyAsync(weights.get(), layout_scratch.get(), weights.size(),
                            cudaMemcpyDeviceToDevice, stream), "install T64K32 weights");
            };
            repack(weights_gate, AW_EXPERT_FF, AW_EMBD);
            repack(weights_up, AW_EXPERT_FF, AW_EMBD);
            repack(weights_down, AW_EMBD, AW_EXPERT_FF);
            aw_cuda_throw(cudaEventRecord(end, stream), "record layout end");
            aw_cuda_throw(cudaEventSynchronize(end), "synchronize layout");
            aw_cuda_throw(cudaEventElapsedTime(&layout_ms, begin, end), "elapsed layout time");
            layout_scratch.reset();
        }

        float down_expand_ms = 0.0f;
        if (config.down_cache == 4) {
            aw_cuda_throw(cudaEventRecord(begin, stream), "record down expansion begin");
            const size_t tile_stages = (size_t) descriptors*(AW_EMBD/AW_T64_ROWS)*
                    (AW_EXPERT_FF/AW_KSTAGE);
            aw_expand_q8_t64_f32<<<(unsigned) std::min<size_t>(tile_stages, 65535), 256, 0, stream>>>(
                    (const char *) weights_down.get(), (float *) weights_down_f32.get(), tile_stages,
                    AW_EMBD, AW_EXPERT_FF);
            aw_cuda_throw(cudaEventRecord(end, stream), "record down expansion end");
            aw_cuda_throw(cudaEventSynchronize(end), "synchronize down expansion");
            aw_cuda_throw(cudaEventElapsedTime(&down_expand_ms, begin, end), "elapsed down expansion time");
            down_expand_ms /= params->active_cells;
        }

        const int n_mtiles64 = (int) tiles_m64.size();
        const int n_mtiles32 = (int) tiles_m32.size();
        const int n_mtiles16 = (int) tiles_m16.size();
        cudaDeviceProp prop;
        aw_cuda_throw(cudaGetDeviceProperties(&prop, params->device), "cudaGetDeviceProperties");
        const int persistent_blocks = prop.multiProcessorCount*2;
        auto mark_stage = [&](cudaEvent_t * marks, int index) {
            if (marks != nullptr) {
                aw_cuda_throw(cudaEventRecord(marks[index], stream), "record stage timing");
            }
        };
        auto launch_q8 = [&](const aw_device_buffer & desc_buffer, int n, int k, bool input_bf16) {
            const auto * desc_ptr = (const aw_work_desc *) desc_buffer.get();
            const auto * tiles64_ptr = (const aw_tile_desc *) tiles_m64_dev.get();
            const auto * tiles32_ptr = (const aw_tile_desc *) tiles_m32_dev.get();
            const bool t64 = config.q8_layout == "t64k32";
            if (n_mtiles64 != 0) {
                if (config.m64_split == 1) {
                    if (input_bf16) {
                        if (t64) {
                            aw_q8_service_m64<true, true, false, 1><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else {
                            aw_q8_service_m64<true, false, false, 1><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        }
                    } else if (t64) {
                        aw_q8_service_m64<false, true, false, 1><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    } else {
                        aw_q8_service_m64<false, false, false, 1><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    }
                } else {
                    if (input_bf16) {
                        if (t64) {
                            aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else {
                            aw_q8_service_m64<true, false><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        }
                    } else if (t64) {
                        aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    } else {
                        aw_q8_service_m64<false, false><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    }
                }
            }
            if (n_mtiles32 != 0) {
                if (input_bf16) {
                    if (t64) {
                        aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    } else {
                        aw_q8_service_m32<true, false><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                } else if (t64) {
                    aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                } else {
                    aw_q8_service_m32<false, false><<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                }
            }
            if (n_mtiles16 != 0) {
                const auto * tiles16_ptr = (const aw_tile_desc *) tiles_m16_dev.get();
                const int pair_blocks = prop.multiProcessorCount*3;
                if (input_bf16) {
                    if (t64) {
                        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    } else {
                        aw_q8_service_m16_pair<true, false><<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    }
                } else if (t64) {
                    aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                } else {
                    aw_q8_service_m16_pair<false, false><<<pair_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                }
            }
        };
        auto service_once = [&](cudaEvent_t * marks = nullptr) {
            mark_stage(marks, 0);
            if (bf16_wire) {
                aw_pack_requests<true><<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                        (const float *) source.get(), wire.get(), (size_t) tokens*AW_EMBD);
            } else {
                aw_pack_requests<false><<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                        (const float *) source.get(), wire.get(), (size_t) tokens*AW_EMBD);
            }
            mark_stage(marks, 1);
            launch_q8(gate_desc_dev, AW_EXPERT_FF, AW_EMBD, bf16_wire);
            mark_stage(marks, 2);
            launch_q8(up_desc_dev, AW_EXPERT_FF, AW_EMBD, bf16_wire);
            mark_stage(marks, 3);
            aw_swiglu<<<aw_grid((size_t) route_rows*AW_EXPERT_FF), 256, 0, stream>>>(
                    (const float *) gate.get(), (const float *) up.get(), (float *) middle.get(),
                    (size_t) route_rows*AW_EXPERT_FF);
            mark_stage(marks, 4);
            if (config.down_cache == 4) {
                const auto * desc_ptr = (const aw_work_desc *) down_desc_dev.get();
                if (n_mtiles64 != 0) {
                    if (config.m64_split == 1) {
                        aw_q8_service_m64<false, false, true, 1><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m64_dev.get(), n_mtiles64,
                                AW_EMBD, AW_EXPERT_FF);
                    } else {
                        aw_q8_service_m64<false, false, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m64_dev.get(), n_mtiles64,
                                AW_EMBD, AW_EXPERT_FF);
                    }
                }
                if (n_mtiles32 != 0) {
                    if (config.m64_split == 1) {
                        aw_q8_service_m64<false, false, true, 1><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m32_dev.get(), n_mtiles32,
                                AW_EMBD, AW_EXPERT_FF);
                    } else {
                        aw_q8_service_m64<false, false, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m32_dev.get(), n_mtiles32,
                                AW_EMBD, AW_EXPERT_FF);
                    }
                }
                if (n_mtiles16 != 0) {
                    if (config.m64_split == 1) {
                        aw_q8_service_m64<false, false, true, 1><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m16_dev.get(), n_mtiles16,
                                AW_EMBD, AW_EXPERT_FF);
                    } else {
                        aw_q8_service_m64<false, false, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m16_dev.get(), n_mtiles16,
                                AW_EMBD, AW_EXPERT_FF);
                    }
                }
            } else {
                launch_q8(down_desc_dev, AW_EMBD, AW_EXPERT_FF, false);
            }
            mark_stage(marks, 5);
            aw_owner_reduce_bf16<<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                    (const float *) route_output.get(), (const int32_t *) token_routes_dev.get(),
                    (const float *) route_weights_dev.get(), (uint16_t *) partial.get(), tokens, AW_EMBD);
            mark_stage(marks, 6);
        };

        for (int warmup = 0; warmup < 2; ++warmup) {
            service_once();
        }
        aw_cuda_throw(cudaGetLastError(), "service warmup kernels");
        aw_cuda_throw(cudaStreamSynchronize(stream), "service warmup");
        if (!aw_bench_barrier.arrive_and_wait()) {
            throw std::runtime_error("four-GPU benchmark barrier was cancelled");
        }
        aw_cuda_throw(cudaEventRecord(begin, stream), "record begin");
        for (int repeat = 0; repeat < params->repeats; ++repeat) {
            service_once();
        }
        aw_cuda_throw(cudaEventRecord(end, stream), "record end");
        aw_cuda_throw(cudaEventSynchronize(end), "synchronize benchmark");
        float total_ms = 0.0f;
        aw_cuda_throw(cudaEventElapsedTime(&total_ms, begin, end), "elapsed time");
        aw_cuda_throw(cudaGetLastError(), "service kernels");

        if (!aw_bench_barrier.arrive_and_wait()) {
            throw std::runtime_error("four-GPU stage barrier was cancelled");
        }
        std::vector<cudaEvent_t> stage_events((size_t) params->repeats*7);
        for (cudaEvent_t & event : stage_events) {
            aw_cuda_throw(cudaEventCreate(&event), "cudaEventCreate(stage)");
        }
        for (int repeat = 0; repeat < params->repeats; ++repeat) {
            service_once(stage_events.data() + repeat*7);
        }
        aw_cuda_throw(cudaEventSynchronize(stage_events.back()), "synchronize stage timings");
        std::array<float, 6> stage_ms{};
        for (int repeat = 0; repeat < params->repeats; ++repeat) {
            for (int stage = 0; stage < 6; ++stage) {
                float value = 0.0f;
                aw_cuda_throw(cudaEventElapsedTime(&value, stage_events[repeat*7 + stage],
                            stage_events[repeat*7 + stage + 1]), "stage elapsed time");
                stage_ms[stage] += value/params->repeats;
            }
        }
        float instrumented_total_ms = 0.0f;
        aw_cuda_throw(cudaEventElapsedTime(&instrumented_total_ms, stage_events.front(), stage_events.back()),
                "instrumented elapsed time");
        instrumented_total_ms /= params->repeats;

        uint16_t observed = 0;
        uint16_t expected = 0;
        int check_passed = 1;
        uint64_t check_mismatches = 0;
        uint64_t check_first_mismatch = std::numeric_limits<uint64_t>::max();
        const size_t check_count = (size_t) tokens*AW_EMBD;
        if (config.check) {
            std::vector<uint16_t> output(check_count);
            aw_cuda_throw(cudaMemcpy(output.data(), partial.get(), partial_bytes, cudaMemcpyDeviceToHost),
                    "copy full check result");
            float reference[17][3][3] = {};
            constexpr std::array<int, 3> split_groups = { 1, 2, 4 };
            for (int residue = 0; residue < 17; ++residue) {
                for (int gate_group = 0; gate_group < 3; ++gate_group) {
                    for (int down_group = 0; down_group < 3; ++down_group) {
                        reference[residue][gate_group][down_group] = aw_expected_route(
                                residue, bf16_wire, split_groups[gate_group], split_groups[down_group]);
                    }
                }
            }
            for (int token = 0; token < tokens; ++token) {
                float sum = 0.0f;
                for (int rank = 0; rank < AW_ROUTES_PER_OWNER_TOKEN; ++rank) {
                    const int route = token_routes[token*AW_ROUTES_PER_OWNER_TOKEN + rank];
                    const int gate_index = route_groups[route] == 1 ? 0 : route_groups[route] == 2 ? 1 : 2;
                    const int down_index = route_down_groups[route] == 1 ? 0 :
                            route_down_groups[route] == 2 ? 1 : 2;
                    sum += 0.5f*reference[token % 17][gate_index][down_index];
                }
                const uint16_t token_expected = aw_float_to_bf16_host(sum);
                if (token == 0) {
                    expected = token_expected;
                }
                const size_t row0 = (size_t) token*AW_EMBD;
                for (int col = 0; col < AW_EMBD; ++col) {
                    const size_t index = row0 + col;
                    if (output[index] != token_expected) {
                        if (check_mismatches == 0) {
                            check_first_mismatch = index;
                        }
                        ++check_mismatches;
                    }
                }
            }
            observed = output[0];
            check_passed = check_mismatches == 0;
        }

        size_t free_after = 0;
        aw_cuda_throw(cudaMemGetInfo(&free_after, &total_bytes), "cudaMemGetInfo(after)");
        const double service_ms = total_ms/params->repeats;
        const double elapsed_ms = service_ms + down_expand_ms;
        const double flops = 6.0*(double) route_rows*AW_EMBD*AW_EXPERT_FF;
        const double gate_issued_rows = (double) tiles_m64.size()*64 + (double) tiles_m32.size()*32 +
                (double) tiles_m16.size()*16;
        const double down_issued_rows = (double) tiles_m64.size()*64 +
                (double) tiles_m32.size()*(config.down_cache == 4 ? 64 : 32) +
                (double) tiles_m16.size()*(config.down_cache == 4 ? 64 : 16);
        const double issued_flops = 2.0*AW_EMBD*AW_EXPERT_FF*(2.0*gate_issued_rows + down_issued_rows);
        result->elapsed_ms = elapsed_ms;
        result->service_ms = service_ms;
        result->instrumented_ms = instrumented_total_ms;
        result->stage_sum_ms = std::accumulate(stage_ms.begin(), stage_ms.end(), 0.0);
        result->effective_tflops = flops/(elapsed_ms*1.0e9);
        result->issued_tflops = issued_flops/(elapsed_ms*1.0e9);
        result->useful_fraction = flops/issued_flops;
        result->request_gib_s = source_bytes/(service_ms*1.0e-3)/(1024.0*1024.0*1024.0);
        result->pack_ms = stage_ms[0];
        result->gate_ms = stage_ms[1];
        result->up_ms = stage_ms[2];
        result->swiglu_ms = stage_ms[3];
        result->down_ms = stage_ms[4];
        result->reduce_ms = stage_ms[5];
        result->layout_ms = layout_ms;
        result->down_expand_ms = down_expand_ms;
        result->device_bytes = expected_alloc;
        result->free_bytes_after = free_after;
        result->descriptors = descriptors;
        result->route_rows = route_rows;
        result->tiles_m64 = n_mtiles64;
        result->tiles_m32 = n_mtiles32;
        result->tiles_m16 = n_mtiles16;
        result->check_ran = config.check ? 1 : 0;
        result->check_passed = check_passed;
        result->check_count = config.check ? check_count : 0;
        result->check_mismatches = check_mismatches;
        result->check_first_mismatch = check_first_mismatch;
        result->check_expected = expected;
        result->check_observed = observed;

        aw_cuda_throw(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
        aw_cuda_throw(cudaEventDestroy(end), "cudaEventDestroy(end)");
        for (cudaEvent_t event : stage_events) {
            aw_cuda_throw(cudaEventDestroy(event), "cudaEventDestroy(stage)");
        }
        aw_cuda_throw(cudaStreamDestroy(stream), "cudaStreamDestroy");
        if (!check_passed) {
            throw std::runtime_error("BF16 owner-partial accuracy check failed");
        }
        return 0;
    } catch (const std::exception & exception) {
        aw_bench_barrier.cancel();
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}
