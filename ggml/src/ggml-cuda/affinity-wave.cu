// [TAG_AFFINITY_WAVE] Phase-1 cross-layer expert service for PCIe Pascal.
//
// This file deliberately stops at the isolated service boundary.  It proves
// the native-Q8 compute and owner-wire dataflow before the much more invasive
// chunk/layer scheduler is allowed to replace the production EP4 graph.

#include "affinity-wave.cuh"
#include "common.cuh"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

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

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif

#if defined(__linux__)
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

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
constexpr int AW_COHORTRAIL_MAX_TILES  = AW_GPU_COUNT*AW_PRIMARY_PER_GPU;
constexpr int AW_EMBD                  = 2048;
constexpr int AW_EXPERT_FF             = 512;
constexpr int AW_SERVICE_PANEL_N       = 256;
constexpr int AW_SERVICE_GU_PANELS     = AW_EXPERT_FF/AW_SERVICE_PANEL_N;
constexpr int AW_SERVICE_DOWN_PANELS   = AW_EMBD/AW_SERVICE_PANEL_N;
constexpr int AW_DIAGONAL_MIN_PANEL_N  = 512;
constexpr int AW_DIAGONAL_MAX_PANELS   =
        AW_EMBD/AW_DIAGONAL_MIN_PANEL_N;
constexpr int AW_DIAGONAL_INPUT_SLOTS  = AW_GPU_COUNT;
constexpr int AW_DIAGONAL_RING_SLOTS   = 2;
constexpr int AW_R44_BUDGET            = 44;
constexpr int AW_R44_SLOTS             = 2;
constexpr int AW_GDN_HEAD_DIM          = 128;
constexpr int AW_GDN_QK_HEADS          = 16;
constexpr int AW_GDN_VALUE_HEADS       = 32;
constexpr int AW_HEADFOLD_QK_HEADS     = 4;
constexpr int AW_HEADFOLD_VALUE_HEADS  = 8;
constexpr int AW_HEADFOLD_PARTS        = 4;

struct aw_config {
    std::string map;
    std::string wire;
    std::string q8_layout;
    std::string q8_kernel;
    std::string q8_engine;
    std::string home;
    std::string gguf_sha256;
    int down_cache = 0;
    int m64_split = 2;
    bool check = false;
    bool dense_t64 = false;
    bool qkv_conv = false;
};

class aw_process_barrier {
public:
    bool arrive_and_wait(int participants) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (broken_ || participants < 1 || participants > AW_GPU_COUNT) {
            return false;
        }
        const int generation = generation_;
        if (arrived_ == 0) {
            participants_ = participants;
        } else if (participants_ != participants) {
            broken_ = true;
            condition_.notify_all();
            return false;
        }
        if (++arrived_ == participants_) {
            arrived_ = 0;
            participants_ = 0;
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
    int participants_ = 0;
    bool broken_ = false;
};

static aw_process_barrier aw_bench_barrier;

struct aw_relay_bench_coord {
    void * local[AW_GPU_COUNT][AW_GPU_COUNT] = {};
    void * gathered[AW_GPU_COUNT][AW_GPU_COUNT] = {};
    cudaEvent_t sent[AW_GPU_COUNT][AW_GPU_COUNT][AW_GPU_COUNT] = {};
    cudaEvent_t consumed[AW_GPU_COUNT][AW_GPU_COUNT] = {};
};

static aw_relay_bench_coord aw_relay_coord;

struct aw_npanel_bench_coord {
    void * recv[AW_GPU_COUNT] = {};
    cudaEvent_t ready[AW_GPU_COUNT][16] = {};
    cudaEvent_t copied[AW_GPU_COUNT][AW_GPU_COUNT][16] = {};
};

static aw_npanel_bench_coord aw_npanel_coord;
static std::mutex aw_layout_mutex;
static std::unordered_set<const void *> aw_t64_tensors;
static std::array<std::atomic<int>, AW_GPU_COUNT> aw_t64_tensor_counts{};
static std::array<int, AW_GPU_COUNT> aw_sm_counts{};

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

static bool aw_p100_exact_enabled() {
    if (!aw_env_on(getenv("GGML_CUDA_AW_P100_EXACT"))) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    const ggml_cuda_device_info & info = ggml_cuda_info();
    return device >= 0 && device < info.device_count &&
            info.devices[device].cc == GGML_CUDA_CC_PASCAL;
}

static int aw_p100_exact_sum_width() {
    const char * value =
            getenv("GGML_CUDA_AW_P100_EXACT_SUM_WIDTH");
    return value != nullptr ? atoi(value) : 4;
}

static nvtxDomainHandle_t aw_trace_domain() {
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("affinitywave");
    return domain;
}

static uint32_t aw_trace_color(uint32_t category) {
    static constexpr uint32_t colors[] = {
        0xff4e79a7, 0xfff28e2b, 0xffe15759, 0xff76b7b2,
        0xff59a14f, 0xffedc949, 0xffaf7aa1, 0xffff9da7,
    };
    return colors[category % (sizeof(colors)/sizeof(colors[0]))];
}

static void aw_trace_event(
        const char * name, uint32_t category, uint64_t payload, bool push) {
    nvtxEventAttributes_t attributes = {};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.category = category;
    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = aw_trace_color(category);
    attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attributes.payload.ullValue = payload;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    if (push) {
        nvtxDomainRangePushEx(aw_trace_domain(), &attributes);
    } else {
        nvtxDomainMarkEx(aw_trace_domain(), &attributes);
    }
}

class aw_trace_scope {
public:
    aw_trace_scope(const char * name, uint32_t category, uint64_t payload) :
        active(ggml_cuda_affinity_wave_trace_enabled()) {
        if (active) {
            aw_trace_event(name, category, payload, true);
        }
    }

    ~aw_trace_scope() {
        if (active) {
            nvtxDomainRangePop(aw_trace_domain());
        }
    }

private:
    bool active;
};

static bool aw_q8_interleave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_KERNEL");
    return value != nullptr && strcmp(value, "interleave") == 0;
}

static bool aw_q8_warpwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "warpwave") == 0;
}

static bool aw_q8_compact_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "compact") == 0;
}

static bool aw_q8_hybrid_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "hybrid") == 0;
}

static bool aw_q8_tailwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "tailwave") == 0;
}

static bool aw_q8_widewave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "widewave") == 0;
}

static bool aw_q8_broadwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "broadwave") == 0;
}

static bool aw_q8_halfpipe_sync_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "halfpipe_sync") == 0;
}

static bool aw_q8_halfpipe_bar_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "halfpipe_bar") == 0;
}

static bool aw_q8_cohortrail_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "cohortrail") == 0;
}

static int aw_q8_cohortrail_p2_ctas() {
    const char * value =
            getenv("GGML_CUDA_AW_COHORTRAIL_P2_CTAS");
    return value != nullptr ? atoi(value) : 3;
}

static int aw_q8_cohortrail_single_ctas() {
    const char * value =
            getenv("GGML_CUDA_AW_COHORTRAIL_SINGLE_CTAS");
    return value != nullptr ? atoi(value) : 3;
}

static bool aw_q8_doublewave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "doublewave") == 0;
}

static bool aw_q8_dualrailwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "dualrailwave") == 0;
}

static bool aw_q8_flexwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "flexwave") == 0;
}

static bool aw_q8_flexwave1_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "flexwave1") == 0;
}

static bool aw_q8_rendezvous_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "rendezvous") == 0;
}

static bool aw_q8_megawave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "megawave") == 0;
}

static bool aw_q8_fabwave_enabled() {
    const char * value = getenv("GGML_CUDA_AW_Q8_ENGINE");
    return value != nullptr && strcmp(value, "fabwave") == 0;
}

static bool aw_q8_fabwave_k8_enabled() {
    const char * value = getenv("GGML_CUDA_AW_FAB_K8");
    return value != nullptr && strcmp(value, "1") == 0;
}

static bool aw_q8_fabwave_qscale_enabled() {
    const char * value = getenv("GGML_CUDA_AW_FAB_QSCALE");
    return value != nullptr && strcmp(value, "1") == 0;
}

static bool aw_q8_fabwave_dual_enabled() {
    const char * value = getenv("GGML_CUDA_AW_FAB_DUAL");
    return value != nullptr && strcmp(value, "1") == 0;
}

static bool aw_q8_fabwave_exact_projection(int n, int k) {
    const char * value = getenv("GGML_CUDA_AW_FAB_EXACT");
    if (value == nullptr) {
        return false;
    }
    if (strcmp(value, "down") == 0) {
        return n == AW_EMBD && k == AW_EXPERT_FF;
    }
    if (strcmp(value, "gateup") == 0) {
        return n == AW_EXPERT_FF && k == AW_EMBD;
    }
    return false;
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
    config.q8_kernel = getenv("GGML_CUDA_AW_Q8_KERNEL") != nullptr ?
            getenv("GGML_CUDA_AW_Q8_KERNEL") : "cuda";
    config.q8_engine = getenv("GGML_CUDA_AW_Q8_ENGINE") != nullptr ?
            getenv("GGML_CUDA_AW_Q8_ENGINE") : "legacy";
    config.home = getenv("GGML_CUDA_AW_HOME") != nullptr ? getenv("GGML_CUDA_AW_HOME") : "chunk";
    config.check = aw_env_on(getenv("GGML_CUDA_AW_CHECK"));
    const char * dense_t64 = getenv("GGML_CUDA_AW_DENSE_T64");
    if (dense_t64 != nullptr && strcmp(dense_t64, "0") != 0 && strcmp(dense_t64, "1") != 0) {
        error = "GGML_CUDA_AW_DENSE_T64 must be 0 or 1";
        return false;
    }
    config.dense_t64 = dense_t64 != nullptr && strcmp(dense_t64, "1") == 0;
    const char * dense_selectors =
            getenv("GGML_CUDA_AW_DENSE_SELECTORS");
    if (dense_selectors != nullptr &&
            strcmp(dense_selectors, "0") != 0 &&
            strcmp(dense_selectors, "exact") != 0) {
        error = "GGML_CUDA_AW_DENSE_SELECTORS must be '0' or 'exact'";
        return false;
    }
    const char * qkv_conv = getenv("GGML_CUDA_AW_QKV_CONV");
    if (qkv_conv != nullptr && strcmp(qkv_conv, "0") != 0 && strcmp(qkv_conv, "1") != 0) {
        error = "GGML_CUDA_AW_QKV_CONV must be 0 or 1";
        return false;
    }
    config.qkv_conv = qkv_conv != nullptr && strcmp(qkv_conv, "1") == 0;
    if (config.wire != "f32" && config.wire != "bf16" && config.wire != "f16") {
        error = "GGML_CUDA_AW_WIRE must be 'f32', 'bf16', or 'f16'";
        return false;
    }
    if (config.q8_layout != "native" && config.q8_layout != "t64k32") {
        error = "GGML_CUDA_AW_Q8_LAYOUT must be 'native' or 't64k32'";
        return false;
    }
    if (config.q8_kernel != "cuda" && config.q8_kernel != "interleave") {
        error = "GGML_CUDA_AW_Q8_KERNEL must be 'cuda' or 'interleave'";
        return false;
    }
    if (config.q8_engine != "legacy" && config.q8_engine != "warpwave" &&
            config.q8_engine != "compact" && config.q8_engine != "hybrid" &&
            config.q8_engine != "tailwave" && config.q8_engine != "widewave" &&
            config.q8_engine != "broadwave" && config.q8_engine != "vectorwave" &&
            config.q8_engine != "halfpipe_sync" && config.q8_engine != "halfpipe_bar" &&
            config.q8_engine != "cohortrail" &&
            config.q8_engine != "crestwave" &&
            config.q8_engine != "doublewave" &&
            config.q8_engine != "dualrailwave" &&
            config.q8_engine != "expandwave" &&
            config.q8_engine != "regbwave" &&
            config.q8_engine != "railwave" &&
            config.q8_engine != "flexwave" &&
            config.q8_engine != "flexwave1" && config.q8_engine != "rendezvous" &&
            config.q8_engine != "megawave" && config.q8_engine != "fabwave") {
        error = "GGML_CUDA_AW_Q8_ENGINE must be 'legacy', 'warpwave', 'compact', 'hybrid', 'tailwave', 'widewave', 'broadwave', 'vectorwave', 'halfpipe_sync', 'halfpipe_bar', 'cohortrail', 'doublewave', 'dualrailwave', 'crestwave', 'regbwave', 'expandwave', 'railwave', 'flexwave', 'flexwave1', 'rendezvous', 'megawave', or 'fabwave'";
        return false;
    }
    const char * cohortrail_p2_ctas =
            getenv("GGML_CUDA_AW_COHORTRAIL_P2_CTAS");
    if (cohortrail_p2_ctas != nullptr &&
            (atoi(cohortrail_p2_ctas) < 1 ||
             atoi(cohortrail_p2_ctas) > 6)) {
        error = "GGML_CUDA_AW_COHORTRAIL_P2_CTAS must be between 1 and 6";
        return false;
    }
    const char * cohortrail_single_ctas =
            getenv("GGML_CUDA_AW_COHORTRAIL_SINGLE_CTAS");
    if (cohortrail_single_ctas != nullptr &&
            (atoi(cohortrail_single_ctas) < 1 ||
             atoi(cohortrail_single_ctas) > 6)) {
        error = "GGML_CUDA_AW_COHORTRAIL_SINGLE_CTAS must be between 1 and 6";
        return false;
    }
    if (config.q8_kernel == "interleave" && config.q8_layout != "t64k32") {
        error = "GGML_CUDA_AW_Q8_KERNEL=interleave requires GGML_CUDA_AW_Q8_LAYOUT=t64k32";
        return false;
    }
    if (config.q8_engine != "legacy" && config.q8_layout != "t64k32") {
        error = "GGML_CUDA_AW_Q8_ENGINE requires GGML_CUDA_AW_Q8_LAYOUT=t64k32";
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
    if (config.q8_engine != "legacy" && config.down_cache != 0) {
        error = "non-legacy GGML_CUDA_AW_Q8_ENGINE requires GGML_CUDA_AW_DOWN_CACHE=0";
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
    if (config.q8_kernel == "interleave" && config.m64_split != 2) {
        error = "GGML_CUDA_AW_Q8_KERNEL=interleave requires GGML_CUDA_AW_M64_SPLIT=2";
        return false;
    }
    if (config.q8_kernel == "interleave" && aw_env_on(getenv("GGML_CUDA_AW_FUSED_GATE_UP"))) {
        error = "GGML_CUDA_AW_Q8_KERNEL=interleave does not support GGML_CUDA_AW_FUSED_GATE_UP";
        return false;
    }
    if (config.q8_engine != "legacy" && aw_env_on(getenv("GGML_CUDA_AW_FUSED_GATE_UP"))) {
        error = "non-legacy GGML_CUDA_AW_Q8_ENGINE does not support GGML_CUDA_AW_FUSED_GATE_UP";
        return false;
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
        aw_sm_counts[device] = prop.multiProcessorCount;
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

struct aw_m16_cohort {
    int32_t tiles[4];
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
        const uintptr_t tagged = (uintptr_t) input;
        const uint16_t * input16 = (const uint16_t *) (tagged & ~(uintptr_t) 1);
        const uint16_t value = input16[index];
        return (tagged & 1) != 0 ?
                __half2float(__ushort_as_half(value)) : aw_bf16_to_float(value);
    } else {
        return ((const float *) input)[index];
    }
}

__device__ __forceinline__ void aw_load_bf16x8(
        const void * input, size_t index, float4 & lo, float4 & hi) {
    const uintptr_t tagged = (uintptr_t) input;
    const uint16_t * input16 = (const uint16_t *) (tagged & ~(uintptr_t) 1);
    const bool fp16 = (tagged & 1) != 0;
    const uint4 packed = *(const uint4 *) (input16 + index);
    const uint32_t words[4] = { packed.x, packed.y, packed.z, packed.w };
    float * lo_values = (float *) &lo;
    float * hi_values = (float *) &hi;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint16_t lo0 = (uint16_t) words[i];
        const uint16_t lo1 = (uint16_t) (words[i] >> 16);
        const uint16_t hi0 = (uint16_t) words[i + 2];
        const uint16_t hi1 = (uint16_t) (words[i + 2] >> 16);
        lo_values[2*i] = fp16 ? __half2float(__ushort_as_half(lo0)) : aw_bf16_to_float(lo0);
        lo_values[2*i + 1] = fp16 ? __half2float(__ushort_as_half(lo1)) : aw_bf16_to_float(lo1);
        hi_values[2*i] = fp16 ? __half2float(__ushort_as_half(hi0)) : aw_bf16_to_float(hi0);
        hi_values[2*i + 1] = fp16 ? __half2float(__ushort_as_half(hi1)) : aw_bf16_to_float(hi1);
    }
}

__device__ __forceinline__ float4 aw_load_bf16x4(
        const void * input, size_t index) {
    const uintptr_t tagged = (uintptr_t) input;
    const uint16_t * input16 = (const uint16_t *) (tagged & ~(uintptr_t) 1);
    const bool fp16 = (tagged & 1) != 0;
    const uint2 packed = *(const uint2 *) (input16 + index);
    const uint32_t words[2] = { packed.x, packed.y };
    float4 result;
    float * values = (float *) &result;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint16_t lo = (uint16_t) words[i];
        const uint16_t hi = (uint16_t) (words[i] >> 16);
        values[2*i] = fp16 ? __half2float(__ushort_as_half(lo)) : aw_bf16_to_float(lo);
        values[2*i + 1] = fp16 ? __half2float(__ushort_as_half(hi)) : aw_bf16_to_float(hi);
    }
    return result;
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 3)
static void aw_q8_warpwave(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ float warp_a[8][16][AW_KSTAGE];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;

    for (int work_id = blockIdx.x*8 + warp; work_id < n_work;
            work_id += gridDim.x*8) {
        const int mt_id = work_id/n_ntiles;
        const int ntile = work_id - mt_id*n_ntiles;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        float acc0[16] = {};
        float acc1[16] = {};

        for (int stage = 0; stage < k/AW_KSTAGE; ++stage) {
            #pragma unroll
            for (int row = 0; row < 16; ++row) {
                float value = 0.0f;
                if (row < tile.rows) {
                    const int local_row = tile.row + row;
                    const int input_row = desc.input_rows != nullptr ?
                            desc.input_rows[local_row] : local_row;
                    value = aw_load_input<BF16_INPUT>(
                            desc.input, (size_t) input_row*k + stage*AW_KSTAGE + lane);
                }
                warp_a[warp][row][lane] = value;
            }
            __syncwarp();

            const int n_kblocks = k/AW_KSTAGE;
            const char * tile_stage = desc.weight +
                    ((size_t) ntile*n_kblocks + stage)*AW_T64_STAGE_BYTES;
            const float scale0 = __half2float(
                    *(const half *) (tile_stage + lane*2));
            const float scale1 = __half2float(
                    *(const half *) (tile_stage + (lane + 32)*2));
            const char * values = tile_stage + AW_T64_ROWS*2;
            const char * values0 = values + lane*AW_KSTAGE;
            const char * values1 = values + (lane + 32)*AW_KSTAGE;

            #pragma unroll
            for (int group = 0; group < AW_KSTAGE/4; ++group) {
                const uint32_t packed0 = *(const uint32_t *) (values0 + group*4);
                const uint32_t packed1 = *(const uint32_t *) (values1 + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = byte*8;
                    const float b0 = (float) (signed char) ((packed0 >> shift) & 0xffu)*scale0;
                    const float b1 = (float) (signed char) ((packed1 >> shift) & 0xffu)*scale1;
                    const int kk = group*4 + byte;
                    #pragma unroll
                    for (int row = 0; row < 16; ++row) {
                        const float a = warp_a[warp][row][kk];
                        acc0[row] += a*b0;
                        acc1[row] += a*b1;
                    }
                }
            }
            __syncwarp();
        }

        #pragma unroll
        for (int row = 0; row < 16; ++row) {
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                output[lane] = acc0[row];
                output[lane + 32] = acc1[row];
            }
        }
    }
}

template<bool BF16_INPUT, int ROWS>
__global__ __launch_bounds__(256, 4)
static void aw_q8_tailwave(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    static_assert(ROWS == 4 || ROWS == 8);
    __shared__ float a_stage[ROWS][AW_KSTAGE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_ngroups = n_ntiles/8;
    const int n_work = n_mtiles*n_ngroups;

    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ntile = (work_id - mt_id*n_ngroups)*8 + warp;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        float acc0[ROWS] = {};
        float acc1[ROWS] = {};

        for (int stage = 0; stage < k/AW_KSTAGE; ++stage) {
            if (tid < ROWS*AW_KSTAGE) {
                const int row = tid/AW_KSTAGE;
                const int kk = tid - row*AW_KSTAGE;
                float value = 0.0f;
                if (row < tile.rows) {
                    const int local_row = tile.row + row;
                    const int input_row = desc.input_rows != nullptr ?
                            desc.input_rows[local_row] : local_row;
                    value = aw_load_input<BF16_INPUT>(
                            desc.input, (size_t) input_row*k + stage*AW_KSTAGE + kk);
                }
                a_stage[row][kk] = value;
            }
            __syncthreads();

            const int n_kblocks = k/AW_KSTAGE;
            const char * tile_stage = desc.weight +
                    ((size_t) ntile*n_kblocks + stage)*AW_T64_STAGE_BYTES;
            const float scale0 = __half2float(
                    *(const half *) (tile_stage + lane*2));
            const float scale1 = __half2float(
                    *(const half *) (tile_stage + (lane + 32)*2));
            const char * values = tile_stage + AW_T64_ROWS*2;
            const char * values0 = values + lane*AW_KSTAGE;
            const char * values1 = values + (lane + 32)*AW_KSTAGE;

            #pragma unroll
            for (int group = 0; group < AW_KSTAGE/4; ++group) {
                const uint32_t packed0 = *(const uint32_t *) (values0 + group*4);
                const uint32_t packed1 = *(const uint32_t *) (values1 + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = byte*8;
                    const float b0 = (float) (signed char) ((packed0 >> shift) & 0xffu)*scale0;
                    const float b1 = (float) (signed char) ((packed1 >> shift) & 0xffu)*scale1;
                    const int kk = group*4 + byte;
                    #pragma unroll
                    for (int row = 0; row < ROWS; ++row) {
                        const float a = a_stage[row][kk];
                        acc0[row] += a*b0;
                        acc1[row] += a*b1;
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                output[lane] = acc0[row];
                output[lane + 32] = acc1[row];
            }
        }
        __syncthreads();
    }
}

template<bool BF16_INPUT, bool SPLIT_K>
__global__ __launch_bounds__(256, SPLIT_K ? 2 : 3)
static void aw_q8_flexwave16(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ float a_stage[2][16][AW_KSTAGE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_ngroups = n_ntiles/8;
    const int n_work = n_mtiles*n_ngroups;

    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ntile = (work_id - mt_id*n_ngroups)*8 + warp;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];

        auto stage_a = [&](int stage, int buffer) {
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int index = tid + part*blockDim.x;
                const int row = index/AW_KSTAGE;
                const int kk = index - row*AW_KSTAGE;
                float value = 0.0f;
                if (row < tile.rows) {
                    const int local_row = tile.row + row;
                    const int input_row = desc.input_rows != nullptr ?
                            desc.input_rows[local_row] : local_row;
                    value = aw_load_input<BF16_INPUT>(
                            desc.input, (size_t) input_row*k + stage*AW_KSTAGE + kk);
                }
                a_stage[buffer][row][kk] = value;
            }
        };

        float acc0[16] = {};
        float acc1[16] = {};
        float acc0_hi[16] = {};
        float acc1_hi[16] = {};
        int buffer = 0;
        stage_a(0, buffer);
        __syncthreads();

        for (int stage = 0; stage < k/AW_KSTAGE; ++stage) {
            const bool last = stage + 1 == k/AW_KSTAGE;
            if (!last) {
                stage_a(stage + 1, buffer ^ 1);
            }

            const int n_kblocks = k/AW_KSTAGE;
            const char * tile_stage = desc.weight +
                    ((size_t) ntile*n_kblocks + stage)*AW_T64_STAGE_BYTES;
            const float scale0 = __half2float(*(const half *) (tile_stage + lane*2));
            const float scale1 = __half2float(
                    *(const half *) (tile_stage + (lane + 32)*2));
            const char * values = tile_stage + AW_T64_ROWS*2;
            const char * values0 = values + lane*AW_KSTAGE;
            const char * values1 = values + (lane + 32)*AW_KSTAGE;

            #pragma unroll
            for (int group = 0; group < AW_KSTAGE/4; ++group) {
                const uint32_t packed0 = *(const uint32_t *) (values0 + group*4);
                const uint32_t packed1 = *(const uint32_t *) (values1 + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = byte*8;
                    const float b0 = (float) (signed char) ((packed0 >> shift) & 0xffu)*scale0;
                    const float b1 = (float) (signed char) ((packed1 >> shift) & 0xffu)*scale1;
                    const int kk = group*4 + byte;
                    if constexpr (!SPLIT_K) {
                        #pragma unroll
                        for (int row = 0; row < 16; ++row) {
                            const float a = a_stage[buffer][row][kk];
                            acc0[row] += a*b0;
                            acc1[row] += a*b1;
                        }
                    } else if (group < AW_KSTAGE/8) {
                        #pragma unroll
                        for (int row = 0; row < 16; ++row) {
                            const float a = a_stage[buffer][row][kk];
                            acc0[row] += a*b0;
                            acc1[row] += a*b1;
                        }
                    } else {
                        #pragma unroll
                        for (int row = 0; row < 16; ++row) {
                            const float a = a_stage[buffer][row][kk];
                            acc0_hi[row] += a*b0;
                            acc1_hi[row] += a*b1;
                        }
                    }
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        #pragma unroll
        for (int row = 0; row < 16; ++row) {
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                if constexpr (SPLIT_K) {
                    output[lane] = acc0[row] + acc0_hi[row];
                    output[lane + 32] = acc1[row] + acc1_hi[row];
                } else {
                    output[lane] = acc0[row];
                    output[lane + 32] = acc1[row];
                }
            }
        }
        __syncthreads();
    }
}

template<int ROWS>
union aw_smem_compact {
    struct {
        float A[AW_KSTAGE][ROWS];
        float B[AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[ROWS][AW_T64_ROWS];
};

template<bool BF16_INPUT, int ROWS>
__global__ __launch_bounds__(256, 3)
static void aw_q8_compact(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    static_assert(ROWS == 16 || ROWS == 32 || ROWS == 64);
    __shared__ aw_smem_compact<ROWS> sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int ROW_GROUPS = ROWS/16;
    constexpr int B_GROUPS = ROWS/16;
    constexpr int B_VALUES = AW_KSTAGE/B_GROUPS;
    const int kgrp = warp/ROW_GROUPS;
    const int wm = warp - kgrp*ROW_GROUPS;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid/B_GROUPS;
    const int b_group = tid - b_r*B_GROUPS;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int ntile = work_id - mt_id*n_ntiles;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        float acc[4][8] = {};

        for (int stage = 0; stage < k/AW_KSTAGE; ++stage) {
            float4 pa0 = {};
            float4 pa1 = {};
            if (a_r < tile.rows) {
                const int local_row = tile.row + a_r;
                const int input_row = desc.input_rows != nullptr ?
                        desc.input_rows[local_row] : local_row;
                const size_t base = (size_t) input_row*k + stage*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, base, pa0, pa1);
                } else {
                    const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                    pa0 = input4[0];
                    pa1 = input4[1];
                }
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[a_k + i][a_r] = a0[i];
                sm.stage.A[a_k + i + 4][a_r] = a1[i];
            }

            const int n_kblocks = k/AW_KSTAGE;
            const char * tile_stage = desc.weight +
                    ((size_t) ntile*n_kblocks + stage)*AW_T64_STAGE_BYTES;
            const float scale = __half2float(*(const half *) (tile_stage + b_r*2));
            const char * values = tile_stage + AW_T64_ROWS*2 + b_r*AW_KSTAGE +
                    b_group*B_VALUES;
            #pragma unroll
            for (int group = 0; group < B_VALUES/4; ++group) {
                const uint32_t packed = *(const uint32_t *) (values + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int kk = b_group*B_VALUES + group*4 + byte;
                    sm.stage.B[kk][b_r] =
                            (float) (signed char) ((packed >> (byte*8)) & 0xffu)*scale;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const int ks = kgrp*(AW_KSTAGE/2) + kk;
                const int row0 = wm*16 + rm*4;
                const float4 av = *(const float4 *) &sm.stage.A[ks][row0];
                const float4 bv0 = *(const float4 *) &sm.stage.B[ks][nt*8];
                const float4 bv1 = *(const float4 *) &sm.stage.B[ks][nt*8 + 4];
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
        __syncthreads();
    }
}

// Cross-layer grouped service.  Every work descriptor owns its
// input/weight/output pointers and (layer, expert, row-offset) identity; a
// projection launch therefore pools all active diagonal cells without making
// tensor addresses layer-global.
template<bool BF16_INPUT, bool T64_LAYOUT,
        bool INTERLEAVE = false,
        bool WARP_QUAD = false,
        int FIXED_N = 0,
        int FIXED_K = 0>
__global__ __launch_bounds__(128, 2)
static void aw_q8_service_m32(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem sm;
    static_assert(FIXED_N == 0 ||
            FIXED_N % AW_T64_ROWS == 0);
    static_assert(FIXED_K == 0 ||
            FIXED_K % AW_KSTAGE == 0);
    static_assert(FIXED_K == 0 ||
            (FIXED_K/AW_KSTAGE) % 2 == 0);
    const int n_stride =
            FIXED_N != 0 ? FIXED_N : n;
    const int k_stride =
            FIXED_K != 0 ? FIXED_K : k;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kgrp = WARP_QUAD ?
            lane & 3 : warp;
    const int mt = WARP_QUAD ?
            warp : lane >> 3;
    const int nt = WARP_QUAD ?
            lane >> 2 : lane & 7;
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
    const int n_ntiles =
            n_stride/AW_T64_ROWS;
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
            gq0 = desc.weight +
                    (size_t) (col0 + b_r0)*
                    (k_stride/AW_KSTAGE)*
                    AW_Q8_BLOCK_BYTES;
            gq1 = desc.weight +
                    (size_t) (col0 + b_r1)*
                    (k_stride/AW_KSTAGE)*
                    AW_Q8_BLOCK_BYTES;
        }

        float4 pa0, pa1;
        unsigned short pq0[4], pq1[4];
        float pd0, pd1;
        auto fetcha = [&](int stage) {
            const size_t base =
                    (size_t) input_row*k_stride +
                    stage*AW_KSTAGE + a_k;
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
                const int n_kblocks =
                        k_stride/AW_KSTAGE;
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
        auto stage_a_part = [&](int buffer, int i) {
            float * fA = (float *) sm.stage.A[buffer];
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            fA[(a_k + i    )*32 + aoff] = a0[i];
            fA[(a_k + i + 4)*32 + aoff] = a1[i];
        };
        auto stage_b_part = [&](int buffer, int i) {
            float * fB = (float *) sm.stage.B[buffer];
            fB[(b_k + 2*i    )*64 + boff0] = (float) (signed char) (pq0[i] & 0xff)*pd0;
            fB[(b_k + 2*i + 1)*64 + boff0] = (float) (signed char) (pq0[i] >> 8  )*pd0;
            fB[(b_k + 2*i    )*64 + boff1] = (float) (signed char) (pq1[i] & 0xff)*pd1;
            fB[(b_k + 2*i + 1)*64 + boff1] = (float) (signed char) (pq1[i] >> 8  )*pd1;
        };
        auto stage_data = [&](int buffer) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                stage_a_part(buffer, i);
                stage_b_part(buffer, i);
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
        #pragma unroll 1
        for (int kb = 0;
                kb < k_stride;
                kb += AW_KSTAGE) {
            const bool last =
                    kb + AW_KSTAGE >= k_stride;
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
                if constexpr (INTERLEAVE) {
                    if ((kk & 1) != 0) {
                        stage_b_part(buffer ^ 1, kk >> 1);
                    }
                }
            }
            if (!last) {
                if constexpr (INTERLEAVE) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        stage_a_part(buffer ^ 1, i);
                    }
                } else {
                    stage_data(buffer ^ 1);
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        if constexpr (WARP_QUAD) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rail_lane =
                            (lane >> 2)*4;
                    const float rail1 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 1);
                    const float rail2 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 2);
                    const float rail3 =
                            __shfl_sync(
                                0xffffffffu,
                                acc[i][j],
                                rail_lane + 3);
                    if ((lane & 3) == 0) {
                        const int row = mt*8 + i;
                        if (row < tile.rows) {
                            const float rail01 =
                                    __fadd_rn(
                                        acc[i][j],
                                        rail1);
                            const float rail012 =
                                    __fadd_rn(
                                        rail01,
                                        rail2);
                            desc.output[
                                    (size_t)
                                        (tile.row + row)*
                                        n_stride +
                                    col0 + nt*8 + j] =
                                    __fadd_rn(
                                        rail012,
                                        rail3);
                        }
                    }
                }
            }
            __syncthreads();
        } else {
            for (int round = 0; round < 8;
                    ++round) {
                if (kgrp != 0) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm.red[j][
                                (kgrp - 1)*32 +
                                lane] =
                                acc[round][j];
                    }
                }
                __syncthreads();
                if (kgrp == 0) {
                    const int row = mt*8 + round;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8;
                                ++j) {
                            const float value =
                                    acc[round][j] +
                                    sm.red[j][lane] +
                                    sm.red[j][32 + lane] +
                                    sm.red[j][64 + lane];
                            desc.output[
                                    (size_t)
                                        (tile.row + row)*
                                        n_stride +
                                    col0 + nt*8 + j] =
                                    value;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

struct aw_smem_m32_raillocal {
    float4 A[2][2][AW_KSTAGE/4][8];
    float4 B[2][2][AW_KSTAGE/4][16];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(64, 4)
static void aw_q8_service_m32_raillocal(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        int k,
        const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m32_raillocal sm;

    const int worker = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int mt = lane >> 3;
    const int nt = lane & 7;
    const int b_r0 = lane;
    const int b_r1 = lane + 32;
    const int bslot0 =
            (((b_r0 >> 2) & 1)*8 + (b_r0 >> 3));
    const int bslot1 =
            (((b_r1 >> 2) & 1)*8 + (b_r1 >> 3));
    const int boff0 = bslot0*4 + (b_r0 & 3);
    const int boff1 = bslot1*4 + (b_r1 & 3);

    n_mtiles = n_mtiles_dev != nullptr ?
            *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int ntile_shift = __ffs(n_ntiles) - 1;
    const int ntile_mask = n_ntiles - 1;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id =
                blockIdx.x*2 + worker;
            work_id < n_work;
            work_id += gridDim.x*2) {
        const int mt_id = work_id >> ntile_shift;
        const int ntile = work_id & ntile_mask;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row =
                tile.row +
                (lane < tile.rows ?
                    lane : tile.rows - 1);
        const int input_row =
                desc.input_rows != nullptr ?
                desc.input_rows[local_row] :
                local_row;
        const int n_kblocks = k/AW_KSTAGE;

        auto fetch = [&](int rail,
                int stage,
                float4 & pa0,
                float4 & pa1,
                uint2 & pq0,
                uint2 & pq1,
                float & scale0,
                float & scale1) {
            const size_t input_base =
                    (size_t) input_row*k +
                    stage*AW_KSTAGE +
                    rail*(AW_KSTAGE/4);
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input, input_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *)
                            desc.input +
                         input_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }

            const char * tile_stage =
                    desc.weight +
                    ((size_t) ntile*n_kblocks +
                     stage)*AW_T64_STAGE_BYTES;
            scale0 = __half2float(
                    *(const half *)
                    (tile_stage + b_r0*2));
            scale1 = __half2float(
                    *(const half *)
                    (tile_stage + b_r1*2));
            const char * values =
                    tile_stage +
                    AW_T64_ROWS*2;
            pq0 = __ldg((const uint2 *)
                    (values +
                     b_r0*AW_KSTAGE +
                     rail*(AW_KSTAGE/4)));
            pq1 = __ldg((const uint2 *)
                    (values +
                     b_r1*AW_KSTAGE +
                     rail*(AW_KSTAGE/4)));
        };

        auto publish = [&](int buffer,
                int kk,
                const float4 & pa0,
                const float4 & pa1,
                const uint2 & pq0,
                const uint2 & pq1,
                float scale0,
                float scale1) {
            const float * a0 =
                    (const float *) &pa0;
            const float * a1 =
                    (const float *) &pa1;
            float * fA =
                    (float *)
                    sm.A[buffer][worker];
            fA[kk*32 + lane] =
                    kk < 4 ?
                    a0[kk] : a1[kk - 4];

            const uint32_t packed0 =
                    kk < 4 ? pq0.x : pq0.y;
            const uint32_t packed1 =
                    kk < 4 ? pq1.x : pq1.y;
            const int shift = (kk & 3)*8;
            float * fB =
                    (float *)
                    sm.B[buffer][worker];
            fB[kk*64 + boff0] =
                    __fmul_rn(
                        (float) (signed char)
                        ((packed0 >> shift) &
                         0xffu),
                        scale0);
            fB[kk*64 + boff1] =
                    __fmul_rn(
                        (float) (signed char)
                        ((packed1 >> shift) &
                         0xffu),
                        scale1);
        };

        float sum[8][8] = {};
        float rail_acc[8][8];
        #pragma unroll 1
        for (int rail = 0; rail < 4;
                ++rail) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8;
                        ++j) {
                    rail_acc[i][j] = 0.0f;
                }
            }

            float4 pa0, pa1;
            uint2 pq0, pq1;
            float scale0, scale1;
            fetch(rail, 0,
                    pa0, pa1, pq0, pq1,
                    scale0, scale1);
            #pragma unroll
            for (int kk = 0;
                    kk < AW_KSTAGE/4;
                    ++kk) {
                publish(0, kk,
                        pa0, pa1, pq0, pq1,
                        scale0, scale1);
            }
            __syncwarp();

            int buffer = 0;
            for (int stage = 0;
                    stage < n_kblocks;
                    ++stage) {
                const bool last =
                        stage + 1 == n_kblocks;
                float4 next_pa0 = {};
                float4 next_pa1 = {};
                uint2 next_pq0 = {};
                uint2 next_pq1 = {};
                float next_scale0 = 0.0f;
                float next_scale1 = 0.0f;
                if (!last) {
                    fetch(rail, stage + 1,
                            next_pa0, next_pa1,
                            next_pq0, next_pq1,
                            next_scale0,
                            next_scale1);
                }

                #pragma unroll
                for (int kk = 0;
                        kk < AW_KSTAGE/4;
                        ++kk) {
                    const float4 va0 =
                            sm.A[buffer][worker]
                                [kk][2*mt];
                    const float4 va1 =
                            sm.A[buffer][worker]
                                [kk][2*mt + 1];
                    const float4 vb0 =
                            sm.B[buffer][worker]
                                [kk][nt];
                    const float4 vb1 =
                            sm.B[buffer][worker]
                                [kk][8 + nt];
                    if (!last) {
                        publish(buffer ^ 1, kk,
                                next_pa0,
                                next_pa1,
                                next_pq0,
                                next_pq1,
                                next_scale0,
                                next_scale1);
                    }
                    const float av[8] = {
                        va0.x, va0.y,
                        va0.z, va0.w,
                        va1.x, va1.y,
                        va1.z, va1.w
                    };
                    const float bv[8] = {
                        vb0.x, vb0.y,
                        vb0.z, vb0.w,
                        vb1.x, vb1.y,
                        vb1.z, vb1.w
                    };
                    #pragma unroll
                    for (int i = 0;
                            i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0;
                                j < 8; ++j) {
                            rail_acc[i][j] =
                                    __fmaf_rn(
                                        av[i],
                                        bv[j],
                                        rail_acc[i][j]);
                        }
                    }
                }
                __syncwarp();
                buffer ^= 1;
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8;
                        ++j) {
                    sum[i][j] =
                            rail == 0 ?
                            rail_acc[i][j] :
                            __fadd_rn(
                                sum[i][j],
                                rail_acc[i][j]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = mt*8 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output +
                        (size_t)
                            (tile.row + row)*n +
                        col0 + nt*8;
                const float4 out0 = {
                    sum[i][0], sum[i][1],
                    sum[i][2], sum[i][3]
                };
                const float4 out1 = {
                    sum[i][4], sum[i][5],
                    sum[i][6], sum[i][7]
                };
                ((float4 *) output)[0] = out0;
                ((float4 *) output)[1] = out1;
            }
        }
        __syncwarp();
    }
}

union aw_smem_m32_bundle {
    struct {
        float4 A[2][2][AW_KSTAGE][8];
        float4 B[2][AW_KSTAGE][16];
    } stage;
    float red[2][8][128];
};

template<bool BF16_INPUT, bool T64_LAYOUT,
        bool INTERLEAVE = false>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m32_bundle(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        const aw_tile_desc * __restrict__ pairs,
        int n_pairs, int n, int k,
        const int * n_pairs_dev = nullptr) {
    __shared__ aw_smem_m32_bundle sm;

    const int group = threadIdx.x >> 7;
    const int tid = threadIdx.x & 127;
    const int kgrp = tid >> 5;
    const int lane = tid & 31;
    const int mt = lane >> 3;
    const int nt = lane & 7;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r0 = tid >> 2;
    const int b_r1 = 32 + (tid >> 2);
    const int xk = tid & 3;
    const int b_k = xk*8;

    const int aslot =
            (((a_r >> 2) & 1)*4 + (a_r >> 3)) ^ xk;
    const int aoff = aslot*4 + (a_r & 3);
    const int bslot0 =
            ((((b_r0 >> 2) & 1)*8 +
              (b_r0 >> 3)) ^ xk);
    const int boff0 = bslot0*4 + (b_r0 & 3);
    const int bslot1 =
            ((((b_r1 >> 2) & 1)*8 +
              (b_r1 >> 3)) ^ xk);
    const int boff1 = bslot1*4 + (b_r1 & 3);
    const int sa0 = (0 + mt) ^ kgrp;
    const int sa1 = (4 + mt) ^ kgrp;
    const int sb0 = (0 + nt) ^ kgrp;
    const int sb1 = (8 + nt) ^ kgrp;

    n_pairs = n_pairs_dev != nullptr ?
            *n_pairs_dev : n_pairs;
    const int n_ntiles = n/64;
    const int n_work = n_pairs*n_ntiles;
    for (int work_id = blockIdx.x;
            work_id < n_work;
            work_id += gridDim.x) {
        const int pair_id = work_id/n_ntiles;
        const int col0 =
                (work_id % n_ntiles)*64;
        const aw_tile_desc pair =
                pairs[pair_id];
        const aw_tile_desc tile =
                tiles[group == 0 ?
                    pair.work : pair.row];
        const aw_work_desc desc =
                descs[tile.work];
        const int local_row =
                tile.row +
                (a_r < tile.rows ?
                    a_r : tile.rows - 1);
        const int input_row =
                desc.input_rows != nullptr ?
                desc.input_rows[local_row] :
                local_row;

        float4 pa0, pa1;
        unsigned short pq0[4], pq1[4];
        float pd0 = 0.0f;
        float pd1 = 0.0f;
        auto fetcha = [&](int stage) {
            const size_t base =
                    (size_t) input_row*k +
                    stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input, base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
        };
        auto fetchb = [&](int stage) {
            const char * p0;
            const char * p1;
            if constexpr (T64_LAYOUT) {
                const int n_kblocks =
                        k/AW_KSTAGE;
                const char * tile_stage =
                        desc.weight +
                        ((size_t)
                            (col0/AW_T64_ROWS)*
                            n_kblocks + stage)*
                            AW_T64_STAGE_BYTES;
                p0 = tile_stage +
                        AW_T64_ROWS*2 +
                        b_r0*AW_KSTAGE - 2;
                p1 = tile_stage +
                        AW_T64_ROWS*2 +
                        b_r1*AW_KSTAGE - 2;
                pd0 = __half2float(
                        *(const half *)
                        (tile_stage + b_r0*2));
                pd1 = __half2float(
                        *(const half *)
                        (tile_stage + b_r1*2));
            } else {
                p0 = desc.weight +
                        (size_t) (col0 + b_r0)*
                            (k/AW_KSTAGE)*
                            AW_Q8_BLOCK_BYTES +
                        (size_t) stage*
                            AW_Q8_BLOCK_BYTES;
                p1 = desc.weight +
                        (size_t) (col0 + b_r1)*
                            (k/AW_KSTAGE)*
                            AW_Q8_BLOCK_BYTES +
                        (size_t) stage*
                            AW_Q8_BLOCK_BYTES;
                pd0 = __half2float(
                        *(const half *) p0);
                pd1 = __half2float(
                        *(const half *) p1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                pq0[i] =
                        *(const unsigned short *)
                        (p0 + 2 + b_k + 2*i);
                pq1[i] =
                        *(const unsigned short *)
                        (p1 + 2 + b_k + 2*i);
            }
        };
        auto stage_a_part =
                [&](int buffer, int i) {
            float * fA =
                    (float *)
                    sm.stage.A[group][buffer];
            const float * a0 =
                    (const float *) &pa0;
            const float * a1 =
                    (const float *) &pa1;
            fA[(a_k + i)*32 + aoff] = a0[i];
            fA[(a_k + i + 4)*32 + aoff] =
                    a1[i];
        };
        auto stage_b_part =
                [&](int buffer, int i) {
            float * fB =
                    (float *) sm.stage.B[buffer];
            fB[(b_k + 2*i)*64 + boff0] =
                    (float) (signed char)
                    (pq0[i] & 0xff)*pd0;
            fB[(b_k + 2*i + 1)*64 + boff0] =
                    (float) (signed char)
                    (pq0[i] >> 8)*pd0;
            fB[(b_k + 2*i)*64 + boff1] =
                    (float) (signed char)
                    (pq1[i] & 0xff)*pd1;
            fB[(b_k + 2*i + 1)*64 + boff1] =
                    (float) (signed char)
                    (pq1[i] >> 8)*pd1;
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
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            stage_a_part(0, i);
        }
        if (group == 0) {
            fetchb(0);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                stage_b_part(0, i);
            }
        }
        __syncthreads();
        int buffer = 0;
        for (int kb = 0; kb < k;
                kb += AW_KSTAGE) {
            const bool last =
                    kb + AW_KSTAGE >= k;
            if (!last) {
                const int next =
                        (kb + AW_KSTAGE)/
                        AW_KSTAGE;
                fetcha(next);
                if (group == 0) {
                    fetchb(next);
                }
            }
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int ks = kgrp*8 + kk;
                const float4 va0 =
                        sm.stage.A[group][
                            buffer][ks][sa0];
                const float4 va1 =
                        sm.stage.A[group][
                            buffer][ks][sa1];
                const float4 vb0 =
                        sm.stage.B[buffer][ks][sb0];
                const float4 vb1 =
                        sm.stage.B[buffer][ks][sb1];
                const float av[8] = {
                    va0.x, va0.y, va0.z, va0.w,
                    va1.x, va1.y, va1.z, va1.w
                };
                const float bv[8] = {
                    vb0.x, vb0.y, vb0.z, vb0.w,
                    vb1.x, vb1.y, vb1.z, vb1.w
                };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] +=
                                av[i]*bv[j];
                    }
                }
                if constexpr (INTERLEAVE) {
                    if (group == 0 &&
                            (kk & 1) != 0) {
                        stage_b_part(
                                buffer ^ 1,
                                kk >> 1);
                    }
                }
            }
            if (!last) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    stage_a_part(
                            buffer ^ 1, i);
                }
                if constexpr (!INTERLEAVE) {
                    if (group == 0) {
                        #pragma unroll
                        for (int i = 0;
                                i < 4; ++i) {
                            stage_b_part(
                                    buffer ^ 1, i);
                        }
                    }
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        for (int round = 0;
                round < 8; ++round) {
            if (kgrp != 0) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sm.red[group][j][
                        (kgrp - 1)*32 + lane] =
                            acc[round][j];
                }
            }
            __syncthreads();
            if (kgrp == 0) {
                const int row =
                        mt*8 + round;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0;
                            j < 8; ++j) {
                        const float value =
                                acc[round][j] +
                                sm.red[group][j][
                                    lane] +
                                sm.red[group][j][
                                    32 + lane] +
                                sm.red[group][j][
                                    64 + lane];
                        desc.output[
                            (size_t)
                                (tile.row + row)*n +
                            col0 + nt*8 + j] =
                                value;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ static void aw_pair_build_tile_bundles(
        const aw_work_desc * descs,
        const aw_tile_desc * tiles,
        int32_t * tile_counts,
        aw_tile_desc * pairs,
        aw_tile_desc * singles,
        int tile_count_index) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    constexpr int max_tiles =
            AW_GPU_COUNT*AW_PRIMARY_PER_GPU;
    bool used[max_tiles] = {};
    const int count =
            tile_counts[tile_count_index];
    int pair_count = 0;
    int single_count = 0;
    for (int first = 0; first < count; ++first) {
        if (used[first]) {
            continue;
        }
        int second = -1;
        const aw_work_desc first_desc =
                descs[tiles[first].work];
        for (int candidate = first + 1;
                candidate < count; ++candidate) {
            if (!used[candidate] &&
                    descs[tiles[candidate].work].
                        weight == first_desc.weight) {
                second = candidate;
                break;
            }
        }
        if (second >= 0) {
            pairs[pair_count++] = {
                first, second, 0
            };
            used[second] = true;
        } else {
            singles[single_count++] =
                    tiles[first];
        }
        used[first] = true;
    }
    tile_counts[4] = pair_count;
    tile_counts[5] = single_count;
}

__global__ static void aw_build_m16_cohorts(
        const aw_work_desc * descs,
        const aw_tile_desc * tiles,
        int n_tiles,
        const int * n_tiles_dev,
        int32_t * cohort_counts,
        aw_m16_cohort * cohorts_p2,
        aw_m16_cohort * cohorts_p3,
        aw_m16_cohort * cohorts_p4,
        aw_tile_desc * singles) {
    if (blockIdx.x != 0) {
        return;
    }
    n_tiles = n_tiles_dev != nullptr ? *n_tiles_dev : n_tiles;
    const int tid = threadIdx.x;
    __shared__ const char *
            weights[AW_COHORTRAIL_MAX_TILES];
    __shared__ int ranks[AW_COHORTRAIL_MAX_TILES];
    __shared__ int cohort_sizes[
            AW_COHORTRAIL_MAX_TILES];

    if (tid < n_tiles) {
        weights[tid] =
                descs[tiles[tid].work].weight;
    } else {
        weights[tid] = nullptr;
    }
    __syncthreads();

    int rank = 0;
    int total = 0;
    if (tid < n_tiles) {
        const char * weight = weights[tid];
        for (int candidate = 0;
                candidate < n_tiles; ++candidate) {
            if (weights[candidate] == weight) {
                total++;
                rank += candidate < tid;
            }
        }
        ranks[tid] = rank;
        cohort_sizes[tid] =
                rank % 4 == 0 ?
                min(4, total - rank) : 0;
    } else {
        ranks[tid] = 0;
        cohort_sizes[tid] = 0;
    }
    __syncthreads();

    const int cohort_size = cohort_sizes[tid];
    if (cohort_size != 0) {
        int output_index = 0;
        for (int candidate = 0;
                candidate < tid; ++candidate) {
            output_index +=
                    cohort_sizes[candidate] ==
                        cohort_size;
        }
        if (cohort_size == 1) {
            singles[output_index] = tiles[tid];
        } else {
            aw_m16_cohort cohort = {
                {-1, -1, -1, -1}
            };
            for (int candidate = tid;
                    candidate < n_tiles; ++candidate) {
                if (weights[candidate] ==
                            weights[tid] &&
                        ranks[candidate] >= rank &&
                        ranks[candidate] <
                            rank + cohort_size) {
                    cohort.tiles[
                            ranks[candidate] - rank] =
                            candidate;
                }
            }
            if (cohort_size == 4) {
                cohorts_p4[output_index] = cohort;
            } else if (cohort_size == 3) {
                cohorts_p3[output_index] = cohort;
            } else {
                cohorts_p2[output_index] = cohort;
            }
        }
    }
    if (tid < 4) {
        int count = 0;
        const int size = tid == 0 ? 2 :
                tid == 1 ? 3 :
                tid == 2 ? 4 : 1;
        for (int candidate = 0;
                candidate < n_tiles; ++candidate) {
            count += cohort_sizes[candidate] == size;
        }
        cohort_counts[tid] = count;
    }
}

union aw_smem_m64 {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[AW_T64_ROWS][AW_T64_ROWS];
};

template<bool BF16_INPUT, bool T64_LAYOUT, bool F32_WEIGHT = false, int SPLIT_K = 2,
        bool INTERLEAVE = false>
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
        auto stage_a_part = [&](int buffer, int i) {
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            sm.stage.A[buffer][a_k + i][a_r] = a0[i];
            sm.stage.A[buffer][a_k + i + 4][a_r] = a1[i];
        };
        auto stage_b_part = [&](int buffer, int i) {
            const float * b0 = (const float *) &pb0;
            const float * b1 = (const float *) &pb1;
            if constexpr (F32_WEIGHT) {
                sm.stage.B[buffer][b_k + i][b_r] = b0[i];
                sm.stage.B[buffer][b_k + i + 4][b_r] = b1[i];
            } else {
                sm.stage.B[buffer][b_k + 2*i][b_r] = (float) (signed char) (pq[i] & 0xff)*pd;
                sm.stage.B[buffer][b_k + 2*i + 1][b_r] = (float) (signed char) (pq[i] >> 8)*pd;
            }
        };
        auto stage_data = [&](int buffer) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                stage_a_part(buffer, i);
                stage_b_part(buffer, i);
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
                    if constexpr (INTERLEAVE) {
                        static_assert(SPLIT_K == 2);
                        if ((kk & 3) == 3) {
                            stage_b_part(buffer ^ 1, kk >> 2);
                        }
                    }
                }
            }
            if (!last) {
                if constexpr (INTERLEAVE) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        stage_a_part(buffer ^ 1, i);
                    }
                } else {
                    stage_data(buffer ^ 1);
                }
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

template<bool BF16_INPUT, bool FUSE_SWIGLU = false>
__global__ __launch_bounds__(512, 1)
static void aw_q8_service_m64_gate_up(
        const aw_work_desc * __restrict__ gate_descs,
        const aw_work_desc * __restrict__ up_descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        int k,
        const int * n_mtiles_dev = nullptr,
        float * middle = nullptr) {
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
                        const float value =
                                acc[i][j] +
                                sm.red[projection][row][nt*8 + j];
                        if constexpr (FUSE_SWIGLU) {
                            sm.red[projection][row][nt*8 + j] = value;
                        } else {
                            desc.output[
                                    (size_t) (tile.row + row)*n +
                                    col0 + nt*8 + j] = value;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if constexpr (FUSE_SWIGLU) {
            if (projection == 0 && kgrp == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = output_row0 + i;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int col = nt*8 + j;
                            const float gate =
                                    sm.red[0][row][col];
                            const float up =
                                    sm.red[1][row][col];
                            middle[
                                    (size_t) (desc.row_offset + tile.row + row)*n +
                                    col0 + col] =
                                    (gate/(1.0f + expf(-gate)))*up;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(512, 1)
static void aw_q8_service_m64_n128(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_gate_up sm;

    const int nhalf = threadIdx.x >> 8;
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
    const int n_ntiles = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int col0 = (work_id - mt_id*n_ntiles)*2*AW_T64_ROWS + nhalf*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float4 pa0, pa1;
        unsigned short pq[4];
        float pd;
        auto fetcha = [&](int stage) {
            if (nhalf != 0) {
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
            if (nhalf == 0) {
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
                sm.stage.B[buffer][nhalf][b_k + 2*i][b_r] =
                        (float) (signed char) (pq[i] & 0xff)*pd;
                sm.stage.B[buffer][nhalf][b_k + 2*i + 1][b_r] =
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
                const float4 bv0 = *(const float4 *) &sm.stage.B[buffer][nhalf][ks][nt*8];
                const float4 bv1 = *(const float4 *) &sm.stage.B[buffer][nhalf][ks][nt*8 + 4];
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
                    sm.red[nhalf][output_row0 + i][nt*8 + j] = acc[i][j];
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
                                acc[i][j] + sm.red[nhalf][row][nt*8 + j];
                    }
                }
            }
        }
        __syncthreads();
    }
}

struct aw_smem_m64_n128_256 {
    float A[2][AW_KSTAGE][AW_T64_ROWS];
    float B[AW_KSTAGE][2*AW_T64_ROWS];
};

template<bool BF16_INPUT, bool F32_WEIGHT = false>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_256(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_256 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 1;
    const int b_half = tid & 1;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        int buffer = 0;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            if constexpr (F32_WEIGHT) {
                const float4 * values = (const float4 *) ((const float *) desc.weight +
                        (size_t) (col0 + b_r)*k + b_half*16);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 loaded = values[i];
                    const float * value = (const float *) &loaded;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sm.B[b_half*16 + i*4 + j][b_r] = value[j];
                    }
                }
            } else {
                const char * tile_stage = desc.weight +
                        (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
                const float scale = __half2float(*(const half *) (tile_stage + b_local*2));
                const char * values = tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_half*16;
                uint32_t packed[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    packed[i] = *(const uint32_t *) (values + i*4);
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[b_half*16 + i*4 + byte][b_r] =
                                (float) (signed char) ((packed[i] >> (byte*8)) & 0xffu)*scale;
                    }
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == k/AW_KSTAGE;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[4] = {};
            float4 next_weight[4];
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }

                if constexpr (F32_WEIGHT) {
                    const float4 * next_values = (const float4 *) ((const float *) desc.weight +
                            (size_t) (col0 + b_r)*k +
                            (stage + 1)*AW_KSTAGE + b_half*16);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        next_weight[i] = next_values[i];
                    }
                } else {
                    const char * next_tile_stage = desc.weight +
                            ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                    next_scale = __half2float(*(const half *) (next_tile_stage + b_local*2));
                    const char * next_values = next_tile_stage + AW_T64_ROWS*2 +
                            b_local*AW_KSTAGE + b_half*16;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        next_packed[i] = *(const uint32_t *) (next_values + i*4);
                    }
                }
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 = *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 = *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc0[i][j] += av[i]*bv[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc1[i][j] += av[i]*bv[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (!last) {
                if constexpr (F32_WEIGHT) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float * value = (const float *) &next_weight[i];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            sm.B[b_half*16 + i*4 + j][b_r] = value[j];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const uint32_t value = next_packed[group];
                        #pragma unroll
                        for (int byte = 0; byte < 4; ++byte) {
                            sm.B[b_half*16 + group*4 + byte][b_r] =
                                    (float) (signed char) ((value >> (byte*8)) & 0xffu)*next_scale;
                        }
                    }
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_halfpipe_sync(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_256 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int producer_half = warp >> 2;
    const int b_r = (warp & 3)*32 + lane;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int ngroup_shift = __ffs(n_ngroups) - 1;
    const int ngroup_mask = n_ngroups - 1;
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id >> ngroup_shift;
        const int ngroup = work_id & ngroup_mask;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        size_t input_stage_base =
                (size_t) input_row*k + a_k;
        const char * weight_stage =
                desc.weight +
                (size_t) ntile*n_kblocks*
                    AW_T64_STAGE_BYTES;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        int buffer = 0;
        {
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input,
                        input_stage_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         input_stage_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            const float scale =
                    __half2float(
                        *(const half *)
                        (weight_stage +
                         b_local*2));
            const char * values =
                    weight_stage +
                    AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + producer_half*16;
            const uint4 packed4 =
                    __ldg((const uint4 *) values);
            const uint32_t packed[4] = {
                packed4.x, packed4.y,
                packed4.z, packed4.w
            };
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[producer_half*16 + group*4 + byte][b_r] =
                            (float) (signed char)
                            ((packed[group] >>
                              (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        input_stage_base +
                        AW_KSTAGE;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }
            }

            if (stage != 0 && producer_half == 1) {
                const float scale =
                        __half2float(*(const half *)
                        (weight_stage + b_local*2));
                const char * values =
                        weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + 16;
                const uint4 packed4 =
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[16 + group*4 + byte][b_r] =
                                (float) (signed char)
                                ((packed[group] >>
                                  (byte*8)) & 0xffu)*scale;
                    }
                }
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc0[i][j] += av[i]*bv[j];
                        }
                    }
                }
            }
            __syncthreads();

            if (!last && producer_half == 0) {
                const char * next_weight_stage =
                        weight_stage +
                        AW_T64_STAGE_BYTES;
                const float scale =
                        __half2float(*(const half *)
                        (next_weight_stage +
                         b_local*2));
                const char * values =
                        next_weight_stage +
                        AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE;
                const uint4 packed4 =
                        __ldg((const uint4 *) values);
                const uint32_t packed[4] = {
                    packed4.x, packed4.y,
                    packed4.z, packed4.w
                };
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                            sm.B[group*4 + byte][b_r] =
                                    (float) (signed char)
                                    ((packed[group] >>
                                      (byte*8)) & 0xffu)*scale;
                    }
                }
            }

            #pragma unroll
            for (int group = 4; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc1[i][j] += av[i]*bv[j];
                        }
                    }
                }
            }
            if (!last) {
                __syncthreads();
            }
            input_stage_base += AW_KSTAGE;
            weight_stage += AW_T64_STAGE_BYTES;
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}

template<int BARRIER>
__device__ __forceinline__ void aw_halfpipe_arrive_256() {
    asm volatile("bar.arrive %0, 256;" :: "n"(BARRIER) : "memory");
}

template<int BARRIER>
__device__ __forceinline__ void aw_halfpipe_sync_256() {
    asm volatile("bar.sync %0, 256;" :: "n"(BARRIER) : "memory");
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_halfpipe_bar(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_256 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int producer_half = warp >> 2;
    const int b_r = (warp & 3)*32 + lane;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        int buffer = 0;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            const char * tile_stage =
                    desc.weight + (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale =
                    __half2float(*(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + producer_half*16;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                const uint32_t packed =
                        *(const uint32_t *) (values + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[producer_half*16 + group*4 + byte][b_r] =
                            (float) (signed char)
                            ((packed >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        (size_t) input_row*k +
                        (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input +
                            next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }
            }

            if (stage != 0 && producer_half == 1) {
                aw_halfpipe_sync_256<2>();
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc0[i][j] += av[i]*bv[j];
                        }
                    }
                }
            }

            if (!last) {
                if (producer_half == 0) {
                    aw_halfpipe_sync_256<1>();
                    const char * tile_stage = desc.weight +
                            ((size_t) ntile*n_kblocks + stage + 1)*
                            AW_T64_STAGE_BYTES;
                    const float scale =
                            __half2float(*(const half *)
                            (tile_stage + b_local*2));
                    const char * values = tile_stage + AW_T64_ROWS*2 +
                            b_local*AW_KSTAGE;
                    #pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const uint32_t packed =
                                *(const uint32_t *) (values + group*4);
                        #pragma unroll
                        for (int byte = 0; byte < 4; ++byte) {
                            sm.B[group*4 + byte][b_r] =
                                    (float) (signed char)
                                    ((packed >> (byte*8)) & 0xffu)*scale;
                        }
                    }
                    aw_halfpipe_arrive_256<2>();
                } else {
                    aw_halfpipe_arrive_256<1>();
                }
            }
            if (stage != 0 && producer_half == 0) {
                aw_halfpipe_sync_256<4>();
            }

            #pragma unroll
            for (int group = 4; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            acc1[i][j] += av[i]*bv[j];
                        }
                    }
                }
            }

            if (!last) {
                if (producer_half == 1) {
                    aw_halfpipe_sync_256<3>();
                    const char * tile_stage = desc.weight +
                            ((size_t) ntile*n_kblocks + stage + 1)*
                            AW_T64_STAGE_BYTES;
                    const float scale =
                            __half2float(*(const half *)
                            (tile_stage + b_local*2));
                    const char * values = tile_stage + AW_T64_ROWS*2 +
                            b_local*AW_KSTAGE + 16;
                    #pragma unroll
                    for (int group = 0; group < 4; ++group) {
                        const uint32_t packed =
                                *(const uint32_t *) (values + group*4);
                        #pragma unroll
                        for (int byte = 0; byte < 4; ++byte) {
                            sm.B[16 + group*4 + byte][b_r] =
                                    (float) (signed char)
                                    ((packed >> (byte*8)) & 0xffu)*scale;
                        }
                    }
                    aw_halfpipe_arrive_256<4>();
                } else {
                    aw_halfpipe_arrive_256<3>();
                }
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_vector(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_256 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 1;
    const int b_half = tid & 1;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row =
                tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ?
                desc.input_rows[local_row] : local_row;

        float acc0[4][8] = {};
        float acc1[4][8] = {};
        int buffer = 0;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale = __half2float(
                    *(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_half*16;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t packed =
                        *(const uint32_t *) (values + i*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[b_half*16 + i*4 + byte][b_r] =
                            (float) (signed char)
                            ((packed >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[4] = {};
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k +
                        (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(
                            desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *)
                            ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*
                        AW_T64_STAGE_BYTES;
                next_scale = __half2float(
                        *(const half *) (next_tile_stage + b_local*2));
                const char * next_values =
                        next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_half*16;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    next_packed[i] =
                            *(const uint32_t *) (next_values + i*4);
                }
            }

            const int row0 = warp*8 + (lane >> 4)*4;
            const int thread_col = (lane & 15)*8;
            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av4 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 bv0 =
                            *(const float4 *) &sm.B[ks][thread_col];
                    const float4 bv1 =
                            *(const float4 *) &sm.B[ks][thread_col + 4];
                    const float av[4] = {
                        av4.x, av4.y, av4.z, av4.w,
                    };
                    const float bv[8] = {
                        bv0.x, bv0.y, bv0.z, bv0.w,
                        bv1.x, bv1.y, bv1.z, bv1.w,
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc0[i][j] += av[i]*bv[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc1[i][j] += av[i]*bv[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    const uint32_t value = next_packed[group];
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[b_half*16 + group*4 + byte][b_r] =
                                (float) (signed char)
                                ((value >> (byte*8)) & 0xffu)*next_scale;
                    }
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8 + (lane >> 4)*4;
        const int thread_col = (lane & 15)*8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n +
                        col0 + thread_col;
                const float4 out0 = {
                    acc0[i][0] + acc1[i][0],
                    acc0[i][1] + acc1[i][1],
                    acc0[i][2] + acc1[i][2],
                    acc0[i][3] + acc1[i][3],
                };
                const float4 out1 = {
                    acc0[i][4] + acc1[i][4],
                    acc0[i][5] + acc1[i][5],
                    acc0[i][6] + acc1[i][6],
                    acc0[i][7] + acc1[i][7],
                };
                ((float4 *) output)[0] = out0;
                ((float4 *) output)[1] = out1;
            }
        }
        __syncthreads();
    }
}

struct aw_smem_m64_n128_double {
    float A[2][AW_KSTAGE][AW_T64_ROWS];
    float B[2][AW_KSTAGE][2*AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(512, 1)
static void aw_q8_service_m64_n128_double(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_double sm;

    const int cohort = threadIdx.x >> 8;
    const int tid = threadIdx.x & 255;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = threadIdx.x >> 2;
    const int b_k = (threadIdx.x & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;

        if (cohort == 0) {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[0][a_k + i][a_r] = a0[i];
                sm.A[0][a_k + i + 4][a_r] = a1[i];
            }
        }
        {
            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale =
                    __half2float(*(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_k;
            #pragma unroll
            for (int group = 0; group < 2; ++group) {
                const uint32_t packed =
                        *(const uint32_t *) (values + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[0][b_k + group*4 + byte][b_r] =
                            (float) (signed char)
                            ((packed >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        float acc0[4][4] = {};
        float acc1[4][4] = {};
        int buffer = 0;
        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                if (cohort == 0) {
                    float4 pa0, pa1;
                    const size_t a_base = (size_t) input_row*k +
                            (stage + 1)*AW_KSTAGE + a_k;
                    if constexpr (BF16_INPUT) {
                        aw_load_bf16x8(desc.input, a_base, pa0, pa1);
                    } else {
                        const float4 * input4 =
                                (const float4 *) ((const float *) desc.input + a_base);
                        pa0 = input4[0];
                        pa1 = input4[1];
                    }
                    const float * a0 = (const float *) &pa0;
                    const float * a1 = (const float *) &pa1;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sm.A[buffer ^ 1][a_k + i][a_r] = a0[i];
                        sm.A[buffer ^ 1][a_k + i + 4][a_r] = a1[i];
                    }
                }
                const char * tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                const float scale =
                        __half2float(*(const half *) (tile_stage + b_local*2));
                const char * values = tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_k;
                #pragma unroll
                for (int group = 0; group < 2; ++group) {
                    const uint32_t packed =
                            *(const uint32_t *) (values + group*4);
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[buffer ^ 1][b_k + group*4 + byte][b_r] =
                                (float) (signed char)
                                ((packed >> (byte*8)) & 0xffu)*scale;
                    }
                }
            }

            const int row0 = cohort*32 + warp*4;
            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float a[4] = { av.x, av.y, av.z, av.w };
                    const float b[4] = {
                        sm.B[buffer][ks][lane],
                        sm.B[buffer][ks][lane + 32],
                        sm.B[buffer][ks][lane + 64],
                        sm.B[buffer][ks][lane + 96],
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc0[i][j] += a[i]*b[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc1[i][j] += a[i]*b[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        const int row0 = cohort*32 + warp*4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}

union aw_smem_m64_n128_dualrail {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[AW_KSTAGE][2*AW_T64_ROWS];
    } stage;
    float red[AW_T64_ROWS][2*AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(512, 2)
static void aw_q8_service_m64_n128_dualrail(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_dualrail sm;

    const int tid = threadIdx.x;
    const int rail = tid >> 8;
    const int local_tid = tid & 255;
    const int warp = local_tid >> 5;
    const int lane = local_tid & 31;
    const int a_r = tid >> 3;
    const int a_k = (tid & 7)*4;
    const int b_r = tid >> 2;
    const int b_quarter = tid & 3;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ?
                desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;

        float acc[8][4] = {};
        int buffer = 0;
        {
            float4 input;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                input = aw_load_bf16x4(desc.input, a_base);
            } else {
                input = *(const float4 *) ((const float *) desc.input + a_base);
            }
            const float * values = (const float *) &input;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[0][a_k + i][a_r] = values[i];
            }

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale = __half2float(
                    *(const half *) (tile_stage + b_local*2));
            const char * qvalues = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_quarter*8;
            const uint32_t packed0 = *(const uint32_t *) qvalues;
            const uint32_t packed1 = *(const uint32_t *) (qvalues + 4);
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                sm.stage.B[b_quarter*8 + byte][b_r] =
                        (float) (signed char)
                        ((packed0 >> (byte*8)) & 0xffu)*scale;
                sm.stage.B[b_quarter*8 + 4 + byte][b_r] =
                        (float) (signed char)
                        ((packed1 >> (byte*8)) & 0xffu)*scale;
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a;
            float next_scale = 0.0f;
            uint32_t next_packed0 = 0;
            uint32_t next_packed1 = 0;
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k +
                        (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    next_a = aw_load_bf16x4(desc.input, next_a_base);
                } else {
                    next_a = *(const float4 *)
                            ((const float *) desc.input + next_a_base);
                }
                const float * values = (const float *) &next_a;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.stage.A[buffer ^ 1][a_k + i][a_r] = values[i];
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*
                        AW_T64_STAGE_BYTES;
                next_scale = __half2float(
                        *(const half *) (next_tile_stage + b_local*2));
                const char * next_qvalues =
                        next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_quarter*8;
                next_packed0 = *(const uint32_t *) next_qvalues;
                next_packed1 = *(const uint32_t *) (next_qvalues + 4);
            }

            const int row0 = warp*8;
            const int rail_k0 = rail*(AW_KSTAGE/2);
            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const int ks = rail_k0 + kk;
                const float4 av0 =
                        *(const float4 *) &sm.stage.A[buffer][ks][row0];
                const float4 av1 =
                        *(const float4 *) &sm.stage.A[buffer][ks][row0 + 4];
                const float av[8] = {
                    av0.x, av0.y, av0.z, av0.w,
                    av1.x, av1.y, av1.z, av1.w,
                };
                const float bv[4] = {
                    sm.stage.B[ks][lane],
                    sm.stage.B[ks][lane + 32],
                    sm.stage.B[ks][lane + 64],
                    sm.stage.B[ks][lane + 96],
                };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += av[i]*bv[j];
                    }
                }
            }
            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.stage.B[b_quarter*8 + byte][b_r] =
                            (float) (signed char)
                            ((next_packed0 >> (byte*8)) & 0xffu)*next_scale;
                    sm.stage.B[b_quarter*8 + 4 + byte][b_r] =
                            (float) (signed char)
                            ((next_packed1 >> (byte*8)) & 0xffu)*next_scale;
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        if (rail == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sm.red[row0 + i][lane + j*32] = acc[i][j];
                }
            }
        }
        __syncthreads();
        if (rail == 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + i;
                if (row < tile.rows) {
                    float * output =
                            desc.output +
                            (size_t) (tile.row + row)*n + col0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        output[lane + j*32] =
                                acc[i][j] +
                                sm.red[row][lane + j*32];
                    }
                }
            }
        }
        __syncthreads();
    }
}

struct aw_smem_m128_n128_512 {
    float A[2][AW_KSTAGE][2*AW_T64_ROWS];
    float B[AW_KSTAGE][2*AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(512, 1)
static void aw_q8_service_m128_n128_512(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m128_n128_512 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 2;
    const int b_quarter = tid & 3;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ?
                desc.input_rows[local_row] : local_row;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        int buffer = 0;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        {
            float4 pa0;
            float4 pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale =
                    __half2float(*(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_quarter*8;
            uint32_t packed[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                packed[i] = *(const uint32_t *) (values + i*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[b_quarter*8 + i*4 + byte][b_r] =
                            (float) (signed char)
                            ((packed[i] >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a0;
            float4 next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[2] = {};
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k +
                        (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(
                            desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 = (const float4 *)
                            ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*
                        AW_T64_STAGE_BYTES;
                next_scale = __half2float(
                        *(const half *) (next_tile_stage + b_local*2));
                const char * next_values =
                        next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_quarter*8;
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    next_packed[i] =
                            *(const uint32_t *) (next_values + i*4);
                }
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc0[i][j] += av[i]*bv[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc1[i][j] += av[i]*bv[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int group = 0; group < 2; ++group) {
                    const uint32_t value = next_packed[group];
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[b_quarter*8 + group*4 + byte][b_r] =
                                (float) (signed char)
                                ((value >> (byte*8)) & 0xffu)*next_scale;
                    }
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
        __syncthreads();
    }
}

struct aw_smem_m64_n128_regb {
    float A[2][AW_KSTAGE][AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_regb(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_regb sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ?
                desc.input_rows[local_row] : local_row;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        int buffer = 0;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }
        }
        __syncthreads();

        const int n_kblocks = k/AW_KSTAGE;
        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            if (!last) {
                float4 next_a0, next_a1;
                const size_t next_a_base =
                        (size_t) input_row*k +
                        (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(
                            desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *)
                            ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * a0 = (const float *) &next_a0;
                const float * a1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = a0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = a1[i];
                }
            }

            const char * tile_stage0 = desc.weight +
                    ((size_t) (2*ngroup)*n_kblocks + stage)*
                    AW_T64_STAGE_BYTES;
            const char * tile_stage1 = tile_stage0 +
                    (size_t) n_kblocks*AW_T64_STAGE_BYTES;
            const float scale0 = __half2float(
                    *(const half *) (tile_stage0 + lane*2));
            const float scale1 = __half2float(
                    *(const half *) (tile_stage0 + (lane + 32)*2));
            const float scale2 = __half2float(
                    *(const half *) (tile_stage1 + lane*2));
            const float scale3 = __half2float(
                    *(const half *) (tile_stage1 + (lane + 32)*2));
            const char * values00 =
                    tile_stage0 + AW_T64_ROWS*2 + lane*AW_KSTAGE;
            const char * values01 = values00 + 32*AW_KSTAGE;
            const char * values10 =
                    tile_stage1 + AW_T64_ROWS*2 + lane*AW_KSTAGE;
            const char * values11 = values10 + 32*AW_KSTAGE;

            const int row0 = warp*8;
            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                const uint32_t packed0 =
                        *(const uint32_t *) (values00 + group*4);
                const uint32_t packed1 =
                        *(const uint32_t *) (values01 + group*4);
                const uint32_t packed2 =
                        *(const uint32_t *) (values10 + group*4);
                const uint32_t packed3 =
                        *(const uint32_t *) (values11 + group*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const int shift = byte*8;
                    const float bv[4] = {
                        (float) (signed char)
                                ((packed0 >> shift) & 0xffu)*scale0,
                        (float) (signed char)
                                ((packed1 >> shift) & 0xffu)*scale1,
                        (float) (signed char)
                                ((packed2 >> shift) & 0xffu)*scale2,
                        (float) (signed char)
                                ((packed3 >> shift) & 0xffu)*scale3,
                    };
                    const int ks = group*4 + byte;
                    const float4 av0 =
                            *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 =
                            *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc0[i][j] += av[i]*bv[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc1[i][j] += av[i]*bv[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output =
                        desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc0[i][j] + acc1[i][j];
                }
            }
        }
    }
}

struct aw_smem_m64_n128_rail {
    float A[2][AW_KSTAGE/2][AW_T64_ROWS];
    float B[AW_KSTAGE/2][2*AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 3)
static void aw_q8_service_m64_n128_rail(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        int k,
        float * __restrict__ partial) {
    __shared__ aw_smem_m64_n128_rail sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*4;
    const int b_r = tid >> 1;
    const int b_half = tid & 1;
    const int n_kblocks = k/AW_KSTAGE;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups*2;

    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int rail = work_id & 1;
        const int tile_work = work_id >> 1;
        const int mt_id = tile_work/n_ngroups;
        const int ngroup = tile_work - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;

        float acc[8][4] = {};
        int buffer = 0;
        {
            const size_t a_base =
                    (size_t) input_row*k + rail*(AW_KSTAGE/2) + a_k;
            float4 input;
            if constexpr (BF16_INPUT) {
                input = aw_load_bf16x4(desc.input, a_base);
            } else {
                input = *(const float4 *) ((const float *) desc.input + a_base);
            }
            const float * values = (const float *) &input;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[0][a_k + i][a_r] = values[i];
            }

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale = __half2float(*(const half *) (tile_stage + b_local*2));
            const char * qvalues = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + rail*(AW_KSTAGE/2) + b_half*8;
            uint32_t packed[2];
            packed[0] = *(const uint32_t *) (qvalues + 0);
            packed[1] = *(const uint32_t *) (qvalues + 4);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[b_half*8 + i*4 + byte][b_r] =
                            (float) (signed char) ((packed[i] >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a;
            float next_scale = 0.0f;
            uint32_t next_packed[2] = {};
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE +
                        rail*(AW_KSTAGE/2) + a_k;
                if constexpr (BF16_INPUT) {
                    next_a = aw_load_bf16x4(desc.input, next_a_base);
                } else {
                    next_a = *(const float4 *) ((const float *) desc.input + next_a_base);
                }
                const float * values = (const float *) &next_a;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = values[i];
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                next_scale = __half2float(*(const half *) (next_tile_stage + b_local*2));
                const char * next_qvalues = next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + rail*(AW_KSTAGE/2) + b_half*8;
                next_packed[0] = *(const uint32_t *) (next_qvalues + 0);
                next_packed[1] = *(const uint32_t *) (next_qvalues + 4);
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const float4 av0 = *(const float4 *) &sm.A[buffer][kk][row0];
                const float4 av1 = *(const float4 *) &sm.A[buffer][kk][row0 + 4];
                const float av[8] = {
                    av0.x, av0.y, av0.z, av0.w,
                    av1.x, av1.y, av1.z, av1.w,
                };
                const float bv[4] = {
                    sm.B[kk][lane],
                    sm.B[kk][lane + 32],
                    sm.B[kk][lane + 64],
                    sm.B[kk][lane + 96],
                };
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += av[i]*bv[j];
                    }
                }
            }
            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t value = next_packed[i];
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[b_half*8 + i*4 + byte][b_r] =
                                (float) (signed char) ((value >> (byte*8)) & 0xffu)*next_scale;
                    }
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output;
                if (rail == 0) {
                    output = desc.output + (size_t) (tile.row + row)*n + col0;
                } else {
                    output = partial +
                            (size_t) (desc.row_offset + tile.row + row)*n + col0;
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc[i][j];
                }
            }
        }
        __syncthreads();
    }
}

__global__ static void aw_q8_service_m64_n128_rail_add(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        const float * __restrict__ partial) {
    const size_t tile_elements = (size_t) AW_T64_ROWS*n;
    const size_t count = (size_t) n_mtiles*tile_elements;
    for (size_t index = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            index < count; index += (size_t) gridDim.x*blockDim.x) {
        const int mt_id = (int) (index/tile_elements);
        const size_t within = index - (size_t) mt_id*tile_elements;
        const int row = (int) (within/n);
        const int col = (int) (within - (size_t) row*n);
        const aw_tile_desc tile = tiles[mt_id];
        if (row < tile.rows) {
            const aw_work_desc desc = descs[tile.work];
            float * output = desc.output + (size_t) (tile.row + row)*n + col;
            const size_t partial_index =
                    (size_t) (desc.row_offset + tile.row + row)*n + col;
            *output += partial[partial_index];
        }
    }
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 2)
static void aw_q8_service_m64_n128_256_panel(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        int k,
        float * __restrict__ partial,
        int route_rows,
        int stage_begin,
        int stage_count,
        const void * panel_input,
        int panel_stride) {
    __shared__ aw_smem_m64_n128_256 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 1;
    const int b_half = tid & 1;
    const int n_kblocks = k/AW_KSTAGE;
    const int stage_end = stage_begin + stage_count;

    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;

        float acc0[8][4] = {};
        float acc1[8][4] = {};
        const int row0 = warp*8;
        if (stage_begin != 0) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + i;
                if (row < tile.rows) {
                    const size_t output_base =
                            (size_t) (desc.row_offset + tile.row + row)*n + col0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const size_t index = output_base + lane + j*32;
                        acc0[i][j] = partial[index];
                        acc1[i][j] = partial[(size_t) route_rows*n + index];
                    }
                }
            }
        }

        int buffer = 0;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;
        {
            float4 pa0, pa1;
            const void * input = panel_input != nullptr ? panel_input : desc.input;
            const size_t a_base = panel_input != nullptr ?
                    (size_t) input_row*panel_stride + a_k :
                    (size_t) input_row*k + stage_begin*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(input, a_base, pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *) ((const float *) input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 = (const float *) &pa0;
            const float * a1 = (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.A[buffer][a_k + i][a_r] = a0[i];
                sm.A[buffer][a_k + i + 4][a_r] = a1[i];
            }

            const char * tile_stage = desc.weight +
                    ((size_t) ntile*n_kblocks + stage_begin)*AW_T64_STAGE_BYTES;
            const float scale = __half2float(*(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_half*16;
            uint32_t packed[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                packed[i] = *(const uint32_t *) (values + i*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.B[b_half*16 + i*4 + byte][b_r] =
                            (float) (signed char) ((packed[i] >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        for (int stage = stage_begin; stage < stage_end; ++stage) {
            const bool last = stage + 1 == stage_end;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[4] = {};
            if (!last) {
                const void * input = panel_input != nullptr ? panel_input : desc.input;
                const size_t next_a_base = panel_input != nullptr ?
                        (size_t) input_row*panel_stride +
                                (stage + 1 - stage_begin)*AW_KSTAGE + a_k :
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * na0 = (const float *) &next_a0;
                const float * na1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.A[buffer ^ 1][a_k + i][a_r] = na0[i];
                    sm.A[buffer ^ 1][a_k + i + 4][a_r] = na1[i];
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                next_scale = __half2float(*(const half *) (next_tile_stage + b_local*2));
                const char * next_values = next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_half*16;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    next_packed[i] = *(const uint32_t *) (next_values + i*4);
                }
            }

            #pragma unroll
            for (int group = 0; group < 8; ++group) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int ks = group*4 + kk;
                    const float4 av0 = *(const float4 *) &sm.A[buffer][ks][row0];
                    const float4 av1 = *(const float4 *) &sm.A[buffer][ks][row0 + 4];
                    const float av[8] = {
                        av0.x, av0.y, av0.z, av0.w,
                        av1.x, av1.y, av1.z, av1.w,
                    };
                    const float bv[4] = {
                        sm.B[ks][lane],
                        sm.B[ks][lane + 32],
                        sm.B[ks][lane + 64],
                        sm.B[ks][lane + 96],
                    };
                    if (group < 4) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc0[i][j] += av[i]*bv[j];
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            #pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                acc1[i][j] += av[i]*bv[j];
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    const uint32_t value = next_packed[group];
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.B[b_half*16 + group*4 + byte][b_r] =
                                (float) (signed char) ((value >> (byte*8)) & 0xffu)*next_scale;
                    }
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const bool final_panel = stage_end == n_kblocks;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                const size_t output_base =
                        (size_t) (desc.row_offset + tile.row + row)*n + col0;
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const size_t index = output_base + lane + j*32;
                    if (final_panel) {
                        output[lane + j*32] = acc0[i][j] + acc1[i][j];
                    } else {
                        partial[index] = acc0[i][j];
                        partial[(size_t) route_rows*n + index] = acc1[i][j];
                    }
                }
            }
        }
        __syncthreads();
    }
}

template<bool DUAL_RAIL>
struct aw_smem_fab_m64_n128 {
    half2 A[2][AW_KSTAGE/2][AW_T64_ROWS];
    half2 A_low[DUAL_RAIL ? 2 : 1][AW_KSTAGE/2][AW_T64_ROWS];
    half2 B[AW_KSTAGE/2][2*AW_T64_ROWS];
    half scales[2*AW_T64_ROWS];
};

static __device__ __forceinline__ void aw_split_half2(float x, float y, half2 & high, half2 & low) {
    high = __floats2half2_rn(x, y);
    const float2 high_f = __half22float2(high);
    low = __floats2half2_rn(x - high_f.x, y - high_f.y);
}

template<bool BF16_INPUT, int FAB_CHUNKS = 2, bool POST_SCALE = false, bool DUAL_RAIL = false>
__global__ __launch_bounds__(256, 2)
static void aw_q8_fab_m64_n128(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    static_assert(AW_KSTAGE % (2*FAB_CHUNKS) == 0, "invalid FabWave promotion interval");
    static_assert(!DUAL_RAIL || POST_SCALE, "dual-rail FabWave requires post scaling");
    __shared__ aw_smem_fab_m64_n128<DUAL_RAIL> sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 1;
    const int b_half = tid & 1;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;
        const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
        const int b_local = b_r % AW_T64_ROWS;

        float acc[8][4] = {};
        int buffer = 0;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            if constexpr (DUAL_RAIL) {
                aw_split_half2(pa0.x, pa0.y,
                        sm.A[0][a_k/2 + 0][a_r], sm.A_low[0][a_k/2 + 0][a_r]);
                aw_split_half2(pa0.z, pa0.w,
                        sm.A[0][a_k/2 + 1][a_r], sm.A_low[0][a_k/2 + 1][a_r]);
                aw_split_half2(pa1.x, pa1.y,
                        sm.A[0][a_k/2 + 2][a_r], sm.A_low[0][a_k/2 + 2][a_r]);
                aw_split_half2(pa1.z, pa1.w,
                        sm.A[0][a_k/2 + 3][a_r], sm.A_low[0][a_k/2 + 3][a_r]);
            } else {
                sm.A[0][a_k/2 + 0][a_r] = __floats2half2_rn(pa0.x, pa0.y);
                sm.A[0][a_k/2 + 1][a_r] = __floats2half2_rn(pa0.z, pa0.w);
                sm.A[0][a_k/2 + 2][a_r] = __floats2half2_rn(pa1.x, pa1.y);
                sm.A[0][a_k/2 + 3][a_r] = __floats2half2_rn(pa1.z, pa1.w);
            }

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const half scale_h = *(const half *) (tile_stage + b_local*2);
            const float scale = __half2float(scale_h);
            if constexpr (POST_SCALE) {
                sm.scales[b_r] = scale_h;
            }
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_half*16;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t packed = *(const uint32_t *) (values + i*4);
                const float q0 = (float) (signed char) (packed & 0xffu);
                const float q1 = (float) (signed char) ((packed >> 8) & 0xffu);
                const float q2 = (float) (signed char) ((packed >> 16) & 0xffu);
                const float q3 = (float) (signed char) ((packed >> 24) & 0xffu);
                sm.B[b_half*8 + 2*i + 0][b_r] =
                        __floats2half2_rn(POST_SCALE ? q0 : q0*scale, POST_SCALE ? q1 : q1*scale);
                sm.B[b_half*8 + 2*i + 1][b_r] =
                        __floats2half2_rn(POST_SCALE ? q2 : q2*scale, POST_SCALE ? q3 : q3*scale);
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[4] = {};
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                if constexpr (DUAL_RAIL) {
                    aw_split_half2(next_a0.x, next_a0.y,
                            sm.A[buffer ^ 1][a_k/2 + 0][a_r],
                            sm.A_low[buffer ^ 1][a_k/2 + 0][a_r]);
                    aw_split_half2(next_a0.z, next_a0.w,
                            sm.A[buffer ^ 1][a_k/2 + 1][a_r],
                            sm.A_low[buffer ^ 1][a_k/2 + 1][a_r]);
                    aw_split_half2(next_a1.x, next_a1.y,
                            sm.A[buffer ^ 1][a_k/2 + 2][a_r],
                            sm.A_low[buffer ^ 1][a_k/2 + 2][a_r]);
                    aw_split_half2(next_a1.z, next_a1.w,
                            sm.A[buffer ^ 1][a_k/2 + 3][a_r],
                            sm.A_low[buffer ^ 1][a_k/2 + 3][a_r]);
                } else {
                    sm.A[buffer ^ 1][a_k/2 + 0][a_r] =
                            __floats2half2_rn(next_a0.x, next_a0.y);
                    sm.A[buffer ^ 1][a_k/2 + 1][a_r] =
                            __floats2half2_rn(next_a0.z, next_a0.w);
                    sm.A[buffer ^ 1][a_k/2 + 2][a_r] =
                            __floats2half2_rn(next_a1.x, next_a1.y);
                    sm.A[buffer ^ 1][a_k/2 + 3][a_r] =
                            __floats2half2_rn(next_a1.z, next_a1.w);
                }

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                next_scale = __half2float(*(const half *) (next_tile_stage + b_local*2));
                const char * next_values = next_tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_half*16;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    next_packed[i] = *(const uint32_t *) (next_values + i*4);
                }
            }

            const int row0 = warp*8;
            float scales[4];
            if constexpr (POST_SCALE) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    scales[j] = __half2float(sm.scales[lane + j*32]);
                }
            }
            #pragma unroll
            for (int chunk = 0; chunk < FAB_CHUNKS; ++chunk) {
                #pragma unroll
                for (int rail = 0; rail < (DUAL_RAIL ? 2 : 1); ++rail) {
                half2 hacc[8][4];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        hacc[i][j] = __float2half2_rn(0.0f);
                    }
                }
                #pragma unroll
                for (int kp = 0; kp < AW_KSTAGE/(2*FAB_CHUNKS); ++kp) {
                    const int pair = chunk*(AW_KSTAGE/(2*FAB_CHUNKS)) + kp;
                    const half2 bv[4] = {
                        sm.B[pair][lane],
                        sm.B[pair][lane + 32],
                        sm.B[pair][lane + 64],
                        sm.B[pair][lane + 96],
                    };
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const half2 av = rail == 0 ?
                                sm.A[buffer][pair][row0 + i] :
                                sm.A_low[buffer][pair][row0 + i];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            hacc[i][j] = __hfma2(av, bv[j], hacc[i][j]);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 partial = __half22float2(hacc[i][j]);
                        const float value = partial.x + partial.y;
                        acc[i][j] += POST_SCALE ? value*scales[j] : value;
                    }
                }
                }
            }

            __syncthreads();
            if (!last) {
                if constexpr (POST_SCALE) {
                    sm.scales[b_r] = __float2half(next_scale);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t packed = next_packed[i];
                    const float q0 = (float) (signed char) (packed & 0xffu);
                    const float q1 = (float) (signed char) ((packed >> 8) & 0xffu);
                    const float q2 = (float) (signed char) ((packed >> 16) & 0xffu);
                    const float q3 = (float) (signed char) ((packed >> 24) & 0xffu);
                    sm.B[b_half*8 + 2*i + 0][b_r] =
                            __floats2half2_rn(
                                    POST_SCALE ? q0 : q0*next_scale,
                                    POST_SCALE ? q1 : q1*next_scale);
                    sm.B[b_half*8 + 2*i + 1][b_r] =
                            __floats2half2_rn(
                                    POST_SCALE ? q2 : q2*next_scale,
                                    POST_SCALE ? q3 : q3*next_scale);
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    output[lane + j*32] = acc[i][j];
                }
            }
        }
        __syncthreads();
    }
}

struct aw_smem_fab_m64_n64 {
    half2 A[2][AW_KSTAGE/2][AW_T64_ROWS];
    half2 B[AW_KSTAGE/2][AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(256, 3)
static void aw_q8_fab_m64_n64(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_fab_m64_n64 sm;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_r = tid >> 2;
    const int a_k = (tid & 3)*8;
    const int b_r = tid >> 2;
    const int b_k = (tid & 3)*8;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_mtiles*n_ntiles;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ntiles;
        const int ntile = work_id - mt_id*n_ntiles;
        const int col0 = ntile*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
        const int input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
        const int n_kblocks = k/AW_KSTAGE;

        float acc[8][2] = {};
        int buffer = 0;
        {
            float4 pa0, pa1;
            const size_t a_base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, a_base, pa0, pa1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + a_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            sm.A[0][a_k/2 + 0][a_r] = __floats2half2_rn(pa0.x, pa0.y);
            sm.A[0][a_k/2 + 1][a_r] = __floats2half2_rn(pa0.z, pa0.w);
            sm.A[0][a_k/2 + 2][a_r] = __floats2half2_rn(pa1.x, pa1.y);
            sm.A[0][a_k/2 + 3][a_r] = __floats2half2_rn(pa1.z, pa1.w);

            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale = __half2float(*(const half *) (tile_stage + b_r*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_r*AW_KSTAGE + b_k;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const uint32_t packed = *(const uint32_t *) (values + i*4);
                const float q0 = (float) (signed char) (packed & 0xffu);
                const float q1 = (float) (signed char) ((packed >> 8) & 0xffu);
                const float q2 = (float) (signed char) ((packed >> 16) & 0xffu);
                const float q3 = (float) (signed char) ((packed >> 24) & 0xffu);
                sm.B[b_k/2 + 2*i + 0][b_r] =
                        __floats2half2_rn(q0*scale, q1*scale);
                sm.B[b_k/2 + 2*i + 1][b_r] =
                        __floats2half2_rn(q2*scale, q3*scale);
            }
        }
        __syncthreads();

        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed[2] = {};
            if (!last) {
                const size_t next_a_base =
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, next_a_base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input + next_a_base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                sm.A[buffer ^ 1][a_k/2 + 0][a_r] =
                        __floats2half2_rn(next_a0.x, next_a0.y);
                sm.A[buffer ^ 1][a_k/2 + 1][a_r] =
                        __floats2half2_rn(next_a0.z, next_a0.w);
                sm.A[buffer ^ 1][a_k/2 + 2][a_r] =
                        __floats2half2_rn(next_a1.x, next_a1.y);
                sm.A[buffer ^ 1][a_k/2 + 3][a_r] =
                        __floats2half2_rn(next_a1.z, next_a1.w);

                const char * next_tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                next_scale = __half2float(*(const half *) (next_tile_stage + b_r*2));
                const char * next_values = next_tile_stage + AW_T64_ROWS*2 +
                        b_r*AW_KSTAGE + b_k;
                next_packed[0] = *(const uint32_t *) (next_values + 0);
                next_packed[1] = *(const uint32_t *) (next_values + 4);
            }

            const int row0 = warp*8;
            #pragma unroll
            for (int chunk = 0; chunk < 2; ++chunk) {
                half2 hacc[8][2];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        hacc[i][j] = __float2half2_rn(0.0f);
                    }
                }
                #pragma unroll
                for (int kp = 0; kp < AW_KSTAGE/4; ++kp) {
                    const int pair = chunk*(AW_KSTAGE/4) + kp;
                    const half2 bv0 = sm.B[pair][lane];
                    const half2 bv1 = sm.B[pair][lane + 32];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const half2 av = sm.A[buffer][pair][row0 + i];
                        hacc[i][0] = __hfma2(av, bv0, hacc[i][0]);
                        hacc[i][1] = __hfma2(av, bv1, hacc[i][1]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 partial0 = __half22float2(hacc[i][0]);
                    const float2 partial1 = __half22float2(hacc[i][1]);
                    acc[i][0] += partial0.x + partial0.y;
                    acc[i][1] += partial1.x + partial1.y;
                }
            }

            __syncthreads();
            if (!last) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t packed = next_packed[i];
                    const float q0 = (float) (signed char) (packed & 0xffu);
                    const float q1 = (float) (signed char) ((packed >> 8) & 0xffu);
                    const float q2 = (float) (signed char) ((packed >> 16) & 0xffu);
                    const float q3 = (float) (signed char) ((packed >> 24) & 0xffu);
                    sm.B[b_k/2 + 2*i + 0][b_r] =
                            __floats2half2_rn(q0*next_scale, q1*next_scale);
                    sm.B[b_k/2 + 2*i + 1][b_r] =
                            __floats2half2_rn(q2*next_scale, q3*next_scale);
                }
                __syncthreads();
            }
            buffer ^= 1;
        }

        const int row0 = warp*8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + i;
            if (row < tile.rows) {
                float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                output[lane] = acc[i][0];
                output[lane + 32] = acc[i][1];
            }
        }
        __syncthreads();
    }
}

union aw_smem_m64_n128_1024 {
    struct {
        float A[2][AW_KSTAGE][AW_T64_ROWS];
        float B[AW_KSTAGE][2*AW_T64_ROWS];
    } stage;
    float red[AW_T64_ROWS][2*AW_T64_ROWS];
};

template<bool BF16_INPUT>
__global__ __launch_bounds__(1024, 1)
static void aw_q8_service_m64_n128_1024(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles, int n, int k, const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m64_n128_1024 sm;

    const int tid = threadIdx.x;
    const int split = tid >> 9;
    const int local_tid = tid & 511;
    const int row0 = (local_tid >> 5)*4;
    const int lane = local_tid & 31;

    n_mtiles = n_mtiles_dev != nullptr ? *n_mtiles_dev : n_mtiles;
    const int n_ngroups = n/(2*AW_T64_ROWS);
    const int n_work = n_mtiles*n_ngroups;
    for (int work_id = blockIdx.x; work_id < n_work; work_id += gridDim.x) {
        const int mt_id = work_id/n_ngroups;
        const int ngroup = work_id - mt_id*n_ngroups;
        const int col0 = ngroup*2*AW_T64_ROWS;
        const aw_tile_desc tile = tiles[mt_id];
        const aw_work_desc desc = descs[tile.work];
        const int n_kblocks = k/AW_KSTAGE;

        int a_r = 0;
        int a_k = 0;
        int input_row = 0;
        if (tid < 256) {
            a_r = tid >> 2;
            a_k = (tid & 3)*8;
            const int local_row = tile.row + (a_r < tile.rows ? a_r : tile.rows - 1);
            input_row = desc.input_rows != nullptr ? desc.input_rows[local_row] : local_row;
            float4 a0, a1;
            const size_t base = (size_t) input_row*k + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(desc.input, base, a0, a1);
            } else {
                const float4 * input4 = (const float4 *) ((const float *) desc.input + base);
                a0 = input4[0];
                a1 = input4[1];
            }
            const float * va0 = (const float *) &a0;
            const float * va1 = (const float *) &a1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[0][a_k + i][a_r] = va0[i];
                sm.stage.A[0][a_k + i + 4][a_r] = va1[i];
            }
        }

        int b_r = 0;
        int b_quarter = 0;
        if (tid >= 256 && tid < 768) {
            const int b_tid = tid - 256;
            b_r = b_tid >> 2;
            b_quarter = b_tid & 3;
            const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
            const int b_local = b_r % AW_T64_ROWS;
            const char * tile_stage = desc.weight +
                    (size_t) ntile*n_kblocks*AW_T64_STAGE_BYTES;
            const float scale = __half2float(*(const half *) (tile_stage + b_local*2));
            const char * values = tile_stage + AW_T64_ROWS*2 +
                    b_local*AW_KSTAGE + b_quarter*8;
            #pragma unroll
            for (int word = 0; word < 2; ++word) {
                const uint32_t packed = *(const uint32_t *) (values + word*4);
                #pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    sm.stage.B[b_quarter*8 + word*4 + byte][b_r] =
                            (float) (signed char) ((packed >> (byte*8)) & 0xffu)*scale;
                }
            }
        }
        __syncthreads();

        float acc[4][4] = {};
        int buffer = 0;
        for (int stage = 0; stage < n_kblocks; ++stage) {
            const bool last = stage + 1 == n_kblocks;
            float4 next_a0, next_a1;
            float next_scale = 0.0f;
            uint32_t next_packed0 = 0;
            uint32_t next_packed1 = 0;
            if (!last && tid < 256) {
                const size_t base =
                        (size_t) input_row*k + (stage + 1)*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(desc.input, base, next_a0, next_a1);
                } else {
                    const float4 * input4 =
                            (const float4 *) ((const float *) desc.input + base);
                    next_a0 = input4[0];
                    next_a1 = input4[1];
                }
                const float * va0 = (const float *) &next_a0;
                const float * va1 = (const float *) &next_a1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.stage.A[buffer ^ 1][a_k + i][a_r] = va0[i];
                    sm.stage.A[buffer ^ 1][a_k + i + 4][a_r] = va1[i];
                }
            }
            if (!last && tid >= 256 && tid < 768) {
                const int ntile = 2*ngroup + b_r/AW_T64_ROWS;
                const int b_local = b_r % AW_T64_ROWS;
                const char * tile_stage = desc.weight +
                        ((size_t) ntile*n_kblocks + stage + 1)*AW_T64_STAGE_BYTES;
                next_scale = __half2float(*(const half *) (tile_stage + b_local*2));
                const char * values = tile_stage + AW_T64_ROWS*2 +
                        b_local*AW_KSTAGE + b_quarter*8;
                next_packed0 = *(const uint32_t *) values;
                next_packed1 = *(const uint32_t *) (values + 4);
            }

            #pragma unroll
            for (int kk = 0; kk < AW_KSTAGE/2; ++kk) {
                const int ks = split*(AW_KSTAGE/2) + kk;
                const float4 av = *(const float4 *) &sm.stage.A[buffer][ks][row0];
                const float bv[4] = {
                    sm.stage.B[ks][lane],
                    sm.stage.B[ks][lane + 32],
                    sm.stage.B[ks][lane + 64],
                    sm.stage.B[ks][lane + 96],
                };
                const float va[4] = { av.x, av.y, av.z, av.w };
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += va[i]*bv[j];
                    }
                }
            }
            __syncthreads();
            if (!last && tid >= 256 && tid < 768) {
                const uint32_t packed[2] = { next_packed0, next_packed1 };
                #pragma unroll
                for (int word = 0; word < 2; ++word) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        sm.stage.B[b_quarter*8 + word*4 + byte][b_r] =
                                (float) (signed char)
                                ((packed[word] >> (byte*8)) & 0xffu)*next_scale;
                    }
                }
            }
            if (!last) {
                __syncthreads();
            }
            buffer ^= 1;
        }

        if (split == 1) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sm.red[row0 + i][lane + j*32] = acc[i][j];
                }
            }
        }
        __syncthreads();
        if (split == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = row0 + i;
                if (row < tile.rows) {
                    float * output = desc.output + (size_t) (tile.row + row)*n + col0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        output[lane + j*32] = acc[i][j] + sm.red[row][lane + j*32];
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

union aw_smem_m16_rail_group {
    float A[2][AW_KSTAGE/2][16];
    float red[16][AW_T64_ROWS];
};

struct aw_smem_m16_rail_pair {
    aw_smem_m16_rail_group group[2];
};

__device__ __forceinline__ void aw_cohortrail_group_sync(
        int group);

template<int WORD>
__device__ __forceinline__ uint32_t aw_uint4_word(
        const uint4 & value) {
    static_assert(WORD >= 0 && WORD < 4);
    if constexpr (WORD == 0) {
        return value.x;
    } else if constexpr (WORD == 1) {
        return value.y;
    } else if constexpr (WORD == 2) {
        return value.z;
    } else {
        return value.w;
    }
}

template<bool BF16_INPUT>
__global__ __launch_bounds__(128, 5)
static void aw_q8_service_m16_rail_pair(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        int n_mtiles,
        int n,
        int k,
        const int * n_mtiles_dev = nullptr) {
    __shared__ aw_smem_m16_rail_pair sm;

    const int group = threadIdx.x >> 6;
    const int local_tid = threadIdx.x & 63;
    const int kgrp = local_tid >> 5;
    const int lane = local_tid & 31;
    const int a_r = lane >> 1;
    const int a_k = (lane & 1)*8;
    const int b_r0 = lane;
    const int b_r1 = lane + 32;

    n_mtiles = n_mtiles_dev != nullptr ?
            *n_mtiles_dev : n_mtiles;
    const int n_ntiles = n/AW_T64_ROWS;
    const int ntile_shift = __ffs(n_ntiles) - 1;
    const int ntile_mask = n_ntiles - 1;
    const int n_pairs = (n_mtiles + 1)/2;
    const int n_work = n_pairs*n_ntiles;
    for (int work_id = blockIdx.x;
            work_id < n_work;
            work_id += gridDim.x) {
        const int pair = work_id >> ntile_shift;
        const int ntile = work_id & ntile_mask;
        const int col0 = ntile*AW_T64_ROWS;
        const int tile_index = pair*2 + group;
        const bool active = tile_index < n_mtiles;
        const aw_tile_desc tile =
                tiles[active ?
                    tile_index : n_mtiles - 1];
        const aw_work_desc desc =
                descs[tile.work];
        const int local_row =
                tile.row +
                (a_r < tile.rows ?
                    a_r : tile.rows - 1);
        const int input_row =
                desc.input_rows != nullptr ?
                desc.input_rows[local_row] :
                local_row;
        const int n_kblocks = k/AW_KSTAGE;

        const char * weight_stage =
                desc.weight +
                (size_t) ntile*n_kblocks*
                    AW_T64_STAGE_BYTES;
        size_t input_stage_base =
                (size_t) input_row*k +
                kgrp*(AW_KSTAGE/2) +
                a_k;

        auto fetch_q = [&](const char * tile_stage,
                uint4 & q0,
                uint4 & q1,
                float & scale0,
                float & scale1) {
            scale0 = __half2float(
                    *(const half *)
                    (tile_stage + b_r0*2));
            scale1 = __half2float(
                    *(const half *)
                    (tile_stage + b_r1*2));
            const char * values =
                    tile_stage +
                    AW_T64_ROWS*2;
            q0 = __ldg((const uint4 *)
                    (values +
                     b_r0*AW_KSTAGE +
                     kgrp*(AW_KSTAGE/2)));
            q1 = __ldg((const uint4 *)
                    (values +
                     b_r1*AW_KSTAGE +
                     kgrp*(AW_KSTAGE/2)));
        };

        float acc0[16] = {};
        float acc1[16] = {};
        uint4 q0, q1;
        float scale0, scale1;
        fetch_q(weight_stage,
                q0, q1, scale0, scale1);
        for (int stage = 0;
                stage < n_kblocks; ++stage) {
            float4 pa0, pa1;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input, input_stage_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *)
                            desc.input +
                         input_stage_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }
            const float * a0 =
                    (const float *) &pa0;
            const float * a1 =
                    (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.group[group].A[kgrp]
                        [a_k + i][a_r] =
                        a0[i];
                sm.group[group].A[kgrp]
                        [a_k + i + 4][a_r] =
                        a1[i];
            }
            __syncwarp();

            uint4 next_q0 = {};
            uint4 next_q1 = {};
            float next_scale0 = 0.0f;
            float next_scale1 = 0.0f;
            const bool last =
                    stage + 1 == n_kblocks;
            if (!last) {
                fetch_q(
                        weight_stage +
                            AW_T64_STAGE_BYTES,
                        next_q0, next_q1,
                        next_scale0,
                        next_scale1);
            }

            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                uint32_t packed0;
                uint32_t packed1;
                if (word == 0) {
                    packed0 =
                            aw_uint4_word<0>(q0);
                    packed1 =
                            aw_uint4_word<0>(q1);
                } else if (word == 1) {
                    packed0 =
                            aw_uint4_word<1>(q0);
                    packed1 =
                            aw_uint4_word<1>(q1);
                } else if (word == 2) {
                    packed0 =
                            aw_uint4_word<2>(q0);
                    packed1 =
                            aw_uint4_word<2>(q1);
                } else {
                    packed0 =
                            aw_uint4_word<3>(q0);
                    packed1 =
                            aw_uint4_word<3>(q1);
                }
                #pragma unroll
                for (int byte = 0;
                        byte < 4; ++byte) {
                    const int kk =
                            word*4 + byte;
                    const float bv0 =
                            __fmul_rn(
                                (float)
                                (signed char)
                                ((packed0 >>
                                  (byte*8)) & 0xffu),
                                scale0);
                    const float bv1 =
                            __fmul_rn(
                                (float)
                                (signed char)
                                ((packed1 >>
                                  (byte*8)) & 0xffu),
                                scale1);
                    #pragma unroll
                    for (int rb = 0;
                            rb < 4; ++rb) {
                        const float4 av =
                                *(const float4 *)
                                &sm.group[group].
                                    A[kgrp][kk][rb*4];
                        acc0[rb*4 + 0] =
                                __fmaf_rn(
                                    av.x, bv0,
                                    acc0[rb*4 + 0]);
                        acc1[rb*4 + 0] =
                                __fmaf_rn(
                                    av.x, bv1,
                                    acc1[rb*4 + 0]);
                        acc0[rb*4 + 1] =
                                __fmaf_rn(
                                    av.y, bv0,
                                    acc0[rb*4 + 1]);
                        acc1[rb*4 + 1] =
                                __fmaf_rn(
                                    av.y, bv1,
                                    acc1[rb*4 + 1]);
                        acc0[rb*4 + 2] =
                                __fmaf_rn(
                                    av.z, bv0,
                                    acc0[rb*4 + 2]);
                        acc1[rb*4 + 2] =
                                __fmaf_rn(
                                    av.z, bv1,
                                    acc1[rb*4 + 2]);
                        acc0[rb*4 + 3] =
                                __fmaf_rn(
                                    av.w, bv0,
                                    acc0[rb*4 + 3]);
                        acc1[rb*4 + 3] =
                                __fmaf_rn(
                                    av.w, bv1,
                                    acc1[rb*4 + 3]);
                    }
                }
            }
            __syncwarp();
            if (!last) {
                q0 = next_q0;
                q1 = next_q1;
                scale0 = next_scale0;
                scale1 = next_scale1;
                weight_stage +=
                        AW_T64_STAGE_BYTES;
                input_stage_base += AW_KSTAGE;
            }
        }

        aw_cohortrail_group_sync(group);
        if (kgrp == 1) {
            #pragma unroll
            for (int row = 0; row < 16; ++row) {
                sm.group[group].red[row][b_r0] =
                        acc0[row];
                sm.group[group].red[row][b_r1] =
                        acc1[row];
            }
        }
        aw_cohortrail_group_sync(group);
        if (active && kgrp == 0) {
            #pragma unroll
            for (int row = 0; row < 16; ++row) {
                if (row < tile.rows) {
                    float * output =
                            desc.output +
                            (size_t)
                                (tile.row + row)*n +
                            col0;
                    output[b_r0] =
                            __fadd_rn(
                                acc0[row],
                                sm.group[group].
                                    red[row][b_r0]);
                    output[b_r1] =
                            __fadd_rn(
                                acc1[row],
                                sm.group[group].
                                    red[row][b_r1]);
                }
            }
        }
        aw_cohortrail_group_sync(group);
    }
}

union aw_smem_m16_bundle {
    struct {
        float A[2][AW_KSTAGE][16];
        float B[AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[2][16][AW_T64_ROWS];
};

template<bool BF16_INPUT, bool T64_LAYOUT>
__global__ __launch_bounds__(128, 5)
static void aw_q8_service_m16_bundle(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        const aw_tile_desc * __restrict__ pairs,
        int n_pairs, int n, int k,
        const int * n_pairs_dev = nullptr) {
    __shared__ aw_smem_m16_bundle sm;

    const int group = threadIdx.x >> 6;
    const int local_tid = threadIdx.x & 63;
    const int kgrp = local_tid >> 5;
    const int lane = local_tid & 31;
    const int rm = lane >> 3;
    const int nt = lane & 7;
    const int a_r = local_tid >> 2;
    const int a_k = (local_tid & 3)*8;
    const int b_r = local_tid;

    n_pairs = n_pairs_dev != nullptr ?
            *n_pairs_dev : n_pairs;
    const int n_ntiles = n/AW_T64_ROWS;
    const int n_work = n_pairs*n_ntiles;
    for (int work_id = blockIdx.x;
            work_id < n_work;
            work_id += gridDim.x) {
        const int pair_id = work_id/n_ntiles;
        const int col0 =
                (work_id % n_ntiles)*AW_T64_ROWS;
        const aw_tile_desc pair =
                pairs[pair_id];
        const aw_tile_desc tile =
                tiles[group == 0 ?
                    pair.work : pair.row];
        const aw_work_desc desc =
                descs[tile.work];
        const int local_row =
                tile.row +
                (a_r < tile.rows ?
                    a_r : tile.rows - 1);
        const int input_row =
                desc.input_rows != nullptr ?
                desc.input_rows[local_row] :
                local_row;

        float acc[4][8] = {};
        for (int stage = 0;
                stage < k/AW_KSTAGE;
                ++stage) {
            float4 pa0, pa1;
            const size_t input_base =
                    (size_t) input_row*k +
                    stage*AW_KSTAGE + a_k;
            if constexpr (BF16_INPUT) {
                aw_load_bf16x8(
                        desc.input, input_base,
                        pa0, pa1);
            } else {
                const float4 * input4 =
                        (const float4 *)
                        ((const float *) desc.input +
                         input_base);
                pa0 = input4[0];
                pa1 = input4[1];
            }

            const float * a0 =
                    (const float *) &pa0;
            const float * a1 =
                    (const float *) &pa1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.stage.A[group][
                    a_k + i][a_r] = a0[i];
                sm.stage.A[group][
                    a_k + i + 4][a_r] = a1[i];
            }
            if (group == 0) {
                float scale;
                const char * values;
                if constexpr (T64_LAYOUT) {
                    const int n_kblocks =
                            k/AW_KSTAGE;
                    const char * tile_stage =
                            desc.weight +
                            ((size_t)
                                (col0/AW_T64_ROWS)*
                                n_kblocks + stage)*
                                AW_T64_STAGE_BYTES;
                    scale = __half2float(
                            *(const half *)
                            (tile_stage + b_r*2));
                    values = tile_stage +
                            AW_T64_ROWS*2 +
                            b_r*AW_KSTAGE;
                } else {
                    const char * block =
                            desc.weight +
                            ((size_t)
                                (col0 + b_r)*
                                (k/AW_KSTAGE) +
                             stage)*
                                AW_Q8_BLOCK_BYTES;
                    scale = __half2float(
                            *(const half *) block);
                    values = block + 2;
                }
                #pragma unroll
                for (int i = 0;
                        i < AW_KSTAGE/2; ++i) {
                    const unsigned short
                            pair_values =
                            *(const unsigned short *)
                            (values + 2*i);
                    sm.stage.B[2*i][b_r] =
                            (float) (signed char)
                            (pair_values & 0xff)*
                            scale;
                    sm.stage.B[2*i + 1][b_r] =
                            (float) (signed char)
                            (pair_values >> 8)*
                            scale;
                }
            }
            __syncthreads();

            #pragma unroll
            for (int kk = 0;
                    kk < AW_KSTAGE/2; ++kk) {
                const int ks =
                        kgrp*(AW_KSTAGE/2) + kk;
                const float4 av =
                        *(const float4 *)
                        &sm.stage.A[group][ks][
                            rm*4];
                const float4 bv0 =
                        *(const float4 *)
                        &sm.stage.B[ks][nt*8];
                const float4 bv1 =
                        *(const float4 *)
                        &sm.stage.B[ks][
                            nt*8 + 4];
                const float a[4] = {
                    av.x, av.y, av.z, av.w
                };
                const float b[8] = {
                    bv0.x, bv0.y, bv0.z, bv0.w,
                    bv1.x, bv1.y, bv1.z, bv1.w
                };
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] +=
                                a[i]*b[j];
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
                    sm.red[group][rm*4 + i][
                        nt*8 + j] = acc[i][j];
                }
            }
        }
        __syncthreads();
        if (kgrp == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = rm*4 + i;
                if (row < tile.rows) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        desc.output[
                            (size_t)
                                (tile.row + row)*n +
                            col0 + nt*8 + j] =
                                acc[i][j] +
                                sm.red[group][row][
                                    nt*8 + j];
                    }
                }
            }
        }
        __syncthreads();
    }
}

template<int BARRIER, int THREADS>
__device__ __forceinline__ void aw_cohortrail_arrive() {
    asm volatile(
            "bar.arrive %0, %1;"
            :: "n"(BARRIER), "n"(THREADS)
            : "memory");
}

template<int BARRIER, int THREADS>
__device__ __forceinline__ void aw_cohortrail_sync() {
    asm volatile(
            "bar.sync %0, %1;"
            :: "n"(BARRIER), "n"(THREADS)
            : "memory");
}

template<int THREADS>
__device__ __forceinline__ void aw_cohortrail_arrive(
        int barrier) {
    asm volatile(
            "bar.arrive %0, %1;"
            :: "r"(barrier), "n"(THREADS)
            : "memory");
}

template<int THREADS>
__device__ __forceinline__ void aw_cohortrail_sync(
        int barrier) {
    asm volatile(
            "bar.sync %0, %1;"
            :: "r"(barrier), "n"(THREADS)
            : "memory");
}

__device__ __forceinline__ void aw_cohortrail_group_sync(
        int group) {
    aw_cohortrail_sync<64>(4 + group);
}

template<int COHORT, bool DOUBLE_A>
union aw_smem_m16_cohortrail {
    struct {
        float A[DOUBLE_A ? 2 : 1][COHORT]
                [AW_KSTAGE][16];
        float B[2][AW_KSTAGE][AW_T64_ROWS];
    } stage;
    float red[COHORT][16][AW_T64_ROWS];
};

template<bool BF16_INPUT, int COHORT,
        int PRODUCER_THREADS,
        int FIXED_N = 0,
        int FIXED_K = 0>
__global__ __launch_bounds__(
        COHORT*64 + 64, COHORT == 2 ? 3 : 2)
static void aw_q8_service_m16_cohortrail(
        const aw_work_desc * __restrict__ descs,
        const aw_tile_desc * __restrict__ tiles,
        const aw_m16_cohort * __restrict__ cohorts,
        int n_cohorts,
        int n,
        int k,
        const int * n_cohorts_dev = nullptr) {
    static_assert(COHORT >= 2 && COHORT <= 4);
    static_assert(PRODUCER_THREADS == 32 ||
            PRODUCER_THREADS == 64);
    static_assert(FIXED_N == 0 ||
            FIXED_N % AW_T64_ROWS == 0);
    static_assert(FIXED_K == 0 ||
            FIXED_K % AW_KSTAGE == 0);
    static_assert(FIXED_K == 0 ||
            (FIXED_K/AW_KSTAGE) % 2 == 0);
    constexpr bool RAIL_A = true;
    constexpr bool DOUBLE_A = false;
    constexpr int CONSUMER_THREADS = COHORT*64;
    constexpr int CTA_THREADS =
            CONSUMER_THREADS + PRODUCER_THREADS;
    __shared__ aw_smem_m16_cohortrail<
            COHORT, DOUBLE_A> sm;
    const int n_stride =
            FIXED_N != 0 ? FIXED_N : n;
    const int k_stride =
            FIXED_K != 0 ? FIXED_K : k;

    n_cohorts = n_cohorts_dev != nullptr ?
            *n_cohorts_dev : n_cohorts;
    const int n_ntiles =
            n_stride/AW_T64_ROWS;
    const int ntile_shift = __ffs(n_ntiles) - 1;
    const int ntile_mask = n_ntiles - 1;
    const int n_work = n_cohorts*n_ntiles;
    for (int work_id = blockIdx.x;
            work_id < n_work;
            work_id += gridDim.x) {
        const int cohort_id =
                work_id >> ntile_shift;
        const int col0 =
                (work_id & ntile_mask)*
                AW_T64_ROWS;
        const aw_m16_cohort * cohort =
                cohorts + cohort_id;
        const int n_kblocks =
                k_stride/AW_KSTAGE;

        if (threadIdx.x < CONSUMER_THREADS) {
            const int group = threadIdx.x >> 6;
            const int local_tid = threadIdx.x & 63;
            const int kgrp = local_tid >> 5;
            const int lane = local_tid & 31;
            const int rm = lane >> 3;
            const int nt = lane & 7;
            const int a_r = RAIL_A ?
                    lane >> 1 : local_tid >> 2;
            const int a_k = RAIL_A ?
                    kgrp*(AW_KSTAGE/2) +
                        (lane & 1)*8 :
                    (local_tid & 3)*8;
            const aw_tile_desc tile =
                    tiles[cohort->tiles[group]];
            const aw_work_desc desc =
                    descs[tile.work];
            const int local_row =
                    tile.row +
                    (a_r < tile.rows ?
                        a_r : tile.rows - 1);
            const int input_row =
                    desc.input_rows != nullptr ?
                    desc.input_rows[local_row] :
                    local_row;

            auto fetch_a = [&](int stage,
                    float4 & pa0, float4 & pa1) {
                const size_t base =
                        (size_t) input_row*k_stride +
                        stage*AW_KSTAGE + a_k;
                if constexpr (BF16_INPUT) {
                    aw_load_bf16x8(
                            desc.input, base,
                            pa0, pa1);
                } else {
                    const float4 * input4 =
                            (const float4 *)
                            ((const float *) desc.input +
                             base);
                    pa0 = input4[0];
                    pa1 = input4[1];
                }
            };
            auto stage_a = [&](int buffer,
                    const float4 & pa0,
                    const float4 & pa1) {
                const float * a0 =
                        (const float *) &pa0;
                const float * a1 =
                        (const float *) &pa1;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sm.stage.A[buffer][group]
                            [a_k + i][a_r] = a0[i];
                    sm.stage.A[buffer][group]
                            [a_k + i + 4][a_r] =
                            a1[i];
                }
            };

            float acc[4][8] = {};
            float4 pa0, pa1;
            fetch_a(0, pa0, pa1);
            stage_a(0, pa0, pa1);
            #pragma unroll 1
            for (int stage = 0;
                    stage < n_kblocks; ++stage) {
                const int buffer = stage & 1;
                aw_cohortrail_sync<CTA_THREADS>(
                        buffer);

                float4 next_a0, next_a1;
                const bool last =
                        stage + 1 == n_kblocks;
                if (!last) {
                    fetch_a(stage + 1,
                            next_a0, next_a1);
                }

                #pragma unroll
                for (int kk = 0;
                        kk < AW_KSTAGE/2; ++kk) {
                    const int ks =
                            kgrp*(AW_KSTAGE/2) + kk;
                    const float4 av =
                            *(const float4 *)
                            &sm.stage.A[
                                DOUBLE_A ? buffer : 0]
                                [group][ks][rm*4];
                    const float4 bv0 =
                            *(const float4 *)
                            &sm.stage.B[buffer][ks][
                                nt*8];
                    const float4 bv1 =
                            *(const float4 *)
                            &sm.stage.B[buffer][ks][
                                nt*8 + 4];
                    const float a[4] = {
                        av.x, av.y, av.z, av.w
                    };
                    const float b[8] = {
                        bv0.x, bv0.y, bv0.z, bv0.w,
                        bv1.x, bv1.y, bv1.z, bv1.w
                    };
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            acc[i][j] += a[i]*b[j];
                        }
                    }
                }

                if constexpr (!RAIL_A) {
                    aw_cohortrail_group_sync(group);
                }
                aw_cohortrail_arrive<CTA_THREADS>(
                        2 + buffer);
                if (!last) {
                    stage_a(DOUBLE_A ?
                            (buffer ^ 1) : 0,
                            next_a0, next_a1);
                }
            }

            aw_cohortrail_sync<0, CTA_THREADS>();
            if (kgrp == 1) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sm.red[group][rm*4 + i][
                                nt*8 + j] =
                                acc[i][j];
                    }
                }
            }
            aw_cohortrail_group_sync(group);
            if (kgrp == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = rm*4 + i;
                    if (row < tile.rows) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            desc.output[
                                    (size_t)
                                        (tile.row + row)*
                                        n_stride +
                                    col0 + nt*8 + j] =
                                    __fadd_rn(
                                        acc[i][j],
                                        sm.red[group][row][
                                            nt*8 + j]);
                        }
                    }
                }
            }
            aw_cohortrail_sync<1, CTA_THREADS>();
        } else {
            const int producer_tid =
                    threadIdx.x - CONSUMER_THREADS;
            const aw_tile_desc first_tile =
                    tiles[cohort->tiles[0]];
            const aw_work_desc first_desc =
                    descs[first_tile.work];
            #pragma unroll 1
            for (int stage = 0;
                    stage < n_kblocks; ++stage) {
                const int buffer = stage & 1;
                const char * tile_stage =
                        first_desc.weight +
                        ((size_t)
                            (col0/AW_T64_ROWS)*
                            n_kblocks + stage)*
                            AW_T64_STAGE_BYTES;
                if constexpr (PRODUCER_THREADS == 64) {
                    const float scale =
                            __half2float(
                                *(const half *)
                                (tile_stage +
                                 producer_tid*2));
                    const char * values =
                            tile_stage +
                            AW_T64_ROWS*2 +
                            producer_tid*AW_KSTAGE;
                    const uint4 packed0 =
                            __ldg((const uint4 *)
                                values);
                    const uint4 packed1 =
                            __ldg((const uint4 *)
                                (values + 16));
                    if (stage >= 2) {
                        aw_cohortrail_sync<
                                CTA_THREADS>(
                                2 + buffer);
                    }
                    const uint32_t packed[8] = {
                        packed0.x, packed0.y,
                        packed0.z, packed0.w,
                        packed1.x, packed1.y,
                        packed1.z, packed1.w,
                    };
                    #pragma unroll
                    for (int word = 0;
                            word < 8; ++word) {
                        #pragma unroll
                        for (int byte = 0;
                                byte < 4; ++byte) {
                            sm.stage.B[buffer]
                                    [word*4 + byte]
                                    [producer_tid] =
                                    (float) (signed char)
                                    ((packed[word] >>
                                      (byte*8)) & 0xffu)*
                                    scale;
                        }
                    }
                } else {
                    if (stage >= 2) {
                        aw_cohortrail_sync<
                                CTA_THREADS>(
                                2 + buffer);
                    }
                    for (int b_r = producer_tid;
                            b_r < AW_T64_ROWS;
                            b_r += PRODUCER_THREADS) {
                        const float scale =
                                __half2float(
                                    *(const half *)
                                    (tile_stage + b_r*2));
                        const char * values =
                                tile_stage +
                                AW_T64_ROWS*2 +
                                b_r*AW_KSTAGE;
                        #pragma unroll
                        for (int i = 0;
                                i < AW_KSTAGE/2; ++i) {
                            const unsigned short packed =
                                    *(const unsigned short *)
                                    (values + 2*i);
                            sm.stage.B[buffer][2*i][b_r] =
                                    (float) (signed char)
                                    (packed & 0xff)*scale;
                            sm.stage.B[buffer][2*i + 1][b_r] =
                                    (float) (signed char)
                                    (packed >> 8)*scale;
                        }
                    }
                }
                aw_cohortrail_arrive<CTA_THREADS>(
                        buffer);
            }

            if constexpr (FIXED_K != 0) {
                aw_cohortrail_sync<2,
                        CTA_THREADS>();
                aw_cohortrail_sync<3,
                        CTA_THREADS>();
            } else {
                const int drain_begin =
                        n_kblocks > 2 ?
                        n_kblocks - 2 : 0;
                for (int stage = drain_begin;
                        stage < n_kblocks; ++stage) {
                    aw_cohortrail_sync<
                            CTA_THREADS>(
                            2 + (stage & 1));
                }
            }
            aw_cohortrail_arrive<0, CTA_THREADS>();
            aw_cohortrail_sync<1, CTA_THREADS>();
        }
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

__global__ static void aw_pack_requests_f16(const float * src, void * dst, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        ((half *) dst)[i] = __float2half(src[i]);
    }
}

__global__ __launch_bounds__(128, 2)
static void aw_headfold_gdn_segment(
        const float * q,
        const float * k,
        const float * v,
        const float * gate,
        const float * beta,
        float * state,
        float * output,
        int tokens) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z*blockDim.y + threadIdx.y;
    const int qk_head = head % AW_HEADFOLD_QK_HEADS;
    state += ((size_t) head*AW_GDN_HEAD_DIM + col)*AW_GDN_HEAD_DIM;
    output += (size_t) head*AW_GDN_HEAD_DIM;

    float state_shard[4];
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        state_shard[row] = state[row*32 + lane];
    }

    constexpr float scale = 0.08838834764831845f;
    for (int token = 0; token < tokens; ++token) {
        const float * q_token =
                q + ((size_t) token*AW_HEADFOLD_QK_HEADS + qk_head)*
                    AW_GDN_HEAD_DIM;
        const float * k_token =
                k + ((size_t) token*AW_HEADFOLD_QK_HEADS + qk_head)*
                    AW_GDN_HEAD_DIM;
        const float * v_token =
                v + ((size_t) token*AW_HEADFOLD_VALUE_HEADS + head)*
                    AW_GDN_HEAD_DIM;
        const size_t scalar =
                (size_t) token*AW_HEADFOLD_VALUE_HEADS + head;
        const unsigned int mask = __activemask();
        const float beta_value = __shfl_sync(
                mask, lane == 0 ? beta[scalar] : 0.0f, 0, 32);
        const float gate_value = __shfl_sync(
                mask, lane == 0 ? expf(gate[scalar]) : 0.0f, 0, 32);
        const float v_col = __shfl_sync(
                mask, lane == 0 ? v_token[col] : 0.0f, 0, 32);

        float q_shard[4];
        float k_shard[4];
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            const int index = row*32 + lane;
            q_shard[row] = q_token[index];
            k_shard[row] = k_token[index];
        }

        float kv = 0.0f;
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            kv += state_shard[row]*k_shard[row];
        }
        kv = warp_reduce_sum<32>(kv);
        const float delta = (v_col - gate_value*kv)*beta_value;

        float attention = 0.0f;
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            state_shard[row] =
                    gate_value*state_shard[row] +
                    k_shard[row]*delta;
            attention += state_shard[row]*q_shard[row];
        }
        attention = warp_reduce_sum<32>(attention);
        if (lane == 0) {
            output[col] = attention*scale;
        }
        output += AW_HEADFOLD_VALUE_HEADS*AW_GDN_HEAD_DIM;
    }

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        state[row*32 + lane] = state_shard[row];
    }
}

__global__ static void aw_swiglu(const float * gate, const float * up, float * output, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const float g = gate[i];
        output[i] = (g/(1.0f + expf(-g)))*up[i];
    }
}

__global__ static void aw_swiglu_tiles(
        const aw_work_desc * gate_descs,
        const aw_work_desc * up_descs,
        const aw_tile_desc * tiles,
        int tile_count,
        float * middle,
        int n) {
    for (int tile_index = blockIdx.x; tile_index < tile_count;
            tile_index += gridDim.x) {
        const aw_tile_desc tile = tiles[tile_index];
        const aw_work_desc gate_desc = gate_descs[tile.work];
        const aw_work_desc up_desc = up_descs[tile.work];
        const int count = tile.rows*n;
        for (int index = threadIdx.x; index < count;
                index += blockDim.x) {
            const int row = index/n;
            const int col = index - row*n;
            const size_t local =
                    (size_t) (tile.row + row)*n + col;
            const float gate = gate_desc.output[local];
            middle[
                    (size_t) (gate_desc.row_offset + tile.row + row)*n +
                    col] =
                    (gate/(1.0f + expf(-gate)))*
                    up_desc.output[local];
        }
    }
}

__global__ static void aw_swiglu_panel(
        const float * gate,
        const float * up,
        float * middle,
        int rows,
        int panel_n,
        int output_n,
        int output_col) {
    const size_t count = (size_t) rows*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < count; i += (size_t) gridDim.x*blockDim.x) {
        const int row = (int) (i/panel_n);
        const int col = (int) (i - (size_t) row*panel_n);
        const float g = gate[i];
        middle[(size_t) row*output_n + output_col + col] =
                (g/(1.0f + expf(-g)))*up[i];
    }
}

__global__ static void aw_live_swiglu_panel(
        const float * gate,
        const float * up,
        float * middle,
        const int32_t * total_routes,
        int panel_n,
        int output_n,
        int output_col) {
    const size_t count = (size_t) *total_routes*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < count; i += (size_t) gridDim.x*blockDim.x) {
        const int row = (int) (i/panel_n);
        const int col = (int) (i - (size_t) row*panel_n);
        const float g = gate[i];
        middle[(size_t) row*output_n + output_col + col] =
                (g/(1.0f + expf(-g)))*up[i];
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

__global__ static void aw_owner_reduce_bf16_panel(
        const float * routes,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * output,
        int tokens,
        int panel_n,
        int output_n,
        int output_col) {
    const size_t count = (size_t) tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < count; i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col = (int) (i - (size_t) token*panel_n);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < AW_ROUTES_PER_OWNER_TOKEN; ++rank) {
            const int slot = token*AW_ROUTES_PER_OWNER_TOKEN + rank;
            sum += route_weights[slot]*
                    routes[(size_t) token_routes[slot]*panel_n + col];
        }
        output[(size_t) token*output_n + output_col + col] =
                aw_float_to_bf16(sum);
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

static constexpr int AW_ROUTE_SEGMENT_SLOTS = 256;

__global__ static void aw_live_count_route_segments(
        const int32_t * ids,
        int32_t * counts,
        int32_t * segment_counts,
        int cells,
        int owner,
        int tokens,
        int segments_per_cell) {
    __shared__ int local_counts[AW_PRIMARY_PER_GPU];

    if (threadIdx.x < AW_PRIMARY_PER_GPU) {
        local_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    const int segment = blockIdx.x;
    const int cell = segment/segments_per_cell;
    const int cell_segment = segment - cell*segments_per_cell;
    if (cell < cells) {
        const int cell_slot_begin = cell*tokens*8;
        const int cell_slot_end = cell_slot_begin + tokens*8;
        const int slot =
                cell_slot_begin +
                cell_segment*AW_ROUTE_SEGMENT_SLOTS +
                threadIdx.x;
        if (slot < cell_slot_end) {
            const int local_expert =
                    ids[slot] - owner*AW_PRIMARY_PER_GPU;
            if (local_expert >= 0 &&
                    local_expert < AW_PRIMARY_PER_GPU) {
                atomicAdd(&local_counts[local_expert], 1);
            }
        }
    }
    __syncthreads();

    if (cell < cells && threadIdx.x < AW_PRIMARY_PER_GPU) {
        const int count = local_counts[threadIdx.x];
        segment_counts[
                (size_t) segment*AW_PRIMARY_PER_GPU +
                threadIdx.x] = count;
        if (count != 0) {
            atomicAdd(&counts[
                        cell*AW_PRIMARY_PER_GPU +
                        threadIdx.x], count);
        }
    }
}

__global__ static void aw_live_prefix_route_segments(
        const int32_t * offsets,
        const int32_t * segment_counts,
        int32_t * segment_offsets,
        int cells,
        int segments_per_cell) {
    const int cell = blockIdx.x;
    const int expert = threadIdx.x;
    if (cell >= cells || expert >= AW_PRIMARY_PER_GPU) {
        return;
    }

    int row = offsets[cell*AW_PRIMARY_PER_GPU + expert];
    for (int segment = 0; segment < segments_per_cell; ++segment) {
        const size_t index =
                ((size_t) cell*segments_per_cell + segment)*
                AW_PRIMARY_PER_GPU + expert;
        segment_offsets[index] = row;
        row += segment_counts[index];
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
        aw_tile_desc * tiles_warp,
        aw_tile_desc * tiles_m8,
        aw_tile_desc * tiles_m4,
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
        int gate_up_stride,
        int down_stride,
        int descriptors) {
    if (blockIdx.x != 0) {
        return;
    }
    if (blockDim.x != 256 ||
            descriptors != AW_PRIMARY_PER_GPU) {
        if (threadIdx.x != 0) {
            return;
        }
        tile_counts[0] = 0;
        tile_counts[1] = 0;
        tile_counts[2] = 0;
        tile_counts[3] = 0;
        tile_counts[4] = 0;
        tile_counts[5] = 0;
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
                gate + (size_t) row0*gate_up_stride,
                route_input + row0, row0, rows, cell, expert
            };
            up_desc[desc] = {
                input, weight_ptrs[cell*3 + AW_PROJECTION_UP] + (size_t) expert*q8_gate_bytes,
                up + (size_t) row0*gate_up_stride,
                route_input + row0, row0, rows, cell, expert
            };
            down_desc[desc] = {
                middle + (size_t) row0*AW_EXPERT_FF,
                weight_ptrs[cell*3 + AW_PROJECTION_DOWN] + (size_t) expert*q8_down_bytes,
                route_output + (size_t) row0*down_stride,
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
                const int tail_rows = rows - row;
                tiles_m16[tile_counts[2]++] = { desc, row, tail_rows };
                int tail_row = row;
                int remaining = tail_rows;
                while (remaining > 4) {
                    const int tile_rows = min(8, remaining);
                    tiles_m8[tile_counts[4]++] = { desc, tail_row, tile_rows };
                    tail_row += tile_rows;
                    remaining -= tile_rows;
                }
                if (remaining != 0) {
                    tiles_m4[tile_counts[5]++] = { desc, tail_row, remaining };
                }
            }
            for (int warp_row = 0; warp_row < rows; warp_row += 16) {
                tiles_warp[tile_counts[3]++] = { desc, warp_row, min(16, rows - warp_row) };
            }
            row0 += rows;
        }
        offsets[descriptors] = row0;
        return;
    }

    static_assert(AW_PRIMARY_PER_GPU == 64);
    __shared__ int scan[7][AW_PRIMARY_PER_GPU];
    const int tid = threadIdx.x;
    int local[7] = {};
    if (tid < AW_PRIMARY_PER_GPU) {
        const int rows = counts[tid];
        local[0] = rows;
        int row = 0;
        while (rows - row >= 48) {
            local[1]++;
            row += min(64, rows - row);
        }
        if (rows - row > 16) {
            local[2]++;
            row += min(32, rows - row);
        }
        if (row < rows) {
            local[3]++;
            int remaining = rows - row;
            while (remaining > 4) {
                local[5]++;
                remaining -= min(8, remaining);
            }
            local[6] += remaining != 0;
        }
        for (int warp_row = 0; warp_row < rows;
                warp_row += 16) {
            local[4]++;
        }
        #pragma unroll
        for (int queue = 0; queue < 7; ++queue) {
            scan[queue][tid] = local[queue];
        }
    }
    __syncthreads();

    for (int stride = 1; stride < AW_PRIMARY_PER_GPU;
            stride *= 2) {
        int previous[7] = {};
        if (tid < AW_PRIMARY_PER_GPU && tid >= stride) {
            #pragma unroll
            for (int queue = 0; queue < 7; ++queue) {
                previous[queue] =
                        scan[queue][tid - stride];
            }
        }
        __syncthreads();
        if (tid < AW_PRIMARY_PER_GPU && tid >= stride) {
            #pragma unroll
            for (int queue = 0; queue < 7; ++queue) {
                scan[queue][tid] += previous[queue];
            }
        }
        __syncthreads();
    }

    if (tid >= AW_PRIMARY_PER_GPU) {
        return;
    }
    const int rows = local[0];
    const int row0 = scan[0][tid] - rows;
    offsets[tid] = row0;
    cursors[tid] = row0;
    const int cell = tid/AW_PRIMARY_PER_GPU;
    const int expert = tid % AW_PRIMARY_PER_GPU;
    const size_t q8_gate_bytes =
            (size_t) AW_EXPERT_FF*(AW_EMBD/AW_KSTAGE)*
            AW_Q8_BLOCK_BYTES;
    const size_t q8_down_bytes =
            (size_t) AW_EMBD*(AW_EXPERT_FF/AW_KSTAGE)*
            AW_Q8_BLOCK_BYTES;
    gate_desc[tid] = {
        input,
        weight_ptrs[cell*3 + AW_PROJECTION_GATE] +
            (size_t) expert*q8_gate_bytes,
        gate + (size_t) row0*gate_up_stride,
        route_input + row0, row0, rows, cell, expert
    };
    up_desc[tid] = {
        input,
        weight_ptrs[cell*3 + AW_PROJECTION_UP] +
            (size_t) expert*q8_gate_bytes,
        up + (size_t) row0*gate_up_stride,
        route_input + row0, row0, rows, cell, expert
    };
    down_desc[tid] = {
        middle + (size_t) row0*AW_EXPERT_FF,
        weight_ptrs[cell*3 + AW_PROJECTION_DOWN] +
            (size_t) expert*q8_down_bytes,
        route_output + (size_t) row0*down_stride,
        nullptr, row0, rows, cell, expert
    };

    int queue_offset[6];
    #pragma unroll
    for (int queue = 0; queue < 6; ++queue) {
        queue_offset[queue] =
                scan[queue + 1][tid] -
                local[queue + 1];
    }
    int row = 0;
    while (rows - row >= 48) {
        const int tile_rows = min(64, rows - row);
        tiles_m64[queue_offset[0]++] = {
            tid, row, tile_rows
        };
        row += tile_rows;
    }
    if (rows - row > 16) {
        const int tile_rows = min(32, rows - row);
        tiles_m32[queue_offset[1]++] = {
            tid, row, tile_rows
        };
        row += tile_rows;
    }
    if (row < rows) {
        const int tail_rows = rows - row;
        tiles_m16[queue_offset[2]++] = {
            tid, row, tail_rows
        };
        int tail_row = row;
        int remaining = tail_rows;
        while (remaining > 4) {
            const int tile_rows = min(8, remaining);
            tiles_m8[queue_offset[4]++] = {
                tid, tail_row, tile_rows
            };
            tail_row += tile_rows;
            remaining -= tile_rows;
        }
        if (remaining != 0) {
            tiles_m4[queue_offset[5]++] = {
                tid, tail_row, remaining
            };
        }
    }
    for (int warp_row = 0; warp_row < rows;
            warp_row += 16) {
        tiles_warp[queue_offset[3]++] = {
            tid, warp_row, min(16, rows - warp_row)
        };
    }
    if (tid < 6) {
        tile_counts[tid] =
                scan[tid + 1][AW_PRIMARY_PER_GPU - 1];
    }
    if (tid == 0) {
        offsets[AW_PRIMARY_PER_GPU] =
                scan[0][AW_PRIMARY_PER_GPU - 1];
    }
}

__global__ static void aw_r44_assign_and_count(
        const int32_t * ids,
        const int32_t * cache_lookup,
        int32_t * assignment,
        int32_t * counts,
        int total_tokens,
        int stride_tokens,
        int physical) {
    for (int token = blockIdx.x*blockDim.x + threadIdx.x;
            token < total_tokens;
            token += gridDim.x*blockDim.x) {
        const int home = token/stride_tokens;
        int chosen[AW_GPU_COUNT];
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT; ++logical) {
            bool present = false;
            bool cached = true;
            #pragma unroll
            for (int rank = 0; rank < 8; ++rank) {
                const int expert = ids[token*8 + rank];
                if (expert >= 0 &&
                        expert/AW_PRIMARY_PER_GPU == logical) {
                    present = true;
                    cached = cached &&
                            cache_lookup[
                                home*AW_EXPERTS + expert] >= 0;
                }
            }
            chosen[logical] =
                    logical == home || (present && cached) ?
                    home : logical;
            assignment[token*AW_GPU_COUNT + logical] =
                    chosen[logical];
        }
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int expert = ids[token*8 + rank];
            if (expert < 0 || expert >= AW_EXPERTS) {
                continue;
            }
            const int logical = expert/AW_PRIMARY_PER_GPU;
            if (chosen[logical] != physical) {
                continue;
            }
            const int cache_slot =
                    cache_lookup[physical*AW_EXPERTS + expert];
            const int desc = logical == physical ?
                    expert % AW_PRIMARY_PER_GPU :
                    AW_PRIMARY_PER_GPU + cache_slot;
            atomicAdd(&counts[desc], 1);
        }
    }
}

__global__ static void aw_group_assign_and_count(
        int32_t * ids,
        int32_t * assignment,
        int32_t * counts,
        int total_tokens,
        int stride_tokens,
        bool pool_rows,
        bool logical_ids,
        int physical,
        int physical0,
        int physical1,
        int physical2,
        int physical3,
        int logical0,
        int logical1,
        int logical2,
        int logical3) {
    for (int token = blockIdx.x*blockDim.x + threadIdx.x;
            token < total_tokens;
            token += gridDim.x*blockDim.x) {
        assignment[token*AW_GPU_COUNT + 0] = physical0;
        assignment[token*AW_GPU_COUNT + 1] = physical1;
        assignment[token*AW_GPU_COUNT + 2] = physical2;
        assignment[token*AW_GPU_COUNT + 3] = physical3;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert < 0 || expert >= AW_EXPERTS) {
                continue;
            }
            int physical_group =
                    expert/AW_PRIMARY_PER_GPU;
            int logical = physical_group == 0 ? logical0 :
                    physical_group == 1 ? logical1 :
                    physical_group == 2 ? logical2 : logical3;
            const int local =
                    expert % AW_PRIMARY_PER_GPU;
            if (logical_ids) {
                logical = physical_group;
                physical_group =
                        logical == 0 ? physical0 :
                        logical == 1 ? physical1 :
                        logical == 2 ? physical2 : physical3;
            } else {
                ids[slot] =
                        logical*AW_PRIMARY_PER_GPU + local;
            }
            if (physical_group == physical) {
                const int cell = token/stride_tokens;
                const int desc =
                        pool_rows ? local :
                            cell*AW_PRIMARY_PER_GPU + local;
                atomicAdd(&counts[desc], 1);
            }
        }
    }
}

__global__ static void aw_r44_build_plan(
        const int32_t * counts,
        const int32_t * desc_experts,
        int32_t * offsets,
        int32_t * tile_counts,
        aw_tile_desc * tiles_m64,
        aw_tile_desc * tiles_m32,
        aw_tile_desc * tiles_m16,
        aw_tile_desc * tiles_warp,
        aw_tile_desc * tiles_m8,
        aw_tile_desc * tiles_m4,
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
        int gate_up_stride,
        int down_stride,
        int descriptors) {
    if (blockIdx.x != 0) {
        return;
    }
    if (blockDim.x != 256 || descriptors > 256) {
        if (threadIdx.x != 0) {
            return;
        }
        for (int i = 0; i < 6; ++i) {
            tile_counts[i] = 0;
        }
        int row0 = 0;
        for (int desc = 0; desc < descriptors; ++desc) {
            const int rows = counts[desc];
            const int expert = desc_experts[desc];
            offsets[desc] = row0;
            gate_desc[desc] = {
                input, weight_ptrs[desc*3 + AW_PROJECTION_GATE],
                gate + (size_t) row0*gate_up_stride,
                route_input + row0, row0, rows, 0, expert
            };
            up_desc[desc] = {
                input, weight_ptrs[desc*3 + AW_PROJECTION_UP],
                up + (size_t) row0*gate_up_stride,
                route_input + row0, row0, rows, 0, expert
            };
            down_desc[desc] = {
                middle + (size_t) row0*AW_EXPERT_FF,
                weight_ptrs[desc*3 + AW_PROJECTION_DOWN],
                route_output + (size_t) row0*down_stride,
                nullptr, row0, rows, 0, expert
            };
            int row = 0;
            while (rows - row >= 48) {
                const int tile_rows = min(64, rows - row);
                tiles_m64[tile_counts[0]++] = {
                    desc, row, tile_rows
                };
                row += tile_rows;
            }
            if (rows - row > 16) {
                const int tile_rows = min(32, rows - row);
                tiles_m32[tile_counts[1]++] = {
                    desc, row, tile_rows
                };
                row += tile_rows;
            }
            if (row < rows) {
                const int tail_rows = rows - row;
                tiles_m16[tile_counts[2]++] = {
                    desc, row, tail_rows
                };
                int tail_row = row;
                int remaining = tail_rows;
                while (remaining > 4) {
                    const int tile_rows = min(8, remaining);
                    tiles_m8[tile_counts[4]++] = {
                        desc, tail_row, tile_rows
                    };
                    tail_row += tile_rows;
                    remaining -= tile_rows;
                }
                if (remaining != 0) {
                    tiles_m4[tile_counts[5]++] = {
                        desc, tail_row, remaining
                    };
                }
            }
            for (int warp_row = 0; warp_row < rows;
                    warp_row += 16) {
                tiles_warp[tile_counts[3]++] = {
                    desc, warp_row, min(16, rows - warp_row)
                };
            }
            row0 += rows;
        }
        offsets[descriptors] = row0;
        return;
    }

    __shared__ int scan[7][256];
    const int tid = threadIdx.x;
    int local[7] = {};
    if (tid < descriptors) {
        const int rows = counts[tid];
        local[0] = rows;
        int row = 0;
        while (rows - row >= 48) {
            local[1]++;
            row += min(64, rows - row);
        }
        if (rows - row > 16) {
            local[2]++;
            row += min(32, rows - row);
        }
        if (row < rows) {
            local[3]++;
            int remaining = rows - row;
            while (remaining > 4) {
                local[5]++;
                remaining -= min(8, remaining);
            }
            local[6] += remaining != 0;
        }
        for (int warp_row = 0; warp_row < rows;
                warp_row += 16) {
            local[4]++;
        }
    }
    #pragma unroll
    for (int queue = 0; queue < 7; ++queue) {
        scan[queue][tid] = local[queue];
    }
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 256; stride *= 2) {
        int previous[7] = {};
        if (tid >= stride) {
            #pragma unroll
            for (int queue = 0; queue < 7; ++queue) {
                previous[queue] =
                        scan[queue][tid - stride];
            }
        }
        __syncthreads();
        if (tid >= stride) {
            #pragma unroll
            for (int queue = 0; queue < 7; ++queue) {
                scan[queue][tid] += previous[queue];
            }
        }
        __syncthreads();
    }

    if (tid < descriptors) {
        const int rows = local[0];
        const int row0 = scan[0][tid] - rows;
        const int expert = desc_experts[tid];
        offsets[tid] = row0;
        gate_desc[tid] = {
            input, weight_ptrs[tid*3 + AW_PROJECTION_GATE],
            gate + (size_t) row0*gate_up_stride,
            route_input + row0, row0, rows, 0, expert
        };
        up_desc[tid] = {
            input, weight_ptrs[tid*3 + AW_PROJECTION_UP],
            up + (size_t) row0*gate_up_stride,
            route_input + row0, row0, rows, 0, expert
        };
        down_desc[tid] = {
            middle + (size_t) row0*AW_EXPERT_FF,
            weight_ptrs[tid*3 + AW_PROJECTION_DOWN],
            route_output + (size_t) row0*down_stride,
            nullptr, row0, rows, 0, expert
        };

        int queue_offset[6];
        #pragma unroll
        for (int queue = 0; queue < 6; ++queue) {
            queue_offset[queue] =
                    scan[queue + 1][tid] -
                    local[queue + 1];
        }
        int row = 0;
        while (rows - row >= 48) {
            const int tile_rows = min(64, rows - row);
            tiles_m64[queue_offset[0]++] = {
                tid, row, tile_rows
            };
            row += tile_rows;
        }
        if (rows - row > 16) {
            const int tile_rows = min(32, rows - row);
            tiles_m32[queue_offset[1]++] = {
                tid, row, tile_rows
            };
            row += tile_rows;
        }
        if (row < rows) {
            const int tail_rows = rows - row;
            tiles_m16[queue_offset[2]++] = {
                tid, row, tail_rows
            };
            int tail_row = row;
            int remaining = tail_rows;
            while (remaining > 4) {
                const int tile_rows = min(8, remaining);
                tiles_m8[queue_offset[4]++] = {
                    tid, tail_row, tile_rows
                };
                tail_row += tile_rows;
                remaining -= tile_rows;
            }
            if (remaining != 0) {
                tiles_m4[queue_offset[5]++] = {
                    tid, tail_row, remaining
                };
            }
        }
        for (int warp_row = 0; warp_row < rows;
                warp_row += 16) {
            tiles_warp[queue_offset[3]++] = {
                tid, warp_row, min(16, rows - warp_row)
            };
        }
    }
    if (tid < 6) {
        tile_counts[tid] = scan[tid + 1][255];
    }
    if (tid == 0) {
        offsets[descriptors] = scan[0][255];
    }
}

__global__ static void aw_r44_fill_routes(
        const int32_t * ids,
        const int32_t * assignment,
        const int32_t * offsets,
        const int32_t * desc_experts,
        int32_t * route_input,
        int32_t * token_routes,
        int descriptors,
        int physical,
        int total_slots) {
    const int desc = blockIdx.x;
    if (desc >= descriptors) {
        return;
    }
    __shared__ int warp_rows[8];
    __shared__ int row_end;
    const int tid = threadIdx.x;
    const int warp = tid/32;
    const int lane = tid%32;
    const int expert = desc_experts[desc];
    const int logical = expert/AW_PRIMARY_PER_GPU;
    if (tid == 0) {
        row_end = offsets[desc];
    }
    __syncthreads();
    for (int base = 0; base < total_slots;
            base += blockDim.x) {
        const int slot = base + tid;
        const int token = slot/8;
        const bool match = slot < total_slots &&
                ids[slot] == expert &&
                assignment[token*AW_GPU_COUNT + logical] ==
                    physical;
        const unsigned int matches =
                __ballot_sync(0xffffffffu, match);
        if (lane == 0) {
            warp_rows[warp] = __popc(matches);
        }
        __syncthreads();
        if (tid == 0) {
            int row = row_end;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int count = warp_rows[i];
                warp_rows[i] = row;
                row += count;
            }
            row_end = row;
        }
        __syncthreads();
        if (match) {
            const unsigned int lane_mask =
                    lane == 0 ? 0u : (1u << lane) - 1u;
            const int row = warp_rows[warp] +
                    __popc(matches & lane_mask);
            route_input[row] = token;
            token_routes[slot] = row;
        }
        __syncthreads();
    }
}

__global__ static void aw_pair_count_routes(
        const int32_t * ids,
        int32_t * counts,
        int total_slots,
        int stride_tokens,
        int group0,
        int group1) {
    for (int slot = blockIdx.x*blockDim.x + threadIdx.x;
            slot < total_slots;
            slot += gridDim.x*blockDim.x) {
        const int expert = ids[slot];
        if (expert < 0 || expert >= AW_EXPERTS) {
            continue;
        }
        const int logical =
                expert/AW_PRIMARY_PER_GPU;
        const int group_slot =
                logical == group0 ? 0 :
                logical == group1 ? 1 : -1;
        if (group_slot >= 0) {
            const int cell =
                    (slot/8)/stride_tokens;
            atomicAdd(&counts[
                        cell*2*AW_PRIMARY_PER_GPU +
                        group_slot*AW_PRIMARY_PER_GPU +
                        expert % AW_PRIMARY_PER_GPU], 1);
        }
    }
}

__device__ __forceinline__ uint32_t aw_pair_encode_rank_mask(
        unsigned int mask) {
    uint32_t packed = (uint32_t) __popc(mask);
    int index = 0;
    #pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
        if ((mask & (1u << rank)) != 0) {
            packed |= (uint32_t) rank << (4 + 3*index);
            ++index;
        }
    }
    return packed;
}

template <bool BUILD_META>
__global__ static void aw_pair_pack_routes(
        const int32_t * source_ids,
        size_t ids_stride,
        const float * source_weights,
        size_t weights_stride,
        int32_t * ids,
        float * weights,
        uint2 * route_meta,
        int logical0,
        int logical1,
        int tokens) {
    const int slot =
            blockIdx.x*blockDim.x + threadIdx.x;
    const bool valid = slot < tokens*8;
    int expert = -1;
    float weight = 0.0f;
    if (valid) {
        const int token = slot/8;
        const int rank = slot - token*8;
        expert = ((const int32_t *)
                ((const char *) source_ids +
                 (size_t) token*ids_stride))[rank];
        weight = ((const float *)
                ((const char *) source_weights +
                 (size_t) token*weights_stride))[rank];
        ids[slot] = expert;
        weights[slot] = weight;
    }

    if constexpr (BUILD_META) {
        const unsigned int active = __activemask();
        const unsigned int logical0_mask = __ballot_sync(
                active, valid &&
                expert >= logical0*AW_PRIMARY_PER_GPU &&
                expert < (logical0 + 1)*AW_PRIMARY_PER_GPU);
        const unsigned int logical1_mask = __ballot_sync(
                active, valid &&
                expert >= logical1*AW_PRIMARY_PER_GPU &&
                expert < (logical1 + 1)*AW_PRIMARY_PER_GPU);
        const int lane = threadIdx.x & 31;
        const int rank = lane & 7;
        if (valid && rank == 0) {
            const int token = slot/8;
            const int shift = lane & ~7;
            route_meta[token] = make_uint2(
                    aw_pair_encode_rank_mask(
                        (logical0_mask >> shift) & 0xffu),
                    aw_pair_encode_rank_mask(
                        (logical1_mask >> shift) & 0xffu));
        }
    }
}

__global__ static void aw_pair_fill_routes(
        const int32_t * ids,
        const int32_t * offsets,
        const int32_t * desc_experts,
        int32_t * route_input,
        int32_t * token_routes,
        int descriptors,
        int stride_tokens) {
    const int desc = blockIdx.x;
    if (desc >= descriptors) {
        return;
    }
    __shared__ int warp_rows[8];
    __shared__ int row_end;
    const int tid = threadIdx.x;
    const int warp = tid/32;
    const int lane = tid%32;
    const int cell =
            desc/(2*AW_PRIMARY_PER_GPU);
    const int expert = desc_experts[desc];
    const int slot_begin =
            cell*stride_tokens*8;
    const int slot_end =
            slot_begin + stride_tokens*8;
    if (tid == 0) {
        row_end = offsets[desc];
    }
    __syncthreads();
    for (int base = slot_begin; base < slot_end;
            base += blockDim.x) {
        const int slot = base + tid;
        const bool match =
                slot < slot_end &&
                ids[slot] == expert;
        const unsigned int matches =
                __ballot_sync(0xffffffffu, match);
        if (lane == 0) {
            warp_rows[warp] = __popc(matches);
        }
        __syncthreads();
        if (tid == 0) {
            int row = row_end;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int count = warp_rows[i];
                warp_rows[i] = row;
                row += count;
            }
            row_end = row;
        }
        __syncthreads();
        if (match) {
            const unsigned int lane_mask =
                    lane == 0 ? 0u :
                    (1u << lane) - 1u;
            const int row = warp_rows[warp] +
                    __popc(matches & lane_mask);
            route_input[row] = slot/8;
            token_routes[slot] = row;
        }
        __syncthreads();
    }
}

__global__ static void aw_pair_group_reduce_panel(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * partial,
        int total_tokens,
        int logical,
        int panel_n) {
    const size_t count =
            (size_t) total_tokens*panel_n;
    for (size_t i =
            (size_t) blockIdx.x*blockDim.x +
                threadIdx.x;
            i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col =
                (int) (i -
                    (size_t) token*panel_n);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert >=
                    logical*AW_PRIMARY_PER_GPU &&
                    expert <
                    (logical + 1)*
                        AW_PRIMARY_PER_GPU) {
                sum += route_weights[slot]*
                        routes[
                            (size_t)
                                token_routes[slot]*
                                panel_n + col];
            }
        }
        partial[
            ((size_t) logical*total_tokens +
             token)*panel_n + col] =
                aw_float_to_bf16(sum);
    }
}

template <int ITEMS>
__global__ static void aw_pair_group_reduce_panel_p100(
        const float * routes,
        const int32_t * token_routes,
        const float * route_weights,
        const uint2 * route_meta,
        uint16_t * partial,
        int total_tokens,
        int logical0,
        int logical1,
        int panel_n) {
    static_assert(ITEMS == 2 || ITEMS == 4);
    __shared__ int route_counts[2];
    __shared__ int rows[2][8];
    __shared__ float weights[2][8];

    const int token = blockIdx.x;
    if (token >= total_tokens) {
        return;
    }
    if (threadIdx.x < 16) {
        const int group = threadIdx.x >> 3;
        const int index = threadIdx.x & 7;
        const uint2 metadata = route_meta[token];
        const uint32_t word =
                group == 0 ? metadata.x : metadata.y;
        const int count = min(
                (int) (word & 0xfu), 8);
        if (index == 0) {
            route_counts[group] = count;
        }
        if (index < count) {
            const int rank =
                    (int) ((word >> (4 + 3*index)) & 7u);
            const int slot = token*8 + rank;
            rows[group][index] = token_routes[slot];
            weights[group][index] = route_weights[slot];
        }
    }
    __syncthreads();

    const int col0 = threadIdx.x*ITEMS;
    if (col0 >= panel_n) {
        return;
    }
    const int logical[2] = {logical0, logical1};
    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        float sum[ITEMS] = {};
        #pragma unroll 1
        for (int index = 0; index < route_counts[group];
                ++index) {
            const float weight = weights[group][index];
            const float * source =
                    routes + (size_t) rows[group][index]*
                        panel_n + col0;
            float values[ITEMS];
            if constexpr (ITEMS == 4) {
                const float4 packed =
                        *(const float4 *) source;
                values[0] = packed.x;
                values[1] = packed.y;
                values[2] = packed.z;
                values[3] = packed.w;
            } else {
                const float2 packed =
                        *(const float2 *) source;
                values[0] = packed.x;
                values[1] = packed.y;
            }
            #pragma unroll
            for (int item = 0; item < ITEMS; ++item) {
                sum[item] = __fmaf_rn(
                        weight, values[item], sum[item]);
            }
        }
        uint16_t * destination =
                partial +
                ((size_t) logical[group]*total_tokens +
                 token)*panel_n + col0;
        if constexpr (ITEMS == 4) {
            const uint32_t lo =
                    (uint32_t) aw_float_to_bf16(sum[0]) |
                    (uint32_t) aw_float_to_bf16(sum[1]) << 16;
            const uint32_t hi =
                    (uint32_t) aw_float_to_bf16(sum[2]) |
                    (uint32_t) aw_float_to_bf16(sum[3]) << 16;
            *(uint2 *) destination = make_uint2(lo, hi);
        } else {
            *(uint32_t *) destination =
                    (uint32_t) aw_float_to_bf16(sum[0]) |
                    (uint32_t) aw_float_to_bf16(sum[1]) << 16;
        }
    }
}

template <bool NCCL_ORDER = false>
__global__ static void aw_pair_sum_groups_peer_panel(
        const uint16_t * partial0,
        const uint16_t * partial1,
        int active0,
        int owner0,
        int owner1,
        int owner2,
        int owner3,
        float * output,
        int total_tokens,
        int token_offset,
        int order_token_offset,
        int tokens,
        int panel_n,
        int output_col) {
    const size_t count =
            (size_t) tokens*panel_n;
    for (size_t i =
            (size_t) blockIdx.x*blockDim.x +
                threadIdx.x;
            i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int local_token =
                (int) (i/panel_n);
        const int col =
                (int) (i -
                    (size_t) local_token*panel_n);
        const int token =
                token_offset + local_token;
        const size_t global =
                (size_t) (order_token_offset +
                    local_token)*AW_EMBD +
                output_col + col;
        float sum = 0.0f;
        #pragma unroll
        for (int step = 0;
                step < AW_GPU_COUNT; ++step) {
            const int logical = NCCL_ORDER ?
                    (int) ((global/32768 + 1 + step) %
                        AW_GPU_COUNT) : step;
            const int owner =
                    logical == 0 ? owner0 :
                    logical == 1 ? owner1 :
                    logical == 2 ? owner2 : owner3;
            const uint16_t * source =
                    owner == active0 ?
                    partial0 : partial1;
            const size_t index =
                    ((size_t) logical*
                        total_tokens + token)*
                        panel_n + col;
            sum = aw_bf16_to_float(
                    aw_float_to_bf16(
                        sum +
                        aw_bf16_to_float(
                            source[index])));
        }
        output[
            (size_t) local_token*AW_EMBD +
            output_col + col] = sum;
    }
}

template <bool NCCL_ORDER, int ITEMS>
__global__ static void aw_pair_sum_groups_peer_panel_p100(
        const uint16_t * group0,
        const uint16_t * group1,
        const uint16_t * group2,
        const uint16_t * group3,
        float * output,
        int tokens,
        int order_token_offset,
        int panel_n,
        int output_col) {
    static_assert(ITEMS == 2 || ITEMS == 4);
    const uint16_t * groups[AW_GPU_COUNT] = {
        group0, group1, group2, group3
    };
    for (int token = blockIdx.x; token < tokens;
            token += gridDim.x) {
        const int col0 = threadIdx.x*ITEMS;
        if (col0 >= panel_n) {
            continue;
        }

        uint32_t packed2[AW_GPU_COUNT];
        uint2 packed4[AW_GPU_COUNT];
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            const uint16_t * source =
                    groups[logical] +
                    (size_t) token*panel_n + col0;
            if constexpr (ITEMS == 4) {
                packed4[logical] =
                        *(const uint2 *) source;
            } else {
                packed2[logical] =
                        *(const uint32_t *) source;
            }
        }

        float values[AW_GPU_COUNT][ITEMS];
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            if constexpr (ITEMS == 4) {
                values[logical][0] = aw_bf16_to_float(
                        (uint16_t) packed4[logical].x);
                values[logical][1] = aw_bf16_to_float(
                        (uint16_t) (packed4[logical].x >> 16));
                values[logical][2] = aw_bf16_to_float(
                        (uint16_t) packed4[logical].y);
                values[logical][3] = aw_bf16_to_float(
                        (uint16_t) (packed4[logical].y >> 16));
            } else {
                values[logical][0] = aw_bf16_to_float(
                        (uint16_t) packed2[logical]);
                values[logical][1] = aw_bf16_to_float(
                        (uint16_t) (packed2[logical] >> 16));
            }
        }

        const int first = NCCL_ORDER ?
                (((order_token_offset + token) >> 4) + 1) &
                    (AW_GPU_COUNT - 1) : 0;
        float sum[ITEMS] = {};
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            const int logical =
                    (first + step) & (AW_GPU_COUNT - 1);
            #pragma unroll
            for (int item = 0; item < ITEMS; ++item) {
                sum[item] = aw_bf16_to_float(
                        aw_float_to_bf16(
                            __fadd_rn(
                                sum[item],
                                values[logical][item])));
            }
        }

        float * destination =
                output + (size_t) token*AW_EMBD +
                output_col + col0;
        if constexpr (ITEMS == 4) {
            *(float4 *) destination = make_float4(
                    sum[0], sum[1], sum[2], sum[3]);
        } else {
            *(float2 *) destination =
                    make_float2(sum[0], sum[1]);
        }
    }
}

template <bool NCCL_ORDER, int ITEMS>
__global__ static void aw_pair_sum_groups_local_panel_p100(
        const uint16_t * group0,
        const uint16_t * group1,
        const uint16_t * group2,
        const uint16_t * group3,
        float * output,
        int tokens,
        int order_token_offset,
        int panel_n,
        int output_col) {
    static_assert(ITEMS == 2 || ITEMS == 4);
    const uint16_t * groups[AW_GPU_COUNT] = {
        group0, group1, group2, group3
    };
    for (int token = blockIdx.x; token < tokens;
            token += gridDim.x) {
        const int col0 = threadIdx.x*ITEMS;
        if (col0 >= panel_n) {
            continue;
        }

        uint32_t packed2[AW_GPU_COUNT];
        uint2 packed4[AW_GPU_COUNT];
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            const uint16_t * source =
                    groups[logical] +
                    (size_t) token*panel_n + col0;
            if constexpr (ITEMS == 4) {
                packed4[logical] =
                        *(const uint2 *) source;
            } else {
                packed2[logical] =
                        *(const uint32_t *) source;
            }
        }

        float values[AW_GPU_COUNT][ITEMS];
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            if constexpr (ITEMS == 4) {
                values[logical][0] = aw_bf16_to_float(
                        (uint16_t) packed4[logical].x);
                values[logical][1] = aw_bf16_to_float(
                        (uint16_t) (packed4[logical].x >> 16));
                values[logical][2] = aw_bf16_to_float(
                        (uint16_t) packed4[logical].y);
                values[logical][3] = aw_bf16_to_float(
                        (uint16_t) (packed4[logical].y >> 16));
            } else {
                values[logical][0] = aw_bf16_to_float(
                        (uint16_t) packed2[logical]);
                values[logical][1] = aw_bf16_to_float(
                        (uint16_t) (packed2[logical] >> 16));
            }
        }

        const int first = NCCL_ORDER ?
                (((order_token_offset + token) >> 4) + 1) &
                    (AW_GPU_COUNT - 1) : 0;
        float sum[ITEMS] = {};
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            const int logical =
                    (first + step) & (AW_GPU_COUNT - 1);
            #pragma unroll
            for (int item = 0; item < ITEMS; ++item) {
                sum[item] = aw_bf16_to_float(
                        aw_float_to_bf16(
                            __fadd_rn(
                                sum[item],
                                values[logical][item])));
            }
        }

        float * destination =
                output + (size_t) token*AW_EMBD +
                output_col + col0;
        if constexpr (ITEMS == 4) {
            *(float4 *) destination = make_float4(
                    sum[0], sum[1], sum[2], sum[3]);
        } else {
            *(float2 *) destination =
                    make_float2(sum[0], sum[1]);
        }
    }
}

__global__ static void aw_r44_gather_selected_inputs(
        const float * input0,
        const float * input1,
        const float * input2,
        const float * input3,
        const int32_t * ids,
        const int32_t * assignment,
        float * output,
        int total_tokens,
        int stride_tokens,
        int physical,
        int32_t * selected_count) {
    const int token = blockIdx.x;
    if (token >= total_tokens) {
        return;
    }
    __shared__ int selected;
    if (threadIdx.x == 0) {
        selected = 0;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int expert = ids[token*8 + rank];
            if (expert >= 0 && expert < AW_EXPERTS &&
                    assignment[token*AW_GPU_COUNT +
                        expert/AW_PRIMARY_PER_GPU] == physical) {
                selected = 1;
                break;
            }
        }
    }
    __syncthreads();
    if (!selected) {
        return;
    }
    if (threadIdx.x == 0 && selected_count != nullptr) {
        atomicAdd(selected_count, 1);
    }
    const int home = token/stride_tokens;
    const int local_token = token - home*stride_tokens;
    const float * source = home == 0 ? input0 :
            home == 1 ? input1 :
            home == 2 ? input2 : input3;
    for (int k = threadIdx.x; k < AW_EMBD; k += blockDim.x) {
        output[(size_t) token*AW_EMBD + k] =
                source[(size_t) local_token*AW_EMBD + k];
    }
}

__global__ static void aw_live_build_panel_descs(
        const aw_work_desc * base,
        aw_work_desc * panels,
        int descriptors,
        int panel_count,
        size_t weight_panel_bytes) {
    const int count = descriptors*panel_count;
    for (int index = blockIdx.x*blockDim.x + threadIdx.x; index < count;
            index += gridDim.x*blockDim.x) {
        const int panel = index/descriptors;
        const int desc = index - panel*descriptors;
        aw_work_desc value = base[desc];
        value.weight += (size_t) panel*weight_panel_bytes;
        panels[index] = value;
    }
}

__global__ static void aw_live_build_shared_plan(
        int32_t * tile_counts,
        aw_tile_desc * tiles_m64,
        aw_tile_desc * tiles_m32,
        aw_tile_desc * tiles_m16,
        aw_tile_desc * tiles_warp,
        aw_tile_desc * tiles_m8,
        aw_tile_desc * tiles_m4,
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
    tile_counts[3] = 0;
    tile_counts[4] = 0;
    tile_counts[5] = 0;
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
        const int tail_rows = rows - row;
        tiles_m16[tile_counts[2]++] = { 0, row, tail_rows };
        int tail_row = row;
        int remaining = tail_rows;
        while (remaining > 4) {
            const int tile_rows = min(8, remaining);
            tiles_m8[tile_counts[4]++] = { 0, tail_row, tile_rows };
            tail_row += tile_rows;
            remaining -= tile_rows;
        }
        if (remaining != 0) {
            tiles_m4[tile_counts[5]++] = { 0, tail_row, remaining };
        }
    }
    for (int warp_row = 0; warp_row < rows; warp_row += 16) {
        tiles_warp[tile_counts[3]++] = { 0, warp_row, min(16, rows - warp_row) };
    }
}

__global__ static void aw_live_fill_routes_deterministic(
        const int32_t * ids,
        const int32_t * offsets,
        int32_t * route_input,
        int32_t * token_routes,
        int descriptors,
        int owner,
        int tokens) {
    const int desc = blockIdx.x;
    if (desc >= descriptors) {
        return;
    }

    __shared__ int warp_rows[8];
    __shared__ int row_end;

    const int tid = threadIdx.x;
    const int warp = tid/32;
    const int lane = tid%32;
    const int cell = desc/AW_PRIMARY_PER_GPU;
    const int expert = owner*AW_PRIMARY_PER_GPU + desc % AW_PRIMARY_PER_GPU;
    const int slot_begin = cell*tokens*8;
    const int slot_end = slot_begin + tokens*8;

    if (tid == 0) {
        row_end = offsets[desc];
    }
    __syncthreads();

    for (int base = slot_begin; base < slot_end; base += blockDim.x) {
        const int slot = base + tid;
        const bool match = slot < slot_end && ids[slot] == expert;
        const unsigned int matches = __ballot_sync(0xffffffffu, match);
        if (lane == 0) {
            warp_rows[warp] = __popc(matches);
        }
        __syncthreads();
        if (tid == 0) {
            int row = row_end;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int count = warp_rows[i];
                warp_rows[i] = row;
                row += count;
            }
            row_end = row;
        }
        __syncthreads();
        if (match) {
            const unsigned int lane_mask = lane == 0 ? 0u : (1u << lane) - 1u;
            const int row = warp_rows[warp] + __popc(matches & lane_mask);
            const int token = (slot/8) % tokens;
            route_input[row] = cell*tokens + token;
            token_routes[slot] = row;
        }
        __syncthreads();
    }
}

__global__ static void aw_live_fill_routes_atomic(
        const int32_t * ids,
        int32_t * cursors,
        int32_t * route_input,
        int32_t * token_routes,
        int total_slots,
        int owner,
        int tokens) {
    for (int slot = blockIdx.x*blockDim.x + threadIdx.x;
            slot < total_slots; slot += gridDim.x*blockDim.x) {
        const int expert = ids[slot];
        if (expert >= owner*AW_PRIMARY_PER_GPU &&
                expert < (owner + 1)*AW_PRIMARY_PER_GPU) {
            const int cell = slot/(tokens*8);
            const int desc =
                    cell*AW_PRIMARY_PER_GPU +
                    expert % AW_PRIMARY_PER_GPU;
            const int row = atomicAdd(&cursors[desc], 1);
            route_input[row] =
                    cell*tokens + (slot/8) % tokens;
            token_routes[slot] = row;
        }
    }
}

__global__ static void aw_live_fill_routes_stable8(
        const int32_t * ids,
        const int32_t * offsets,
        int32_t * route_input,
        int32_t * token_routes,
        int cells,
        int owner,
        int tokens) {
    const int cell = blockIdx.x;
    if (cell >= cells) {
        return;
    }

    const int warp = threadIdx.x/32;
    const int lane = threadIdx.x%32;
    const int expert_base =
            owner*AW_PRIMARY_PER_GPU + warp*8;
    const int desc_base =
            cell*AW_PRIMARY_PER_GPU + warp*8;
    const int lane_offset =
            lane < 8 ? offsets[desc_base + lane] : 0;
    int row_base[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        row_base[i] = __shfl_sync(0xffffffffu, lane_offset, i);
    }

    const int slot_begin = cell*tokens*8;
    const int slot_end = slot_begin + tokens*8;
    for (int base = slot_begin; base < slot_end; base += 32) {
        const int slot = base + lane;
        const int expert = slot < slot_end ? ids[slot] : -1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned int matches =
                    __ballot_sync(0xffffffffu, expert == expert_base + i);
            if (expert == expert_base + i) {
                const unsigned int lane_mask =
                        lane == 0 ? 0u : (1u << lane) - 1u;
                const int row =
                        row_base[i] + __popc(matches & lane_mask);
                route_input[row] =
                        cell*tokens + (slot/8) % tokens;
                token_routes[slot] = row;
            }
            row_base[i] += __popc(matches);
        }
    }
}

__global__ static void aw_live_fill_routes_stable256(
        const int32_t * ids,
        const int32_t * segment_offsets,
        int32_t * route_input,
        int32_t * token_routes,
        int cells,
        int owner,
        int tokens,
        int segments_per_cell) {
    __shared__ int warp_counts[8][AW_PRIMARY_PER_GPU];

    const int segment = blockIdx.x;
    const int cell = segment/segments_per_cell;
    const int cell_segment = segment - cell*segments_per_cell;
    const int lane = threadIdx.x%32;
    const int warp = threadIdx.x/32;
    const int cell_slot_begin = cell*tokens*8;
    const int cell_slot_end = cell_slot_begin + tokens*8;
    const int slot =
            cell_slot_begin +
            cell_segment*AW_ROUTE_SEGMENT_SLOTS +
            threadIdx.x;
    const int local_expert = cell < cells && slot < cell_slot_end ?
            ids[slot] - owner*AW_PRIMARY_PER_GPU : -1;
    int local_rank = 0;
    const unsigned int lane_mask =
            lane == 0 ? 0u : (1u << lane) - 1u;

    #pragma unroll 1
    for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
        const unsigned int matches =
                __ballot_sync(0xffffffffu, local_expert == expert);
        if (lane == 0) {
            warp_counts[warp][expert] = __popc(matches);
        }
        if (local_expert == expert) {
            local_rank = __popc(matches & lane_mask);
        }
    }
    __syncthreads();

    if (local_expert >= 0 && local_expert < AW_PRIMARY_PER_GPU) {
        #pragma unroll
        for (int prior_warp = 0; prior_warp < 8; ++prior_warp) {
            if (prior_warp < warp) {
                local_rank += warp_counts[prior_warp][local_expert];
            }
        }
        const size_t segment_index =
                ((size_t) cell*segments_per_cell + cell_segment)*
                AW_PRIMARY_PER_GPU + local_expert;
        const int row =
                segment_offsets[segment_index] + local_rank;
        route_input[row] =
                cell*tokens + (slot/8) % tokens;
        token_routes[slot] = row;
    }
}

__global__ static void aw_live_fill_routes_radix256(
        const int32_t * ids,
        const int32_t * segment_offsets,
        int32_t * route_input,
        int32_t * token_routes,
        int cells,
        int owner,
        int tokens,
        int segments_per_cell) {
    __shared__ int keys[2][AW_ROUTE_SEGMENT_SLOTS];
    __shared__ int slots[2][AW_ROUTE_SEGMENT_SLOTS];
    __shared__ int warp_ones[8];
    __shared__ int key_starts[AW_PRIMARY_PER_GPU];

    const int segment = blockIdx.x;
    const int cell = segment/segments_per_cell;
    const int cell_segment = segment - cell*segments_per_cell;
    const int lane = threadIdx.x%32;
    const int warp = threadIdx.x/32;
    const int cell_slot_begin = cell*tokens*8;
    const int cell_slot_end = cell_slot_begin + tokens*8;
    const int slot =
            cell_slot_begin +
            cell_segment*AW_ROUTE_SEGMENT_SLOTS +
            threadIdx.x;
    const int local_expert = cell < cells && slot < cell_slot_end ?
            ids[slot] - owner*AW_PRIMARY_PER_GPU : -1;
    keys[0][threadIdx.x] =
            local_expert >= 0 && local_expert < AW_PRIMARY_PER_GPU ?
            local_expert : AW_PRIMARY_PER_GPU;
    slots[0][threadIdx.x] = slot;
    __syncthreads();

    const unsigned int lane_mask =
            lane == 0 ? 0u : (1u << lane) - 1u;
    int source = 0;
    #pragma unroll
    for (int bit = 0; bit < 7; ++bit) {
        const int target = 1 - source;
        const int key = keys[source][threadIdx.x];
        const bool one = ((key >> bit) & 1) != 0;
        const unsigned int matches =
                __ballot_sync(0xffffffffu, one);
        if (lane == 0) {
            warp_ones[warp] = __popc(matches);
        }
        __syncthreads();
        int ones_before = __popc(matches & lane_mask);
        int total_ones = 0;
        #pragma unroll
        for (int prior_warp = 0; prior_warp < 8;
                ++prior_warp) {
            const int count = warp_ones[prior_warp];
            total_ones += count;
            if (prior_warp < warp) {
                ones_before += count;
            }
        }
        const int zeros_before = threadIdx.x - ones_before;
        const int destination = one ?
                AW_ROUTE_SEGMENT_SLOTS - total_ones + ones_before :
                zeros_before;
        keys[target][destination] = key;
        slots[target][destination] = slots[source][threadIdx.x];
        __syncthreads();
        source = target;
    }

    const int key = keys[source][threadIdx.x];
    if (threadIdx.x < AW_PRIMARY_PER_GPU) {
        key_starts[threadIdx.x] = 0;
    }
    __syncthreads();
    if (key < AW_PRIMARY_PER_GPU &&
            (threadIdx.x == 0 ||
             keys[source][threadIdx.x - 1] != key)) {
        key_starts[key] = threadIdx.x;
    }
    __syncthreads();
    if (key < AW_PRIMARY_PER_GPU) {
        const int original_slot = slots[source][threadIdx.x];
        const size_t segment_index =
                ((size_t) cell*segments_per_cell + cell_segment)*
                AW_PRIMARY_PER_GPU + key;
        const int row =
                segment_offsets[segment_index] +
                threadIdx.x - key_starts[key];
        route_input[row] =
                cell*tokens + (original_slot/8) % tokens;
        token_routes[original_slot] = row;
    }
}

__global__ static void aw_live_fill_routes_segment8(
        const int32_t * ids,
        const int32_t * segment_offsets,
        int32_t * route_input,
        int32_t * token_routes,
        int cells,
        int owner,
        int tokens,
        int segments_per_cell) {
    if (threadIdx.x >= 8) {
        return;
    }

    const int segment = blockIdx.x;
    const int cell = segment/segments_per_cell;
    const int cell_segment = segment - cell*segments_per_cell;
    const int expert_base = threadIdx.x*8;
    const size_t segment_index =
            ((size_t) cell*segments_per_cell + cell_segment)*
            AW_PRIMARY_PER_GPU;
    int row_base[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        row_base[i] =
                segment_offsets[segment_index + expert_base + i];
    }

    const int cell_slot_begin = cell*tokens*8;
    const int cell_slot_end = cell_slot_begin + tokens*8;
    const int segment_slot_begin =
            cell_slot_begin +
            cell_segment*AW_ROUTE_SEGMENT_SLOTS;
    const int segment_slot_end =
            min(segment_slot_begin + AW_ROUTE_SEGMENT_SLOTS,
                    cell_slot_end);
    for (int slot = segment_slot_begin;
            slot < segment_slot_end; ++slot) {
        const int local_expert =
                ids[slot] - owner*AW_PRIMARY_PER_GPU;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (local_expert == expert_base + i) {
                const int row = row_base[i]++;
                route_input[row] =
                        cell*tokens + (slot/8) % tokens;
                token_routes[slot] = row;
            }
        }
    }
}

template <bool BF16>
__global__ static void aw_live_owner_reduce(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        void * partial,
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
        if constexpr (BF16) {
            ((uint16_t *) partial)[i] = aw_float_to_bf16(sum);
        } else {
            ((float *) partial)[i] = sum;
        }
    }
}

template <bool BF16>
__global__ static void aw_live_owner_reduce_panel(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        void * partial,
        int total_tokens,
        int owner,
        int panel_n,
        int output_n,
        int output_col) {
    const size_t count = (size_t) total_tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col = (int) (i - (size_t) token*panel_n);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert >= owner*AW_PRIMARY_PER_GPU && expert < (owner + 1)*AW_PRIMARY_PER_GPU) {
                sum += route_weights[slot]*routes[(size_t) token_routes[slot]*panel_n + col];
            }
        }
        const size_t output_index = (size_t) token*output_n + output_col + col;
        if constexpr (BF16) {
            ((uint16_t *) partial)[output_index] = aw_float_to_bf16(sum);
        } else {
            ((float *) partial)[output_index] = sum;
        }
    }
}

__global__ static void aw_live_owner_reduce_panel_w4(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * partial,
        int total_tokens,
        int owner,
        int panel_n,
        int output_n,
        int output_col) {
    __shared__ int rows[8];
    __shared__ float weights[8];

    for (int token = blockIdx.x; token < total_tokens;
            token += gridDim.x) {
        if (threadIdx.x < 8) {
            const int rank = threadIdx.x;
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            const bool active =
                    expert >= owner*AW_PRIMARY_PER_GPU &&
                    expert < (owner + 1)*AW_PRIMARY_PER_GPU;
            rows[rank] = active ? token_routes[slot] : -1;
            weights[rank] = active ?
                    route_weights[slot] : 0.0f;
        }
        __syncthreads();

        for (int col0 = threadIdx.x*4;
                col0 < panel_n;
                col0 += blockDim.x*4) {
            float sum[4] = {};
            #pragma unroll
            for (int rank = 0; rank < 8; ++rank) {
                const int row = rows[rank];
                if (row >= 0) {
                    const float weight = weights[rank];
                    const float * values =
                            routes + (size_t) row*panel_n +
                            col0;
                    #pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        if (col0 + item < panel_n) {
                            sum[item] = __fmaf_rn(
                                    weight, values[item],
                                    sum[item]);
                        }
                    }
                }
            }
            const size_t output =
                    (size_t) token*output_n +
                    output_col + col0;
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                if (col0 + item < panel_n) {
                    partial[output + item] =
                            aw_float_to_bf16(sum[item]);
                }
            }
        }
        __syncthreads();
    }
}

__global__ static void aw_r44_owner_reduce_panel(
        const float * routes,
        const int32_t * ids,
        const int32_t * assignment,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * partial,
        int total_tokens,
        int stride_tokens,
        int physical,
        int panel_n) {
    const size_t count = (size_t) total_tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x +
            threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col =
                (int) (i - (size_t) token*panel_n);
        float sum[AW_GPU_COUNT] = {};
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert < 0 || expert >= AW_EXPERTS) {
                continue;
            }
            const int logical = expert/AW_PRIMARY_PER_GPU;
            if (assignment[token*AW_GPU_COUNT + logical] ==
                    physical) {
                sum[logical] += route_weights[slot]*
                        routes[
                            (size_t) token_routes[slot]*
                                panel_n + col];
            }
        }
        #pragma unroll
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            if (assignment[token*AW_GPU_COUNT + logical] ==
                    physical) {
                const int home = token/stride_tokens;
                const int local_token =
                        token - home*stride_tokens;
                const int partial_token =
                        home == physical ?
                            local_token*AW_GPU_COUNT + logical :
                            AW_GPU_COUNT*stride_tokens +
                                (token < physical*stride_tokens ?
                                    token : token - stride_tokens);
                const size_t output =
                        (size_t) partial_token*panel_n + col;
                partial[output] =
                        aw_float_to_bf16(sum[logical]);
            }
        }
    }
}

__global__ static void aw_group_owner_reduce_panel(
        const float * routes,
        const int32_t * ids,
        const int32_t * token_routes,
        const float * route_weights,
        uint16_t * partial,
        int total_tokens,
        int stride_tokens,
        int physical,
        int logical,
        int panel_n) {
    const size_t count = (size_t) total_tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x +
            threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col =
                (int) (i - (size_t) token*panel_n);
        float sum = 0.0f;
        #pragma unroll
        for (int rank = 0; rank < 8; ++rank) {
            const int slot = token*8 + rank;
            const int expert = ids[slot];
            if (expert >= logical*AW_PRIMARY_PER_GPU &&
                    expert <
                        (logical + 1)*AW_PRIMARY_PER_GPU) {
                sum += route_weights[slot]*
                        routes[
                            (size_t) token_routes[slot]*
                                panel_n + col];
            }
        }
        const int home = token/stride_tokens;
        const int local_token =
                token - home*stride_tokens;
        const int partial_token =
                home == physical ?
                    local_token*AW_GPU_COUNT + logical :
                    AW_GPU_COUNT*stride_tokens +
                        (token < physical*stride_tokens ?
                            token : token - stride_tokens);
        partial[(size_t) partial_token*panel_n + col] =
                aw_float_to_bf16(sum);
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

template <bool BF16, bool NCCL_ORDER = false>
__global__ static void aw_live_sum_owners(
        const void * recv,
        float * output,
        int total_tokens,
        int token_offset,
        int tokens) {
    const size_t count = (size_t) tokens*AW_EMBD;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const size_t global = (size_t) token_offset*AW_EMBD + i;
        float sum = 0.0f;
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            int owner = step;
            if constexpr (NCCL_ORDER) {
                owner = (int) (((global/32768) + 1 + step) % AW_GPU_COUNT);
            }
            const size_t index = (size_t) owner*total_tokens*AW_EMBD + global;
            if constexpr (BF16) {
                sum = aw_bf16_to_float(aw_float_to_bf16(
                            sum + aw_bf16_to_float(((const uint16_t *) recv)[index])));
            } else {
                sum += ((const float *) recv)[index];
            }
        }
        output[i] = sum;
    }
}

template <bool NCCL_ORDER = false>
__global__ static void aw_live_sum_owners_panel(
        const uint16_t * recv,
        float * output,
        int tokens,
        int panel_n,
        int output_col) {
    const size_t count = (size_t) tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < count; i += (size_t) gridDim.x*blockDim.x) {
        const int token = (int) (i/panel_n);
        const int col = (int) (i - (size_t) token*panel_n);
        const size_t global =
                (size_t) token*AW_EMBD + output_col + col;
        float sum = 0.0f;
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            int owner = step;
            if constexpr (NCCL_ORDER) {
                owner = (int) (((global/32768) + 1 + step) %
                        AW_GPU_COUNT);
            }
            const size_t index =
                    ((size_t) owner*tokens + token)*panel_n + col;
            sum = aw_bf16_to_float(aw_float_to_bf16(
                        sum + aw_bf16_to_float(recv[index])));
        }
        output[(size_t) token*AW_EMBD + output_col + col] = sum;
    }
}

template <bool NCCL_ORDER, int ITEMS>
__global__ static void aw_live_sum_owners_panel_p100(
        const uint16_t * recv,
        float * output,
        int tokens,
        int panel_n,
        int output_col) {
    static_assert(ITEMS == 2 || ITEMS == 4);
    for (int token = blockIdx.x; token < tokens;
            token += gridDim.x) {
        for (int col0 = threadIdx.x*ITEMS;
                col0 < panel_n;
                col0 += blockDim.x*ITEMS) {
            const size_t global =
                    (size_t) token*AW_EMBD +
                    output_col + col0;
            float values[AW_GPU_COUNT][ITEMS];
            #pragma unroll
            for (int step = 0; step < AW_GPU_COUNT; ++step) {
                int owner = step;
                if constexpr (NCCL_ORDER) {
                    owner = (int) (((global/32768) +
                                1 + step) %
                            AW_GPU_COUNT);
                }
                const uint16_t * source =
                        recv +
                        ((size_t) owner*tokens + token)*
                            panel_n + col0;
                #pragma unroll
                for (int item = 0; item < ITEMS; ++item) {
                    values[step][item] =
                            col0 + item < panel_n ?
                            aw_bf16_to_float(source[item]) :
                            0.0f;
                }
            }

            float sum[ITEMS] = {};
            #pragma unroll
            for (int step = 0; step < AW_GPU_COUNT; ++step) {
                #pragma unroll
                for (int item = 0; item < ITEMS; ++item) {
                    sum[item] = aw_bf16_to_float(
                            aw_float_to_bf16(
                                __fadd_rn(
                                    sum[item],
                                    values[step][item])));
                }
            }
            float * destination =
                    output + (size_t) token*AW_EMBD +
                    output_col + col0;
            #pragma unroll
            for (int item = 0; item < ITEMS; ++item) {
                if (col0 + item < panel_n) {
                    destination[item] = sum[item];
                }
            }
        }
    }
}

template <bool BF16, bool NCCL_ORDER = false>
__global__ static void aw_live_sum_owners_peer(
        const void * partial0,
        const void * partial1,
        const void * partial2,
        const void * partial3,
        float * output,
        int total_tokens,
        int token_offset,
        int tokens) {
    const size_t count = (size_t) tokens*AW_EMBD;
    const size_t work_count = count/4;
    for (size_t work = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            work < work_count;
            work += (size_t) gridDim.x*blockDim.x) {
        const size_t local = work*4;
        const size_t global = (size_t) token_offset*AW_EMBD + local;
        float sum[4] = {};
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            int owner = step;
            if constexpr (NCCL_ORDER) {
                owner = (int) (((global/32768) + 1 + step) % AW_GPU_COUNT);
            }
            const void * owner_partial = owner == 0 ? partial0 :
                    owner == 1 ? partial1 :
                    owner == 2 ? partial2 : partial3;
            #pragma unroll
            for (int item = 0; item < 4; ++item) {
                if constexpr (BF16) {
                    sum[item] = aw_bf16_to_float(aw_float_to_bf16(
                                sum[item] +
                                aw_bf16_to_float(
                                    ((const uint16_t *) owner_partial)
                                        [global + item])));
                } else {
                    sum[item] +=
                            ((const float *) owner_partial)[global + item];
                }
            }
        }
        #pragma unroll
        for (int item = 0; item < 4; ++item) {
            output[local + item] = sum[item];
        }
    }
}

__global__ static void aw_r44_sum_owners_peer_panel(
        const uint16_t * partial0,
        const uint16_t * partial1,
        const uint16_t * partial2,
        const uint16_t * partial3,
        const int32_t * assignment,
        float * output,
        int total_tokens,
        int token_offset,
        int tokens,
        int stride_tokens,
        int panel_n,
        int output_col) {
    const size_t count = (size_t) tokens*panel_n;
    for (size_t i = (size_t) blockIdx.x*blockDim.x +
            threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        const int local_token = (int) (i/panel_n);
        const int col =
                (int) (i - (size_t) local_token*panel_n);
        const int token = token_offset + local_token;
        const size_t global =
                (size_t) token*AW_EMBD + output_col + col;
        float sum = 0.0f;
        #pragma unroll
        for (int step = 0; step < AW_GPU_COUNT; ++step) {
            const int logical =
                    (int) ((global/32768 + 1 + step) %
                            AW_GPU_COUNT);
            const int physical =
                    assignment[token*AW_GPU_COUNT + logical];
            const uint16_t * source =
                    physical == 0 ? partial0 :
                    physical == 1 ? partial1 :
                    physical == 2 ? partial2 : partial3;
            const int home = token/stride_tokens;
            const int local_token =
                    token - home*stride_tokens;
            const int partial_token =
                    home == physical ?
                        local_token*AW_GPU_COUNT + logical :
                        AW_GPU_COUNT*stride_tokens +
                            (token < physical*stride_tokens ?
                                token : token - stride_tokens);
            const size_t index =
                    (size_t) partial_token*panel_n + col;
            sum = aw_bf16_to_float(aw_float_to_bf16(
                        sum + aw_bf16_to_float(source[index])));
        }
        output[(size_t) local_token*AW_EMBD +
                output_col + col] = sum;
    }
}

__global__ static void aw_live_unpack_bf16(const uint16_t * input, float * output, size_t count) {
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        output[i] = aw_bf16_to_float(input[i]);
    }
}

__global__ static void aw_live_compare_f32_bits(
        const float * reference,
        const float * candidate,
        size_t count,
        unsigned long long * errors) {
    unsigned long long local_errors = 0;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
            i < count;
            i += (size_t) gridDim.x*blockDim.x) {
        local_errors +=
                __float_as_uint(reference[i]) !=
                __float_as_uint(candidate[i]);
    }
    if (local_errors != 0) {
        atomicAdd(errors, local_errors);
    }
}

#ifdef GGML_USE_NCCL
static std::array<ncclComm_t, AW_GPU_COUNT> aw_live_nccl_comms = {};
static std::once_flag aw_live_nccl_once;

static void aw_live_nccl_init() {
    std::array<int, AW_GPU_COUNT> devices = { 0, 1, 2, 3 };
    const ncclResult_t status = ncclCommInitAll(
            aw_live_nccl_comms.data(), AW_GPU_COUNT, devices.data());
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string("ncclCommInitAll failed: ") +
                ncclGetErrorString(status));
    }
}

static void aw_live_nccl_throw(ncclResult_t status, const char * operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}
#endif

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
    void ensure_managed(size_t size) {
        if (size <= size_) {
            return;
        }
        reset();
        const cudaError_t status = cudaMallocManaged(
                &ptr_, size, cudaMemAttachGlobal);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                    std::string("cudaMallocManaged(") +
                    std::to_string(size) + ") failed: " +
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

static void aw_live_dump(
        const char * directory,
        const char * kind,
        int layer,
        int home,
        int owner,
        const void * data,
        size_t count,
        size_t element_size) {
    char path[1024];
    const int length = snprintf(path, sizeof(path), "%s/l%d-h%d-o%d-%s.bin",
            directory, layer, home, owner, kind);
    if (length <= 0 || (size_t) length >= sizeof(path)) {
        throw std::runtime_error("live dump path is too long");
    }
    FILE * file = fopen(path, "wb");
    if (file == nullptr) {
        throw std::runtime_error(std::string("open live dump: ") + strerror(errno));
    }
    const uint64_t header[4] = {
        UINT64_C(0x41574c4956454431),
        (uint64_t) count,
        (uint64_t) element_size,
        0,
    };
    bool ok = fwrite(header, sizeof(header), 1, file) == 1;
    ok = fwrite(data, element_size, count, file) == count && ok;
    ok = fclose(file) == 0 && ok;
    if (!ok) {
        throw std::runtime_error("write live dump failed");
    }
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

static float aw_expected_route(int token, int wire_format, int gate_split_groups, int down_split_groups) {
    std::vector<float> input(AW_EMBD);
    for (int k = 0; k < AW_EMBD; ++k) {
        const size_t index = (size_t) token*AW_EMBD + k;
        input[k] = (float) ((int) (index % 17) - 8)*(1.0f/32.0f);
        if (wire_format == 1) {
            input[k] = aw_bf16_to_float_host(aw_float_to_bf16_host(input[k]));
        } else if (wire_format == 2) {
            input[k] = __half2float(__float2half(input[k]));
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
    aw_device_buffer route_segment_counts;
    aw_device_buffer route_segment_offsets;
    aw_device_buffer tile_counts;
    aw_device_buffer tiles_m64;
    aw_device_buffer tiles_m32;
    aw_device_buffer tiles_m16;
    aw_device_buffer tiles_warp;
    aw_device_buffer tiles_m8;
    aw_device_buffer tiles_m4;
    aw_device_buffer cohort_counts;
    aw_device_buffer cohorts_p2;
    aw_device_buffer cohorts_p3;
    aw_device_buffer cohorts_p4;
    aw_device_buffer cohort_singles;
    aw_device_buffer gate_desc;
    aw_device_buffer up_desc;
    aw_device_buffer down_desc;
    aw_device_buffer gate_panel_desc;
    aw_device_buffer up_panel_desc;
    aw_device_buffer down_panel_desc;
    aw_device_buffer weight_ptrs;
    aw_device_buffer r44_lookup;
    aw_device_buffer r44_assignment;
    aw_device_buffer r44_desc_experts;
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

    void ensure(
            int new_device,
            int new_cells,
            int new_tokens,
            bool n256,
            bool bf16_partial,
            bool r44,
            bool diagonal_panel) {
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
        if (!diagonal_panel) {
            input.ensure(total_tokens*AW_EMBD*sizeof(float));
        }
        ids.ensure(max_routes*sizeof(int32_t));
        weights.ensure(max_routes*sizeof(float));
        counts.ensure(descriptors*sizeof(int32_t));
        offsets.ensure((descriptors + 1)*sizeof(int32_t));
        cursors.ensure(descriptors*sizeof(int32_t));
        const size_t route_segments =
                (tokens*8 + AW_ROUTE_SEGMENT_SLOTS - 1)/
                AW_ROUTE_SEGMENT_SLOTS;
        const size_t route_segment_entries =
                (size_t) cells*route_segments*AW_PRIMARY_PER_GPU;
        route_segment_counts.ensure(
                route_segment_entries*sizeof(int32_t));
        route_segment_offsets.ensure(
                route_segment_entries*sizeof(int32_t));
        tile_counts.ensure(6*sizeof(int32_t));
        tiles_m64.ensure(max_tiles*sizeof(aw_tile_desc));
        tiles_m32.ensure(max_tiles*sizeof(aw_tile_desc));
        tiles_m16.ensure(max_tiles*sizeof(aw_tile_desc));
        tiles_warp.ensure(max_tiles*sizeof(aw_tile_desc));
        const size_t max_tail_tiles = 2*(size_t) descriptors;
        tiles_m8.ensure(max_tail_tiles*sizeof(aw_tile_desc));
        tiles_m4.ensure(max_tail_tiles*sizeof(aw_tile_desc));
        cohort_counts.ensure(4*sizeof(int32_t));
        cohorts_p2.ensure(
                descriptors*sizeof(aw_m16_cohort));
        cohorts_p3.ensure(
                descriptors*sizeof(aw_m16_cohort));
        cohorts_p4.ensure(
                descriptors*sizeof(aw_m16_cohort));
        cohort_singles.ensure(
                descriptors*sizeof(aw_tile_desc));
        gate_desc.ensure(descriptors*sizeof(aw_work_desc));
        up_desc.ensure(descriptors*sizeof(aw_work_desc));
        down_desc.ensure(descriptors*sizeof(aw_work_desc));
        gate_panel_desc.ensure(n256 ?
                AW_SERVICE_GU_PANELS*descriptors*sizeof(aw_work_desc) : 0);
        up_panel_desc.ensure(n256 ?
                AW_SERVICE_GU_PANELS*descriptors*sizeof(aw_work_desc) : 0);
        down_panel_desc.ensure(n256 ?
                AW_SERVICE_DOWN_PANELS*descriptors*sizeof(aw_work_desc) :
                diagonal_panel ?
                    AW_DIAGONAL_MAX_PANELS*descriptors*
                        sizeof(aw_work_desc) : 0);
        weight_ptrs.ensure(std::max(
                    (size_t) cells*6,
                    r44 ? (size_t) (AW_PRIMARY_PER_GPU +
                        AW_R44_BUDGET)*3 : 0)*sizeof(const char *));
        r44_lookup.ensure(r44 ?
                (size_t) AW_GPU_COUNT*AW_EXPERTS*sizeof(int32_t) : 0);
        r44_assignment.ensure(r44 ?
                (size_t) total_tokens*AW_GPU_COUNT*sizeof(int32_t) : 0);
        r44_desc_experts.ensure(r44 ?
                (size_t) (AW_PRIMARY_PER_GPU +
                    AW_R44_BUDGET)*sizeof(int32_t) : 0);
        route_input.ensure(max_routes*sizeof(int32_t));
        token_routes.ensure(max_routes*sizeof(int32_t));
        const size_t gate_up_n = n256 ? AW_SERVICE_PANEL_N : AW_EXPERT_FF;
        const size_t down_n = n256 ? AW_SERVICE_PANEL_N : AW_EMBD;
        if (diagonal_panel) {
            // Large data buffers are shared by all serial diagonal groups.
        } else if (r44) {
            gate.ensure_managed(
                    max_routes*gate_up_n*sizeof(float));
            up.ensure_managed(
                    max_routes*gate_up_n*sizeof(float));
            middle.ensure_managed(
                    max_routes*AW_EXPERT_FF*sizeof(float));
        } else {
            gate.ensure(max_routes*gate_up_n*sizeof(float));
            up.ensure(max_routes*gate_up_n*sizeof(float));
            middle.ensure(max_routes*AW_EXPERT_FF*sizeof(float));
            route_output.ensure(max_routes*down_n*sizeof(float));
        }
        const size_t partial_element_size = bf16_partial ? sizeof(uint16_t) : sizeof(float);
        if (diagonal_panel) {
            // Owner panels use the bounded shared ring.
        } else if (r44) {
            partial.ensure(
                    (size_t) (2*AW_GPU_COUNT - 1)*tokens*
                        AW_SERVICE_PANEL_N*partial_element_size);
        } else {
            partial.ensure(
                    total_tokens*AW_EMBD*partial_element_size);
        }
        if (!diagonal_panel && !r44 &&
                (!aw_env_on(getenv("GGML_CUDA_AW_DIRECT_OWNER")) ||
                aw_env_on(getenv("GGML_CUDA_AW_DIRECT_OWNER_CHECK")))) {
            recv.ensure((size_t) AW_GPU_COUNT*total_tokens*AW_EMBD*partial_element_size);
        }
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

struct aw_diagonal_scratch {
    int device = -1;
    int tokens = 0;
    std::array<aw_device_buffer, AW_DIAGONAL_INPUT_SLOTS> input;
    aw_device_buffer middle;
    aw_device_buffer route_output;
    std::array<aw_device_buffer, AW_DIAGONAL_RING_SLOTS> partial;
    std::array<aw_device_buffer, AW_DIAGONAL_RING_SLOTS> recv;
    cudaStream_t publish_stream = nullptr;
    std::array<cudaEvent_t, AW_DIAGONAL_INPUT_SLOTS> input_free{};
    std::array<cudaEvent_t, AW_DIAGONAL_RING_SLOTS> partial_free{};
    std::array<cudaEvent_t, AW_DIAGONAL_RING_SLOTS> partial_ready{};
    std::array<cudaEvent_t, AW_DIAGONAL_RING_SLOTS> copied{};
    std::array<cudaEvent_t, AW_DIAGONAL_RING_SLOTS> recv_free{};

    void ensure(int new_device, int new_tokens, int panel_n) {
        const int previous_tokens = tokens;
        device = new_device;
        tokens = std::max(tokens, new_tokens);
        aw_cuda_throw(cudaSetDevice(device),
                "set diagonal scratch device");
        size_t free_before = 0;
        size_t total_bytes = 0;
        const bool report_memory =
                aw_env_on(getenv("GGML_CUDA_AW_MEMORY")) &&
                tokens > previous_tokens;
        if (report_memory) {
            aw_cuda_throw(cudaMemGetInfo(
                        &free_before, &total_bytes),
                    "query diagonal memory before allocation");
        }
        const size_t max_routes = (size_t) tokens*8;
        for (aw_device_buffer & buffer : input) {
            buffer.ensure((size_t) tokens*AW_EMBD*sizeof(float));
        }
        middle.ensure(max_routes*AW_EXPERT_FF*sizeof(float));
        route_output.ensure(
                max_routes*panel_n*sizeof(float));
        const int ring_slots = std::min(
                AW_DIAGONAL_RING_SLOTS, AW_EMBD/panel_n);
        for (int slot = 0; slot < AW_DIAGONAL_RING_SLOTS; ++slot) {
            partial[slot].ensure(
                    slot < ring_slots ?
                    (size_t) tokens*panel_n*
                        sizeof(uint16_t) : 0);
            recv[slot].ensure(
                    slot < ring_slots ?
                    (size_t) AW_GPU_COUNT*tokens*
                        panel_n*sizeof(uint16_t) : 0);
        }
        if (publish_stream == nullptr) {
            int least_priority = 0;
            int greatest_priority = 0;
            aw_cuda_throw(cudaDeviceGetStreamPriorityRange(
                        &least_priority, &greatest_priority),
                    "query diagonal stream priority");
            aw_cuda_throw(cudaStreamCreateWithPriority(
                        &publish_stream, cudaStreamNonBlocking,
                        greatest_priority),
                    "create diagonal publish stream");
            auto create_event = [](cudaEvent_t * event) {
                aw_cuda_throw(cudaEventCreateWithFlags(
                            event, cudaEventDisableTiming),
                        "create diagonal event");
            };
            for (int slot = 0; slot < AW_DIAGONAL_INPUT_SLOTS;
                    ++slot) {
                create_event(&input_free[slot]);
                aw_cuda_throw(cudaEventRecord(
                            input_free[slot], publish_stream),
                        "initialize diagonal input event");
            }
            for (int slot = 0; slot < AW_DIAGONAL_RING_SLOTS;
                    ++slot) {
                create_event(&partial_free[slot]);
                create_event(&partial_ready[slot]);
                create_event(&copied[slot]);
                create_event(&recv_free[slot]);
                aw_cuda_throw(cudaEventRecord(
                            partial_free[slot], publish_stream),
                        "initialize diagonal partial event");
                aw_cuda_throw(cudaEventRecord(
                            recv_free[slot], publish_stream),
                        "initialize diagonal receive event");
            }
            if (ggml_cuda_affinity_wave_trace_enabled()) {
                char name[24];
                snprintf(name, sizeof(name), "aw-publish/d%d", device);
                ggml_cuda_affinity_wave_trace_name_stream(
                        (void *) publish_stream, name);
            }
        }
        if (report_memory) {
            size_t free_after = 0;
            aw_cuda_throw(cudaMemGetInfo(
                        &free_after, &total_bytes),
                    "query diagonal memory after allocation");
            fprintf(stderr,
                    "AffinityWave: diagonal memory device=%d free-before=%zu free-after=%zu used-after=%zu allocation-delta=%zu arena=%zu\n",
                    device, free_before, free_after,
                    total_bytes - free_after,
                    free_before - free_after, bytes());
        }
    }

    size_t bytes() const {
        size_t result = middle.size() + route_output.size();
        for (const aw_device_buffer & buffer : input) {
            result += buffer.size();
        }
        for (int slot = 0; slot < AW_DIAGONAL_RING_SLOTS; ++slot) {
            result += partial[slot].size() + recv[slot].size();
        }
        return result;
    }

    ~aw_diagonal_scratch() {
        if (device >= 0) {
            (void) cudaSetDevice(device);
        }
        if (input_free[0] != nullptr) {
            for (cudaEvent_t event : input_free) {
                (void) cudaEventDestroy(event);
            }
            for (int slot = 0; slot < AW_DIAGONAL_RING_SLOTS;
                    ++slot) {
                (void) cudaEventDestroy(partial_free[slot]);
                (void) cudaEventDestroy(partial_ready[slot]);
                (void) cudaEventDestroy(copied[slot]);
                (void) cudaEventDestroy(recv_free[slot]);
            }
        }
        if (publish_stream != nullptr) {
            (void) cudaStreamDestroy(publish_stream);
        }
    }
};

static std::array<aw_diagonal_scratch, AW_GPU_COUNT>
        aw_diagonal_scratch_states;

struct aw_pair_scratch {
    int device = -1;
    int tokens = 0;
    aw_device_buffer input;
    aw_device_buffer gate;
    aw_device_buffer up;
    aw_device_buffer weight_ptrs;
    aw_device_buffer desc_experts;
    aw_device_buffer route_meta;
    aw_device_buffer partial;
    aw_device_buffer recv;
    std::array<cudaEvent_t, AW_GPU_COUNT> copied{};

    void ensure(int new_device, int new_tokens,
            int panel_n) {
        const int previous_tokens = tokens;
        device = new_device;
        tokens = std::max(tokens, new_tokens);
        aw_cuda_throw(cudaSetDevice(device),
                "set PairWave scratch device");
        size_t free_before = 0;
        size_t total_bytes = 0;
        const bool report_memory =
                aw_env_on(getenv("GGML_CUDA_AW_MEMORY")) &&
                tokens > previous_tokens;
        if (report_memory) {
            aw_cuda_throw(cudaMemGetInfo(
                        &free_before, &total_bytes),
                    "query PairWave memory before allocation");
        }
        const size_t max_routes =
                (size_t) tokens*8;
        const size_t descriptors =
                AW_GPU_COUNT*AW_PRIMARY_PER_GPU;
        input.ensure(
                (size_t) tokens*AW_EMBD*
                sizeof(float));
        gate.ensure(
                max_routes*AW_EXPERT_FF*
                sizeof(float));
        up.ensure(
                max_routes*AW_EXPERT_FF*
                sizeof(float));
        weight_ptrs.ensure(
                descriptors*3*
                sizeof(const char *));
        desc_experts.ensure(
                descriptors*sizeof(int32_t));
        route_meta.ensure(
                (size_t) tokens*sizeof(uint2));
        partial.ensure(
                (size_t) AW_GPU_COUNT*tokens*
                panel_n*sizeof(uint16_t));
        recv.ensure(
                (size_t) AW_GPU_COUNT*tokens*
                panel_n*sizeof(uint16_t));
        if (copied[0] == nullptr) {
            for (cudaEvent_t & event : copied) {
                aw_cuda_throw(cudaEventCreateWithFlags(
                            &event, cudaEventDisableTiming),
                        "create PairWave owner-copy event");
            }
        }
        if (report_memory) {
            size_t free_after = 0;
            aw_cuda_throw(cudaMemGetInfo(
                        &free_after, &total_bytes),
                    "query PairWave memory after allocation");
            fprintf(stderr,
                    "AffinityWave: PairWave memory device=%d free-before=%zu free-after=%zu used-after=%zu allocation-delta=%zu arena=%zu\n",
                    device, free_before, free_after,
                    total_bytes - free_after,
                    free_before - free_after, bytes());
        }
    }

    size_t bytes() const {
        return input.size() + gate.size() +
                up.size() + weight_ptrs.size() +
                desc_experts.size() + route_meta.size() +
                partial.size() + recv.size();
    }

    ~aw_pair_scratch() {
        if (device >= 0) {
            (void) cudaSetDevice(device);
        }
        if (copied[0] != nullptr) {
            for (cudaEvent_t event : copied) {
                (void) cudaEventDestroy(event);
            }
        }
    }
};

static std::array<aw_pair_scratch, AW_GPU_COUNT>
        aw_pair_scratch_states;

using aw_live_weight_table = std::array<
        std::array<
            std::array<const char *, AW_GPU_COUNT*6>,
            AW_GPU_COUNT>,
        AW_GPU_COUNT>;

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

struct aw_r44_placement {
    std::array<std::array<std::array<int32_t, AW_R44_BUDGET>, AW_GPU_COUNT>, AW_LAYERS> cache{};
};

struct aw_r44_slot {
    aw_device_buffer weights;
    cudaEvent_t ready = nullptr;
    cudaEvent_t free = nullptr;
    int layer = -1;
};

struct aw_r44_device_state {
    int device = -1;
    cudaStream_t stream = nullptr;
    std::array<aw_r44_slot, AW_R44_SLOTS> slots;

    ~aw_r44_device_state() {
        if (device >= 0) {
            (void) cudaSetDevice(device);
        }
        for (aw_r44_slot & slot : slots) {
            if (slot.ready != nullptr) {
                (void) cudaEventDestroy(slot.ready);
            }
            if (slot.free != nullptr) {
                (void) cudaEventDestroy(slot.free);
            }
        }
        if (stream != nullptr) {
            (void) cudaStreamDestroy(stream);
        }
    }
};

static std::array<aw_r44_device_state, AW_GPU_COUNT> aw_r44_states;
static std::once_flag aw_r44_placement_once;
static aw_r44_placement aw_r44_map;
static std::string aw_r44_placement_error;

struct aw_group_placement {
    bool enabled = false;
    std::array<std::array<int32_t, AW_GPU_COUNT>,
            AW_LAYERS> physical_for_logical{};
    std::array<std::array<int32_t, AW_GPU_COUNT>,
            AW_LAYERS> logical_for_physical{};
};

static std::once_flag aw_group_placement_once;
static aw_group_placement aw_group_map;
static std::string aw_group_placement_error;

struct aw_pairwave_layer {
    int pair = -1;
    std::array<int32_t, AW_GPU_COUNT> owners{};
    std::array<int32_t, 2> flips{};
    std::array<std::array<int32_t, 2>,
            AW_GPU_COUNT> groups{};
};

struct aw_pairwave_manifest {
    bool enabled = false;
    std::array<aw_pairwave_layer, AW_LAYERS> layers{};
};

static std::once_flag aw_pairwave_once;
static aw_pairwave_manifest aw_pairwave_map;
static std::string aw_pairwave_error;

static void aw_pairwave_load_manifest() {
    const char * path =
            getenv("GGML_CUDA_AW_PAIRWAVE_MANIFEST");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
                std::string("cannot open PairWave manifest: ") +
                path);
    }
    std::string schema;
    int n_layers = 0;
    int n_devices = 0;
    if (!(input >> schema >> n_layers >> n_devices) ||
            schema != "pairwave-v1" ||
            n_layers != AW_LAYERS ||
            n_devices != AW_GPU_COUNT) {
        throw std::runtime_error(
                "invalid PairWave manifest header");
    }
    std::array<bool, AW_LAYERS> seen{};
    std::array<int, 2> layer_counts{};
    std::array<int, 2> attention_counts{};
    for (int row = 0; row < AW_LAYERS; ++row) {
        int layer = -1;
        aw_pairwave_layer value;
        if (!(input >> layer >> value.pair >>
                    value.owners[0] >> value.owners[1] >>
                    value.owners[2] >> value.owners[3] >>
                    value.flips[0] >> value.flips[1])) {
            throw std::runtime_error(
                    "truncated PairWave manifest");
        }
        if (layer < 0 || layer >= AW_LAYERS ||
                seen[layer]) {
            throw std::runtime_error(
                    "invalid or duplicate PairWave layer");
        }
        const int expected_pair =
                layer < AW_LAYERS/2 ?
                layer % 2 : (layer + 1) % 2;
        if (value.pair != expected_pair) {
            throw std::runtime_error(
                    "PairWave layer-pair schedule mismatch");
        }
        std::array<int, AW_GPU_COUNT> owner_counts{};
        for (int device = 0; device < AW_GPU_COUNT;
                ++device) {
            value.groups[device].fill(-1);
        }
        for (int group = 0; group < AW_GPU_COUNT;
                ++group) {
            const int owner = value.owners[group];
            if (owner < value.pair*2 ||
                    owner >= value.pair*2 + 2 ||
                    owner_counts[owner] >= 2) {
                throw std::runtime_error(
                        "invalid PairWave group ownership");
            }
            value.groups[owner][owner_counts[owner]++] =
                    group;
        }
        if (owner_counts[value.pair*2] != 2 ||
                owner_counts[value.pair*2 + 1] != 2) {
            throw std::runtime_error(
                    "PairWave requires two groups per active GPU");
        }
        if ((value.flips[0] != 0 &&
             value.flips[0] != 1) ||
                (value.flips[1] != 0 &&
                 value.flips[1] != 1)) {
            throw std::runtime_error(
                    "invalid PairWave panel orientation");
        }
        aw_pairwave_map.layers[layer] = value;
        seen[layer] = true;
        layer_counts[value.pair]++;
        if (layer % 4 == 3) {
            attention_counts[value.pair]++;
        }
    }
    std::string trailing;
    if (input >> trailing) {
        throw std::runtime_error(
                "trailing data in PairWave manifest");
    }
    if (layer_counts[0] != AW_LAYERS/2 ||
            layer_counts[1] != AW_LAYERS/2 ||
            attention_counts[0] != 5 ||
            attention_counts[1] != 5) {
        throw std::runtime_error(
                "unbalanced PairWave layer assignment");
    }
    aw_pairwave_map.enabled = true;
    fprintf(stderr,
            "AffinityWave: CUDA loaded PairWave manifest %s\n",
            path);
}

static const aw_pairwave_manifest & aw_pairwave_get_manifest() {
    std::call_once(aw_pairwave_once, []() {
        try {
            aw_pairwave_load_manifest();
        } catch (const std::exception & exception) {
            aw_pairwave_error = exception.what();
        }
    });
    if (!aw_pairwave_error.empty()) {
        throw std::runtime_error(aw_pairwave_error);
    }
    return aw_pairwave_map;
}

static void aw_group_load_placement() {
    for (int layer = 0; layer < AW_LAYERS; ++layer) {
        for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
            aw_group_map.physical_for_logical[layer][owner] =
                    owner;
            aw_group_map.logical_for_physical[layer][owner] =
                    owner;
        }
    }

    const char * path =
            getenv("GGML_CUDA_AW_GROUP_PLACEMENT");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
                std::string(
                    "cannot open GGML_CUDA_AW_GROUP_PLACEMENT: ") +
                path);
    }

    std::array<bool, AW_LAYERS> layer_seen{};
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r") ==
                std::string::npos) {
            continue;
        }
        std::istringstream values(line);
        int layer = -1;
        std::array<int32_t, AW_GPU_COUNT> physical{};
        if (!(values >> layer >> physical[0] >> physical[1] >>
                    physical[2] >> physical[3])) {
            throw std::runtime_error(
                    "invalid group placement line " +
                    std::to_string(line_number));
        }
        std::string trailing;
        if (values >> trailing) {
            throw std::runtime_error(
                    "trailing data in group placement line " +
                    std::to_string(line_number));
        }
        if (layer < 0 || layer >= AW_LAYERS ||
                layer_seen[layer]) {
            throw std::runtime_error(
                    "invalid or duplicate group placement layer");
        }
        std::array<bool, AW_GPU_COUNT> owner_seen{};
        for (int logical = 0; logical < AW_GPU_COUNT;
                ++logical) {
            const int owner = physical[logical];
            if (owner < 0 || owner >= AW_GPU_COUNT ||
                    owner_seen[owner]) {
                throw std::runtime_error(
                        "group placement is not a permutation");
            }
            owner_seen[owner] = true;
            aw_group_map.physical_for_logical[layer][logical] =
                    owner;
            aw_group_map.logical_for_physical[layer][owner] =
                    logical;
        }
        layer_seen[layer] = true;
    }
    if (!std::all_of(layer_seen.begin(), layer_seen.end(),
                [](bool value) { return value; })) {
        throw std::runtime_error(
                "group placement must contain all 40 layers");
    }
    aw_group_map.enabled = true;
    fprintf(stderr,
            "AffinityWave: loaded whole-owner group placement %s\n",
            path);
}

static const aw_group_placement & aw_group_get_placement() {
    std::call_once(aw_group_placement_once, []() {
        try {
            aw_group_load_placement();
        } catch (const std::exception & exception) {
            aw_group_placement_error = exception.what();
        }
    });
    if (!aw_group_placement_error.empty()) {
        throw std::runtime_error(aw_group_placement_error);
    }
    return aw_group_map;
}

static const char * aw_r44_mode() {
    const char * mode = getenv("GGML_CUDA_AW_R44");
    return mode != nullptr ? mode : "0";
}

static bool aw_r44_enabled() {
    const char * mode = aw_r44_mode();
    return strcmp(mode, "prefetch") == 0 || strcmp(mode, "service") == 0;
}

static int aw_r44_slot_count() {
    const char * value = getenv("GGML_CUDA_AW_R44_SLOTS");
    if (value == nullptr || value[0] == '\0') {
        return 2;
    }
    if (strcmp(value, "1") == 0) {
        return 1;
    }
    if (strcmp(value, "2") == 0) {
        return 2;
    }
    throw std::runtime_error(
            "GGML_CUDA_AW_R44_SLOTS must be '1' or '2'");
}

static void aw_r44_load_placement() {
    const char * path = getenv("GGML_CUDA_AW_R44_MAP");
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error(
                "GGML_CUDA_AW_R44_MAP is required when GGML_CUDA_AW_R44 is enabled");
    }
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) {
        throw std::runtime_error(
                std::string("cannot open GGML_CUDA_AW_R44_MAP: ") + path);
    }
    std::array<uint32_t, 6> header{};
    input.read((char *) header.data(), sizeof(header));
    constexpr uint32_t magic = 0x34525741u;
    if (!input || header[0] != magic || header[1] != 1 ||
            header[2] != AW_LAYERS || header[3] != AW_GPU_COUNT ||
            header[4] != AW_EXPERTS || header[5] != AW_R44_BUDGET) {
        throw std::runtime_error("invalid R44 placement manifest header");
    }
    input.read((char *) aw_r44_map.cache.data(),
            sizeof(aw_r44_map.cache));
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid R44 placement manifest size");
    }
    for (int layer = 0; layer < AW_LAYERS; ++layer) {
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            std::array<bool, AW_EXPERTS> seen{};
            for (int slot = 0; slot < AW_R44_BUDGET; ++slot) {
                const int expert = aw_r44_map.cache[layer][device][slot];
                if (expert < 0 || expert >= AW_EXPERTS ||
                        expert/AW_PRIMARY_PER_GPU == device ||
                        seen[expert]) {
                    throw std::runtime_error(
                            "invalid R44 cache entry");
                }
                seen[expert] = true;
            }
        }
    }
}

static const aw_r44_placement & aw_r44_get_placement() {
    std::call_once(aw_r44_placement_once, []() {
        try {
            aw_r44_load_placement();
        } catch (const std::exception & exception) {
            aw_r44_placement_error = exception.what();
        }
    });
    if (!aw_r44_placement_error.empty()) {
        throw std::runtime_error(aw_r44_placement_error);
    }
    return aw_r44_map;
}

static aw_r44_slot & aw_r44_prepare_slot(
        int device, int layer, size_t cache_bytes) {
    aw_r44_device_state & state = aw_r44_states[device];
    aw_cuda_throw(cudaSetDevice(device), "set R44 cache device");
    if (state.device < 0) {
        state.device = device;
        aw_cuda_throw(cudaStreamCreateWithFlags(
                    &state.stream, cudaStreamNonBlocking),
                "create R44 copy stream");
        if (ggml_cuda_affinity_wave_trace_enabled()) {
            char name[24];
            snprintf(name, sizeof(name), "aw-r44/d%d", device);
            ggml_cuda_affinity_wave_trace_name_stream(
                    (void *) state.stream, name);
        }
        for (aw_r44_slot & slot : state.slots) {
            aw_cuda_throw(cudaEventCreateWithFlags(
                        &slot.ready, cudaEventDisableTiming),
                    "create R44 ready event");
            aw_cuda_throw(cudaEventCreateWithFlags(
                        &slot.free, cudaEventDisableTiming),
                    "create R44 free event");
            aw_cuda_throw(cudaEventRecord(slot.free, state.stream),
                    "initialize R44 free event");
        }
    }
    aw_r44_slot & slot =
            state.slots[layer % aw_r44_slot_count()];
    slot.weights.ensure(cache_bytes);
    aw_cuda_throw(cudaStreamWaitEvent(state.stream, slot.free, 0),
            "wait for R44 cache slot");
    slot.layer = layer;
    return slot;
}

template<bool BF16_INPUT>
static void aw_launch_cohortrail_m16(
        const aw_work_desc * descs,
        const aw_tile_desc * tiles,
        const int32_t * counts,
        const aw_m16_cohort * cohorts_p2,
        const aw_m16_cohort * cohorts_p3,
        const aw_m16_cohort * cohorts_p4,
        const aw_tile_desc * singles,
        int n,
        int k,
        cudaStream_t stream,
        int sm_count,
        int max_cohort = 4) {
    if (max_cohort >= 4) {
        aw_q8_service_m16_cohortrail<
                BF16_INPUT, 4, 64>
                <<<sm_count*2, 4*64 + 64,
                    0, stream>>>(
                        descs, tiles, cohorts_p4,
                        0, n, k, counts + 2);
    }
    if (max_cohort >= 3) {
        aw_q8_service_m16_cohortrail<
                BF16_INPUT, 3, 64>
                <<<sm_count*2, 3*64 + 64,
                    0, stream>>>(
                        descs, tiles, cohorts_p3,
                        0, n, k, counts + 1);
    }
    aw_q8_service_m16_cohortrail<
            BF16_INPUT, 2, 64>
            <<<sm_count*
                    aw_q8_cohortrail_p2_ctas(),
                2*64 + 64, 0, stream>>>(
                    descs, tiles, cohorts_p2,
                    0, n, k, counts + 0);
    aw_q8_service_m16_rail_pair<BF16_INPUT>
            <<<sm_count*aw_q8_cohortrail_single_ctas(),
                128, 0, stream>>>(
                    descs, singles,
                    0, n, k, counts + 3);
}

template<bool BF16_INPUT>
static void aw_launch_cohortrail_m32(
        const aw_work_desc * descs,
        const aw_tile_desc * tiles,
        int n_mtiles,
        int n,
        int k,
        const int * n_mtiles_dev,
        cudaStream_t stream,
        int retained_blocks) {
    aw_q8_service_m32<
            BF16_INPUT, true, true>
            <<<retained_blocks,
                128, 0, stream>>>(
                    descs, tiles,
                    n_mtiles, n, k,
                    n_mtiles_dev);
}

template<bool BF16_INPUT>
static void aw_live_launch_cohortrail_m16(
        aw_live_state & state,
        const aw_work_desc * descs,
        const aw_tile_desc * tiles,
        int n,
        int k,
        cudaStream_t stream,
        int sm_count,
        int max_cohort) {
    aw_launch_cohortrail_m16<
            BF16_INPUT>(
                    descs, tiles,
                    (const int32_t *)
                        state.cohort_counts.get(),
                    (const aw_m16_cohort *)
                        state.cohorts_p2.get(),
                    (const aw_m16_cohort *)
                        state.cohorts_p3.get(),
                    (const aw_m16_cohort *)
                        state.cohorts_p4.get(),
                    (const aw_tile_desc *)
                        state.cohort_singles.get(),
                    n, k, stream, sm_count,
                    max_cohort);
}

static void aw_live_launch_projection(
        aw_live_state & state,
        const aw_device_buffer & desc,
        int n,
        int k,
        cudaStream_t stream,
        int persistent_blocks,
        int pair_blocks,
        bool bf16_input,
        size_t desc_offset = 0,
        bool bundle_m32 = false,
        bool bundle_m16 = false,
        int bundle_m16_ctas = 5,
        int cohortrail_max_cohort = 4,
        bool cohortrail_queues_ready = false,
        bool cohortrail_singletons_only = false) {
    const auto * desc_ptr =
            (const aw_work_desc *) desc.get() + desc_offset;
    const auto * tile_count = (const int32_t *) state.tile_counts.get();
    if (aw_q8_cohortrail_enabled()) {
        const int sm_count = pair_blocks/3;
        if (!cohortrail_singletons_only &&
                !cohortrail_queues_ready) {
            aw_build_m16_cohorts<<<
                    1, AW_COHORTRAIL_MAX_TILES,
                    0, stream>>>(
                    desc_ptr,
                    (const aw_tile_desc *)
                        state.tiles_m16.get(),
                    0, tile_count + 2,
                    (int32_t *)
                        state.cohort_counts.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p2.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p3.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p4.get(),
                    (aw_tile_desc *)
                        state.cohort_singles.get());
        }
        if (bf16_input) {
            aw_q8_service_m64_n128_halfpipe_sync<true>
                    <<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
            aw_launch_cohortrail_m32<true>(
                    desc_ptr,
                    (const aw_tile_desc *)
                        state.tiles_m32.get(),
                    0, n, k, tile_count + 1,
                    stream, persistent_blocks);
            if (cohortrail_singletons_only) {
                aw_q8_service_m16_rail_pair<true>
                        <<<sm_count*
                                aw_q8_cohortrail_single_ctas(),
                            128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                0, n, k, tile_count + 2);
            } else {
                aw_live_launch_cohortrail_m16<
                        true>(
                            state, desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m16.get(),
                            n, k, stream, sm_count,
                            cohortrail_max_cohort);
            }
        } else {
            aw_q8_service_m64_n128_halfpipe_sync<false>
                    <<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
            aw_launch_cohortrail_m32<false>(
                    desc_ptr,
                    (const aw_tile_desc *)
                        state.tiles_m32.get(),
                    0, n, k, tile_count + 1,
                    stream, persistent_blocks);
            if (cohortrail_singletons_only) {
                aw_q8_service_m16_rail_pair<false>
                        <<<sm_count*
                                aw_q8_cohortrail_single_ctas(),
                            128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                0, n, k, tile_count + 2);
            } else {
                aw_live_launch_cohortrail_m16<
                        false>(
                            state, desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m16.get(),
                            n, k, stream, sm_count,
                            cohortrail_max_cohort);
            }
        }
        return;
    }
    if (aw_q8_compact_enabled()) {
        const int sm_count = pair_blocks/3;
        if (bf16_input) {
            aw_q8_compact<true, 64><<<sm_count*3, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                    0, n, k, tile_count + 0);
            aw_q8_compact<true, 32><<<sm_count*5, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, n, k, tile_count + 1);
            aw_q8_compact<true, 16><<<sm_count*6, 64, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                    0, n, k, tile_count + 2);
        } else {
            aw_q8_compact<false, 64><<<sm_count*3, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                    0, n, k, tile_count + 0);
            aw_q8_compact<false, 32><<<sm_count*5, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, n, k, tile_count + 1);
            aw_q8_compact<false, 16><<<sm_count*6, 64, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                    0, n, k, tile_count + 2);
        }
        return;
    }
    if (aw_q8_warpwave_enabled()) {
        if (bf16_input) {
            aw_q8_warpwave<true><<<pair_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_warp.get(),
                    0, n, k, tile_count + 3);
        } else {
            aw_q8_warpwave<false><<<pair_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_warp.get(),
                    0, n, k, tile_count + 3);
        }
        return;
    }
    if (aw_q8_broadwave_enabled() || aw_q8_halfpipe_sync_enabled() ||
            aw_q8_halfpipe_bar_enabled() ||
            aw_q8_doublewave_enabled() ||
            aw_q8_dualrailwave_enabled() ||
            aw_q8_flexwave_enabled() || aw_q8_flexwave1_enabled() ||
            aw_q8_rendezvous_enabled() || aw_q8_megawave_enabled() ||
            aw_q8_fabwave_enabled()) {
        const bool doublewave = aw_q8_doublewave_enabled();
        const bool dualrailwave = aw_q8_dualrailwave_enabled();
        const bool halfpipe_sync = aw_q8_halfpipe_sync_enabled();
        const bool halfpipe_bar = aw_q8_halfpipe_bar_enabled();
        const bool flexwave = aw_q8_flexwave_enabled() || aw_q8_flexwave1_enabled();
        const bool flexwave1 = aw_q8_flexwave1_enabled();
        const bool megawave = aw_q8_megawave_enabled();
        const bool fabwave = aw_q8_fabwave_enabled();
        const bool fabwave_k8 = aw_q8_fabwave_k8_enabled();
        const bool fabwave_qscale = aw_q8_fabwave_qscale_enabled();
        const bool fabwave_dual = aw_q8_fabwave_dual_enabled();
        const bool fabwave_exact = fabwave && aw_q8_fabwave_exact_projection(n, k);
        const int flex_blocks = (pair_blocks/3)*(flexwave1 ? 6 : 4);
        if (bf16_input) {
            if (doublewave) {
                aw_q8_service_m64_n128_double<true>
                        <<<pair_blocks/3, 512, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (dualrailwave) {
                aw_q8_service_m64_n128_dualrail<true>
                        <<<persistent_blocks, 512, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (halfpipe_sync) {
                aw_q8_service_m64_n128_halfpipe_sync<true>
                        <<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (halfpipe_bar) {
                aw_q8_service_m64_n128_halfpipe_bar<true>
                        <<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (megawave) {
                aw_q8_service_m64_n128_1024<true><<<pair_blocks/3, 1024, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            } else if (fabwave && !fabwave_exact) {
                if (fabwave_dual) {
                    aw_q8_fab_m64_n128<true, 4, true, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_k8 && fabwave_qscale) {
                    aw_q8_fab_m64_n128<true, 4, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_k8) {
                    aw_q8_fab_m64_n128<true, 4><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_qscale) {
                    aw_q8_fab_m64_n128<true, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else {
                    aw_q8_fab_m64_n128<true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                }
            } else {
                aw_q8_service_m64_n128_256<true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            }
            if (bundle_m32) {
                aw_q8_service_m32_bundle<true, true, true>
                        <<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m32.get(),
                            (const aw_tile_desc *)
                                state.tiles_m8.get(),
                            0, n, k, tile_count + 4);
                aw_q8_service_m32<true, true, true>
                        <<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m4.get(),
                            0, n, k, tile_count + 5);
            } else {
                aw_q8_service_m32<true, true, true>
                        <<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m32.get(),
                            0, n, k, tile_count + 1);
            }
            if (flexwave) {
                if (flexwave1) {
                    aw_q8_flexwave16<true, false><<<flex_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                            0, n, k, tile_count + 2);
                } else {
                    aw_q8_flexwave16<true, true><<<flex_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                            0, n, k, tile_count + 2);
                }
            } else {
                if (bundle_m16) {
                    const int bundle_blocks =
                            (pair_blocks/3)*
                            bundle_m16_ctas;
                    aw_q8_service_m16_bundle<true, true>
                            <<<bundle_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                (const aw_tile_desc *)
                                    state.tiles_m8.get(),
                                0, n, k,
                                tile_count + 4);
                    aw_q8_service_m16_pair<true, true>
                            <<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m4.get(),
                                0, n, k,
                                tile_count + 5);
                } else {
                    aw_q8_service_m16_pair<true, true>
                            <<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                0, n, k,
                                tile_count + 2);
                }
            }
        } else {
            if (doublewave) {
                aw_q8_service_m64_n128_double<false>
                        <<<pair_blocks/3, 512, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (dualrailwave) {
                aw_q8_service_m64_n128_dualrail<false>
                        <<<persistent_blocks, 512, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (halfpipe_sync) {
                aw_q8_service_m64_n128_halfpipe_sync<false>
                        <<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (halfpipe_bar) {
                aw_q8_service_m64_n128_halfpipe_bar<false>
                        <<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *) state.tiles_m64.get(),
                                0, n, k, tile_count + 0);
            } else if (megawave) {
                aw_q8_service_m64_n128_1024<false><<<pair_blocks/3, 1024, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            } else if (fabwave && !fabwave_exact) {
                if (fabwave_dual) {
                    aw_q8_fab_m64_n128<false, 4, true, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_k8 && fabwave_qscale) {
                    aw_q8_fab_m64_n128<false, 4, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_k8) {
                    aw_q8_fab_m64_n128<false, 4><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else if (fabwave_qscale) {
                    aw_q8_fab_m64_n128<false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                } else {
                    aw_q8_fab_m64_n128<false><<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                            0, n, k, tile_count + 0);
                }
            } else {
                aw_q8_service_m64_n128_256<false><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            }
            if (bundle_m32) {
                aw_q8_service_m32_bundle<false, true, true>
                        <<<persistent_blocks, 256, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m32.get(),
                            (const aw_tile_desc *)
                                state.tiles_m8.get(),
                            0, n, k, tile_count + 4);
                aw_q8_service_m32<false, true, true>
                        <<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m4.get(),
                            0, n, k, tile_count + 5);
            } else {
                aw_q8_service_m32<false, true, true>
                        <<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr,
                            (const aw_tile_desc *)
                                state.tiles_m32.get(),
                            0, n, k, tile_count + 1);
            }
            if (flexwave) {
                if (flexwave1) {
                    aw_q8_flexwave16<false, false><<<flex_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                            0, n, k, tile_count + 2);
                } else {
                    aw_q8_flexwave16<false, true><<<flex_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                            0, n, k, tile_count + 2);
                }
            } else {
                if (bundle_m16) {
                    const int bundle_blocks =
                            (pair_blocks/3)*
                            bundle_m16_ctas;
                    aw_q8_service_m16_bundle<false, true>
                            <<<bundle_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                (const aw_tile_desc *)
                                    state.tiles_m8.get(),
                                0, n, k,
                                tile_count + 4);
                    aw_q8_service_m16_pair<false, true>
                            <<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m4.get(),
                                0, n, k,
                                tile_count + 5);
                } else {
                    aw_q8_service_m16_pair<false, true>
                            <<<pair_blocks, 128, 0, stream>>>(
                                desc_ptr,
                                (const aw_tile_desc *)
                                    state.tiles_m16.get(),
                                0, n, k,
                                tile_count + 2);
                }
            }
        }
        return;
    }
    if (aw_q8_widewave_enabled()) {
        const int sm_count = pair_blocks/3;
        if (bf16_input) {
            aw_q8_service_m64_n128<true><<<sm_count, 512, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                    0, n, k, tile_count + 0);
            if (aw_q8_interleave_enabled()) {
                aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            } else {
                aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            }
            aw_q8_tailwave<true, 8><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m8.get(),
                    0, n, k, tile_count + 4);
            aw_q8_tailwave<true, 4><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m4.get(),
                    0, n, k, tile_count + 5);
        } else {
            aw_q8_service_m64_n128<false><<<sm_count, 512, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                    0, n, k, tile_count + 0);
            if (aw_q8_interleave_enabled()) {
                aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            } else {
                aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            }
            aw_q8_tailwave<false, 8><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m8.get(),
                    0, n, k, tile_count + 4);
            aw_q8_tailwave<false, 4><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m4.get(),
                    0, n, k, tile_count + 5);
        }
        return;
    }
    if (aw_q8_tailwave_enabled()) {
        const int sm_count = pair_blocks/3;
        const bool interleave = aw_q8_interleave_enabled();
        if (bf16_input) {
            if (interleave) {
                aw_q8_service_m64<true, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
                aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            } else {
                aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
                aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            }
            aw_q8_tailwave<true, 8><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m8.get(),
                    0, n, k, tile_count + 4);
            aw_q8_tailwave<true, 4><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m4.get(),
                    0, n, k, tile_count + 5);
        } else {
            if (interleave) {
                aw_q8_service_m64<false, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
                aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            } else {
                aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
                aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                        0, n, k, tile_count + 1);
            }
            aw_q8_tailwave<false, 8><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m8.get(),
                    0, n, k, tile_count + 4);
            aw_q8_tailwave<false, 4><<<sm_count*8, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m4.get(),
                    0, n, k, tile_count + 5);
        }
        return;
    }
    if (aw_q8_hybrid_enabled()) {
        const int sm_count = pair_blocks/3;
        const bool interleave = aw_q8_interleave_enabled();
        if (bf16_input) {
            if (interleave) {
                aw_q8_service_m64<true, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            } else {
                aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            }
            aw_q8_compact<true, 32><<<sm_count*5, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, n, k, tile_count + 1);
            aw_q8_compact<true, 16><<<sm_count*6, 64, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                    0, n, k, tile_count + 2);
        } else {
            if (interleave) {
                aw_q8_service_m64<false, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            } else {
                aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                        desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(),
                        0, n, k, tile_count + 0);
            }
            aw_q8_compact<false, 32><<<sm_count*5, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, n, k, tile_count + 1);
            aw_q8_compact<false, 16><<<sm_count*6, 64, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(),
                    0, n, k, tile_count + 2);
        }
        return;
    }
    const bool interleave = aw_q8_interleave_enabled();
    if (bf16_input) {
        if (interleave) {
            aw_q8_service_m64<true, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
            aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        } else {
            aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
            aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        }
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(), 0, n, k, tile_count + 2);
    } else {
        if (interleave) {
            aw_q8_service_m64<false, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
            aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        } else {
            aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m64.get(), 0, n, k, tile_count + 0);
            aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                    desc_ptr, (const aw_tile_desc *) state.tiles_m32.get(), 0, n, k, tile_count + 1);
        }
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                desc_ptr, (const aw_tile_desc *) state.tiles_m16.get(), 0, n, k, tile_count + 2);
    }
}

static void aw_live_launch_diagonal_projection(
        aw_live_state & state,
        const aw_device_buffer & desc,
        int n,
        int k,
        cudaStream_t stream,
        int persistent_blocks,
        int pair_blocks,
        size_t desc_offset = 0) {
    aw_live_launch_projection(
            state, desc, n, k, stream,
            persistent_blocks, pair_blocks, false,
            desc_offset, false, false, 5, 4, false,
            aw_p100_exact_enabled());
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
    const bool interleave = aw_q8_interleave_enabled();
    if (bf16_input) {
        aw_q8_service_m64_gate_up<true><<<fused_blocks, 512, 0, stream>>>(
                gate_desc, up_desc, (const aw_tile_desc *) state.tiles_m64.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 0);
        if (interleave) {
            aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        } else {
            aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                    gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        }
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
        if (interleave) {
            aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        } else {
            aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                    up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        }
        aw_q8_service_m16_pair<true, true><<<pair_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
    } else {
        aw_q8_service_m64_gate_up<false><<<fused_blocks, 512, 0, stream>>>(
                gate_desc, up_desc, (const aw_tile_desc *) state.tiles_m64.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 0);
        if (interleave) {
            aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        } else {
            aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                    gate_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        }
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                gate_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
        if (interleave) {
            aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                    up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        } else {
            aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                    up_desc, (const aw_tile_desc *) state.tiles_m32.get(),
                    0, AW_EXPERT_FF, AW_EMBD, tile_count + 1);
        }
        aw_q8_service_m16_pair<false, true><<<pair_blocks, 128, 0, stream>>>(
                up_desc, (const aw_tile_desc *) state.tiles_m16.get(),
                0, AW_EXPERT_FF, AW_EMBD, tile_count + 2);
    }
}

struct aw_headfold_gdn_state {
    int device = -1;
    int segment_tokens = 0;
    aw_device_buffer q;
    aw_device_buffer k;
    aw_device_buffer v;
    aw_device_buffer gate;
    aw_device_buffer beta;
    aw_device_buffer output;
    aw_device_buffer state;
    cudaEvent_t input_ready = nullptr;
    cudaEvent_t output_done = nullptr;

    void ensure(int new_device, int new_segment_tokens) {
        device = new_device;
        segment_tokens = std::max(segment_tokens, new_segment_tokens);
        aw_cuda_throw(cudaSetDevice(device), "set HeadFold GDN device");
        const size_t qk_values =
                (size_t) segment_tokens*AW_HEADFOLD_QK_HEADS*
                AW_GDN_HEAD_DIM;
        const size_t value_values =
                (size_t) segment_tokens*AW_HEADFOLD_VALUE_HEADS*
                AW_GDN_HEAD_DIM;
        const size_t scalar_values =
                (size_t) segment_tokens*AW_HEADFOLD_VALUE_HEADS;
        const size_t state_values =
                (size_t) AW_HEADFOLD_VALUE_HEADS*
                AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM;
        q.ensure(qk_values*sizeof(float));
        k.ensure(qk_values*sizeof(float));
        v.ensure(value_values*sizeof(float));
        gate.ensure(scalar_values*sizeof(float));
        beta.ensure(scalar_values*sizeof(float));
        output.ensure(value_values*sizeof(float));
        state.ensure(state_values*sizeof(float));
        if (input_ready == nullptr) {
            aw_cuda_throw(cudaEventCreateWithFlags(
                        &input_ready, cudaEventDisableTiming),
                    "create HeadFold input event");
            aw_cuda_throw(cudaEventCreateWithFlags(
                        &output_done, cudaEventDisableTiming),
                    "create HeadFold output event");
        }
    }

    ~aw_headfold_gdn_state() {
        if (device >= 0) {
            (void) cudaSetDevice(device);
        }
        if (input_ready != nullptr) {
            (void) cudaEventDestroy(input_ready);
            (void) cudaEventDestroy(output_done);
        }
    }
};

static std::array<aw_headfold_gdn_state, AW_GPU_COUNT>
        aw_headfold_gdn_states;

static void aw_set_error(char * error, size_t capacity, const std::string & message) {
    if (error != nullptr && capacity != 0) {
        snprintf(error, capacity, "%s", message.c_str());
    }
}

} // namespace

bool ggml_cuda_affinity_wave_trace_enabled() {
    static const bool enabled = aw_env_on(getenv("GGML_CUDA_AW_TRACE"));
    return enabled;
}

bool ggml_cuda_affinity_wave_gemm_map_enabled() {
    static const bool enabled = aw_env_on(getenv("GGML_CUDA_AW_GEMM_MAP"));
    return enabled;
}

void ggml_cuda_affinity_wave_trace_push(const char * name, uint32_t category, uint64_t payload) {
    if (ggml_cuda_affinity_wave_trace_enabled() || ggml_cuda_affinity_wave_gemm_map_enabled()) {
        aw_trace_event(name, category, payload, true);
    }
}

void ggml_cuda_affinity_wave_trace_pop() {
    if (ggml_cuda_affinity_wave_trace_enabled() || ggml_cuda_affinity_wave_gemm_map_enabled()) {
        nvtxDomainRangePop(aw_trace_domain());
    }
}

void ggml_cuda_affinity_wave_trace_mark(const char * name, uint32_t category, uint64_t payload) {
    if (ggml_cuda_affinity_wave_trace_enabled()) {
        aw_trace_event(name, category, payload, false);
    }
}

void ggml_cuda_affinity_wave_trace_name_thread(const char * name) {
    if (ggml_cuda_affinity_wave_trace_enabled()) {
#if defined(__linux__)
        nvtxNameOsThreadA((uint32_t) syscall(SYS_gettid), name);
#else
        GGML_UNUSED(name);
#endif
    }
}

void ggml_cuda_affinity_wave_trace_name_stream(void * stream, const char * name) {
    if (ggml_cuda_affinity_wave_trace_enabled()) {
        nvtxNameCudaStreamA((cudaStream_t) stream, name);
    }
}

int ggml_cuda_affinity_wave_headfold_gdn(
        void * const * streams_ptr,
        const ggml_cuda_aw_headfold_gdn_lane * lanes,
        int32_t n_lanes,
        char * error,
        size_t error_capacity) {
    try {
        if (streams_ptr == nullptr || lanes == nullptr ||
                n_lanes != AW_GPU_COUNT) {
            throw std::runtime_error("invalid HeadFold GDN request");
        }
        const int tokens = lanes[0].tokens;
        if (tokens < AW_HEADFOLD_PARTS ||
                tokens % AW_HEADFOLD_PARTS != 0) {
            throw std::runtime_error(
                    "HeadFold GDN tokens must be divisible by four");
        }
        for (int lane = 0; lane < AW_GPU_COUNT; ++lane) {
            if (lanes[lane].tokens != tokens ||
                    lanes[lane].qk_stride <
                        AW_GDN_QK_HEADS*AW_GDN_HEAD_DIM ||
                    lanes[lane].v_stride <
                        AW_GDN_VALUE_HEADS*AW_GDN_HEAD_DIM ||
                    lanes[lane].scalar_stride <
                        AW_GDN_VALUE_HEADS ||
                    lanes[lane].q == nullptr ||
                    lanes[lane].k == nullptr ||
                    lanes[lane].v == nullptr ||
                    lanes[lane].gate == nullptr ||
                    lanes[lane].beta == nullptr ||
                    lanes[lane].state == nullptr ||
                    lanes[lane].output == nullptr ||
                    lanes[lane].state_output == nullptr) {
                throw std::runtime_error(
                        "invalid HeadFold GDN lane descriptor");
            }
        }

        const int segment_tokens = tokens/AW_HEADFOLD_PARTS;
        std::array<cudaStream_t, AW_GPU_COUNT> streams;
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            streams[device] = (cudaStream_t) streams_ptr[device];
            aw_cuda_throw(cudaSetDevice(device),
                    "set HeadFold source device");
            aw_headfold_gdn_states[device].ensure(
                    device, segment_tokens);
            aw_cuda_throw(cudaEventRecord(
                        aw_headfold_gdn_states[device].input_ready,
                        streams[device]),
                    "record HeadFold input ready");
        }

        constexpr size_t qk_panel_pitch =
                (size_t) AW_HEADFOLD_QK_HEADS*AW_GDN_HEAD_DIM*
                sizeof(float);
        constexpr size_t value_panel_pitch =
                (size_t) AW_HEADFOLD_VALUE_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t value_output_pitch =
                (size_t) AW_GDN_VALUE_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t scalar_panel_pitch =
                (size_t) AW_HEADFOLD_VALUE_HEADS*sizeof(float);
        constexpr size_t qk_half_width =
                (size_t) AW_HEADFOLD_QK_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t value_half_width =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t scalar_half_width =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                sizeof(float);
        constexpr size_t state_head_bytes =
                (size_t) AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM*
                sizeof(float);
        constexpr size_t state_half_bytes =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                state_head_bytes;

        for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
            aw_cuda_throw(cudaSetDevice(owner),
                    "set HeadFold owner device");
            aw_headfold_gdn_state & state =
                    aw_headfold_gdn_states[owner];
            cudaStream_t stream = streams[owner];
            for (int source = 0; source < AW_GPU_COUNT; ++source) {
                aw_cuda_throw(cudaStreamWaitEvent(
                            stream,
                            aw_headfold_gdn_states[source].input_ready,
                            0),
                        "wait for HeadFold source");
            }

            const size_t qk_source_col =
                    (size_t) owner*AW_HEADFOLD_QK_HEADS*
                    AW_GDN_HEAD_DIM;
            const size_t value_source_col0 = qk_source_col;
            const size_t value_source_col1 =
                    (size_t) (AW_GDN_QK_HEADS +
                        owner*AW_HEADFOLD_QK_HEADS)*
                    AW_GDN_HEAD_DIM;
            const size_t scalar_source_col0 =
                    (size_t) owner*AW_HEADFOLD_QK_HEADS;
            const size_t scalar_source_col1 =
                    (size_t) AW_GDN_QK_HEADS +
                    owner*AW_HEADFOLD_QK_HEADS;

            aw_cuda_throw(cudaMemcpyAsync(
                        state.state.get(),
                        lanes[0].state +
                            (size_t) owner*
                            (AW_HEADFOLD_VALUE_HEADS/2)*
                            AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM,
                        state_half_bytes,
                        cudaMemcpyDefault, stream),
                    "copy HeadFold initial state low");
            aw_cuda_throw(cudaMemcpyAsync(
                        (char *) state.state.get() +
                            state_half_bytes,
                        lanes[0].state +
                            (size_t) (AW_GDN_QK_HEADS +
                                owner*
                                (AW_HEADFOLD_VALUE_HEADS/2))*
                            AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM,
                        state_half_bytes,
                        cudaMemcpyDefault, stream),
                    "copy HeadFold initial state high");

            for (int source = 0; source < AW_GPU_COUNT; ++source) {
                for (int part = 0; part < AW_HEADFOLD_PARTS;
                        ++part) {
                    const size_t token =
                            (size_t) part*segment_tokens;
                    const float * q_source =
                            lanes[source].q +
                            token*lanes[source].qk_stride +
                            qk_source_col;
                    const float * k_source =
                            lanes[source].k +
                            token*lanes[source].qk_stride +
                            qk_source_col;
                    const float * v_source =
                            lanes[source].v +
                            token*lanes[source].v_stride;
                    const float * gate_source =
                            lanes[source].gate +
                            token*lanes[source].scalar_stride;
                    const float * beta_source =
                            lanes[source].beta +
                            token*lanes[source].scalar_stride;

                    aw_cuda_throw(cudaMemcpy2DAsync(
                                state.q.get(), qk_panel_pitch,
                                q_source,
                                (size_t) lanes[source].qk_stride*
                                    sizeof(float),
                                qk_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold Q");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                state.k.get(), qk_panel_pitch,
                                k_source,
                                (size_t) lanes[source].qk_stride*
                                    sizeof(float),
                                qk_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold K");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                state.v.get(), value_panel_pitch,
                                v_source + value_source_col0,
                                (size_t) lanes[source].v_stride*
                                    sizeof(float),
                                value_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold V low");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                (char *) state.v.get() +
                                    value_half_width,
                                value_panel_pitch,
                                v_source + value_source_col1,
                                (size_t) lanes[source].v_stride*
                                    sizeof(float),
                                value_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold V high");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                state.gate.get(), scalar_panel_pitch,
                                gate_source + scalar_source_col0,
                                (size_t) lanes[source].scalar_stride*
                                    sizeof(float),
                                scalar_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold gate low");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                (char *) state.gate.get() +
                                    scalar_half_width,
                                scalar_panel_pitch,
                                gate_source + scalar_source_col1,
                                (size_t) lanes[source].scalar_stride*
                                    sizeof(float),
                                scalar_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold gate high");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                state.beta.get(), scalar_panel_pitch,
                                beta_source + scalar_source_col0,
                                (size_t) lanes[source].scalar_stride*
                                    sizeof(float),
                                scalar_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold beta low");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                (char *) state.beta.get() +
                                    scalar_half_width,
                                scalar_panel_pitch,
                                beta_source + scalar_source_col1,
                                (size_t) lanes[source].scalar_stride*
                                    sizeof(float),
                                scalar_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "gather HeadFold beta high");

                    const dim3 grid(
                            AW_HEADFOLD_VALUE_HEADS, 1,
                            AW_GDN_HEAD_DIM/4);
                    const dim3 block(32, 4, 1);
                    aw_headfold_gdn_segment<<<grid, block, 0, stream>>>(
                            (const float *) state.q.get(),
                            (const float *) state.k.get(),
                            (const float *) state.v.get(),
                            (const float *) state.gate.get(),
                            (const float *) state.beta.get(),
                            (float *) state.state.get(),
                            (float *) state.output.get(),
                            segment_tokens);

                    float * output =
                            lanes[source].output +
                            token*AW_GDN_VALUE_HEADS*
                            AW_GDN_HEAD_DIM;
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                output + value_source_col0,
                                value_output_pitch,
                                state.output.get(),
                                value_panel_pitch,
                                value_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "scatter HeadFold output low");
                    aw_cuda_throw(cudaMemcpy2DAsync(
                                output + value_source_col1,
                                value_output_pitch,
                                (const char *) state.output.get() +
                                    value_half_width,
                                value_panel_pitch,
                                value_half_width, segment_tokens,
                                cudaMemcpyDefault, stream),
                            "scatter HeadFold output high");
                }
            }

            for (int destination = 0;
                    destination < AW_GPU_COUNT; ++destination) {
                aw_cuda_throw(cudaMemcpyAsync(
                            lanes[destination].state_output +
                                (size_t) owner*
                                (AW_HEADFOLD_VALUE_HEADS/2)*
                                AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM,
                            state.state.get(),
                            state_half_bytes,
                            cudaMemcpyDefault, stream),
                        "scatter HeadFold state low");
                aw_cuda_throw(cudaMemcpyAsync(
                            lanes[destination].state_output +
                                (size_t) (AW_GDN_QK_HEADS +
                                    owner*
                                    (AW_HEADFOLD_VALUE_HEADS/2))*
                                AW_GDN_HEAD_DIM*AW_GDN_HEAD_DIM,
                            (const char *) state.state.get() +
                                state_half_bytes,
                            state_half_bytes,
                            cudaMemcpyDefault, stream),
                        "scatter HeadFold state high");
            }
            aw_cuda_throw(cudaGetLastError(),
                    "launch HeadFold GDN");
            aw_cuda_throw(cudaEventRecord(
                        state.output_done, stream),
                    "record HeadFold owner done");
        }

        for (int destination = 0;
                destination < AW_GPU_COUNT; ++destination) {
            aw_cuda_throw(cudaSetDevice(destination),
                    "set HeadFold destination device");
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaStreamWaitEvent(
                            streams[destination],
                            aw_headfold_gdn_states[owner].output_done,
                            0),
                        "wait for HeadFold owner output");
            }
        }

        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) {
            const size_t scratch =
                    aw_headfold_gdn_states[0].q.size() +
                    aw_headfold_gdn_states[0].k.size() +
                    aw_headfold_gdn_states[0].v.size() +
                    aw_headfold_gdn_states[0].gate.size() +
                    aw_headfold_gdn_states[0].beta.size() +
                    aw_headfold_gdn_states[0].output.size() +
                    aw_headfold_gdn_states[0].state.size();
            fprintf(stderr,
                    "AffinityWave: HeadFold GDN active tokens=%d segment=%d scratch=%.2f MiB/GPU\n",
                    tokens, segment_tokens,
                    scratch/(1024.0*1024.0));
        }
        return 0;
    } catch (const std::exception & exception) {
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}

int ggml_cuda_affinity_wave_pairfold_gdn(
        void * const * streams_ptr,
        const int32_t * devices,
        const int32_t * groups,
        const ggml_cuda_aw_headfold_gdn_lane * lanes,
        int32_t n_lanes,
        char * error,
        size_t error_capacity) {
    try {
        if (streams_ptr == nullptr || devices == nullptr ||
                groups == nullptr || lanes == nullptr ||
                n_lanes != 2 ||
                devices[0] < 0 ||
                devices[0] >= AW_GPU_COUNT ||
                devices[1] < 0 ||
                devices[1] >= AW_GPU_COUNT ||
                devices[0] == devices[1] ||
                devices[0]/2 != devices[1]/2) {
            throw std::runtime_error(
                    "invalid PairFold GDN request");
        }
        std::array<bool, AW_GPU_COUNT> group_seen{};
        for (int i = 0; i < AW_GPU_COUNT; ++i) {
            if (groups[i] < 0 || groups[i] >= AW_GPU_COUNT ||
                    group_seen[groups[i]]) {
                throw std::runtime_error(
                        "invalid PairFold GDN head groups");
            }
            group_seen[groups[i]] = true;
        }
        const int tokens = lanes[0].tokens;
        if (tokens < AW_HEADFOLD_PARTS ||
                tokens % AW_HEADFOLD_PARTS != 0) {
            throw std::runtime_error(
                    "PairFold GDN tokens must be divisible by four");
        }
        for (int lane = 0; lane < n_lanes; ++lane) {
            if (lanes[lane].tokens != tokens ||
                    lanes[lane].qk_stride <
                        AW_GDN_QK_HEADS*AW_GDN_HEAD_DIM ||
                    lanes[lane].v_stride <
                        AW_GDN_VALUE_HEADS*AW_GDN_HEAD_DIM ||
                    lanes[lane].scalar_stride <
                        AW_GDN_VALUE_HEADS ||
                    lanes[lane].q == nullptr ||
                    lanes[lane].k == nullptr ||
                    lanes[lane].v == nullptr ||
                    lanes[lane].gate == nullptr ||
                    lanes[lane].beta == nullptr ||
                    lanes[lane].state == nullptr ||
                    lanes[lane].output == nullptr ||
                    lanes[lane].state_output == nullptr) {
                throw std::runtime_error(
                        "invalid PairFold GDN lane descriptor");
            }
        }

        const int segment_tokens = tokens/AW_HEADFOLD_PARTS;
        std::array<cudaStream_t, 2> streams;
        for (int lane = 0; lane < n_lanes; ++lane) {
            const int device = devices[lane];
            streams[lane] =
                    (cudaStream_t) streams_ptr[device];
            aw_cuda_throw(cudaSetDevice(device),
                    "set PairFold GDN source device");
            aw_headfold_gdn_states[device].ensure(
                    device, segment_tokens);
            aw_cuda_throw(cudaEventRecord(
                        aw_headfold_gdn_states[device].
                            input_ready,
                        streams[lane]),
                    "record PairFold GDN input ready");
        }

        constexpr size_t qk_panel_pitch =
                (size_t) AW_HEADFOLD_QK_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t value_panel_pitch =
                (size_t) AW_HEADFOLD_VALUE_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t value_output_pitch =
                (size_t) AW_GDN_VALUE_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t scalar_panel_pitch =
                (size_t) AW_HEADFOLD_VALUE_HEADS*
                sizeof(float);
        constexpr size_t qk_width =
                (size_t) AW_HEADFOLD_QK_HEADS*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t value_half_width =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t scalar_half_width =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                sizeof(float);
        constexpr size_t state_head_bytes =
                (size_t) AW_GDN_HEAD_DIM*
                AW_GDN_HEAD_DIM*sizeof(float);
        constexpr size_t state_half_bytes =
                (size_t) (AW_HEADFOLD_VALUE_HEADS/2)*
                state_head_bytes;

        for (int owner_slot = 0; owner_slot < 2;
                ++owner_slot) {
            const int owner = devices[owner_slot];
            aw_cuda_throw(cudaSetDevice(owner),
                    "set PairFold GDN owner device");
            aw_headfold_gdn_state & state =
                    aw_headfold_gdn_states[owner];
            cudaStream_t stream = streams[owner_slot];
            for (int source = 0; source < n_lanes;
                    ++source) {
                aw_cuda_throw(cudaStreamWaitEvent(
                            stream,
                            aw_headfold_gdn_states[
                                devices[source]].input_ready,
                            0),
                        "wait for PairFold GDN source");
            }

            for (int group_slot = 0; group_slot < 2;
                    ++group_slot) {
                const int group =
                        groups[owner_slot*2 + group_slot];
                const size_t qk_source_col =
                        (size_t) group*
                        AW_HEADFOLD_QK_HEADS*
                        AW_GDN_HEAD_DIM;
                const size_t value_source_col0 =
                        qk_source_col;
                const size_t value_source_col1 =
                        (size_t) (AW_GDN_QK_HEADS +
                            group*AW_HEADFOLD_QK_HEADS)*
                        AW_GDN_HEAD_DIM;
                const size_t scalar_source_col0 =
                        (size_t) group*
                        AW_HEADFOLD_QK_HEADS;
                const size_t scalar_source_col1 =
                        (size_t) AW_GDN_QK_HEADS +
                        group*AW_HEADFOLD_QK_HEADS;

                aw_cuda_throw(cudaMemcpyAsync(
                            state.state.get(),
                            lanes[0].state +
                                (size_t) group*
                                (AW_HEADFOLD_VALUE_HEADS/2)*
                                AW_GDN_HEAD_DIM*
                                AW_GDN_HEAD_DIM,
                            state_half_bytes,
                            cudaMemcpyDefault, stream),
                        "copy PairFold GDN initial state low");
                aw_cuda_throw(cudaMemcpyAsync(
                            (char *) state.state.get() +
                                state_half_bytes,
                            lanes[0].state +
                                (size_t) (AW_GDN_QK_HEADS +
                                    group*
                                    (AW_HEADFOLD_VALUE_HEADS/2))*
                                AW_GDN_HEAD_DIM*
                                AW_GDN_HEAD_DIM,
                            state_half_bytes,
                            cudaMemcpyDefault, stream),
                        "copy PairFold GDN initial state high");

                for (int source = 0; source < n_lanes;
                        ++source) {
                    for (int part = 0;
                            part < AW_HEADFOLD_PARTS;
                            ++part) {
                        const size_t token =
                                (size_t) part*
                                segment_tokens;
                        const float * q_source =
                                lanes[source].q +
                                token*
                                    lanes[source].qk_stride +
                                qk_source_col;
                        const float * k_source =
                                lanes[source].k +
                                token*
                                    lanes[source].qk_stride +
                                qk_source_col;
                        const float * v_source =
                                lanes[source].v +
                                token*
                                    lanes[source].v_stride;
                        const float * gate_source =
                                lanes[source].gate +
                                token*
                                    lanes[source].scalar_stride;
                        const float * beta_source =
                                lanes[source].beta +
                                token*
                                    lanes[source].scalar_stride;

                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    state.q.get(),
                                    qk_panel_pitch,
                                    q_source,
                                    (size_t)
                                        lanes[source].qk_stride*
                                        sizeof(float),
                                    qk_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN Q");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    state.k.get(),
                                    qk_panel_pitch,
                                    k_source,
                                    (size_t)
                                        lanes[source].qk_stride*
                                        sizeof(float),
                                    qk_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN K");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    state.v.get(),
                                    value_panel_pitch,
                                    v_source +
                                        value_source_col0,
                                    (size_t)
                                        lanes[source].v_stride*
                                        sizeof(float),
                                    value_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN V low");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    (char *) state.v.get() +
                                        value_half_width,
                                    value_panel_pitch,
                                    v_source +
                                        value_source_col1,
                                    (size_t)
                                        lanes[source].v_stride*
                                        sizeof(float),
                                    value_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN V high");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    state.gate.get(),
                                    scalar_panel_pitch,
                                    gate_source +
                                        scalar_source_col0,
                                    (size_t)
                                        lanes[source].
                                            scalar_stride*
                                        sizeof(float),
                                    scalar_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN gate low");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    (char *) state.gate.get() +
                                        scalar_half_width,
                                    scalar_panel_pitch,
                                    gate_source +
                                        scalar_source_col1,
                                    (size_t)
                                        lanes[source].
                                            scalar_stride*
                                        sizeof(float),
                                    scalar_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN gate high");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    state.beta.get(),
                                    scalar_panel_pitch,
                                    beta_source +
                                        scalar_source_col0,
                                    (size_t)
                                        lanes[source].
                                            scalar_stride*
                                        sizeof(float),
                                    scalar_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN beta low");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    (char *) state.beta.get() +
                                        scalar_half_width,
                                    scalar_panel_pitch,
                                    beta_source +
                                        scalar_source_col1,
                                    (size_t)
                                        lanes[source].
                                            scalar_stride*
                                        sizeof(float),
                                    scalar_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "gather PairFold GDN beta high");

                        const dim3 grid(
                                AW_HEADFOLD_VALUE_HEADS,
                                1, AW_GDN_HEAD_DIM/4);
                        const dim3 block(32, 4, 1);
                        aw_headfold_gdn_segment<<<
                                grid, block, 0, stream>>>(
                                (const float *) state.q.get(),
                                (const float *) state.k.get(),
                                (const float *) state.v.get(),
                                (const float *)
                                    state.gate.get(),
                                (const float *)
                                    state.beta.get(),
                                (float *) state.state.get(),
                                (float *) state.output.get(),
                                segment_tokens);

                        float * output =
                                lanes[source].output +
                                token*
                                AW_GDN_VALUE_HEADS*
                                AW_GDN_HEAD_DIM;
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    output +
                                        value_source_col0,
                                    value_output_pitch,
                                    state.output.get(),
                                    value_panel_pitch,
                                    value_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "scatter PairFold GDN output low");
                        aw_cuda_throw(cudaMemcpy2DAsync(
                                    output +
                                        value_source_col1,
                                    value_output_pitch,
                                    (const char *)
                                        state.output.get() +
                                        value_half_width,
                                    value_panel_pitch,
                                    value_half_width,
                                    segment_tokens,
                                    cudaMemcpyDefault, stream),
                                "scatter PairFold GDN output high");
                    }
                }

                for (int destination = 0;
                        destination < n_lanes;
                        ++destination) {
                    aw_cuda_throw(cudaMemcpyAsync(
                                lanes[destination].
                                    state_output +
                                    (size_t) group*
                                    (AW_HEADFOLD_VALUE_HEADS/2)*
                                    AW_GDN_HEAD_DIM*
                                    AW_GDN_HEAD_DIM,
                                state.state.get(),
                                state_half_bytes,
                                cudaMemcpyDefault, stream),
                            "scatter PairFold GDN state low");
                    aw_cuda_throw(cudaMemcpyAsync(
                                lanes[destination].
                                    state_output +
                                    (size_t)
                                    (AW_GDN_QK_HEADS +
                                     group*
                                     (AW_HEADFOLD_VALUE_HEADS/2))*
                                    AW_GDN_HEAD_DIM*
                                    AW_GDN_HEAD_DIM,
                                (const char *)
                                    state.state.get() +
                                    state_half_bytes,
                                state_half_bytes,
                                cudaMemcpyDefault, stream),
                            "scatter PairFold GDN state high");
                }
            }
            aw_cuda_throw(cudaGetLastError(),
                    "launch PairFold GDN");
            aw_cuda_throw(cudaEventRecord(
                        state.output_done, stream),
                    "record PairFold GDN owner done");
        }

        for (int destination = 0;
                destination < n_lanes; ++destination) {
            const int device = devices[destination];
            aw_cuda_throw(cudaSetDevice(device),
                    "set PairFold GDN destination device");
            for (int owner = 0; owner < 2; ++owner) {
                aw_cuda_throw(cudaStreamWaitEvent(
                            streams[destination],
                            aw_headfold_gdn_states[
                                devices[owner]].output_done,
                            0),
                        "wait for PairFold GDN owner output");
            }
        }

        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) {
            const size_t scratch =
                    aw_headfold_gdn_states[devices[0]].q.size() +
                    aw_headfold_gdn_states[devices[0]].k.size() +
                    aw_headfold_gdn_states[devices[0]].v.size() +
                    aw_headfold_gdn_states[devices[0]].gate.size() +
                    aw_headfold_gdn_states[devices[0]].beta.size() +
                    aw_headfold_gdn_states[devices[0]].output.size() +
                    aw_headfold_gdn_states[devices[0]].state.size();
            fprintf(stderr,
                    "AffinityWave: PairFold GDN active tokens=%d segment=%d scratch=%.2f MiB/GPU\n",
                    tokens, segment_tokens,
                    scratch/(1024.0*1024.0));
        }
        return 0;
    } catch (const std::exception & exception) {
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}

int ggml_cuda_affinity_wave_r44_prefetch(
        int32_t layer,
        char * error,
        size_t error_capacity) {
    try {
        if (!aw_r44_enabled()) {
            return 0;
        }
        if (layer < 0 || layer >= AW_LAYERS) {
            throw std::runtime_error("invalid R44 prefetch layer");
        }
        const aw_r44_placement & placement = aw_r44_get_placement();
        constexpr size_t expert_bytes =
                (size_t) AW_EXPERT_FF*(AW_EMBD/AW_KSTAGE)*
                AW_Q8_BLOCK_BYTES;
        constexpr size_t cache_bytes =
                (size_t) AW_R44_BUDGET*3*expert_bytes;
        std::array<std::array<const char *, 3>, AW_GPU_COUNT>
                source_weights{};
        {
            std::lock_guard<std::mutex> lock(aw_layout_mutex);
            for (int source = 0; source < AW_GPU_COUNT; ++source) {
                for (int projection = 0; projection < 3;
                        ++projection) {
                    const auto it = std::find_if(
                            aw_layout_entries.begin(),
                            aw_layout_entries.end(),
                            [&](const aw_layout_entry & entry) {
                        return entry.device == source &&
                                entry.layer == layer &&
                                entry.projection == projection &&
                                aw_t64_tensors.count(entry.data) != 0;
                    });
                    if (it == aw_layout_entries.end() ||
                            it->bytes < AW_PRIMARY_PER_GPU*
                                expert_bytes) {
                        throw std::runtime_error(
                                "R44 weight catalog is incomplete");
                    }
                    source_weights[source][projection] =
                            (const char *) it->data;
                }
            }
        }
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            aw_r44_device_state & state = aw_r44_states[device];
            aw_r44_slot & cache =
                    aw_r44_prepare_slot(device, layer, cache_bytes);
            for (int projection = 0; projection < 3;
                    ++projection) {
                for (int slot = 0; slot < AW_R44_BUDGET; ++slot) {
                    const int expert =
                            placement.cache[layer][device][slot];
                    const int source = expert/AW_PRIMARY_PER_GPU;
                    const int local_expert =
                            expert % AW_PRIMARY_PER_GPU;
                    const char * src =
                            source_weights[source][projection] +
                            (size_t) local_expert*expert_bytes;
                    char * dst = (char *) cache.weights.get() +
                            ((size_t) projection*AW_R44_BUDGET +
                             slot)*expert_bytes;
                    if (source == device) {
                        aw_cuda_throw(cudaMemcpyAsync(
                                    dst, src, expert_bytes,
                                    cudaMemcpyDeviceToDevice,
                                    state.stream),
                                "copy local R44 expert");
                    } else {
                        aw_cuda_throw(cudaMemcpyPeerAsync(
                                    dst, device, src, source,
                                    expert_bytes, state.stream),
                                "copy peer R44 expert");
                    }
                }
            }
            aw_cuda_throw(cudaEventRecord(cache.ready, state.stream),
                    "record R44 cache ready");
            if (strcmp(aw_r44_mode(), "prefetch") == 0) {
                aw_cuda_throw(cudaEventRecord(cache.free, state.stream),
                        "release diagnostic R44 cache");
            }
        }
        static bool reported = false;
        if (!reported) {
            reported = true;
            fprintf(stderr,
                    "AffinityWave: R44 streaming enabled slots=%d bytes/device=%zu mode=%s\n",
                    aw_r44_slot_count(),
                    (size_t) aw_r44_slot_count()*cache_bytes,
                    aw_r44_mode());
        }
        return 0;
    } catch (const std::exception & exception) {
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}

static void aw_live_diagonal_panel(
        const std::array<cudaStream_t, AW_GPU_COUNT> & streams,
        const ggml_cuda_aw_live_cell * cells,
        int n_cells,
        const aw_live_weight_table & weight_ptrs,
        const aw_group_placement & group_placement,
        int live_call,
        const char * debug_sync,
        bool nccl_order_sum,
        int panel_n) {
    const int down_panels = AW_EMBD/panel_n;
    const bool p100_exact = aw_p100_exact_enabled();
    const int exact_sum_width =
            aw_p100_exact_sum_width();
    if (p100_exact && exact_sum_width != 2 &&
            exact_sum_width != 4) {
        throw std::runtime_error(
                "GGML_CUDA_AW_P100_EXACT_SUM_WIDTH must be 2 or 4");
    }
    auto sync_live = [&](cudaStream_t stream,
            const char * operation) {
        if (aw_env_on(debug_sync) &&
                (strcmp(debug_sync, "1") == 0 ||
                 strstr(operation, debug_sync) != nullptr)) {
            aw_cuda_throw(cudaStreamSynchronize(stream), operation);
        }
    };

    for (int cell = 0; cell < n_cells; ++cell) {
        const int home = cells[cell].home_device;
        aw_cuda_throw(cudaSetDevice(home),
                "set diagonal source device");
        aw_cuda_throw(cudaEventRecord(
                    aw_live_states[cell][home].source_ready,
                    streams[home]),
                "record diagonal source ready");
    }

    for (int group = 0; group < n_cells; ++group) {
        const ggml_cuda_aw_live_cell & cell = cells[group];
        const int home = cell.home_device;
        const int input_slot = group;
        const int tokens = cell.tokens;
        const size_t input_bytes =
                (size_t) tokens*AW_EMBD*sizeof(float);
        const size_t route_row_bytes = 8*sizeof(int32_t);
        const size_t route_bytes =
                (size_t) tokens*route_row_bytes;

        aw_cuda_throw(cudaSetDevice(home),
                "set diagonal input device");
        aw_live_state & home_state =
                aw_live_states[group][home];
        cudaStream_t copy_stream = aw_copy_streams.get(home);
        aw_cuda_throw(cudaStreamWaitEvent(
                    copy_stream, home_state.source_ready, 0),
                "wait for diagonal source");
        for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
            aw_cuda_throw(cudaStreamWaitEvent(
                        copy_stream,
                        aw_live_states[group][owner].scratch_free,
                        0),
                    "wait for diagonal metadata");
            aw_cuda_throw(cudaStreamWaitEvent(
                        copy_stream,
                        aw_diagonal_scratch_states[owner].
                            input_free[input_slot],
                        0),
                    "wait for diagonal input slot");
        }

        aw_diagonal_scratch & home_scratch =
                aw_diagonal_scratch_states[home];
        aw_cuda_throw(cudaMemcpyAsync(
                    home_scratch.input[input_slot].get(),
                    cell.input,
                    input_bytes, cudaMemcpyDeviceToDevice,
                    copy_stream),
                "copy diagonal local input");
        aw_cuda_throw(cudaMemcpy2DAsync(
                    home_state.ids.get(), route_row_bytes,
                    cell.ids, cell.ids_stride,
                    route_row_bytes, tokens,
                    cudaMemcpyDeviceToDevice, copy_stream),
                "copy diagonal local ids");
        aw_cuda_throw(cudaMemcpy2DAsync(
                    home_state.weights.get(), route_row_bytes,
                    cell.weights, cell.weights_stride,
                    route_row_bytes, tokens,
                    cudaMemcpyDeviceToDevice, copy_stream),
                "copy diagonal local weights");

        for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
            if (owner == home) {
                continue;
            }
            aw_cuda_throw(cudaMemcpyPeerAsync(
                        aw_diagonal_scratch_states[owner].
                            input[input_slot].get(),
                        owner,
                        home_scratch.input[input_slot].get(),
                        home,
                        input_bytes, copy_stream),
                    "copy diagonal peer input");
            aw_cuda_throw(cudaMemcpyPeerAsync(
                        aw_live_states[group][owner].ids.get(),
                        owner, home_state.ids.get(), home,
                        route_bytes, copy_stream),
                    "copy diagonal peer ids");
            aw_cuda_throw(cudaMemcpyPeerAsync(
                        aw_live_states[group][owner].weights.get(),
                        owner, home_state.weights.get(), home,
                        route_bytes, copy_stream),
                    "copy diagonal peer weights");
        }
        aw_cuda_throw(cudaEventRecord(
                    home_state.input_ready, copy_stream),
                "record diagonal input ready");
        sync_live(copy_stream,
                "synchronize diagonal input copies");
    }

    for (int group = 0; group < n_cells; ++group) {
        const ggml_cuda_aw_live_cell & cell = cells[group];
        const int home = cell.home_device;
        const int input_slot = group;
        const int tokens = cell.tokens;
        const int total_slots = tokens*8;
        const int descriptors = AW_PRIMARY_PER_GPU;

        for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
            aw_cuda_throw(cudaSetDevice(owner),
                    "set diagonal compute device");
            aw_live_state & state = aw_live_states[group][owner];
            aw_diagonal_scratch & scratch =
                    aw_diagonal_scratch_states[owner];
            cudaStream_t stream = streams[owner];
            aw_cuda_throw(cudaStreamWaitEvent(
                        stream,
                        aw_live_states[group][home].input_ready, 0),
                    "wait for diagonal input");
            aw_cuda_throw(cudaMemcpyAsync(
                        state.weight_ptrs.get(),
                        weight_ptrs[group][owner].data(),
                        3*sizeof(const char *),
                        cudaMemcpyHostToDevice, stream),
                    "copy diagonal weight pointers");
            aw_cuda_throw(cudaMemsetAsync(
                        state.counts.get(), 0,
                        descriptors*sizeof(int32_t), stream),
                    "clear diagonal route counts");
            aw_live_count_routes<<<
                    aw_grid(total_slots), 256, 0, stream>>>(
                        (const int32_t *) state.ids.get(),
                        (int32_t *) state.counts.get(),
                        total_slots, owner, tokens);
            aw_live_build_plan<<<
                    1, p100_exact ? 256 : 1,
                    0, stream>>>(
                    (const int32_t *) state.counts.get(),
                    (int32_t *) state.offsets.get(),
                    (int32_t *) state.cursors.get(),
                    (int32_t *) state.tile_counts.get(),
                    (aw_tile_desc *) state.tiles_m64.get(),
                    (aw_tile_desc *) state.tiles_m32.get(),
                    (aw_tile_desc *) state.tiles_m16.get(),
                    (aw_tile_desc *) state.tiles_warp.get(),
                    (aw_tile_desc *) state.tiles_m8.get(),
                    (aw_tile_desc *) state.tiles_m4.get(),
                    (aw_work_desc *) state.gate_desc.get(),
                    (aw_work_desc *) state.up_desc.get(),
                    (aw_work_desc *) state.down_desc.get(),
                    (const char * const *) state.weight_ptrs.get(),
                    scratch.input[input_slot].get(),
                    (int32_t *) state.route_input.get(),
                    (float *) scratch.middle.get(),
                    (float *) scratch.route_output.get(),
                    (float *) scratch.middle.get(),
                    (float *) scratch.route_output.get(),
                    AW_EXPERT_FF,
                    panel_n,
                    descriptors);
            if (down_panels > 1 || !p100_exact) {
                const size_t down_weight_panel_bytes =
                        (size_t) (panel_n/AW_T64_ROWS)*
                        (AW_EXPERT_FF/AW_KSTAGE)*
                        AW_T64_STAGE_BYTES;
                aw_live_build_panel_descs<<<
                        aw_grid((size_t) descriptors*
                            down_panels),
                        256, 0, stream>>>(
                            (const aw_work_desc *)
                                state.down_desc.get(),
                            (aw_work_desc *)
                                state.down_panel_desc.get(),
                            descriptors, down_panels,
                            down_weight_panel_bytes);
            }
            aw_live_fill_routes_deterministic<<<
                    descriptors, 256, 0, stream>>>(
                        (const int32_t *) state.ids.get(),
                        (const int32_t *) state.offsets.get(),
                        (int32_t *) state.route_input.get(),
                        (int32_t *) state.token_routes.get(),
                        descriptors, owner, tokens);

            const int sm_count = aw_sm_counts[owner];
            if (sm_count <= 0) {
                throw std::runtime_error(
                        "diagonal owner SM count was not initialized");
            }
            const int persistent_blocks = sm_count*2;
            const int pair_blocks = sm_count*3;
            aw_live_launch_diagonal_projection(
                    state, state.gate_desc, AW_EXPERT_FF, AW_EMBD,
                    stream, persistent_blocks, pair_blocks);
            sync_live(stream, "synchronize diagonal gate");
            aw_live_launch_diagonal_projection(
                    state, state.up_desc, AW_EXPERT_FF, AW_EMBD,
                    stream, persistent_blocks, pair_blocks);
            aw_cuda_throw(cudaEventRecord(
                        scratch.input_free[input_slot], stream),
                    "record diagonal input free");
            sync_live(stream, "synchronize diagonal up");
            aw_live_swiglu<<<
                    aw_grid((size_t) total_slots*AW_EXPERT_FF),
                    256, 0, stream>>>(
                        (const float *) scratch.middle.get(),
                        (const float *) scratch.route_output.get(),
                        (float *) scratch.middle.get(),
                        (const int32_t *) state.offsets.get() +
                            descriptors);
            aw_cuda_throw(cudaGetLastError(),
                    "launch diagonal gate up");
        }

        for (int panel = 0;
                panel < down_panels; ++panel) {
            const int slot = panel % AW_DIAGONAL_RING_SLOTS;
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaSetDevice(owner),
                        "set diagonal down device");
                aw_live_state & state =
                        aw_live_states[group][owner];
                aw_diagonal_scratch & scratch =
                        aw_diagonal_scratch_states[owner];
                cudaStream_t stream = streams[owner];
                aw_cuda_throw(cudaStreamWaitEvent(
                            stream, scratch.partial_free[slot], 0),
                        "wait for diagonal partial slot");
                const int sm_count = aw_sm_counts[owner];
                const aw_device_buffer & down_desc =
                        down_panels == 1 ?
                        state.down_desc :
                        state.down_panel_desc;
                const size_t desc_offset =
                        down_panels == 1 ?
                        0 : (size_t) panel*descriptors;
                aw_live_launch_diagonal_projection(
                        state, down_desc,
                        panel_n, AW_EXPERT_FF,
                        stream, sm_count*2, sm_count*3,
                        desc_offset);
                if (p100_exact) {
                    const int blocks = std::min(tokens, 224);
                    aw_live_owner_reduce_panel_w4<<<
                            blocks, 128, 0, stream>>>(
                                (const float *)
                                    scratch.route_output.get(),
                                (const int32_t *)
                                    state.ids.get(),
                                (const int32_t *)
                                    state.token_routes.get(),
                                (const float *)
                                    state.weights.get(),
                                (uint16_t *)
                                    scratch.partial[slot].get(),
                                tokens, owner,
                                panel_n,
                                panel_n, 0);
                } else {
                    aw_live_owner_reduce_panel<true><<<
                            aw_grid((size_t) tokens*
                                panel_n),
                            256, 0, stream>>>(
                                (const float *)
                                    scratch.route_output.get(),
                                (const int32_t *) state.ids.get(),
                                (const int32_t *)
                                    state.token_routes.get(),
                                (const float *) state.weights.get(),
                                scratch.partial[slot].get(),
                                tokens, owner,
                                panel_n,
                                panel_n, 0);
                }
                aw_cuda_throw(cudaGetLastError(),
                        "launch diagonal down panel");
                const char * partial_dump =
                        getenv("GGML_CUDA_AW_SERVICE_DUMP");
                const char * partial_dump_layer_env =
                        getenv(
                            "GGML_CUDA_AW_SERVICE_DUMP_LAYER");
                const int partial_dump_layer =
                        partial_dump_layer_env != nullptr ?
                        atoi(partial_dump_layer_env) : 0;
                if (partial_dump != nullptr &&
                        partial_dump[0] != '\0' &&
                        (partial_dump_layer < 0 ||
                         cell.layer == partial_dump_layer)) {
                    aw_cuda_throw(cudaStreamSynchronize(stream),
                            "synchronize diagonal partial dump");
                    int32_t route_rows = 0;
                    aw_cuda_throw(cudaMemcpy(
                                &route_rows,
                                (const int32_t *)
                                    state.offsets.get() +
                                    descriptors,
                                sizeof(route_rows),
                                cudaMemcpyDeviceToHost),
                            "copy diagonal route-row count");
                    std::vector<float> route_values(
                            (size_t) route_rows*panel_n);
                    aw_cuda_throw(cudaMemcpy(
                                route_values.data(),
                                scratch.route_output.get(),
                                route_values.size()*sizeof(float),
                                cudaMemcpyDeviceToHost),
                            "copy diagonal route-output dump");
                    char route_kind[40];
                    snprintf(route_kind, sizeof(route_kind),
                            "routes-p%02d-f32", panel);
                    const int logical_owner =
                            group_placement.logical_for_physical[
                                cell.layer][owner];
                    aw_live_dump(partial_dump, route_kind,
                            cell.layer, home, logical_owner,
                            route_values.data(),
                            route_values.size(), sizeof(float));
                    std::vector<int32_t> route_tokens(route_rows);
                    aw_cuda_throw(cudaMemcpy(
                                route_tokens.data(),
                                state.route_input.get(),
                                route_tokens.size()*sizeof(int32_t),
                                cudaMemcpyDeviceToHost),
                            "copy diagonal route-input dump");
                    aw_live_dump(partial_dump, "route-input-i32",
                            cell.layer, home, logical_owner,
                            route_tokens.data(),
                            route_tokens.size(), sizeof(int32_t));
                    std::vector<uint16_t> values(
                            (size_t) tokens*panel_n);
                    aw_cuda_throw(cudaMemcpy(
                                values.data(),
                                scratch.partial[slot].get(),
                                values.size()*sizeof(uint16_t),
                                cudaMemcpyDeviceToHost),
                            "copy diagonal partial dump");
                    char kind[40];
                    snprintf(kind, sizeof(kind),
                            "partial-p%02d-bf16", panel);
                    aw_live_dump(partial_dump, kind,
                            cell.layer, home, logical_owner,
                            values.data(), values.size(),
                            sizeof(uint16_t));
                }
                aw_cuda_throw(cudaEventRecord(
                            scratch.partial_ready[slot], stream),
                        "record diagonal partial ready");
                if (panel + 1 == down_panels) {
                    aw_cuda_throw(cudaEventRecord(
                                state.compute_done, stream),
                            "record diagonal compute done");
                    aw_cuda_throw(cudaEventRecord(
                                state.scratch_free, stream),
                            "record diagonal metadata free");
                }
                sync_live(stream,
                        "synchronize diagonal down panel");
            }

            aw_diagonal_scratch & home_scratch =
                    aw_diagonal_scratch_states[home];
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaSetDevice(owner),
                        "set diagonal owner-copy device");
                aw_diagonal_scratch & scratch =
                        aw_diagonal_scratch_states[owner];
                cudaStream_t copy_stream = aw_copy_streams.get(owner);
                aw_cuda_throw(cudaStreamWaitEvent(
                            copy_stream,
                            scratch.partial_ready[slot], 0),
                        "wait for diagonal owner partial");
                aw_cuda_throw(cudaStreamWaitEvent(
                            copy_stream,
                            home_scratch.recv_free[slot], 0),
                        "wait for diagonal receive slot");
                const int logical_owner =
                        group_placement.logical_for_physical[
                            cell.layer][owner];
                uint16_t * dst =
                        (uint16_t *) home_scratch.recv[slot].get() +
                        (size_t) logical_owner*tokens*
                            panel_n;
                const size_t bytes =
                        (size_t) tokens*panel_n*
                        sizeof(uint16_t);
                if (owner == home) {
                    aw_cuda_throw(cudaMemcpyAsync(
                                dst, scratch.partial[slot].get(),
                                bytes, cudaMemcpyDeviceToDevice,
                                copy_stream),
                            "copy diagonal local partial");
                } else {
                    aw_cuda_throw(cudaMemcpyPeerAsync(
                                dst, home,
                                scratch.partial[slot].get(), owner,
                                bytes, copy_stream),
                            "copy diagonal peer partial");
                }
                aw_cuda_throw(cudaEventRecord(
                            scratch.copied[slot], copy_stream),
                        "record diagonal owner copy");
                aw_cuda_throw(cudaEventRecord(
                            scratch.partial_free[slot],
                            copy_stream),
                        "record diagonal partial free");
            }

            aw_cuda_throw(cudaSetDevice(home),
                    "set diagonal publish device");
            cudaStream_t publish_stream =
                    home_scratch.publish_stream;
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaStreamWaitEvent(
                            publish_stream,
                            aw_diagonal_scratch_states[owner].
                                copied[slot],
                            0),
                        "wait for diagonal owner copy");
            }
            if (p100_exact && exact_sum_width == 4) {
                const int blocks = std::min(tokens, 224);
                if (nccl_order_sum) {
                    aw_live_sum_owners_panel_p100<true, 4><<<
                            blocks, 128, 0, publish_stream>>>(
                                (const uint16_t *)
                                    home_scratch.recv[slot].get(),
                                cell.output, tokens,
                                panel_n,
                                panel*panel_n);
                } else {
                    aw_live_sum_owners_panel_p100<false, 4><<<
                            blocks, 128, 0, publish_stream>>>(
                                (const uint16_t *)
                                    home_scratch.recv[slot].get(),
                                cell.output, tokens,
                                panel_n,
                                panel*panel_n);
                }
            } else if (p100_exact) {
                const int blocks = std::min(tokens, 448);
                if (nccl_order_sum) {
                    aw_live_sum_owners_panel_p100<true, 2><<<
                            blocks, 256, 0, publish_stream>>>(
                                (const uint16_t *)
                                    home_scratch.recv[slot].get(),
                                cell.output, tokens,
                                panel_n,
                                panel*panel_n);
                } else {
                    aw_live_sum_owners_panel_p100<false, 2><<<
                            blocks, 256, 0, publish_stream>>>(
                                (const uint16_t *)
                                    home_scratch.recv[slot].get(),
                                cell.output, tokens,
                                panel_n,
                                panel*panel_n);
                }
            } else if (nccl_order_sum) {
                aw_live_sum_owners_panel<true><<<
                        aw_grid((size_t) tokens*
                            panel_n),
                        256, 0, publish_stream>>>(
                            (const uint16_t *)
                                home_scratch.recv[slot].get(),
                            cell.output, tokens,
                            panel_n,
                            panel*panel_n);
            } else {
                aw_live_sum_owners_panel<<<
                        aw_grid((size_t) tokens*
                            panel_n),
                        256, 0, publish_stream>>>(
                            (const uint16_t *)
                                home_scratch.recv[slot].get(),
                            cell.output, tokens,
                            panel_n,
                            panel*panel_n);
            }
            aw_cuda_throw(cudaGetLastError(),
                    "launch diagonal owner sum");
            aw_cuda_throw(cudaEventRecord(
                        home_scratch.recv_free[slot],
                        publish_stream),
                    "record diagonal receive free");
        }

        aw_live_state & home_state =
                aw_live_states[group][home];
        aw_cuda_throw(cudaSetDevice(home),
                "set diagonal completion device");
        aw_cuda_throw(cudaEventRecord(
                    home_state.output_ready,
                    aw_diagonal_scratch_states[home].publish_stream),
                "record diagonal output ready");
        sync_live(
                aw_diagonal_scratch_states[home].publish_stream,
                "synchronize diagonal publish");
    }

    for (int group = 0; group < n_cells; ++group) {
        const int home = cells[group].home_device;
        aw_cuda_throw(cudaSetDevice(home),
                "set diagonal publication device");
        aw_cuda_throw(cudaStreamWaitEvent(
                    streams[home],
                    aw_live_states[group][home].output_ready, 0),
                "publish diagonal output");
    }

    const char * service_dump =
            getenv("GGML_CUDA_AW_SERVICE_DUMP");
    const char * dump_layer_env =
            getenv("GGML_CUDA_AW_SERVICE_DUMP_LAYER");
    const int dump_layer =
            dump_layer_env != nullptr ? atoi(dump_layer_env) : 0;
    if (service_dump != nullptr && service_dump[0] != '\0') {
        for (int cell = 0; cell < n_cells; ++cell) {
            if (dump_layer >= 0 &&
                    cells[cell].layer != dump_layer) {
                continue;
            }
            const int home = cells[cell].home_device;
            const size_t count =
                    (size_t) cells[cell].tokens*AW_EMBD;
            aw_cuda_throw(cudaSetDevice(home),
                    "set diagonal dump device");
            aw_cuda_throw(cudaStreamSynchronize(streams[home]),
                    "synchronize diagonal dump");
            std::vector<float> values(count);
            aw_cuda_throw(cudaMemcpy(
                        values.data(), cells[cell].output,
                        count*sizeof(float),
                        cudaMemcpyDeviceToHost),
                    "copy diagonal reduced output");
            aw_live_dump(service_dump, "reduced-output-f32",
                    cells[cell].layer, home, -1,
                    values.data(), count, sizeof(float));
        }
    }

    if (live_call == 0) {
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            fprintf(stderr,
                    "AffinityWave: diagonal panel%d arena device=%d bytes=%zu\n",
                    panel_n, device,
                    aw_diagonal_scratch_states[device].bytes());
        }
    }
}

static void aw_live_pairwave_panel(
        const std::array<cudaStream_t, AW_GPU_COUNT> & streams,
        const ggml_cuda_aw_live_cell * cells,
        int n_cells,
        const aw_pairwave_manifest & manifest,
        int live_call,
        const char * debug_sync,
        bool nccl_order_sum,
        bool pairfold_local) {
    constexpr int groups_per_active = 2;
    constexpr int descriptors_per_cell =
            groups_per_active*AW_PRIMARY_PER_GPU;
    constexpr int descriptors =
            2*descriptors_per_cell;
    constexpr int panel_n = 512;
    constexpr int down_panels =
            AW_EMBD/panel_n;
    if (n_cells != 2 ||
            cells[0].layer != cells[1].layer) {
        throw std::runtime_error(
                "PairWave service requires two cells from one layer");
    }
    const bool p100_exact = aw_p100_exact_enabled();
    const int exact_sum_width =
            aw_p100_exact_sum_width();
    if (p100_exact && exact_sum_width != 2 &&
            exact_sum_width != 4) {
        throw std::runtime_error(
                "GGML_CUDA_AW_P100_EXACT_SUM_WIDTH must be 2 or 4");
    }
    const int layer = cells[0].layer;
    const char * bundle_env =
            getenv("GGML_CUDA_AW_PAIRWAVE_BUNDLE");
    if (bundle_env != nullptr &&
            strcmp(bundle_env, "0") != 0 &&
            strcmp(bundle_env, "m32") != 0 &&
            strcmp(bundle_env, "m16") != 0) {
        throw std::runtime_error(
                "GGML_CUDA_AW_PAIRWAVE_BUNDLE must be '0', 'm32', or 'm16'");
    }
    const bool bundle_m32 =
            bundle_env != nullptr &&
            strcmp(bundle_env, "m32") == 0;
    const bool bundle_m16 =
            bundle_env != nullptr &&
            strcmp(bundle_env, "m16") == 0;
    const char * bundle_ctas_env =
            getenv(
                "GGML_CUDA_AW_PAIRWAVE_M16_CTAS");
    const int bundle_m16_ctas =
            bundle_ctas_env != nullptr ?
            atoi(bundle_ctas_env) : 5;
    if (bundle_m16_ctas < 3 ||
            bundle_m16_ctas > 5) {
        throw std::runtime_error(
                "GGML_CUDA_AW_PAIRWAVE_M16_CTAS must be between 3 and 5");
    }
    const bool pair_stats =
            aw_env_on(getenv(
                "GGML_CUDA_AW_PAIRWAVE_STATS"));
    const aw_pairwave_layer & placement =
            manifest.layers[layer];
    const int active0 = placement.pair*2;
    const int active1 = active0 + 1;
    const int stride_tokens =
            std::max(cells[0].tokens,
                    cells[1].tokens);
    const int total_tokens =
            n_cells*stride_tokens;
    const int total_slots =
            total_tokens*8;
    constexpr size_t expert_bytes =
            (size_t) AW_EXPERT_FF*
            (AW_EMBD/AW_KSTAGE)*
            AW_Q8_BLOCK_BYTES;

    for (int device = 0; device < AW_GPU_COUNT;
            ++device) {
        aw_live_states[0][device].ensure(
                device, AW_GPU_COUNT, stride_tokens,
                false, true, false, true);
    }
    for (int active : {active0, active1}) {
        aw_pair_scratch_states[active].ensure(
                active, total_tokens, panel_n);
    }

    std::array<std::array<const char *, 3>,
            AW_GPU_COUNT> weight_bases{};
    {
        std::lock_guard<std::mutex> lock(
                aw_layout_mutex);
        for (int active : {active0, active1}) {
            for (int projection = 0;
                    projection < 3; ++projection) {
                const auto it = std::find_if(
                        aw_layout_entries.begin(),
                        aw_layout_entries.end(),
                        [&](const aw_layout_entry & entry) {
                    return entry.device == active &&
                            entry.layer == layer &&
                            entry.projection ==
                                projection &&
                            aw_t64_tensors.count(
                                entry.data) != 0;
                });
                if (it == aw_layout_entries.end() ||
                        it->bytes !=
                        (size_t) descriptors_per_cell*
                            expert_bytes) {
                    throw std::runtime_error(
                            "PairWave weight catalog is incomplete");
                }
                weight_bases[active][projection] =
                        (const char *) it->data;
            }
        }
    }

    std::array<std::array<int32_t, descriptors>,
            AW_GPU_COUNT> desc_experts{};
    std::array<std::array<const char *,
            descriptors*3>, AW_GPU_COUNT>
            weight_ptrs{};
    for (int active : {active0, active1}) {
        for (int cell = 0; cell < n_cells; ++cell) {
            for (int group_slot = 0;
                    group_slot < groups_per_active;
                    ++group_slot) {
                const int logical =
                        placement.groups[active][
                            group_slot];
                for (int local = 0;
                        local < AW_PRIMARY_PER_GPU;
                        ++local) {
                    const int desc =
                            cell*descriptors_per_cell +
                            group_slot*
                                AW_PRIMARY_PER_GPU +
                            local;
                    desc_experts[active][desc] =
                            logical*
                                AW_PRIMARY_PER_GPU +
                            local;
                    for (int projection = 0;
                            projection < 3;
                            ++projection) {
                        weight_ptrs[active][
                                desc*3 + projection] =
                                weight_bases[active][
                                    projection] +
                                (size_t) (group_slot*
                                    AW_PRIMARY_PER_GPU +
                                    local)*
                                    expert_bytes;
                    }
                }
            }
        }
    }

    for (int cell = 0; cell < n_cells; ++cell) {
        const int home = cells[cell].home_device;
        aw_cuda_throw(cudaSetDevice(home),
                "set PairWave source device");
        aw_cuda_throw(cudaEventRecord(
                    aw_live_states[0][home].
                        source_ready,
                    streams[home]),
                "record PairWave source ready");
    }

    for (int active : {active0, active1}) {
        aw_cuda_throw(cudaSetDevice(active),
                "set PairWave input device");
        aw_live_state & state =
                aw_live_states[0][active];
        aw_pair_scratch & scratch =
                aw_pair_scratch_states[active];
        cudaStream_t stream = streams[active];
        for (int cell = 0; cell < n_cells; ++cell) {
            const int home =
                    cells[cell].home_device;
            aw_cuda_throw(cudaStreamWaitEvent(
                        stream,
                        aw_live_states[0][home].
                            source_ready, 0),
                    "wait for PairWave source");
            float * input_dst =
                    (float *) scratch.input.get() +
                    (size_t) cell*stride_tokens*
                        AW_EMBD;
            int32_t * ids_dst =
                    (int32_t *) state.ids.get() +
                    (size_t) cell*stride_tokens*8;
            float * weights_dst =
                    (float *) state.weights.get() +
                    (size_t) cell*stride_tokens*8;
            uint2 * route_meta_dst =
                    (uint2 *) scratch.route_meta.get() +
                    (size_t) cell*stride_tokens;
            const size_t input_bytes =
                    (size_t) cells[cell].tokens*
                    AW_EMBD*sizeof(float);
            if (home == active) {
                aw_cuda_throw(cudaMemcpyAsync(
                            input_dst,
                            cells[cell].input,
                            input_bytes,
                            cudaMemcpyDeviceToDevice,
                            stream),
                        "copy PairWave local input");
            } else {
                aw_cuda_throw(cudaMemcpyPeerAsync(
                            input_dst, active,
                            cells[cell].input, home,
                            input_bytes, stream),
                        "copy PairWave peer input");
            }
            if (p100_exact) {
                aw_pair_pack_routes<true><<<
                        aw_grid((size_t)
                            cells[cell].tokens*8),
                        256, 0, stream>>>(
                            cells[cell].ids,
                            cells[cell].ids_stride,
                            cells[cell].weights,
                            cells[cell].weights_stride,
                            ids_dst, weights_dst,
                            route_meta_dst,
                            placement.groups[active][0],
                            placement.groups[active][1],
                            cells[cell].tokens);
            } else {
                aw_pair_pack_routes<false><<<
                        aw_grid((size_t)
                            cells[cell].tokens*8),
                        256, 0, stream>>>(
                            cells[cell].ids,
                            cells[cell].ids_stride,
                            cells[cell].weights,
                            cells[cell].weights_stride,
                            ids_dst, weights_dst,
                            nullptr, 0, 0,
                            cells[cell].tokens);
            }
            const int tail_tokens =
                    stride_tokens -
                    cells[cell].tokens;
            if (tail_tokens > 0) {
                aw_cuda_throw(cudaMemsetAsync(
                            ids_dst +
                                (size_t)
                                    cells[cell].tokens*8,
                            0xff,
                            (size_t) tail_tokens*8*
                                sizeof(int32_t),
                            stream),
                        "clear PairWave padded ids");
                if (p100_exact) {
                    aw_cuda_throw(cudaMemsetAsync(
                                route_meta_dst +
                                    cells[cell].tokens,
                                0,
                                (size_t) tail_tokens*
                                    sizeof(uint2),
                                stream),
                            "clear PairWave route metadata");
                }
            }
        }

        aw_cuda_throw(cudaMemcpyAsync(
                    scratch.weight_ptrs.get(),
                    weight_ptrs[active].data(),
                    weight_ptrs[active].size()*
                        sizeof(const char *),
                    cudaMemcpyHostToDevice,
                    stream),
                "copy PairWave weight pointers");
        aw_cuda_throw(cudaMemcpyAsync(
                    scratch.desc_experts.get(),
                    desc_experts[active].data(),
                    desc_experts[active].size()*
                        sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream),
                "copy PairWave descriptor experts");
        aw_cuda_throw(cudaMemsetAsync(
                    state.counts.get(), 0,
                    descriptors*sizeof(int32_t),
                    stream),
                "clear PairWave counts");
        aw_pair_count_routes<<<
                aw_grid(total_slots), 256, 0,
                stream>>>(
                    (const int32_t *)
                        state.ids.get(),
                    (int32_t *)
                        state.counts.get(),
                    total_slots,
                    stride_tokens,
                    placement.groups[active][0],
                    placement.groups[active][1]);
        aw_r44_build_plan<<<
                1, p100_exact ? 256 : 1,
                0, stream>>>(
                (const int32_t *)
                    state.counts.get(),
                (const int32_t *)
                    scratch.desc_experts.get(),
                (int32_t *) state.offsets.get(),
                (int32_t *) state.tile_counts.get(),
                (aw_tile_desc *)
                    state.tiles_m64.get(),
                (aw_tile_desc *)
                    state.tiles_m32.get(),
                (aw_tile_desc *)
                    state.tiles_m16.get(),
                (aw_tile_desc *)
                    state.tiles_warp.get(),
                (aw_tile_desc *)
                    state.tiles_m8.get(),
                (aw_tile_desc *)
                    state.tiles_m4.get(),
                (aw_work_desc *)
                    state.gate_desc.get(),
                (aw_work_desc *)
                    state.up_desc.get(),
                (aw_work_desc *)
                    state.down_desc.get(),
                (const char * const *)
                    scratch.weight_ptrs.get(),
                scratch.input.get(),
                (int32_t *)
                    state.route_input.get(),
                (float *) scratch.gate.get(),
                (float *) scratch.up.get(),
                (float *) scratch.gate.get(),
                (float *) scratch.up.get(),
                AW_EXPERT_FF, panel_n,
                descriptors);
        aw_cuda_throw(cudaGetLastError(),
                "launch PairWave plan");
        if (aw_q8_cohortrail_enabled()) {
            aw_build_m16_cohorts<<<
                    1, AW_COHORTRAIL_MAX_TILES,
                    0, stream>>>(
                    (const aw_work_desc *)
                        state.gate_desc.get(),
                    (const aw_tile_desc *)
                        state.tiles_m16.get(),
                    0,
                    (const int32_t *)
                        state.tile_counts.get() + 2,
                    (int32_t *)
                        state.cohort_counts.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p2.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p3.get(),
                    (aw_m16_cohort *)
                        state.cohorts_p4.get(),
                    (aw_tile_desc *)
                        state.cohort_singles.get());
        }
        if (bundle_m32 || bundle_m16 ||
                pair_stats) {
            const int bundle_tile_index =
                    bundle_m16 ? 2 : 1;
            aw_pair_build_tile_bundles<<<
                    1, 1, 0, stream>>>(
                        (const aw_work_desc *)
                            state.gate_desc.get(),
                        (const aw_tile_desc *)
                            (bundle_m16 ?
                                state.tiles_m16.get() :
                                state.tiles_m32.get()),
                        (int32_t *)
                            state.tile_counts.get(),
                        (aw_tile_desc *)
                            state.tiles_m8.get(),
                        (aw_tile_desc *)
                            state.tiles_m4.get(),
                        bundle_tile_index);
        }
        if (pair_stats) {
            std::array<int32_t, 6> counts{};
            aw_cuda_throw(cudaMemcpyAsync(
                        counts.data(),
                        state.tile_counts.get(),
                        counts.size()*
                            sizeof(int32_t),
                        cudaMemcpyDeviceToHost,
                        stream),
                    "copy PairWave tile counts");
            aw_cuda_throw(cudaStreamSynchronize(
                        stream),
                    "synchronize PairWave tile counts");
            fprintf(stderr,
                    "AffinityWave: PairWave tiles layer=%d cells=%d,%d active=%d bundle=%s m64=%d m32=%d m16=%d bundle-pairs=%d bundle-singles=%d\n",
                    layer, cells[0].home_device,
                    cells[1].home_device, active,
                    bundle_m16 ? "m16" : "m32",
                    counts[0], counts[1],
                    counts[2], counts[4],
                    counts[5]);
        }
        const size_t down_weight_panel_bytes =
                (size_t) (panel_n/AW_T64_ROWS)*
                (AW_EXPERT_FF/AW_KSTAGE)*
                AW_T64_STAGE_BYTES;
        aw_live_build_panel_descs<<<
                aw_grid((size_t) descriptors*
                    down_panels),
                256, 0, stream>>>(
                    (const aw_work_desc *)
                        state.down_desc.get(),
                    (aw_work_desc *)
                        state.down_panel_desc.get(),
                    descriptors, down_panels,
                    down_weight_panel_bytes);
        aw_pair_fill_routes<<<
                descriptors, 256, 0, stream>>>(
                    (const int32_t *)
                        state.ids.get(),
                    (const int32_t *)
                        state.offsets.get(),
                    (const int32_t *)
                        scratch.desc_experts.get(),
                    (int32_t *)
                        state.route_input.get(),
                    (int32_t *)
                        state.token_routes.get(),
                    descriptors, stride_tokens);
        const int sm_count =
                aw_sm_counts[active];
        if (sm_count <= 0) {
            throw std::runtime_error(
                    "PairWave SM count was not initialized");
        }
        aw_live_launch_projection(
                state, state.gate_desc,
                AW_EXPERT_FF, AW_EMBD,
                stream, sm_count*2,
                sm_count*3, false, 0,
                bundle_m32, bundle_m16,
                bundle_m16_ctas, 2, true);
        aw_live_launch_projection(
                state, state.up_desc,
                AW_EXPERT_FF, AW_EMBD,
                stream, sm_count*2,
                sm_count*3, false, 0,
                bundle_m32, bundle_m16,
                bundle_m16_ctas, 2, true);
        aw_live_swiglu<<<
                aw_grid((size_t) total_slots*
                    AW_EXPERT_FF),
                256, 0, stream>>>(
                    (const float *)
                        scratch.gate.get(),
                    (const float *)
                        scratch.up.get(),
                    (float *) scratch.gate.get(),
                    (const int32_t *)
                        state.offsets.get() +
                        descriptors);
        aw_cuda_throw(cudaGetLastError(),
                "launch PairWave gate/up");
    }

    for (int panel = 0;
            panel < down_panels; ++panel) {
        for (int active : {active0, active1}) {
            aw_cuda_throw(cudaSetDevice(active),
                    "set PairWave down device");
            aw_live_state & state =
                    aw_live_states[0][active];
            aw_pair_scratch & scratch =
                    aw_pair_scratch_states[active];
            cudaStream_t stream =
                    streams[active];
            if (panel > 0) {
                for (int cell = 0;
                        cell < n_cells; ++cell) {
                    const int home =
                            cells[cell].home_device;
                    aw_cuda_throw(
                            cudaStreamWaitEvent(
                                stream,
                                aw_live_states[0][home].
                                    recv_free,
                                0),
                            "wait for PairWave panel use");
                }
            }
            const int sm_count =
                    aw_sm_counts[active];
            aw_live_launch_projection(
                    state,
                    state.down_panel_desc,
                    panel_n, AW_EXPERT_FF,
                    stream, sm_count*2,
                    sm_count*3, false,
                    (size_t) panel*
                        descriptors,
                    bundle_m32, bundle_m16,
                    bundle_m16_ctas, 2, true);
            const bool fused_pair_reduce =
                    p100_exact && panel_n == 512 &&
                    ((uintptr_t) scratch.route_meta.get() %
                        alignof(uint2)) == 0 &&
                    ((uintptr_t) scratch.up.get() %
                        (exact_sum_width == 4 ?
                            alignof(float4) :
                            alignof(float2))) == 0 &&
                    ((uintptr_t) scratch.partial.get() %
                        (exact_sum_width == 4 ?
                            alignof(uint2) :
                            alignof(uint32_t))) == 0;
            if (fused_pair_reduce &&
                    exact_sum_width == 4) {
                aw_pair_group_reduce_panel_p100<4><<<
                        total_tokens, 128, 0, stream>>>(
                            (const float *)
                                scratch.up.get(),
                            (const int32_t *)
                                state.token_routes.get(),
                            (const float *)
                                state.weights.get(),
                            (const uint2 *)
                                scratch.route_meta.get(),
                            (uint16_t *)
                                scratch.partial.get(),
                            total_tokens,
                            placement.groups[active][0],
                            placement.groups[active][1],
                            panel_n);
            } else if (fused_pair_reduce) {
                aw_pair_group_reduce_panel_p100<2><<<
                        total_tokens, 256, 0, stream>>>(
                            (const float *)
                                scratch.up.get(),
                            (const int32_t *)
                                state.token_routes.get(),
                            (const float *)
                                state.weights.get(),
                            (const uint2 *)
                                scratch.route_meta.get(),
                            (uint16_t *)
                                scratch.partial.get(),
                            total_tokens,
                            placement.groups[active][0],
                            placement.groups[active][1],
                            panel_n);
            } else {
                for (int group_slot = 0;
                        group_slot < 2;
                        ++group_slot) {
                    const int logical =
                            placement.groups[active][
                                group_slot];
                    aw_pair_group_reduce_panel<<<
                            aw_grid((size_t)
                                total_tokens*
                                panel_n),
                            256, 0, stream>>>(
                                (const float *)
                                    scratch.up.get(),
                                (const int32_t *)
                                    state.ids.get(),
                                (const int32_t *)
                                    state.token_routes.get(),
                                (const float *)
                                    state.weights.get(),
                                (uint16_t *)
                                    scratch.partial.get(),
                                total_tokens, logical,
                                panel_n);
                }
            }
            aw_cuda_throw(cudaGetLastError(),
                    "launch PairWave down panel");
            aw_cuda_throw(cudaEventRecord(
                        state.compute_done,
                        stream),
                    "record PairWave panel ready");
        }

        for (int cell = 0;
                cell < n_cells; ++cell) {
            const int home =
                    cells[cell].home_device;
            const int token_offset =
                    cell*stride_tokens;
            std::array<const uint16_t *, AW_GPU_COUNT>
                    group_partials{};
            if (pairfold_local) {
                aw_pair_scratch & home_scratch =
                        aw_pair_scratch_states[home];
                for (int logical = 0;
                        logical < AW_GPU_COUNT; ++logical) {
                    const int owner =
                            placement.owners[logical];
                    aw_cuda_throw(cudaSetDevice(owner),
                            "set PairFold owner-return device");
                    cudaStream_t copy_stream =
                            aw_copy_streams.get(owner);
                    aw_cuda_throw(cudaStreamWaitEvent(
                                copy_stream,
                                aw_live_states[0][owner].
                                    compute_done, 0),
                            "wait for PairFold owner partial");
                    if (panel > 0) {
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    copy_stream,
                                    aw_live_states[0][home].
                                        recv_free, 0),
                                "wait for PairFold receive scratch");
                    }
                    const uint16_t * source =
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    owner].partial.get() +
                            ((size_t) logical*
                                total_tokens +
                             token_offset)*panel_n;
                    uint16_t * destination =
                            (uint16_t *)
                                home_scratch.recv.get() +
                            ((size_t) logical*
                                total_tokens +
                             token_offset)*panel_n;
                    const size_t bytes =
                            (size_t) cells[cell].tokens*
                            panel_n*sizeof(uint16_t);
                    if (owner == home) {
                        aw_cuda_throw(cudaMemcpyAsync(
                                    destination, source, bytes,
                                    cudaMemcpyDeviceToDevice,
                                    copy_stream),
                                "copy PairFold local owner partial");
                    } else {
                        aw_cuda_throw(cudaMemcpyPeerAsync(
                                    destination, home,
                                    source, owner, bytes,
                                    copy_stream),
                                "copy PairFold peer owner partial");
                    }
                    aw_cuda_throw(cudaEventRecord(
                                aw_pair_scratch_states[
                                    owner].copied[logical],
                                copy_stream),
                            "record PairFold owner return");
                    group_partials[logical] =
                            destination;
                }
            } else if (p100_exact) {
                for (int logical = 0;
                        logical < AW_GPU_COUNT; ++logical) {
                    const int owner =
                            placement.owners[logical];
                    group_partials[logical] =
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    owner].partial.get() +
                            ((size_t) logical*
                                total_tokens +
                             token_offset)*panel_n;
                }
            }
            aw_cuda_throw(cudaSetDevice(home),
                    "set PairWave output device");
            cudaStream_t stream =
                    streams[home];
            if (pairfold_local) {
                for (int logical = 0;
                        logical < AW_GPU_COUNT; ++logical) {
                    const int owner =
                            placement.owners[logical];
                    aw_cuda_throw(cudaStreamWaitEvent(
                                stream,
                                aw_pair_scratch_states[
                                    owner].copied[logical],
                                0),
                            "wait for PairFold owner return");
                }
            } else {
                aw_cuda_throw(cudaStreamWaitEvent(
                            stream,
                            aw_live_states[0][active0].
                                compute_done, 0),
                        "wait for PairWave owner 0");
                aw_cuda_throw(cudaStreamWaitEvent(
                            stream,
                            aw_live_states[0][active1].
                                compute_done, 0),
                        "wait for PairWave owner 1");
            }
            bool p100_pair_sum = p100_exact &&
                    ((uintptr_t) cells[cell].output %
                        (exact_sum_width == 4 ?
                            alignof(float4) :
                            alignof(float2))) == 0;
            for (const uint16_t * partial :
                    group_partials) {
                p100_pair_sum = p100_pair_sum &&
                        ((uintptr_t) partial %
                            (exact_sum_width == 4 ?
                                alignof(uint2) :
                                alignof(uint32_t))) == 0;
            }
            if (p100_pair_sum &&
                    exact_sum_width == 4) {
                const int blocks =
                        std::min(cells[cell].tokens, 224);
                if (pairfold_local && nccl_order_sum) {
                    aw_pair_sum_groups_local_panel_p100<
                            true, 4><<<
                            blocks, 128, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                cells[cell].reserved*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else if (pairfold_local) {
                    aw_pair_sum_groups_local_panel_p100<
                            false, 4><<<
                            blocks, 128, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                cells[cell].reserved*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else if (nccl_order_sum) {
                    aw_pair_sum_groups_peer_panel_p100<
                            true, 4><<<
                            blocks, 128, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                (pairfold_local ?
                                    cells[cell].reserved :
                                    cells[cell].home_device)*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else {
                    aw_pair_sum_groups_peer_panel_p100<
                            false, 4><<<
                            blocks, 128, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                (pairfold_local ?
                                    cells[cell].reserved :
                                    cells[cell].home_device)*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                }
            } else if (p100_pair_sum) {
                const int blocks =
                        std::min(cells[cell].tokens, 448);
                if (pairfold_local && nccl_order_sum) {
                    aw_pair_sum_groups_local_panel_p100<
                            true, 2><<<
                            blocks, 256, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                cells[cell].reserved*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else if (pairfold_local) {
                    aw_pair_sum_groups_local_panel_p100<
                            false, 2><<<
                            blocks, 256, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                cells[cell].reserved*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else if (nccl_order_sum) {
                    aw_pair_sum_groups_peer_panel_p100<
                            true, 2><<<
                            blocks, 256, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                (pairfold_local ?
                                    cells[cell].reserved :
                                    cells[cell].home_device)*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                } else {
                    aw_pair_sum_groups_peer_panel_p100<
                            false, 2><<<
                            blocks, 256, 0, stream>>>(
                                group_partials[0],
                                group_partials[1],
                                group_partials[2],
                                group_partials[3],
                                cells[cell].output,
                                cells[cell].tokens,
                                (pairfold_local ?
                                    cells[cell].reserved :
                                    cells[cell].home_device)*
                                    stride_tokens,
                                panel_n,
                                panel*panel_n);
                }
            } else if (nccl_order_sum) {
                aw_pair_sum_groups_peer_panel<true><<<
                        aw_grid((size_t)
                            cells[cell].tokens*
                            panel_n),
                        256, 0, stream>>>(
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    active0].
                                    partial.get(),
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    active1].
                                    partial.get(),
                            active0,
                            placement.owners[0],
                            placement.owners[1],
                            placement.owners[2],
                            placement.owners[3],
                            cells[cell].output,
                            total_tokens,
                            token_offset,
                            (pairfold_local ?
                                cells[cell].reserved :
                                cells[cell].home_device)*
                                stride_tokens,
                            cells[cell].tokens,
                            panel_n,
                            panel*panel_n);
            } else {
                aw_pair_sum_groups_peer_panel<<<
                        aw_grid((size_t)
                            cells[cell].tokens*
                            panel_n),
                        256, 0, stream>>>(
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    active0].
                                    partial.get(),
                            (const uint16_t *)
                                aw_pair_scratch_states[
                                    active1].
                                    partial.get(),
                            active0,
                            placement.owners[0],
                            placement.owners[1],
                            placement.owners[2],
                            placement.owners[3],
                            cells[cell].output,
                            total_tokens,
                            token_offset,
                            (pairfold_local ?
                                cells[cell].reserved :
                                cells[cell].home_device)*
                                stride_tokens,
                            cells[cell].tokens,
                            panel_n,
                            panel*panel_n);
            }
            aw_cuda_throw(cudaGetLastError(),
                    "launch PairWave group sum");
            aw_cuda_throw(cudaEventRecord(
                        aw_live_states[0][home].
                            recv_free,
                        stream),
                    "record PairWave panel consumed");
        }
    }

    for (int active : {active0, active1}) {
        aw_cuda_throw(cudaSetDevice(active),
                "set PairWave completion device");
        cudaStream_t stream = streams[active];
        for (int cell = 0;
                cell < n_cells; ++cell) {
            aw_cuda_throw(cudaStreamWaitEvent(
                        stream,
                        aw_live_states[0][
                            cells[cell].home_device].
                            recv_free, 0),
                    "wait for PairWave output use");
        }
        aw_cuda_throw(cudaEventRecord(
                    aw_live_states[0][active].
                        scratch_free,
                    stream),
                "record PairWave scratch free");
    }

    const char * service_dump =
            getenv("GGML_CUDA_AW_SERVICE_DUMP");
    const char * dump_layer_env =
            getenv("GGML_CUDA_AW_SERVICE_DUMP_LAYER");
    const int dump_layer =
            dump_layer_env != nullptr ?
            atoi(dump_layer_env) : 0;
    if (service_dump != nullptr &&
            service_dump[0] != '\0' &&
            (dump_layer < 0 ||
             layer == dump_layer)) {
        for (int cell = 0;
                cell < n_cells; ++cell) {
            const int home =
                    cells[cell].home_device;
            aw_cuda_throw(cudaSetDevice(home),
                    "set PairWave dump device");
            aw_cuda_throw(cudaStreamSynchronize(
                        streams[home]),
                    "synchronize PairWave dump");
            const size_t count =
                    (size_t) cells[cell].tokens*
                    AW_EMBD;
            std::vector<float> values(count);
            aw_cuda_throw(cudaMemcpy(
                        values.data(),
                        cells[cell].output,
                        count*sizeof(float),
                        cudaMemcpyDeviceToHost),
                    "copy PairWave output");
            aw_live_dump(service_dump,
                    "reduced-output-f32",
                    layer,
                    pairfold_local ?
                        cells[cell].reserved : home,
                    -1,
                    values.data(), count,
                    sizeof(float));
        }
    }

    if (live_call == 0) {
        fprintf(stderr,
                "AffinityWave: PairWave panel512 bundle=%s ctas=%d arena active0=%zu active1=%zu bytes\n",
                bundle_m32 ? "m32" :
                bundle_m16 ? "m16" : "0",
                bundle_m16_ctas,
                aw_pair_scratch_states[active0].bytes(),
                aw_pair_scratch_states[active1].bytes());
    }
    if (aw_env_on(debug_sync)) {
        for (int device = 0;
                device < AW_GPU_COUNT; ++device) {
            aw_cuda_throw(cudaSetDevice(device),
                    "set PairWave debug-sync device");
            aw_cuda_throw(cudaStreamSynchronize(
                        streams[device]),
                    "synchronize PairWave service");
        }
    }
}

static int aw_live_service(
        void * const * streams_ptr,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity,
        bool pairfold_local) {
    try {
        static std::atomic<int> live_calls{0};
        const int live_call = live_calls.fetch_add(1);
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service entered cells=%d\n", n_cells);
        }
        if (!ggml_cuda_affinity_wave_t64_enabled() || streams_ptr == nullptr || cells == nullptr ||
                n_cells < 1 || n_cells > AW_GPU_COUNT) {
            throw std::runtime_error("invalid live AffinityWave service request");
        }
        const char * service_env = getenv("GGML_CUDA_AW_SERVICE");
        if (service_env != nullptr &&
                strcmp(service_env, "legacy") != 0 &&
                strcmp(service_env, "n256") != 0) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_SERVICE must be 'legacy' or 'n256'");
        }
        const bool n256_service =
                service_env != nullptr && strcmp(service_env, "n256") == 0;
        const char * diagonal_service_env =
                getenv("GGML_CUDA_AW_DIAGONAL_SERVICE");
        if (diagonal_service_env != nullptr &&
                strcmp(diagonal_service_env, "legacy") != 0 &&
                strcmp(diagonal_service_env, "panel512") != 0 &&
                strcmp(diagonal_service_env, "panel1024") != 0 &&
                strcmp(diagonal_service_env, "panel2048") != 0) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_DIAGONAL_SERVICE must be 'legacy', 'panel512', 'panel1024', or 'panel2048'");
        }
        const bool diagonal_panel =
                !pairfold_local &&
                diagonal_service_env != nullptr &&
                (strcmp(diagonal_service_env, "panel512") == 0 ||
                 strcmp(diagonal_service_env, "panel1024") == 0 ||
                 strcmp(diagonal_service_env, "panel2048") == 0);
        const aw_group_placement & group_placement =
                aw_group_get_placement();
        int diagonal_panel_n = 512;
        if (diagonal_service_env != nullptr &&
                strcmp(diagonal_service_env, "panel1024") == 0) {
            diagonal_panel_n = 1024;
        } else if (diagonal_service_env != nullptr &&
                strcmp(diagonal_service_env, "panel2048") == 0) {
            diagonal_panel_n = 2048;
        }
        const bool r44_service =
                strcmp(aw_r44_mode(), "service") == 0;
        const bool group_service =
                group_placement.enabled && !diagonal_panel;
        const bool selective_service =
                r44_service || group_service;
        const char * group_pool_env =
                getenv("GGML_CUDA_AW_GROUP_POOL");
        if (group_pool_env != nullptr &&
                strcmp(group_pool_env, "0") != 0 &&
                strcmp(group_pool_env, "1") != 0) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_GROUP_POOL must be '0' or '1'");
        }
        const bool group_pool =
                group_service && group_pool_env != nullptr &&
                strcmp(group_pool_env, "1") == 0;
        const char * expert_map_env =
                getenv("GGML_CUDA_MOE_EPLB_EXPERT_MAP");
        const bool group_logical_ids =
                group_service && expert_map_env != nullptr &&
                expert_map_env[0] != '\0';
        if (expert_map_env != nullptr &&
                expert_map_env[0] != '\0' &&
                !group_service) {
            throw std::runtime_error(
                    "GGML_CUDA_MOE_EPLB_EXPERT_MAP requires whole-owner group service");
        }
        const bool r44_stats =
                r44_service &&
                aw_env_on(getenv("GGML_CUDA_AW_R44_STATS"));
        const bool r44_memory =
                r44_service &&
                aw_env_on(getenv("GGML_CUDA_AW_R44_MEMORY"));
        static std::array<size_t, AW_GPU_COUNT>
                r44_memory_base{};
        static std::array<size_t, AW_GPU_COUNT>
                r44_memory_min{};
        if (r44_memory && live_call == 0) {
            for (int device = 0; device < AW_GPU_COUNT; ++device) {
                aw_cuda_throw(cudaSetDevice(device),
                        "set R44 memory baseline device");
                size_t total_bytes = 0;
                aw_cuda_throw(cudaMemGetInfo(
                            &r44_memory_base[device],
                            &total_bytes),
                        "query R44 memory baseline");
                r44_memory_min[device] =
                        r44_memory_base[device];
            }
        }
        const bool shared_service = aw_env_on(getenv("GGML_CUDA_AW_SHARED_SERVICE"));
        const bool fused_gate_up = aw_env_on(getenv("GGML_CUDA_AW_FUSED_GATE_UP"));
        if (n256_service && shared_service) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_SERVICE=n256 does not yet support GGML_CUDA_AW_SHARED_SERVICE");
        }
        if (n256_service && fused_gate_up) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_SERVICE=n256 is incompatible with GGML_CUDA_AW_FUSED_GATE_UP");
        }
        const char * route_scatter_env =
                getenv("GGML_CUDA_AW_ROUTE_SCATTER");
        if (route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "deterministic") != 0 &&
                strcmp(route_scatter_env, "atomic") != 0 &&
                strcmp(route_scatter_env, "stable8") != 0 &&
                strcmp(route_scatter_env, "stable256") != 0 &&
                strcmp(route_scatter_env, "radix256") != 0 &&
                strcmp(route_scatter_env, "segment8") != 0) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_ROUTE_SCATTER must be 'deterministic', 'atomic', 'stable8', 'stable256', 'radix256', or 'segment8'");
        }
        const bool atomic_route_scatter =
                route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "atomic") == 0;
        const bool stable8_route_scatter =
                route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "stable8") == 0;
        const bool stable256_route_scatter =
                route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "stable256") == 0;
        const bool radix256_route_scatter =
                route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "radix256") == 0;
        const bool segment8_route_scatter =
                route_scatter_env != nullptr &&
                strcmp(route_scatter_env, "segment8") == 0;
        const char * debug_sync = getenv("GGML_CUDA_AW_DEBUG_LIVE_SYNC");
        const bool bf16_wire = getenv("GGML_CUDA_AW_WIRE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_WIRE"), "bf16") == 0;
        const bool f16_wire = getenv("GGML_CUDA_AW_WIRE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_WIRE"), "f16") == 0;
        const bool wire16 = bf16_wire || f16_wire;
        const bool bf16_partial = getenv("GGML_CUDA_AW_PARTIAL") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_PARTIAL"), "bf16") == 0;
        const bool nccl_order_sum =
                bf16_partial &&
                aw_env_on(getenv("GGML_CUDA_AW_NCCL_ORDER_SUM"));
        const bool pairwave_service = pairfold_local ||
                aw_env_on(getenv(
                    "GGML_CUDA_AW_PAIRWAVE_SERVICE"));
        const bool direct_owner =
                aw_env_on(getenv("GGML_CUDA_AW_DIRECT_OWNER"));
        const bool direct_owner_check =
                aw_env_on(getenv("GGML_CUDA_AW_DIRECT_OWNER_CHECK"));
        if (direct_owner_check && !direct_owner) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_DIRECT_OWNER_CHECK requires GGML_CUDA_AW_DIRECT_OWNER");
        }
        if (n256_service && direct_owner_check) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_SERVICE=n256 is incompatible with GGML_CUDA_AW_DIRECT_OWNER_CHECK");
        }
        if (diagonal_panel &&
                (n256_service || !bf16_partial || wire16 ||
                 shared_service || fused_gate_up || direct_owner ||
                 direct_owner_check || r44_service ||
                 aw_env_on(getenv("GGML_CUDA_AW_HEADFOLD")) ||
                 (route_scatter_env != nullptr &&
                  strcmp(route_scatter_env, "deterministic") != 0))) {
            throw std::runtime_error(
                    "panelized diagonal service requires legacy service, F32 wire, BF16 partials, deterministic routes, and the non-shared diagonal scheduler");
        }
        if (r44_service &&
                (!n256_service || !bf16_partial || wire16 ||
                 shared_service || fused_gate_up || direct_owner ||
                 direct_owner_check ||
                 !aw_env_on(getenv("GGML_CUDA_AW_HEADFOLD")) ||
                 n_cells != AW_GPU_COUNT)) {
            throw std::runtime_error(
                    "R44 service requires HeadFold, N256, F32 wire, BF16 partials, and the exact non-shared owner path");
        }
        if (group_service &&
                (!n256_service || !bf16_partial || wire16 ||
                 shared_service || fused_gate_up || direct_owner ||
                 direct_owner_check || r44_service ||
                 !aw_env_on(getenv("GGML_CUDA_AW_HEADFOLD")) ||
                 n_cells != AW_GPU_COUNT ||
                 (route_scatter_env != nullptr &&
                  strcmp(route_scatter_env, "deterministic") != 0))) {
            throw std::runtime_error(
                    "whole-owner group service requires HeadFold, N256, F32 wire, BF16 partials, and deterministic routes");
        }
        const aw_pairwave_manifest * pairwave_manifest =
                nullptr;
        if (pairwave_service) {
            const aw_pairwave_manifest & manifest =
                    aw_pairwave_get_manifest();
            if (!manifest.enabled || n256_service ||
                    !bf16_partial || wire16 ||
                    shared_service || fused_gate_up ||
                    direct_owner || direct_owner_check ||
                    r44_service || group_service ||
                    diagonal_panel ||
                    !aw_env_on(getenv(
                        "GGML_CUDA_AW_HEADFOLD")) ||
                    n_cells != 2 ||
                    cells[0].layer != cells[1].layer ||
                    (route_scatter_env != nullptr &&
                     strcmp(route_scatter_env,
                         "deterministic") != 0)) {
                throw std::runtime_error(
                        "PairWave service requires its manifest, HeadFold, legacy service, F32 wire, BF16 partials, two same-layer cells, and deterministic routes");
            }
            pairwave_manifest = &manifest;
        }
        auto sync_live = [&](cudaStream_t stream, const char * operation) {
            if (aw_env_on(debug_sync) &&
                    (strcmp(debug_sync, "1") == 0 || strstr(operation, debug_sync) != nullptr)) {
                aw_cuda_throw(cudaStreamSynchronize(stream), operation);
            }
        };
        const size_t wire_element_size = wire16 ? sizeof(uint16_t) : sizeof(float);
        int max_tokens = 0;
        std::array<int, AW_GPU_COUNT> cell_for_home;
        cell_for_home.fill(-1);
        for (int cell = 0; cell < n_cells; ++cell) {
            if (cells[cell].tokens < 1 || cells[cell].layer < 0 || cells[cell].layer >= AW_LAYERS ||
                    cells[cell].home_device < 0 || cells[cell].home_device >= AW_GPU_COUNT ||
                    cells[cell].input == nullptr || cells[cell].ids == nullptr ||
                    cells[cell].weights == nullptr || cells[cell].output == nullptr ||
                    cells[cell].ids_stride < 8*sizeof(int32_t) ||
                    cells[cell].weights_stride < 8*sizeof(float) ||
                    (shared_service && cells[cell].shared_output == nullptr) ||
                    cell_for_home[cells[cell].home_device] != -1) {
                throw std::runtime_error("invalid live AffinityWave cell descriptor");
            }
            cell_for_home[cells[cell].home_device] = cell;
            max_tokens = std::max(max_tokens, cells[cell].tokens);
        }
        {
            const char * input_dump =
                    getenv("GGML_CUDA_AW_SERVICE_DUMP");
            const char * input_dump_layer_env =
                    getenv("GGML_CUDA_AW_SERVICE_DUMP_LAYER");
            const int input_dump_layer =
                    input_dump_layer_env != nullptr ?
                    atoi(input_dump_layer_env) : 0;
            if (input_dump != nullptr &&
                    input_dump[0] != '\0') {
                for (int cell = 0; cell < n_cells; ++cell) {
                    if (input_dump_layer >= 0 &&
                            cells[cell].layer !=
                                input_dump_layer) {
                        continue;
                    }
                    const int home =
                            cells[cell].home_device;
                    const size_t count =
                            (size_t) cells[cell].tokens*
                                AW_EMBD;
                    aw_cuda_throw(cudaSetDevice(home),
                            "set service input dump device");
                    aw_cuda_throw(cudaStreamSynchronize(
                                (cudaStream_t)
                                    streams_ptr[home]),
                            "synchronize service input dump");
                    std::vector<float> values(count);
                    aw_cuda_throw(cudaMemcpy(
                                values.data(),
                                cells[cell].input,
                                count*sizeof(float),
                                cudaMemcpyDeviceToHost),
                            "copy service input dump");
                    aw_live_dump(input_dump,
                            "service-input-f32",
                            cells[cell].layer,
                            pairfold_local ?
                                cells[cell].reserved : home,
                            -1,
                            values.data(), count,
                            sizeof(float));
                    const size_t slots =
                            (size_t) cells[cell].tokens*8;
                    const size_t route_row_bytes =
                            8*sizeof(int32_t);
                    std::vector<int32_t> ids(slots);
                    std::vector<float> weights(slots);
                    aw_cuda_throw(cudaMemcpy2D(
                                ids.data(), route_row_bytes,
                                cells[cell].ids,
                                cells[cell].ids_stride,
                                route_row_bytes,
                                cells[cell].tokens,
                                cudaMemcpyDeviceToHost),
                            "copy service id dump");
                    aw_cuda_throw(cudaMemcpy2D(
                                weights.data(), route_row_bytes,
                                cells[cell].weights,
                                cells[cell].weights_stride,
                                route_row_bytes,
                                cells[cell].tokens,
                                cudaMemcpyDeviceToHost),
                            "copy service weight dump");
                    aw_live_dump(input_dump,
                            "service-ids-i32",
                            cells[cell].layer,
                            pairfold_local ?
                                cells[cell].reserved : home,
                            -1,
                            ids.data(), slots, sizeof(int32_t));
                    aw_live_dump(input_dump,
                            "service-weights-f32",
                            cells[cell].layer,
                            pairfold_local ?
                                cells[cell].reserved : home,
                            -1,
                            weights.data(), slots, sizeof(float));
                }
            }
        }
        const bool debug_routes = aw_env_on(getenv("GGML_CUDA_AW_DEBUG_ROUTES"));
        const char * route_dump = getenv("GGML_CUDA_AW_ROUTE_DUMP");
        const bool dump_routes = route_dump != nullptr && route_dump[0] != '\0';
        if (debug_routes || dump_routes) {
            for (int cell = 0; cell < n_cells; ++cell) {
                aw_cuda_throw(cudaSetDevice(cells[cell].home_device), "set route verification device");
                aw_cuda_throw(cudaStreamSynchronize((cudaStream_t) streams_ptr[cells[cell].home_device]),
                        "synchronize route verification source");
                const size_t slots = (size_t) cells[cell].tokens*8;
                const size_t route_row_bytes = 8*sizeof(int32_t);
                std::vector<int32_t> ids(slots);
                std::vector<float> weights(slots);
                aw_cuda_throw(cudaMemcpy2D(ids.data(), route_row_bytes,
                            cells[cell].ids, cells[cell].ids_stride,
                            route_row_bytes, cells[cell].tokens, cudaMemcpyDeviceToHost),
                        "copy route verification ids");
                aw_cuda_throw(cudaMemcpy2D(weights.data(), route_row_bytes,
                            cells[cell].weights, cells[cell].weights_stride,
                            route_row_bytes, cells[cell].tokens, cudaMemcpyDeviceToHost),
                        "copy route verification weights");
                std::array<size_t, AW_GPU_COUNT> owner_counts{};
                size_t invalid = 0;
                double weight_sum = 0.0;
                float weight_max = 0.0f;
                int32_t id_min = std::numeric_limits<int32_t>::max();
                int32_t id_max = std::numeric_limits<int32_t>::min();
                for (size_t slot = 0; slot < slots; ++slot) {
                    id_min = std::min(id_min, ids[slot]);
                    id_max = std::max(id_max, ids[slot]);
                    if (ids[slot] >= 0 && ids[slot] < AW_EXPERTS) {
                        ++owner_counts[ids[slot]/AW_PRIMARY_PER_GPU];
                    } else {
                        ++invalid;
                    }
                    weight_sum += weights[slot];
                    weight_max = std::max(weight_max, std::abs(weights[slot]));
                }
                if (debug_routes) {
                    fprintf(stderr,
                            "AffinityWave: routes layer=%d home=%d ids=%d..%d owners=%zu,%zu,%zu,%zu invalid=%zu weight-sum=%.8g weight-max=%.8g\n",
                            cells[cell].layer, cells[cell].home_device, id_min, id_max,
                            owner_counts[0], owner_counts[1], owner_counts[2], owner_counts[3],
                            invalid, weight_sum, weight_max);
                }
                if (dump_routes) {
                    struct route_record_header {
                        uint32_t magic;
                        int32_t layer;
                        int32_t home;
                        int32_t tokens;
                        int32_t routes;
                    };
                    const route_record_header header = {
                        0x41575254u,
                        cells[cell].layer,
                        cells[cell].home_device,
                        cells[cell].tokens,
                        8,
                    };
                    const std::ios::openmode mode = std::ios::out |
                            std::ios::binary |
                            (live_call == 0 && cell == 0 ? std::ios::trunc : std::ios::app);
                    std::ofstream output(route_dump, mode);
                    if (!output) {
                        throw std::runtime_error("cannot open AffinityWave route dump");
                    }
                    output.write((const char *) &header, sizeof(header));
                    output.write((const char *) ids.data(), ids.size()*sizeof(int32_t));
                    if (!output) {
                        throw std::runtime_error("cannot write AffinityWave route dump");
                    }
                }
            }
        }
        if (pairwave_service) {
            std::array<cudaStream_t, AW_GPU_COUNT>
                    pair_streams{};
            for (int device = 0;
                    device < AW_GPU_COUNT; ++device) {
                pair_streams[device] =
                        (cudaStream_t) streams_ptr[device];
            }
            aw_live_pairwave_panel(
                    pair_streams, cells, n_cells,
                    *pairwave_manifest, live_call,
                    debug_sync, nccl_order_sum,
                    pairfold_local);
            return 0;
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
        if (aw_env_on(getenv("GGML_CUDA_AW_HEADFOLD"))) {
            if (n_cells != AW_GPU_COUNT) {
                throw std::runtime_error(
                        "HeadFold service requires four same-layer cells");
            }
            grouped_cells = n_cells;
            group_offsets[++n_groups] = grouped_cells;
        } else {
            const char * group_pattern =
                    getenv("GGML_CUDA_AW_GROUP_PATTERN");
            if (group_pattern != nullptr &&
                    group_pattern[0] != '\0') {
                for (const char * p = group_pattern;
                        *p != '\0' && grouped_cells < n_cells; ++p) {
                    if (*p < '1' || *p > '4' ||
                            n_groups >= AW_GPU_COUNT) {
                        throw std::runtime_error(
                                "GGML_CUDA_AW_GROUP_PATTERN must contain one to four digits from 1 to 4");
                    }
                    const int width =
                            std::min(*p - '0',
                                    n_cells - grouped_cells);
                    grouped_cells += width;
                    group_offsets[++n_groups] = grouped_cells;
                }
            }
            while (grouped_cells < n_cells) {
                if (n_groups >= AW_GPU_COUNT) {
                    throw std::runtime_error(
                            "GGML_CUDA_AW_GROUP_PATTERN does not cover the active cells");
                }
                grouped_cells += std::min(
                        group_cells_limit,
                        n_cells - grouped_cells);
                group_offsets[++n_groups] = grouped_cells;
            }
        }
        auto group_begin = [&](int group) { return group_offsets[group]; };
        auto group_size = [&](int group) { return group_offsets[group + 1] - group_offsets[group]; };
        if (diagonal_panel) {
            if (n_groups != n_cells) {
                throw std::runtime_error(
                        "panelized diagonal service requires one cell per group");
            }
            for (int group = 0; group < n_groups; ++group) {
                if (group_begin(group) != group ||
                        group_size(group) != 1) {
                    throw std::runtime_error(
                            "panelized diagonal service requires GGML_CUDA_AW_GROUP_PATTERN=1111");
                }
            }
        }
        std::array<int, AW_GPU_COUNT> group_max_tokens{};
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                group_max_tokens[group] = std::max(group_max_tokens[group], cells[cell_begin + group_cell].tokens);
            }
        }
        std::array<cudaStream_t, AW_GPU_COUNT> streams;
        std::array<size_t, AW_GPU_COUNT>
                diagonal_memory_base{};
        if (diagonal_panel && live_call == 0) {
            for (int device = 0; device < AW_GPU_COUNT; ++device) {
                aw_cuda_throw(cudaSetDevice(device),
                        "set diagonal memory baseline device");
                size_t total_bytes = 0;
                aw_cuda_throw(cudaMemGetInfo(
                            &diagonal_memory_base[device],
                            &total_bytes),
                        "query diagonal memory baseline");
            }
        }
        for (int device = 0; device < AW_GPU_COUNT; ++device) {
            streams[device] = (cudaStream_t) streams_ptr[device];
            (void) aw_copy_streams.get(device);
            if (ggml_cuda_affinity_wave_trace_enabled()) {
                char name[24];
                snprintf(name, sizeof(name), "aw-main/d%d", device);
                ggml_cuda_affinity_wave_trace_name_stream(streams[device], name);
            }
            for (int group = 0; group < n_groups; ++group) {
                aw_live_states[group][device].ensure(
                        device, group_size(group), max_tokens,
                        n256_service, bf16_partial,
                        selective_service,
                        diagonal_panel);
            }
            if (diagonal_panel) {
                aw_diagonal_scratch_states[device].ensure(
                        device, max_tokens, diagonal_panel_n);
            }
        }
        if (diagonal_panel && live_call == 0) {
            for (int device = 0; device < AW_GPU_COUNT; ++device) {
                aw_cuda_throw(cudaSetDevice(device),
                        "set diagonal memory sample device");
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                aw_cuda_throw(cudaMemGetInfo(
                            &free_bytes, &total_bytes),
                        "query diagonal memory sample");
                fprintf(stderr,
                        "AffinityWave: diagonal panel%d allocation device=%d bytes=%zu\n",
                        diagonal_panel_n, device,
                        diagonal_memory_base[device] - free_bytes);
            }
        }
        if (live_call == 0) {
            fprintf(stderr,
                    "AffinityWave: live service scratch ready max_tokens=%d service=%s\n",
                    max_tokens, n256_service ? "n256" : "legacy");
            if (r44_memory) {
                for (int device = 0; device < AW_GPU_COUNT;
                        ++device) {
                    aw_cuda_throw(cudaSetDevice(device),
                            "set R44 memory reserve device");
                    size_t free_bytes = 0;
                    size_t total_bytes = 0;
                    aw_cuda_throw(cudaMemGetInfo(
                                &free_bytes, &total_bytes),
                            "query R44 memory reserve");
                    fprintf(stderr,
                            "AffinityWave: R44 memory reserve device=%d bytes=%zu\n",
                            device,
                            r44_memory_base[device] -
                                free_bytes);
                }
            }
        }

        aw_live_weight_table weight_ptrs{};
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

        constexpr int R44_DESCRIPTORS =
                AW_PRIMARY_PER_GPU + AW_R44_BUDGET;
        const int selective_descriptors = r44_service ?
                R44_DESCRIPTORS : group_pool ?
                AW_PRIMARY_PER_GPU :
                AW_GPU_COUNT*AW_PRIMARY_PER_GPU;
        std::array<int32_t, AW_GPU_COUNT*AW_EXPERTS>
                r44_lookup{};
        std::array<std::array<int32_t, R44_DESCRIPTORS>,
                AW_GPU_COUNT> selective_desc_experts{};
        std::array<std::array<const char *,
                R44_DESCRIPTORS*3>, AW_GPU_COUNT>
                selective_weight_ptrs{};
        std::array<const float *, AW_GPU_COUNT>
                selective_inputs{};
        if (selective_service) {
            if (n_groups != 1 || group_size(0) != AW_GPU_COUNT) {
                throw std::runtime_error(
                        "selective service requires one four-cell layer group");
            }
            const int layer = cells[0].layer;
            for (int cell = 1; cell < n_cells; ++cell) {
                if (cells[cell].layer != layer) {
                    throw std::runtime_error(
                            "selective service cells must share one layer");
                }
            }
            for (int cell = 0; cell < n_cells; ++cell) {
                const int home = cells[cell].home_device;
                if (home < 0 || home >= AW_GPU_COUNT ||
                        selective_inputs[home] != nullptr) {
                    throw std::runtime_error(
                            "selective service requires one input per home");
                }
                selective_inputs[home] = cells[cell].input;
            }
            constexpr size_t expert_bytes =
                    (size_t) AW_EXPERT_FF*
                    (AW_EMBD/AW_KSTAGE)*AW_Q8_BLOCK_BYTES;
            for (int device = 0;
                    device < AW_GPU_COUNT; ++device) {
                const int logical =
                        group_service ?
                        group_placement.logical_for_physical[
                            layer][device] : device;
                for (int desc = 0;
                        desc < AW_PRIMARY_PER_GPU; ++desc) {
                    const int expert =
                            logical*AW_PRIMARY_PER_GPU + desc;
                    selective_desc_experts[device][desc] =
                            expert;
                    for (int projection = 0;
                            projection < 3; ++projection) {
                        selective_weight_ptrs[device][desc*3 +
                                projection] =
                                weight_ptrs[0][device][projection] +
                                (size_t) desc*expert_bytes;
                    }
                }
            }
            if (r44_service) {
                r44_lookup.fill(-1);
                const aw_r44_placement & placement =
                        aw_r44_get_placement();
                for (int device = 0;
                        device < AW_GPU_COUNT; ++device) {
                    aw_r44_slot & cache =
                            aw_r44_states[device].slots[
                                layer % aw_r44_slot_count()];
                    if (cache.layer != layer ||
                            cache.weights.size() <
                                (size_t) AW_R44_BUDGET*3*
                                    expert_bytes) {
                        throw std::runtime_error(
                                "R44 layer cache was not prefetched");
                    }
                for (int slot = 0; slot < AW_R44_BUDGET;
                        ++slot) {
                    const int expert =
                            placement.cache[layer][device][slot];
                    r44_lookup[
                        device*AW_EXPERTS + expert] = slot;
                    const int desc =
                            AW_PRIMARY_PER_GPU + slot;
                    selective_desc_experts[device][desc] =
                            expert;
                    for (int projection = 0;
                            projection < 3; ++projection) {
                        selective_weight_ptrs[device][desc*3 +
                                projection] =
                                (const char *) cache.weights.get() +
                                ((size_t) projection*
                                    AW_R44_BUDGET + slot)*
                                    expert_bytes;
                    }
                }
                }
            }
        }

        if (diagonal_panel) {
            aw_live_diagonal_panel(
                    streams, cells, n_cells, weight_ptrs,
                    group_placement,
                    live_call, debug_sync, nccl_order_sum,
                    diagonal_panel_n);
            if (live_call == 0) {
                fprintf(stderr,
                        "AffinityWave: diagonal panel%d first call complete\n",
                        diagonal_panel_n);
            }
            aw_cuda_throw(cudaSetDevice(0),
                    "restore diagonal scheduler device");
            return 0;
        }

        {
        aw_trace_scope input_scope("service/inputs", 5, (uint64_t) live_call);
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            char trace_name[96];
            snprintf(trace_name, sizeof(trace_name),
                    "service/input/call=%d/group=%d/layers=%d-%d",
                    live_call, group, cells[cell_begin].layer,
                    cells[cell_begin + group_cells - 1].layer);
            aw_trace_scope group_scope(trace_name, 5,
                    ((uint64_t) live_call << 32) | (uint32_t) group);
            const int stride_tokens = group_max_tokens[group];
            const size_t input_cell_stride_bytes = (size_t) stride_tokens*AW_EMBD*wire_element_size;
            const size_t route_cell_stride_bytes = (size_t) stride_tokens*8*sizeof(int32_t);
            for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                const int cell = cell_begin + group_cell;
                const int home = cells[cell].home_device;
                const size_t input_elements = (size_t) cells[cell].tokens*AW_EMBD;
                const size_t input_cell_bytes = input_elements*wire_element_size;
                const size_t route_row_bytes = 8*sizeof(int32_t);
                const size_t route_cell_bytes = (size_t) cells[cell].tokens*route_row_bytes;
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
                if (selective_service) {
                    // Selected rows are gathered after the device placement
                    // is known.
                } else if (bf16_wire) {
                    aw_pack_requests<true><<<aw_grid(input_elements), 256, 0, copy_stream>>>(
                            cells[cell].input, home_input, input_elements);
                    aw_cuda_throw(cudaGetLastError(), "pack BF16 live input");
                } else if (f16_wire) {
                    aw_pack_requests_f16<<<aw_grid(input_elements), 256, 0, copy_stream>>>(
                            cells[cell].input, home_input, input_elements);
                    aw_cuda_throw(cudaGetLastError(), "pack FP16 live input");
                } else {
                    aw_cuda_throw(cudaMemcpyAsync(home_input, cells[cell].input, input_cell_bytes,
                                cudaMemcpyDeviceToDevice, copy_stream), "copy local live input");
                }
                char * home_ids = (char *) home_state.ids.get() +
                        (size_t) group_cell*route_cell_stride_bytes;
                char * home_weights = (char *) home_state.weights.get() +
                        (size_t) group_cell*route_cell_stride_bytes;
                aw_cuda_throw(cudaMemcpy2DAsync(home_ids, route_row_bytes,
                            cells[cell].ids, cells[cell].ids_stride,
                            route_row_bytes, cells[cell].tokens,
                            cudaMemcpyDeviceToDevice, copy_stream), "copy local live ids");
                aw_cuda_throw(cudaMemcpy2DAsync(home_weights, route_row_bytes,
                            cells[cell].weights, cells[cell].weights_stride,
                            route_row_bytes, cells[cell].tokens,
                            cudaMemcpyDeviceToDevice, copy_stream), "copy local live weights");
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
                    if (owner != home) {
                        if (!selective_service) {
                            aw_cuda_throw(cudaMemcpyPeerAsync(input_dst, owner, home_input, home,
                                        input_cell_bytes, copy_stream), "copy peer live input");
                        }
                        aw_cuda_throw(cudaMemcpyPeerAsync(ids_dst, owner, home_ids, home,
                                    route_cell_bytes, copy_stream), "copy peer live ids");
                        aw_cuda_throw(cudaMemcpyPeerAsync(weights_dst, owner, home_weights, home,
                                    route_cell_bytes, copy_stream), "copy peer live weights");
                    }
                }
                aw_cuda_throw(cudaEventRecord(home_state.input_ready, copy_stream),
                        "record live input ready");
                aw_cuda_throw(cudaStreamWaitEvent(streams[home], home_state.input_ready, 0),
                        "preserve live input source");
                sync_live(copy_stream, "synchronize live input copies");
            }
        }
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service inputs queued\n");
        }

        {
        aw_trace_scope compute_scope("service/compute", 6, (uint64_t) live_call);
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const size_t input_cell_stride_bytes = (size_t) stride_tokens*AW_EMBD*wire_element_size;
            const int total_tokens = group_cells*stride_tokens;
            const int total_slots = total_tokens*8;
            const int descriptors = selective_service ?
                    selective_descriptors :
                    group_cells*AW_PRIMARY_PER_GPU;
            const int route_segments_per_cell =
                    (stride_tokens*8 + AW_ROUTE_SEGMENT_SLOTS - 1)/
                    AW_ROUTE_SEGMENT_SLOTS;
            if (selective_service) {
                for (int owner = 0; owner < AW_GPU_COUNT;
                        ++owner) {
                    aw_cuda_throw(cudaSetDevice(owner),
                            "set selective count device");
                    aw_live_state & state =
                            aw_live_states[group][owner];
                    cudaStream_t stream = streams[owner];
                    if (r44_service) {
                        aw_r44_slot & cache =
                                aw_r44_states[owner].slots[
                                    cells[cell_begin].layer %
                                        aw_r44_slot_count()];
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    stream, cache.ready, 0),
                                "wait for R44 weights");
                    }
                    for (int group_cell = 0;
                            group_cell < group_cells; ++group_cell) {
                        const int home =
                                cells[cell_begin +
                                    group_cell].home_device;
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    stream,
                                aw_live_states[group][home].
                                    input_ready,
                                0),
                                "wait for selective input");
                        const int tail_tokens =
                                stride_tokens -
                                cells[cell_begin +
                                    group_cell].tokens;
                        if (tail_tokens > 0) {
                            char * ids_tail =
                                    (char *) state.ids.get() +
                                    ((size_t) group_cell*
                                        stride_tokens +
                                     cells[cell_begin +
                                        group_cell].tokens)*
                                        8*sizeof(int32_t);
                            aw_cuda_throw(cudaMemsetAsync(
                                        ids_tail, 0xff,
                                        (size_t) tail_tokens*8*
                                            sizeof(int32_t),
                                        stream),
                                    "clear selective padded ids");
                        }
                    }
                    const char * const * host_weight_ptrs =
                            r44_service || group_pool ?
                            selective_weight_ptrs[owner].data() :
                            weight_ptrs[group][owner].data();
                    const size_t host_weight_count =
                            r44_service || group_pool ?
                            (size_t) descriptors*3 :
                            (size_t) group_cells*3;
                    aw_cuda_throw(cudaMemcpyAsync(
                                state.weight_ptrs.get(),
                                host_weight_ptrs,
                                host_weight_count*
                                    sizeof(const char *),
                                cudaMemcpyHostToDevice, stream),
                            "copy selective weight pointers");
                    if (r44_service) {
                        aw_cuda_throw(cudaMemcpyAsync(
                                    state.r44_lookup.get(),
                                    r44_lookup.data(),
                                    sizeof(r44_lookup),
                                    cudaMemcpyHostToDevice, stream),
                                "copy R44 lookup");
                    }
                    if (r44_service || group_pool) {
                        aw_cuda_throw(cudaMemcpyAsync(
                                    state.r44_desc_experts.get(),
                                    selective_desc_experts[owner].data(),
                                    (size_t) descriptors*
                                        sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream),
                                "copy selective descriptor experts");
                    }
                    aw_cuda_throw(cudaMemsetAsync(
                                state.counts.get(), 0,
                                (size_t) descriptors*
                                    sizeof(int32_t),
                                stream),
                            "clear selective route counts");
                    if (r44_service) {
                        aw_r44_assign_and_count<<<
                                aw_grid(total_tokens), 256, 0,
                                stream>>>(
                                    (const int32_t *)
                                        state.ids.get(),
                                    (const int32_t *)
                                        state.r44_lookup.get(),
                                    (int32_t *)
                                        state.r44_assignment.get(),
                                    (int32_t *) state.counts.get(),
                                    total_tokens, stride_tokens,
                                    owner);
                    } else {
                        const int layer =
                                cells[cell_begin].layer;
                        const auto & physical =
                                group_placement.
                                    physical_for_logical[layer];
                        const auto & logical =
                                group_placement.
                                    logical_for_physical[layer];
                        aw_group_assign_and_count<<<
                                aw_grid(total_tokens), 256, 0,
                                stream>>>(
                                    (int32_t *) state.ids.get(),
                                    (int32_t *)
                                        state.r44_assignment.get(),
                                    (int32_t *) state.counts.get(),
                                    total_tokens, stride_tokens,
                                    group_pool, group_logical_ids,
                                    owner,
                                    physical[0], physical[1],
                                    physical[2], physical[3],
                                    logical[0], logical[1],
                                    logical[2], logical[3]);
                    }
                }
            }
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                char trace_name[112];
                snprintf(trace_name, sizeof(trace_name),
                        "service/compute/call=%d/group=%d/owner=%d/layers=%d-%d",
                        live_call, group, owner, cells[cell_begin].layer,
                        cells[cell_begin + group_cells - 1].layer);
                aw_trace_scope owner_scope(trace_name, 6,
                        ((uint64_t) live_call << 32) |
                        ((uint64_t) group << 16) | (uint32_t) owner);
                aw_cuda_throw(cudaSetDevice(owner), "set live owner device");
                aw_live_state & state = aw_live_states[group][owner];
                cudaStream_t compute_stream = streams[owner];
                if (!selective_service) {
                    for (int group_cell = 0;
                            group_cell < group_cells; ++group_cell) {
                        const int home =
                                cells[cell_begin +
                                    group_cell].home_device;
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    compute_stream,
                                    aw_live_states[group][home].
                                        input_ready,
                                    0),
                                "wait for live input");
                        const int tail_tokens =
                                stride_tokens -
                                cells[cell_begin +
                                    group_cell].tokens;
                        if (tail_tokens > 0) {
                            char * ids_tail =
                                    (char *) state.ids.get() +
                                    ((size_t) group_cell*
                                        stride_tokens +
                                     cells[cell_begin +
                                        group_cell].tokens)*
                                        8*sizeof(int32_t);
                            aw_cuda_throw(cudaMemsetAsync(
                                        ids_tail, 0xff,
                                        (size_t) tail_tokens*8*
                                            sizeof(int32_t),
                                        compute_stream),
                                    "clear live padded ids");
                        }
                    }
                    aw_cuda_throw(cudaMemcpyAsync(
                                state.weight_ptrs.get(),
                                weight_ptrs[group][owner].data(),
                                (size_t) group_cells*
                                    (shared_service ? 6 : 3)*
                                    sizeof(const char *),
                                cudaMemcpyHostToDevice,
                                compute_stream),
                            "copy live weight pointers");
                    aw_cuda_throw(cudaMemsetAsync(
                                state.counts.get(), 0,
                                (size_t) descriptors*
                                    sizeof(int32_t),
                                compute_stream),
                            "clear live route counts");
                    if (stable256_route_scatter ||
                            radix256_route_scatter ||
                            segment8_route_scatter) {
                        aw_live_count_route_segments<<<
                                group_cells*
                                    route_segments_per_cell,
                                AW_ROUTE_SEGMENT_SLOTS, 0,
                                compute_stream>>>(
                                    (const int32_t *)
                                        state.ids.get(),
                                    (int32_t *)
                                        state.counts.get(),
                                    (int32_t *)
                                        state.route_segment_counts.get(),
                                    group_cells, owner,
                                    stride_tokens,
                                    route_segments_per_cell);
                    } else {
                        aw_live_count_routes<<<
                                aw_grid(total_slots), 256, 0,
                                compute_stream>>>(
                                    (const int32_t *)
                                        state.ids.get(),
                                    (int32_t *)
                                        state.counts.get(),
                                    total_slots, owner,
                                    stride_tokens);
                    }
                }
                const void * plan_input = f16_wire ?
                        (const void *) ((uintptr_t) state.input.get() | 1) : state.input.get();
                if (selective_service &&
                        (r44_service || group_pool)) {
                    aw_r44_build_plan<<<1, 1, 0,
                            compute_stream>>>(
                                (const int32_t *) state.counts.get(),
                                (const int32_t *)
                                    state.r44_desc_experts.get(),
                                (int32_t *) state.offsets.get(),
                                (int32_t *) state.tile_counts.get(),
                                (aw_tile_desc *)
                                    state.tiles_m64.get(),
                                (aw_tile_desc *)
                                    state.tiles_m32.get(),
                                (aw_tile_desc *)
                                    state.tiles_m16.get(),
                                (aw_tile_desc *)
                                    state.tiles_warp.get(),
                                (aw_tile_desc *)
                                    state.tiles_m8.get(),
                                (aw_tile_desc *)
                                    state.tiles_m4.get(),
                                (aw_work_desc *)
                                    state.gate_desc.get(),
                                (aw_work_desc *)
                                    state.up_desc.get(),
                                (aw_work_desc *)
                                    state.down_desc.get(),
                                (const char * const *)
                                    state.weight_ptrs.get(),
                                plan_input,
                                (int32_t *)
                                    state.route_input.get(),
                                (float *) state.gate.get(),
                                (float *) state.up.get(),
                                (float *) state.middle.get(),
                                (float *) state.gate.get(),
                                AW_SERVICE_PANEL_N,
                                AW_SERVICE_PANEL_N,
                                descriptors);
                } else {
                    aw_live_build_plan<<<1, 1, 0, compute_stream>>>(
                            (const int32_t *) state.counts.get(),
                            (int32_t *) state.offsets.get(), (int32_t *) state.cursors.get(),
                            (int32_t *) state.tile_counts.get(),
                            (aw_tile_desc *) state.tiles_m64.get(),
                            (aw_tile_desc *) state.tiles_m32.get(),
                            (aw_tile_desc *) state.tiles_m16.get(),
                            (aw_tile_desc *) state.tiles_warp.get(),
                            (aw_tile_desc *) state.tiles_m8.get(),
                            (aw_tile_desc *) state.tiles_m4.get(),
                            (aw_work_desc *) state.gate_desc.get(),
                            (aw_work_desc *) state.up_desc.get(),
                            (aw_work_desc *) state.down_desc.get(),
                            (const char * const *) state.weight_ptrs.get(),
                            plan_input,
                            (int32_t *) state.route_input.get(),
                            (float *) state.gate.get(), (float *) state.up.get(),
                            (float *) state.middle.get(),
                            selective_service ?
                                (float *) state.gate.get() :
                                (float *) state.route_output.get(),
                            n256_service ? AW_SERVICE_PANEL_N : AW_EXPERT_FF,
                            n256_service ? AW_SERVICE_PANEL_N : AW_EMBD,
                            descriptors);
                }
                if (n256_service) {
                    const size_t gate_up_weight_panel_bytes =
                            (size_t) (AW_SERVICE_PANEL_N/AW_T64_ROWS)*
                            (AW_EMBD/AW_KSTAGE)*AW_T64_STAGE_BYTES;
                    const size_t down_weight_panel_bytes =
                            (size_t) (AW_SERVICE_PANEL_N/AW_T64_ROWS)*
                            (AW_EXPERT_FF/AW_KSTAGE)*AW_T64_STAGE_BYTES;
                    aw_live_build_panel_descs<<<
                            aw_grid((size_t) descriptors*AW_SERVICE_GU_PANELS),
                            256, 0, compute_stream>>>(
                                (const aw_work_desc *) state.gate_desc.get(),
                                (aw_work_desc *) state.gate_panel_desc.get(),
                                descriptors, AW_SERVICE_GU_PANELS,
                                gate_up_weight_panel_bytes);
                    aw_live_build_panel_descs<<<
                            aw_grid((size_t) descriptors*AW_SERVICE_GU_PANELS),
                            256, 0, compute_stream>>>(
                                (const aw_work_desc *) state.up_desc.get(),
                                (aw_work_desc *) state.up_panel_desc.get(),
                                descriptors, AW_SERVICE_GU_PANELS,
                                gate_up_weight_panel_bytes);
                    aw_live_build_panel_descs<<<
                            aw_grid((size_t) descriptors*AW_SERVICE_DOWN_PANELS),
                            256, 0, compute_stream>>>(
                                (const aw_work_desc *) state.down_desc.get(),
                                (aw_work_desc *) state.down_panel_desc.get(),
                                descriptors, AW_SERVICE_DOWN_PANELS,
                                down_weight_panel_bytes);
                }
                if (selective_service) {
                    if (r44_service || group_pool) {
                        aw_r44_fill_routes<<<
                                descriptors, 256, 0,
                                compute_stream>>>(
                                    (const int32_t *)
                                        state.ids.get(),
                                    (const int32_t *)
                                        state.r44_assignment.get(),
                                    (const int32_t *)
                                        state.offsets.get(),
                                    (const int32_t *)
                                        state.r44_desc_experts.get(),
                                    (int32_t *)
                                        state.route_input.get(),
                                    (int32_t *)
                                        state.token_routes.get(),
                                    descriptors, owner,
                                    total_slots);
                    } else {
                        const int logical =
                                group_placement.
                                    logical_for_physical[
                                        cells[cell_begin].layer][owner];
                        aw_live_fill_routes_deterministic<<<
                                descriptors, 256, 0,
                                compute_stream>>>(
                                    (const int32_t *)
                                        state.ids.get(),
                                    (const int32_t *)
                                        state.offsets.get(),
                                    (int32_t *)
                                        state.route_input.get(),
                                    (int32_t *)
                                        state.token_routes.get(),
                                    descriptors, logical,
                                    stride_tokens);
                    }
                    if (r44_stats) {
                        aw_cuda_throw(cudaMemsetAsync(
                                    state.cursors.get(), 0,
                                    sizeof(int32_t), compute_stream),
                                "clear R44 selected count");
                    }
                    aw_r44_gather_selected_inputs<<<
                            total_tokens, 256, 0, compute_stream>>>(
                                selective_inputs[0],
                                selective_inputs[1],
                                selective_inputs[2],
                                selective_inputs[3],
                                (const int32_t *) state.ids.get(),
                                (const int32_t *)
                                    state.r44_assignment.get(),
                                (float *) state.input.get(),
                                total_tokens, stride_tokens, owner,
                                r44_stats ?
                                    (int32_t *) state.cursors.get() :
                                    nullptr);
                } else if (stable256_route_scatter ||
                        radix256_route_scatter ||
                        segment8_route_scatter) {
                    aw_live_prefix_route_segments<<<
                            group_cells, AW_PRIMARY_PER_GPU, 0,
                            compute_stream>>>(
                                (const int32_t *) state.offsets.get(),
                                (const int32_t *)
                                    state.route_segment_counts.get(),
                                (int32_t *)
                                    state.route_segment_offsets.get(),
                                group_cells, route_segments_per_cell);
                    if (segment8_route_scatter) {
                        aw_live_fill_routes_segment8<<<
                                group_cells*route_segments_per_cell,
                                32, 0, compute_stream>>>(
                                    (const int32_t *) state.ids.get(),
                                    (const int32_t *)
                                        state.route_segment_offsets.get(),
                                    (int32_t *) state.route_input.get(),
                                    (int32_t *) state.token_routes.get(),
                                    group_cells, owner, stride_tokens,
                                    route_segments_per_cell);
                    } else if (radix256_route_scatter) {
                        aw_live_fill_routes_radix256<<<
                                group_cells*route_segments_per_cell,
                                AW_ROUTE_SEGMENT_SLOTS, 0,
                                compute_stream>>>(
                                    (const int32_t *) state.ids.get(),
                                    (const int32_t *)
                                        state.route_segment_offsets.get(),
                                    (int32_t *) state.route_input.get(),
                                    (int32_t *) state.token_routes.get(),
                                    group_cells, owner, stride_tokens,
                                    route_segments_per_cell);
                    } else {
                        aw_live_fill_routes_stable256<<<
                                group_cells*route_segments_per_cell,
                                AW_ROUTE_SEGMENT_SLOTS, 0,
                                compute_stream>>>(
                                    (const int32_t *) state.ids.get(),
                                    (const int32_t *)
                                        state.route_segment_offsets.get(),
                                    (int32_t *) state.route_input.get(),
                                    (int32_t *) state.token_routes.get(),
                                    group_cells, owner, stride_tokens,
                                    route_segments_per_cell);
                    }
                } else if (atomic_route_scatter) {
                    aw_live_fill_routes_atomic<<<
                            aw_grid(total_slots), 256, 0, compute_stream>>>(
                                (const int32_t *) state.ids.get(),
                                (int32_t *) state.cursors.get(),
                                (int32_t *) state.route_input.get(),
                                (int32_t *) state.token_routes.get(),
                                total_slots, owner, stride_tokens);
                } else if (stable8_route_scatter) {
                    aw_live_fill_routes_stable8<<<
                            group_cells, 256, 0, compute_stream>>>(
                                (const int32_t *) state.ids.get(),
                                (const int32_t *) state.offsets.get(),
                                (int32_t *) state.route_input.get(),
                                (int32_t *) state.token_routes.get(),
                                group_cells, owner, stride_tokens);
                } else {
                    aw_live_fill_routes_deterministic<<<
                            descriptors, 256, 0, compute_stream>>>(
                                (const int32_t *) state.ids.get(),
                                (const int32_t *) state.offsets.get(),
                                (int32_t *) state.route_input.get(),
                                (int32_t *) state.token_routes.get(),
                                descriptors, owner, stride_tokens);
                }

                const int sm_count = aw_sm_counts[owner];
                if (sm_count <= 0) {
                    throw std::runtime_error("live owner SM count was not initialized");
                }
                const int persistent_blocks = sm_count*2;
                const int pair_blocks = sm_count*3;
                if (n256_service) {
                    for (int panel = 0; panel < AW_SERVICE_GU_PANELS; ++panel) {
                        const size_t desc_offset = (size_t) panel*descriptors;
                        aw_live_launch_projection(
                                state, state.gate_panel_desc,
                                AW_SERVICE_PANEL_N, AW_EMBD,
                                compute_stream, persistent_blocks,
                                pair_blocks, wire16, desc_offset);
                        sync_live(compute_stream,
                                "synchronize live N256 gate");
                        aw_live_launch_projection(
                                state, state.up_panel_desc,
                                AW_SERVICE_PANEL_N, AW_EMBD,
                                compute_stream, persistent_blocks,
                                pair_blocks, wire16, desc_offset);
                        sync_live(compute_stream,
                                "synchronize live N256 up");
                        aw_live_swiglu_panel<<<
                                aw_grid((size_t) total_slots*
                                    AW_SERVICE_PANEL_N),
                                256, 0, compute_stream>>>(
                                    (const float *) state.gate.get(),
                                    (const float *) state.up.get(),
                                    (float *) state.middle.get(),
                                    (const int32_t *) state.offsets.get() +
                                        descriptors,
                                    AW_SERVICE_PANEL_N, AW_EXPERT_FF,
                                    panel*AW_SERVICE_PANEL_N);
                    }
                    sync_live(compute_stream,
                            "synchronize live N256 swiglu");
                    for (int panel = 0;
                            !selective_service &&
                            panel < AW_SERVICE_DOWN_PANELS; ++panel) {
                        const size_t desc_offset =
                                (size_t) panel*descriptors;
                        aw_live_launch_projection(
                                state, state.down_panel_desc,
                                AW_SERVICE_PANEL_N, AW_EXPERT_FF,
                                compute_stream, persistent_blocks,
                                pair_blocks, false, desc_offset);
                        sync_live(compute_stream,
                                "synchronize live N256 down");
                        if (bf16_partial) {
                            aw_live_owner_reduce_panel<true><<<
                                    aw_grid((size_t) total_tokens*
                                        AW_SERVICE_PANEL_N),
                                    256, 0, compute_stream>>>(
                                        (const float *)
                                            state.route_output.get(),
                                        (const int32_t *) state.ids.get(),
                                        (const int32_t *)
                                            state.token_routes.get(),
                                        (const float *)
                                            state.weights.get(),
                                        state.partial.get(), total_tokens,
                                        owner, AW_SERVICE_PANEL_N,
                                        AW_EMBD,
                                        panel*AW_SERVICE_PANEL_N);
                        } else {
                            aw_live_owner_reduce_panel<false><<<
                                    aw_grid((size_t) total_tokens*
                                        AW_SERVICE_PANEL_N),
                                    256, 0, compute_stream>>>(
                                        (const float *)
                                            state.route_output.get(),
                                        (const int32_t *) state.ids.get(),
                                        (const int32_t *)
                                            state.token_routes.get(),
                                        (const float *)
                                            state.weights.get(),
                                        state.partial.get(), total_tokens,
                                        owner, AW_SERVICE_PANEL_N,
                                        AW_EMBD,
                                        panel*AW_SERVICE_PANEL_N);
                        }
                    }
                } else if (fused_gate_up) {
                    aw_live_launch_gate_up(state, compute_stream, sm_count,
                            persistent_blocks, pair_blocks, wire16);
                    sync_live(compute_stream, "synchronize live fused gate/up");
                } else {
                    aw_live_launch_projection(state, state.gate_desc, AW_EXPERT_FF, AW_EMBD,
                            compute_stream, persistent_blocks, pair_blocks, wire16);
                    sync_live(compute_stream, "synchronize live gate");
                    aw_live_launch_projection(state, state.up_desc, AW_EXPERT_FF, AW_EMBD,
                            compute_stream, persistent_blocks, pair_blocks, wire16);
                    sync_live(compute_stream, "synchronize live up");
                }
                if (!n256_service) {
                    aw_live_swiglu<<<aw_grid((size_t) total_slots*AW_EXPERT_FF), 256, 0, compute_stream>>>(
                            (const float *) state.gate.get(), (const float *) state.up.get(),
                            (float *) state.middle.get(), (const int32_t *) state.offsets.get() + descriptors);
                    sync_live(compute_stream, "synchronize live swiglu");
                    aw_live_launch_projection(state, state.down_desc, AW_EMBD, AW_EXPERT_FF,
                            compute_stream, persistent_blocks, pair_blocks, false);
                    sync_live(compute_stream, "synchronize live down");
                    if (bf16_partial) {
                        aw_live_owner_reduce<true><<<aw_grid((size_t) total_tokens*AW_EMBD), 256, 0, compute_stream>>>(
                                (const float *) state.route_output.get(), (const int32_t *) state.ids.get(),
                                (const int32_t *) state.token_routes.get(), (const float *) state.weights.get(),
                                state.partial.get(), total_tokens, owner);
                    } else {
                        aw_live_owner_reduce<false><<<aw_grid((size_t) total_tokens*AW_EMBD), 256, 0, compute_stream>>>(
                                (const float *) state.route_output.get(), (const int32_t *) state.ids.get(),
                                (const int32_t *) state.token_routes.get(), (const float *) state.weights.get(),
                                state.partial.get(), total_tokens, owner);
                    }
                }
                aw_cuda_throw(cudaGetLastError(), "launch live owner service");
                sync_live(compute_stream, "synchronize live owner reduce");
                if (!selective_service) {
                    aw_cuda_throw(cudaEventRecord(
                                state.compute_done,
                                compute_stream),
                            "record live owner compute done");
                }
                if (shared_service) {
                    for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                        const int cell = cell_begin + group_cell;
                        if (cells[cell].home_device != owner) {
                            continue;
                        }
                        const char * const * shared_weights =
                                (const char * const *) state.weight_ptrs.get() + group_cells*3 + group_cell*3;
                        const void * shared_input = (const char *) state.input.get() +
                                (size_t) group_cell*input_cell_stride_bytes;
                        if (f16_wire) {
                            shared_input = (const void *) ((uintptr_t) shared_input | 1);
                        }
                        aw_live_build_shared_plan<<<1, 1, 0, compute_stream>>>(
                                (int32_t *) state.tile_counts.get(),
                                (aw_tile_desc *) state.tiles_m64.get(),
                                (aw_tile_desc *) state.tiles_m32.get(),
                                (aw_tile_desc *) state.tiles_m16.get(),
                                (aw_tile_desc *) state.tiles_warp.get(),
                                (aw_tile_desc *) state.tiles_m8.get(),
                                (aw_tile_desc *) state.tiles_m4.get(),
                                (aw_work_desc *) state.gate_desc.get(),
                                (aw_work_desc *) state.up_desc.get(),
                                (aw_work_desc *) state.down_desc.get(),
                                shared_weights, shared_input,
                                (float *) state.gate.get(), (float *) state.up.get(),
                                (float *) state.middle.get(), cells[cell].shared_output, cells[cell].tokens);
                        aw_live_launch_projection(state, state.gate_desc, AW_EXPERT_FF, AW_EMBD,
                                compute_stream, persistent_blocks, pair_blocks, wire16);
                        aw_live_launch_projection(state, state.up_desc, AW_EXPERT_FF, AW_EMBD,
                                compute_stream, persistent_blocks, pair_blocks, wire16);
                        aw_swiglu<<<aw_grid((size_t) cells[cell].tokens*AW_EXPERT_FF), 256, 0, compute_stream>>>(
                                (const float *) state.gate.get(), (const float *) state.up.get(),
                                (float *) state.middle.get(), (size_t) cells[cell].tokens*AW_EXPERT_FF);
                        aw_live_launch_projection(state, state.down_desc, AW_EMBD, AW_EXPERT_FF,
                                compute_stream, persistent_blocks, pair_blocks, false);
                        aw_cuda_throw(cudaGetLastError(), "launch live shared expert");
                    }
                }
                if (!selective_service) {
                    aw_cuda_throw(cudaEventRecord(
                                state.scratch_free,
                                compute_stream),
                            "record live compute scratch free");
                }
            }
        }
        }
        if (selective_service) {
            const int group = 0;
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const int total_tokens = group_cells*stride_tokens;
            for (int panel = 0;
                    panel < AW_SERVICE_DOWN_PANELS; ++panel) {
                const size_t desc_offset =
                        (size_t) panel*
                            selective_descriptors;
                for (int physical = 0;
                        physical < AW_GPU_COUNT; ++physical) {
                    aw_cuda_throw(cudaSetDevice(physical),
                            "set R44 down device");
                    aw_live_state & state =
                            aw_live_states[group][physical];
                    cudaStream_t stream = streams[physical];
                    const int sm_count = aw_sm_counts[physical];
                    aw_live_launch_projection(
                            state, state.down_panel_desc,
                            AW_SERVICE_PANEL_N, AW_EXPERT_FF,
                            stream, sm_count*2, sm_count*3,
                            false, desc_offset);
                    if (group_service && !group_pool) {
                        const int logical =
                                group_placement.
                                    logical_for_physical[
                                        cells[cell_begin].layer][physical];
                        aw_group_owner_reduce_panel<<<
                                aw_grid((size_t) total_tokens*
                                    AW_SERVICE_PANEL_N),
                                256, 0, stream>>>(
                                    (const float *)
                                        state.gate.get(),
                                    (const int32_t *)
                                        state.ids.get(),
                                    (const int32_t *)
                                        state.token_routes.get(),
                                    (const float *)
                                        state.weights.get(),
                                    (uint16_t *)
                                        state.partial.get(),
                                    total_tokens, stride_tokens,
                                    physical, logical,
                                    AW_SERVICE_PANEL_N);
                    } else {
                        aw_r44_owner_reduce_panel<<<
                                aw_grid((size_t) total_tokens*
                                    AW_SERVICE_PANEL_N),
                                256, 0, stream>>>(
                                    (const float *)
                                        state.gate.get(),
                                    (const int32_t *)
                                        state.ids.get(),
                                    (const int32_t *)
                                        state.r44_assignment.get(),
                                    (const int32_t *)
                                        state.token_routes.get(),
                                    (const float *)
                                        state.weights.get(),
                                    (uint16_t *)
                                        state.partial.get(),
                                    total_tokens, stride_tokens,
                                    physical,
                                    AW_SERVICE_PANEL_N);
                    }
                    aw_cuda_throw(cudaGetLastError(),
                            "launch R44 down panel");
                    aw_cuda_throw(cudaEventRecord(
                                state.compute_done, stream),
                            "record R44 panel ready");
                }
                for (int home = 0; home < AW_GPU_COUNT;
                        ++home) {
                    aw_cuda_throw(cudaSetDevice(home),
                            "set R44 gather device");
                    cudaStream_t stream = streams[home];
                    for (int physical = 0;
                            physical < AW_GPU_COUNT; ++physical) {
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    stream,
                                    aw_live_states[group][physical].
                                        compute_done,
                                    0),
                                "wait for R44 panel");
                    }
                    const int cell = cell_for_home[home];
                    const int group_cell =
                            cell - cell_begin;
                    aw_r44_sum_owners_peer_panel<<<
                            aw_grid((size_t) cells[cell].tokens*
                                AW_SERVICE_PANEL_N),
                            256, 0, stream>>>(
                                (const uint16_t *)
                                    aw_live_states[group][0].
                                        partial.get(),
                                (const uint16_t *)
                                    aw_live_states[group][1].
                                        partial.get(),
                                (const uint16_t *)
                                    aw_live_states[group][2].
                                        partial.get(),
                                (const uint16_t *)
                                    aw_live_states[group][3].
                                        partial.get(),
                                (const int32_t *)
                                    aw_live_states[group][home].
                                        r44_assignment.get(),
                                cells[cell].output,
                                total_tokens,
                                group_cell*stride_tokens,
                                cells[cell].tokens,
                                stride_tokens,
                                AW_SERVICE_PANEL_N,
                                panel*AW_SERVICE_PANEL_N);
                    aw_cuda_throw(cudaGetLastError(),
                            "launch R44 owner gather");
                    aw_cuda_throw(cudaEventRecord(
                                aw_live_states[group][home].
                                    recv_free,
                                stream),
                            "record R44 panel consumed");
                }
                for (int physical = 0;
                        physical < AW_GPU_COUNT; ++physical) {
                    aw_cuda_throw(cudaSetDevice(physical),
                            "set R44 panel release device");
                    cudaStream_t stream = streams[physical];
                    for (int home = 0; home < AW_GPU_COUNT;
                            ++home) {
                        aw_cuda_throw(cudaStreamWaitEvent(
                                    stream,
                                    aw_live_states[group][home].
                                        recv_free,
                                    0),
                                "wait for R44 panel consumption");
                    }
                }
            }
            for (int device = 0; device < AW_GPU_COUNT;
                    ++device) {
                aw_cuda_throw(cudaSetDevice(device),
                        "set R44 completion device");
                aw_live_state & state =
                        aw_live_states[group][device];
                cudaStream_t stream = streams[device];
                aw_cuda_throw(cudaEventRecord(
                            state.output_ready, stream),
                        "record R44 output ready");
                aw_cuda_throw(cudaEventRecord(
                            state.scratch_free, stream),
                        "record R44 scratch free");
                if (r44_service) {
                    aw_r44_slot & cache =
                            aw_r44_states[device].slots[
                                cells[cell_begin].layer %
                                    aw_r44_slot_count()];
                    aw_cuda_throw(cudaEventRecord(
                                cache.free, stream),
                            "record R44 cache free");
                }
                sync_live(stream,
                        "synchronize R44 service");
                if (r44_stats) {
                    int32_t route_rows = 0;
                    int32_t selected_tokens = 0;
                    aw_cuda_throw(cudaMemcpy(
                                &route_rows,
                                (const int32_t *)
                                    state.offsets.get() +
                                    selective_descriptors,
                                sizeof(route_rows),
                                cudaMemcpyDeviceToHost),
                            "copy R44 route-row count");
                    aw_cuda_throw(cudaMemcpy(
                                &selected_tokens,
                                state.cursors.get(),
                                sizeof(selected_tokens),
                                cudaMemcpyDeviceToHost),
                            "copy R44 selected-token count");
                    fprintf(stderr,
                            "AffinityWave: R44 stats layer=%d device=%d routes=%d selected=%d\n",
                            cells[cell_begin].layer, device,
                            route_rows, selected_tokens);
                }
            }
            const char * service_dump =
                    getenv("GGML_CUDA_AW_SERVICE_DUMP");
            const char * service_dump_layer_env =
                    getenv("GGML_CUDA_AW_SERVICE_DUMP_LAYER");
            const int service_dump_layer =
                    service_dump_layer_env != nullptr ?
                    atoi(service_dump_layer_env) : 0;
            if (service_dump != nullptr &&
                    service_dump[0] != '\0' &&
                    (service_dump_layer < 0 ||
                     cells[cell_begin].layer ==
                        service_dump_layer)) {
                for (int home = 0; home < AW_GPU_COUNT;
                        ++home) {
                    const int cell = cell_for_home[home];
                    const size_t count =
                            (size_t) cells[cell].tokens*
                                AW_EMBD;
                    aw_cuda_throw(cudaSetDevice(home),
                            "set R44 dump device");
                    aw_cuda_throw(cudaStreamSynchronize(
                                streams[home]),
                            "synchronize R44 dump");
                    std::vector<float> values(count);
                    aw_cuda_throw(cudaMemcpy(
                                values.data(),
                                cells[cell].output,
                                count*sizeof(float),
                                cudaMemcpyDeviceToHost),
                            "copy R44 reduced output");
                    aw_live_dump(service_dump,
                            "reduced-output-f32",
                            cells[cell].layer, home, -1,
                            values.data(), count,
                            sizeof(float));
                }
            }
            if (live_call == 0) {
                fprintf(stderr,
                        "AffinityWave: %s %s N256 service queued descriptors=%d\n",
                        r44_service ? "R44" : "GroupWave",
                        r44_service || group_pool ? "pooled" : "lane-exact",
                        selective_descriptors);
            }
            if (r44_memory) {
                for (int device = 0; device < AW_GPU_COUNT;
                        ++device) {
                    aw_cuda_throw(cudaSetDevice(device),
                            "set R44 memory sample device");
                    aw_cuda_throw(cudaStreamSynchronize(
                                streams[device]),
                            "synchronize R44 memory sample");
                    size_t free_bytes = 0;
                    size_t total_bytes = 0;
                    aw_cuda_throw(cudaMemGetInfo(
                                &free_bytes, &total_bytes),
                            "query R44 memory sample");
                    if (free_bytes < r44_memory_min[device]) {
                        r44_memory_min[device] = free_bytes;
                        fprintf(stderr,
                                "AffinityWave: R44 memory high-water call=%d layer=%d device=%d bytes=%zu\n",
                                live_call, cells[cell_begin].layer,
                                device,
                                r44_memory_base[device] -
                                    free_bytes);
                    }
                }
            }
            aw_cuda_throw(cudaSetDevice(0),
                    "restore R44 service device");
            return 0;
        }
        if (live_call == 0) {
            fprintf(stderr, "AffinityWave: live service owners queued\n");
        }

        if (aw_env_on(getenv("GGML_CUDA_AW_VERIFY_LOCAL"))) {
            for (int group = 0; group < n_groups; ++group) {
                const int cell_begin = group_begin(group);
                const int group_cells = group_size(group);
                const int stride_tokens = group_max_tokens[group];
                for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                    const int cell = cell_begin + group_cell;
                    const int owner = cells[cell].home_device;
                    if (cells[cell].reference == nullptr) {
                        throw std::runtime_error("local verification reference is missing");
                    }
                    aw_live_state & state = aw_live_states[group][owner];
                    aw_cuda_throw(cudaSetDevice(owner), "set local verification device");
                    aw_cuda_throw(cudaStreamSynchronize(streams[owner]), "synchronize local verification");
                    const size_t count = (size_t) cells[cell].tokens*AW_EMBD;
                    std::vector<uint16_t> observed_bf16;
                    std::vector<float> observed_f32;
                    std::vector<float> reference(count);
                    const size_t partial_offset = (size_t) group_cell*stride_tokens*AW_EMBD;
                    if (bf16_partial) {
                        observed_bf16.resize(count);
                        aw_cuda_throw(cudaMemcpy(observed_bf16.data(),
                                    (const uint16_t *) state.partial.get() + partial_offset,
                                    count*sizeof(uint16_t), cudaMemcpyDeviceToHost),
                                "copy local BF16 verification partial");
                    } else {
                        observed_f32.resize(count);
                        aw_cuda_throw(cudaMemcpy(observed_f32.data(),
                                    (const float *) state.partial.get() + partial_offset,
                                    count*sizeof(float), cudaMemcpyDeviceToHost),
                                "copy local FP32 verification partial");
                    }
                    aw_cuda_throw(cudaMemcpy(reference.data(), cells[cell].reference,
                                count*sizeof(float), cudaMemcpyDeviceToHost),
                            "copy local verification reference");
                    double sum2 = 0.0;
                    double ref2 = 0.0;
                    double observed2 = 0.0;
                    float max_abs = 0.0f;
                    size_t nonfinite = 0;
                    for (size_t i = 0; i < count; ++i) {
                        const float value = bf16_partial ?
                                aw_bf16_to_float_host(observed_bf16[i]) : observed_f32[i];
                        const float diff = value - reference[i];
                        if (!std::isfinite(value) || !std::isfinite(reference[i])) {
                            ++nonfinite;
                            continue;
                        }
                        sum2 += (double) diff*diff;
                        ref2 += (double) reference[i]*reference[i];
                        observed2 += (double) value*value;
                        max_abs = std::max(max_abs, std::abs(diff));
                    }
                    fprintf(stderr,
                            "AffinityWave: local verify layer=%d home=%d tokens=%d rms=%.8g rel=%.8g observed=%.8g reference=%.8g max=%.8g nonfinite=%zu\n",
                            cells[cell].layer, owner, cells[cell].tokens,
                            std::sqrt(sum2/count), std::sqrt(sum2/std::max(ref2, 1.0e-30)),
                            std::sqrt(observed2/count), std::sqrt(ref2/count),
                            max_abs, nonfinite);
                }
            }
        }

        const char * service_dump = getenv("GGML_CUDA_AW_SERVICE_DUMP");
        const char * service_dump_layer_env = getenv("GGML_CUDA_AW_SERVICE_DUMP_LAYER");
        const int service_dump_layer = service_dump_layer_env != nullptr ?
                atoi(service_dump_layer_env) : 0;
        if (bf16_partial && service_dump != nullptr && service_dump[0] != '\0') {
            for (int group = 0; group < n_groups; ++group) {
                const int cell_begin = group_begin(group);
                const int stride_tokens = group_max_tokens[group];
                for (int group_cell = 0; group_cell < group_size(group); ++group_cell) {
                    const int cell = cell_begin + group_cell;
                    if (service_dump_layer >= 0 &&
                            cells[cell].layer !=
                                service_dump_layer) {
                        continue;
                    }
                    const size_t count = (size_t) cells[cell].tokens*AW_EMBD;
                    const size_t offset = (size_t) group_cell*stride_tokens*AW_EMBD;
                    for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                        aw_cuda_throw(cudaSetDevice(owner), "set live dump owner device");
                        aw_cuda_throw(cudaStreamSynchronize(streams[owner]),
                                "synchronize live dump owner partial");
                        std::vector<uint16_t> values(count);
                        aw_cuda_throw(cudaMemcpy(values.data(),
                                    (const uint16_t *) aw_live_states[group][owner].partial.get() + offset,
                                    count*sizeof(uint16_t), cudaMemcpyDeviceToHost),
                                "copy live dump owner partial");
                        aw_live_dump(service_dump, "partial-bf16", cells[cell].layer,
                                cells[cell].home_device, owner, values.data(), count, sizeof(uint16_t));
                    }
                }
            }
        }

        const bool nccl_sum = bf16_partial && aw_env_on(getenv("GGML_CUDA_AW_NCCL_SUM"));
        if (direct_owner && nccl_sum) {
            throw std::runtime_error(
                    "GGML_CUDA_AW_DIRECT_OWNER is incompatible with GGML_CUDA_AW_NCCL_SUM");
        }
        {
        aw_trace_scope output_scope("service/output-and-reduce", 7, (uint64_t) live_call);
        if (nccl_sum) {
#ifdef GGML_USE_NCCL
            std::call_once(aw_live_nccl_once, aw_live_nccl_init);
            for (int group = 0; group < n_groups; ++group) {
                const int cell_begin = group_begin(group);
                const int group_cells = group_size(group);
                char trace_name[96];
                snprintf(trace_name, sizeof(trace_name),
                        "service/nccl-sum/call=%d/group=%d/layers=%d-%d",
                        live_call, group, cells[cell_begin].layer,
                        cells[cell_begin + group_cells - 1].layer);
                aw_trace_scope group_scope(trace_name, 7,
                        ((uint64_t) live_call << 32) | (uint32_t) group);
                const int stride_tokens = group_max_tokens[group];
                const int total_tokens = group_cells*stride_tokens;
                const size_t count = (size_t) total_tokens*AW_EMBD;

                aw_live_nccl_throw(ncclGroupStart(), "start live NCCL sum");
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_cuda_throw(cudaSetDevice(owner), "set live NCCL owner device");
                    aw_live_state & state = aw_live_states[group][owner];
                    aw_cuda_throw(cudaStreamWaitEvent(streams[owner], state.compute_done, 0),
                            "wait for live NCCL owner compute");
                    aw_live_nccl_throw(ncclAllReduce(
                                state.partial.get(), state.partial.get(), count,
                                ncclBfloat16, ncclSum,
                                aw_live_nccl_comms[owner], streams[owner]),
                            "enqueue live NCCL all-reduce");
                }
                aw_live_nccl_throw(ncclGroupEnd(), "finish live NCCL sum");

                for (int group_cell = 0; group_cell < group_cells; ++group_cell) {
                    const int cell = cell_begin + group_cell;
                    const int home = cells[cell].home_device;
                    const size_t cell_count = (size_t) cells[cell].tokens*AW_EMBD;
                    const uint16_t * input =
                            (const uint16_t *) aw_live_states[group][home].partial.get() +
                            (size_t) group_cell*stride_tokens*AW_EMBD;
                    aw_cuda_throw(cudaSetDevice(home), "set live NCCL output device");
                    aw_live_unpack_bf16<<<aw_grid(cell_count), 256, 0, streams[home]>>>(
                            input, cells[cell].output, cell_count);
                    aw_cuda_throw(cudaGetLastError(), "unpack live NCCL sum");
                    sync_live(streams[home], "synchronize live NCCL owner sum");
                    if (service_dump != nullptr &&
                            service_dump[0] != '\0' &&
                            (service_dump_layer < 0 ||
                             cells[cell].layer ==
                                service_dump_layer)) {
                        aw_cuda_throw(cudaStreamSynchronize(streams[home]),
                                "synchronize live dump reduced output");
                        std::vector<uint16_t> values(cell_count);
                        aw_cuda_throw(cudaMemcpy(values.data(), input,
                                    cell_count*sizeof(uint16_t), cudaMemcpyDeviceToHost),
                                "copy live dump reduced output");
                        aw_live_dump(service_dump, "reduced-bf16", cells[cell].layer,
                                home, -1, values.data(), cell_count, sizeof(uint16_t));
                    }
                }
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_cuda_throw(cudaSetDevice(owner), "set live NCCL release device");
                    aw_live_state & state = aw_live_states[group][owner];
                    aw_cuda_throw(cudaEventRecord(state.output_ready, streams[owner]),
                            "record live NCCL output ready");
                    aw_cuda_throw(cudaEventRecord(state.scratch_free, streams[owner]),
                            "record live NCCL scratch free");
                }
            }
#else
            throw std::runtime_error("AffinityWave NCCL sum requested without NCCL support");
#endif
        } else if (direct_owner) {
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const int total_tokens = group_cells*stride_tokens;
            for (int group_cell = 0; group_cell < group_cells;
                    ++group_cell) {
                const int cell = cell_begin + group_cell;
                const int home = cells[cell].home_device;
                char trace_name[112];
                snprintf(trace_name, sizeof(trace_name),
                        "service/direct-owner/call=%d/group=%d/home=%d/layer=%d",
                        live_call, group, home, cells[cell].layer);
                aw_trace_scope home_scope(trace_name, 8,
                        ((uint64_t) live_call << 32) |
                        ((uint64_t) group << 16) |
                        (uint32_t) home);
                aw_cuda_throw(cudaSetDevice(home),
                        "set direct owner home device");
                cudaStream_t direct_stream =
                        aw_copy_streams.get(home);
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_cuda_throw(cudaStreamWaitEvent(
                                direct_stream,
                                aw_live_states[group][owner].compute_done,
                                0),
                            "wait for direct owner partial");
                }
                const size_t work =
                        (size_t) cells[cell].tokens*AW_EMBD/4;
                if (bf16_partial) {
                    if (nccl_order_sum) {
                        aw_live_sum_owners_peer<true, true>
                                <<<aw_grid(work), 256, 0,
                                    direct_stream>>>(
                                        aw_live_states[group][0].partial.get(),
                                        aw_live_states[group][1].partial.get(),
                                        aw_live_states[group][2].partial.get(),
                                        aw_live_states[group][3].partial.get(),
                                        cells[cell].output,
                                        total_tokens,
                                        group_cell*stride_tokens,
                                        cells[cell].tokens);
                    } else {
                        aw_live_sum_owners_peer<true>
                                <<<aw_grid(work), 256, 0,
                                    direct_stream>>>(
                                        aw_live_states[group][0].partial.get(),
                                        aw_live_states[group][1].partial.get(),
                                        aw_live_states[group][2].partial.get(),
                                        aw_live_states[group][3].partial.get(),
                                        cells[cell].output,
                                        total_tokens,
                                        group_cell*stride_tokens,
                                        cells[cell].tokens);
                    }
                } else {
                    aw_live_sum_owners_peer<false>
                            <<<aw_grid(work), 256, 0, direct_stream>>>(
                                    aw_live_states[group][0].partial.get(),
                                    aw_live_states[group][1].partial.get(),
                                    aw_live_states[group][2].partial.get(),
                                    aw_live_states[group][3].partial.get(),
                                    cells[cell].output,
                                    total_tokens,
                                    group_cell*stride_tokens,
                                    cells[cell].tokens);
                }
                aw_cuda_throw(cudaGetLastError(),
                        "launch direct owner sum");
                if (direct_owner_check && cells[cell].layer == 0) {
                    aw_live_state & home_state =
                            aw_live_states[group][home];
                    const size_t token_offset =
                            (size_t) group_cell*stride_tokens;
                    const size_t count =
                            (size_t) cells[cell].tokens*AW_EMBD;
                    const size_t element_size =
                            bf16_partial ?
                            sizeof(uint16_t) : sizeof(float);
                    for (int owner = 0;
                            owner < AW_GPU_COUNT;
                            ++owner) {
                        char * recv_dst =
                                (char *) home_state.recv.get() +
                                ((size_t) owner*total_tokens +
                                 token_offset)*AW_EMBD*element_size;
                        const char * partial_src =
                                (const char *)
                                aw_live_states[group][owner].partial.get() +
                                token_offset*AW_EMBD*element_size;
                        if (owner == home) {
                            aw_cuda_throw(cudaMemcpyAsync(
                                        recv_dst,
                                        partial_src,
                                        count*element_size,
                                        cudaMemcpyDeviceToDevice,
                                        direct_stream),
                                    "copy direct owner check local partial");
                        } else {
                            aw_cuda_throw(cudaMemcpyPeerAsync(
                                        recv_dst,
                                        home,
                                        partial_src,
                                        owner,
                                        count*element_size,
                                        direct_stream),
                                    "copy direct owner check peer partial");
                        }
                    }
                    float * reference =
                            (float *) home_state.route_output.get();
                    if (bf16_partial) {
                        if (nccl_order_sum) {
                            aw_live_sum_owners<true, true>
                                    <<<aw_grid(count), 256, 0,
                                        direct_stream>>>(
                                            home_state.recv.get(),
                                            reference,
                                            total_tokens,
                                            (int) token_offset,
                                            cells[cell].tokens);
                        } else {
                            aw_live_sum_owners<true>
                                    <<<aw_grid(count), 256, 0,
                                        direct_stream>>>(
                                            home_state.recv.get(),
                                            reference,
                                            total_tokens,
                                            (int) token_offset,
                                            cells[cell].tokens);
                        }
                    } else {
                        aw_live_sum_owners<false>
                                <<<aw_grid(count), 256, 0,
                                    direct_stream>>>(
                                        home_state.recv.get(),
                                        reference,
                                        total_tokens,
                                        (int) token_offset,
                                        cells[cell].tokens);
                    }
                    auto * errors =
                            (unsigned long long *)
                            home_state.counts.get();
                    aw_cuda_throw(cudaMemsetAsync(
                                errors, 0,
                                sizeof(unsigned long long),
                                direct_stream),
                            "clear direct owner check");
                    aw_live_compare_f32_bits<<<
                            aw_grid(count), 256, 0,
                            direct_stream>>>(
                                reference,
                                cells[cell].output,
                                count,
                                errors);
                    aw_cuda_throw(cudaGetLastError(),
                            "launch direct owner check");
                    aw_cuda_throw(cudaStreamSynchronize(
                                direct_stream),
                            "synchronize direct owner check");
                    unsigned long long host_errors = 0;
                    aw_cuda_throw(cudaMemcpy(
                                &host_errors,
                                errors,
                                sizeof(host_errors),
                                cudaMemcpyDeviceToHost),
                            "copy direct owner check");
                    if (host_errors != 0) {
                        throw std::runtime_error(
                                "direct owner output is not bitwise exact");
                    }
                }
                aw_cuda_throw(cudaEventRecord(
                            aw_live_states[group][home].recv_free,
                            direct_stream),
                        "record direct owner consumed");
                aw_cuda_throw(cudaStreamWaitEvent(
                            streams[home],
                            aw_live_states[group][home].recv_free,
                            0),
                        "publish direct owner output");
                sync_live(direct_stream,
                        "synchronize direct owner sum");
            }
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                aw_cuda_throw(cudaSetDevice(owner),
                        "set direct owner release device");
                cudaStream_t release_stream =
                        aw_copy_streams.get(owner);
                for (int group_cell = 0;
                        group_cell < group_cells;
                        ++group_cell) {
                    const int home =
                            cells[cell_begin + group_cell].home_device;
                    aw_cuda_throw(cudaStreamWaitEvent(
                                release_stream,
                                aw_live_states[group][home].recv_free,
                                0),
                            "wait for direct owner consumption");
                }
                aw_cuda_throw(cudaEventRecord(
                            aw_live_states[group][owner].output_ready,
                            release_stream),
                        "record direct owner output ready");
            }
        }
        } else {
        for (int group = 0; group < n_groups; ++group) {
            const int cell_begin = group_begin(group);
            const int group_cells = group_size(group);
            const int stride_tokens = group_max_tokens[group];
            const size_t partial_element_size = bf16_partial ? sizeof(uint16_t) : sizeof(float);
            const size_t partial_cell_stride_bytes =
                    (size_t) stride_tokens*AW_EMBD*partial_element_size;
            const int total_tokens = group_cells*stride_tokens;
            for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                char trace_name[112];
                snprintf(trace_name, sizeof(trace_name),
                        "service/output/call=%d/group=%d/owner=%d/layers=%d-%d",
                        live_call, group, owner, cells[cell_begin].layer,
                        cells[cell_begin + group_cells - 1].layer);
                aw_trace_scope owner_scope(trace_name, 7,
                        ((uint64_t) live_call << 32) |
                        ((uint64_t) group << 16) | (uint32_t) owner);
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
                            ((size_t) owner*total_tokens +
                                (size_t) group_cell*stride_tokens)*AW_EMBD*partial_element_size;
                    const size_t partial_cell_bytes =
                            (size_t) cells[cell].tokens*AW_EMBD*partial_element_size;
                    if (owner == home) {
                        aw_cuda_throw(cudaMemcpyAsync(recv_dst, partial_src, partial_cell_bytes,
                                    cudaMemcpyDeviceToDevice, copy_stream), "copy local owner partial");
                    } else {
                        aw_cuda_throw(cudaMemcpyPeerAsync(recv_dst, home, partial_src, owner,
                                    partial_cell_bytes, copy_stream), "copy peer owner partial");
                    }
                }
                aw_cuda_throw(cudaEventRecord(src.output_ready, copy_stream), "record live output ready");
                sync_live(copy_stream, "synchronize live output copies");
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
                char trace_name[96];
                snprintf(trace_name, sizeof(trace_name),
                        "service/reduce/call=%d/group=%d/home=%d/layer=%d",
                        live_call, group, home, cells[cell].layer);
                aw_trace_scope home_scope(trace_name, 8,
                        ((uint64_t) live_call << 32) |
                        ((uint64_t) group << 16) | (uint32_t) home);
                aw_cuda_throw(cudaSetDevice(home), "set live output home device");
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    aw_cuda_throw(cudaStreamWaitEvent(streams[home],
                                aw_live_states[group][owner].output_ready, 0), "wait for live owner output");
                }
                if (bf16_partial) {
                    if (nccl_order_sum) {
                        aw_live_sum_owners<true, true>
                                <<<aw_grid((size_t) cells[cell].tokens*AW_EMBD), 256, 0, streams[home]>>>(
                                        aw_live_states[group][home].recv.get(), cells[cell].output,
                                        total_tokens, group_cell*stride_tokens, cells[cell].tokens);
                    } else {
                        aw_live_sum_owners<true>
                                <<<aw_grid((size_t) cells[cell].tokens*AW_EMBD), 256, 0, streams[home]>>>(
                                        aw_live_states[group][home].recv.get(), cells[cell].output,
                                        total_tokens, group_cell*stride_tokens, cells[cell].tokens);
                    }
                } else {
                    aw_live_sum_owners<false><<<aw_grid((size_t) cells[cell].tokens*AW_EMBD), 256, 0, streams[home]>>>(
                            aw_live_states[group][home].recv.get(), cells[cell].output,
                            total_tokens, group_cell*stride_tokens, cells[cell].tokens);
                }
                aw_cuda_throw(cudaGetLastError(), "launch live owner sum");
                aw_cuda_throw(cudaEventRecord(aw_live_states[group][home].recv_free, streams[home]),
                        "record live receive scratch free");
                sync_live(streams[home], "synchronize live owner sum");
            }
        }
        }
        }
        if (service_dump != nullptr && service_dump[0] != '\0') {
            for (int cell = 0; cell < n_cells; ++cell) {
                if (service_dump_layer >= 0 &&
                        cells[cell].layer !=
                            service_dump_layer) {
                    continue;
                }
                const int home = cells[cell].home_device;
                const size_t count =
                        (size_t) cells[cell].tokens*AW_EMBD;
                aw_cuda_throw(cudaSetDevice(home),
                        "set reduced-output dump device");
                aw_cuda_throw(cudaStreamSynchronize(streams[home]),
                        "synchronize reduced-output dump");
                std::vector<float> values(count);
                aw_cuda_throw(cudaMemcpy(
                            values.data(), cells[cell].output,
                            count*sizeof(float),
                            cudaMemcpyDeviceToHost),
                        "copy reduced-output dump");
                aw_live_dump(service_dump, "reduced-output-f32",
                        cells[cell].layer, home, -1,
                        values.data(), count, sizeof(float));
            }
        }
        if (aw_env_on(getenv("GGML_CUDA_AW_VERIFY_LOCAL"))) {
            for (int cell = 0; cell < n_cells; ++cell) {
                int group = 0;
                while (cell >= group_offsets[group + 1]) {
                    ++group;
                }
                const int group_cell = cell - group_begin(group);
                const int stride_tokens = group_max_tokens[group];
                const int total_tokens = group_size(group)*stride_tokens;
                const int home = cells[cell].home_device;
                aw_cuda_throw(cudaSetDevice(home), "set service verification device");
                aw_cuda_throw(cudaStreamSynchronize(streams[home]), "synchronize service verification");
                const size_t count = (size_t) cells[cell].tokens*AW_EMBD;
                std::vector<float> observed(count);
                std::vector<float> reference(count);
                std::vector<float> direct(count, 0.0f);
                std::vector<float> received(count, 0.0f);
                aw_cuda_throw(cudaMemcpy(observed.data(), cells[cell].output,
                            count*sizeof(float), cudaMemcpyDeviceToHost),
                        "copy service verification output");
                aw_cuda_throw(cudaMemcpy(reference.data(), cells[cell].reference,
                            count*sizeof(float), cudaMemcpyDeviceToHost),
                        "copy service verification reference");
                for (int owner = 0; owner < AW_GPU_COUNT; ++owner) {
                    if (bf16_partial) {
                        std::vector<uint16_t> values(count);
                        aw_cuda_throw(cudaSetDevice(owner), "set direct verification device");
                        aw_cuda_throw(cudaMemcpy(values.data(),
                                    (const uint16_t *) aw_live_states[group][owner].partial.get() +
                                        (size_t) group_cell*stride_tokens*AW_EMBD,
                                    count*sizeof(uint16_t), cudaMemcpyDeviceToHost),
                                "copy direct BF16 verification partial");
                        for (size_t i = 0; i < count; ++i) {
                            direct[i] += aw_bf16_to_float_host(values[i]);
                        }
                        aw_cuda_throw(cudaSetDevice(home), "restore receive verification device");
                        aw_cuda_throw(cudaMemcpy(values.data(),
                                    (const uint16_t *) aw_live_states[group][home].recv.get() +
                                        ((size_t) owner*total_tokens +
                                            (size_t) group_cell*stride_tokens)*AW_EMBD,
                                    count*sizeof(uint16_t), cudaMemcpyDeviceToHost),
                                "copy received BF16 verification partial");
                        for (size_t i = 0; i < count; ++i) {
                            received[i] += aw_bf16_to_float_host(values[i]);
                        }
                    } else {
                        std::vector<float> values(count);
                        aw_cuda_throw(cudaSetDevice(owner), "set direct verification device");
                        aw_cuda_throw(cudaMemcpy(values.data(),
                                    (const float *) aw_live_states[group][owner].partial.get() +
                                        (size_t) group_cell*stride_tokens*AW_EMBD,
                                    count*sizeof(float), cudaMemcpyDeviceToHost),
                                "copy direct FP32 verification partial");
                        for (size_t i = 0; i < count; ++i) {
                            direct[i] += values[i];
                        }
                        aw_cuda_throw(cudaSetDevice(home), "restore receive verification device");
                        aw_cuda_throw(cudaMemcpy(values.data(),
                                    (const float *) aw_live_states[group][home].recv.get() +
                                        ((size_t) owner*total_tokens +
                                            (size_t) group_cell*stride_tokens)*AW_EMBD,
                                    count*sizeof(float), cudaMemcpyDeviceToHost),
                                "copy received FP32 verification partial");
                        for (size_t i = 0; i < count; ++i) {
                            received[i] += values[i];
                        }
                    }
                }
                double sum2 = 0.0;
                double direct2 = 0.0;
                double received2 = 0.0;
                double output_received2 = 0.0;
                double ref2 = 0.0;
                float max_abs = 0.0f;
                size_t nonfinite = 0;
                for (size_t i = 0; i < count; ++i) {
                    const float diff = observed[i] - reference[i];
                    if (!std::isfinite(observed[i]) || !std::isfinite(reference[i])) {
                        ++nonfinite;
                        continue;
                    }
                    sum2 += (double) diff*diff;
                    direct2 += (double) (direct[i] - reference[i])*(direct[i] - reference[i]);
                    received2 += (double) (received[i] - reference[i])*(received[i] - reference[i]);
                    output_received2 += (double) (observed[i] - received[i])*(observed[i] - received[i]);
                    ref2 += (double) reference[i]*reference[i];
                    max_abs = std::max(max_abs, std::abs(diff));
                }
                fprintf(stderr,
                        "AffinityWave: service verify layer=%d home=%d tokens=%d rel=%.8g direct=%.8g recv=%.8g out-recv=%.8g max=%.8g nonfinite=%zu\n",
                        cells[cell].layer, home, cells[cell].tokens,
                        std::sqrt(sum2/std::max(ref2, 1.0e-30)),
                        std::sqrt(direct2/std::max(ref2, 1.0e-30)),
                        std::sqrt(received2/std::max(ref2, 1.0e-30)),
                        std::sqrt(output_received2/std::max(ref2, 1.0e-30)), max_abs, nonfinite);
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

int ggml_cuda_affinity_wave_live_service(
        void * const * streams_ptr,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity) {
    return aw_live_service(streams_ptr, cells, n_cells,
            error, error_capacity, false);
}

int ggml_cuda_affinity_wave_pairfold_service(
        void * const * streams_ptr,
        const ggml_cuda_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity) {
    return aw_live_service(streams_ptr, cells, n_cells,
            error, error_capacity, true);
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
    const char * pairwave_env =
            getenv("GGML_CUDA_AW_PAIRWAVE_MANIFEST");
    const bool pairwave =
            pairwave_env != nullptr &&
            pairwave_env[0] != '\0';
    if (count == 1 || count % (AW_LAYERS*3) == 0 ||
            (pairwave && count == (AW_LAYERS/2)*3)) {
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
    const char * r44 = aw_r44_mode();
    if (strcmp(r44, "0") != 0 &&
            strcmp(r44, "prefetch") != 0 &&
            strcmp(r44, "service") != 0) {
        GGML_ABORT(
                "GGML_CUDA_AW_R44 must be '0', 'prefetch', or 'service'");
    }
    if (aw_r44_enabled()) {
        try {
            (void) aw_r44_get_placement();
        } catch (const std::exception & exception) {
            GGML_ABORT("AffinityWave R44 configuration error: %s",
                    exception.what());
        }
    }
    try {
        (void) aw_pairwave_get_manifest();
    } catch (const std::exception & exception) {
        GGML_ABORT("AffinityWave PairWave configuration error: %s",
                exception.what());
    }
    aw_config config;
    std::string error;
    if (!aw_load_config(config, error)) {
        GGML_ABORT("AffinityWave configuration error: %s", error.c_str());
    }
    GGML_LOG_INFO("AffinityWave: expert service enabled; map=%s wire=%s layout=%s kernel=%s engine=%s "
            "down-cache=%d m64-split=%d home=%s check=%d dense-t64=%d qkv-conv=%d\n",
            config.map.c_str(), config.wire.c_str(),
            config.q8_layout.c_str(), config.q8_kernel.c_str(), config.q8_engine.c_str(),
            config.down_cache, config.m64_split, config.home.c_str(), config.check ? 1 : 0,
            config.dense_t64 ? 1 : 0, config.qkv_conv ? 1 : 0);
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
        if (params->device < 0 || params->device >= AW_GPU_COUNT || params->participants < 1 ||
                params->participants > AW_GPU_COUNT || params->device >= params->participants ||
                params->active_cells < 1 ||
                params->active_cells > AW_GPU_COUNT || params->tokens_per_cell < 32 ||
                params->tokens_per_cell % 32 != 0 || params->repeats < 1 ||
                params->route_pattern < 0 || params->route_pattern > 2) {
            throw std::runtime_error("invalid service benchmark dimensions");
        }
        if (params->route_pattern != 0 && params->tokens_per_cell != 2048) {
            throw std::runtime_error("mixed routing currently requires 2048 tokens per cell");
        }
        aw_cuda_throw(cudaSetDevice(params->device), "cudaSetDevice");

        int service_group_cells = config.q8_engine == "rendezvous" ||
                aw_env_on(getenv("GGML_CUDA_AW_BENCH_COALESCE")) ? params->active_cells : 1;
        const char * group_cells_env = getenv("GGML_CUDA_AW_BENCH_GROUP_CELLS");
        if (group_cells_env != nullptr && group_cells_env[0] != '\0') {
            char * end = nullptr;
            const long parsed = strtol(group_cells_env, &end, 10);
            if (end == group_cells_env || *end != '\0' || parsed < 1 ||
                    parsed > params->active_cells || params->active_cells % parsed != 0) {
                throw std::runtime_error(
                        "GGML_CUDA_AW_BENCH_GROUP_CELLS must divide the active cell count");
            }
            service_group_cells = (int) parsed;
        }
        const bool service_coalesce = service_group_cells > 1;
        const int descriptor_groups = params->active_cells/service_group_cells;
        const int descriptors = descriptor_groups*AW_PRIMARY_PER_GPU;
        int service_kpanels = 1;
        const char * kpanels_env = getenv("GGML_CUDA_AW_BENCH_KPANELS");
        if (kpanels_env != nullptr && kpanels_env[0] != '\0') {
            char * end = nullptr;
            const long parsed = strtol(kpanels_env, &end, 10);
            if (end == kpanels_env || *end != '\0' || parsed < 1 ||
                    AW_EMBD/AW_KSTAGE % parsed != 0) {
                throw std::runtime_error(
                        "GGML_CUDA_AW_BENCH_KPANELS must divide the expert K blocks");
            }
            service_kpanels = (int) parsed;
        }
        if (service_kpanels > 1 &&
                (config.q8_engine != "broadwave" || config.q8_layout != "t64k32")) {
            throw std::runtime_error(
                    "K-panel diagnostic requires BroadWave with T64K32 weights");
        }
        const bool panel_broadwave =
                config.q8_engine == "broadwave" ||
                config.q8_engine == "halfpipe_sync";
        int service_gupanels = 1;
        const char * gupanels_env =
                getenv("GGML_CUDA_AW_BENCH_GUPANELS");
        if (gupanels_env != nullptr && gupanels_env[0] != '\0') {
            char * end = nullptr;
            const long parsed = strtol(gupanels_env, &end, 10);
            if (end == gupanels_env || *end != '\0' ||
                    (parsed != 1 && parsed != 2 && parsed != 4)) {
                throw std::runtime_error(
                        "GGML_CUDA_AW_BENCH_GUPANELS must be 1, 2, or 4");
            }
            service_gupanels = (int) parsed;
        }
        if (service_gupanels > 1 &&
                (service_kpanels != 1 ||
                 !panel_broadwave ||
                 config.q8_layout != "t64k32")) {
            throw std::runtime_error(
                    "gate/up panel diagnostic requires unpanelled-K BroadWave-family T64K32");
        }
        int service_npanels = 1;
        const char * npanels_env = getenv("GGML_CUDA_AW_BENCH_NPANELS");
        if (npanels_env != nullptr && npanels_env[0] != '\0') {
            char * end = nullptr;
            const long parsed = strtol(npanels_env, &end, 10);
            if (end == npanels_env || *end != '\0' ||
                    parsed < 1 || parsed > AW_EMBD/(2*AW_T64_ROWS) ||
                    (parsed & (parsed - 1)) != 0) {
                throw std::runtime_error(
                        "GGML_CUDA_AW_BENCH_NPANELS must be 1, 2, 4, 8, or 16");
            }
            service_npanels = (int) parsed;
        }
        if (service_npanels > 1 &&
                (!panel_broadwave ||
                 config.q8_layout != "t64k32" ||
                 config.down_cache != 0)) {
            throw std::runtime_error(
                    "N-panel diagnostic requires uncached BroadWave-family T64K32");
        }
        const bool npanel_copy =
                aw_env_on(getenv("GGML_CUDA_AW_BENCH_NPANEL_COPY"));
        const bool npanel_pair_copy =
                aw_env_on(getenv("GGML_CUDA_AW_BENCH_NPANEL_PAIR_COPY"));
        const bool npanel_any_copy = npanel_copy || npanel_pair_copy;
        if (npanel_copy && npanel_pair_copy) {
            throw std::runtime_error(
                    "N-panel all-owner and pair-copy diagnostics are mutually exclusive");
        }
        if (npanel_any_copy &&
                (service_npanels == 1 ||
                 params->participants != AW_GPU_COUNT ||
                 params->active_cells != AW_GPU_COUNT ||
                 config.wire != "f32")) {
            throw std::runtime_error(
                    "N-panel copy diagnostic requires four F32 participants and multiple panels");
        }
        if (npanel_pair_copy && service_group_cells != 2) {
            throw std::runtime_error(
                    "N-panel pair-copy diagnostic requires two-cell descriptor groups");
        }
        const bool bench_fused_swiglu =
                aw_env_on(getenv("GGML_CUDA_AW_BENCH_FUSED_SWIGLU"));
        if (bench_fused_swiglu &&
                (service_kpanels != 1 ||
                 service_gupanels != 1 ||
                 config.q8_engine != "broadwave" ||
                 config.q8_layout != "t64k32")) {
            throw std::runtime_error(
                    "fused SwiGLU diagnostic requires unpanelled BroadWave T64K32 gate/up");
        }
        const bool relay_copy =
                aw_env_on(getenv("GGML_CUDA_AW_BENCH_KPANEL_COPY"));
        if (relay_copy && (service_kpanels == 1 ||
                params->participants != AW_GPU_COUNT ||
                params->active_cells != AW_GPU_COUNT ||
                config.wire != "f32")) {
            throw std::runtime_error(
                    "K-panel copy diagnostic requires four F32 participants and at least two panels");
        }
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
        const bool f16_wire = config.wire == "f16";
        const bool wire16 = bf16_wire || f16_wire;
        const int wire_format = bf16_wire ? 1 : f16_wire ? 2 : 0;
        const size_t wire_bytes = (size_t) tokens*AW_EMBD*(wire16 ? sizeof(uint16_t) : sizeof(float));
        const size_t narrow_bytes = (size_t) route_rows*AW_EXPERT_FF*sizeof(float);
        const size_t wide_bytes = (size_t) route_rows*AW_EMBD*sizeof(float);
        const int service_gu_panel_n = AW_EXPERT_FF/service_gupanels;
        const size_t gate_up_panel_bytes =
                (size_t) route_rows*service_gu_panel_n*sizeof(float);
        const int service_panel_n = AW_EMBD/service_npanels;
        const size_t route_output_bytes =
                (size_t) route_rows*service_panel_n*sizeof(float);
        const size_t partial_bytes = (size_t) tokens*AW_EMBD*sizeof(uint16_t);
        const size_t npanel_recv_bytes =
                npanel_any_copy ? AW_GPU_COUNT*partial_bytes : 0;
        const int panel_k = AW_EMBD/service_kpanels;
        const size_t relay_local_bytes = relay_copy ?
                (size_t) params->tokens_per_cell*AW_EMBD*sizeof(float) : 0;
        const size_t relay_gathered_bytes = relay_copy ? source_bytes : 0;
        const bool expandwave = config.q8_engine == "expandwave";
        const bool railwave = config.q8_engine == "railwave";
        const bool cohortrail =
                config.q8_engine == "cohortrail";
        const size_t down_f32_bytes =
                (config.down_cache == 4 || expandwave) ?
                (size_t) descriptors*AW_EMBD*AW_EXPERT_FF*sizeof(float) : 0;
        const size_t rail_partial_bytes = railwave ? wide_bytes : 0;

        std::vector<int32_t> route_input(route_rows);
        std::vector<int32_t> token_routes((size_t) tokens*AW_ROUTES_PER_OWNER_TOKEN);
        std::vector<float> route_weights(token_routes.size(), 0.5f);
        std::vector<int32_t> route_groups(route_rows);
        std::vector<int32_t> desc_row0(descriptors), desc_rows(descriptors);
        std::vector<aw_tile_desc> tiles_m128, tiles_m64, tiles_m32, tiles_m16;
        std::vector<aw_tile_desc> tiles_warp, tiles_m8, tiles_m4;
        std::vector<aw_work_desc> gate_desc(descriptors), up_desc(descriptors), down_desc(descriptors);
        std::vector<aw_work_desc> gate_panel_desc(
                service_gupanels > 1 ?
                (size_t) service_gupanels*descriptors : 0);
        std::vector<aw_work_desc> up_panel_desc(
                service_gupanels > 1 ?
                (size_t) service_gupanels*descriptors : 0);
        std::vector<aw_work_desc> down_panel_desc(
                service_npanels > 1 ? (size_t) service_npanels*descriptors : 0);
        std::vector<aw_work_desc> down_desc_f32(expandwave ? descriptors : 0);
        tiles_m128.reserve((route_rows + 127)/128);
        tiles_m64.reserve((route_rows + 63)/64);
        tiles_m32.reserve((route_rows + 31)/32);
        tiles_m16.reserve((route_rows + 15)/16);
        tiles_warp.reserve((route_rows + 15)/16 + descriptors);
        tiles_m8.reserve(2*descriptors);
        tiles_m4.reserve(descriptors);

        std::vector<int> route_expert((size_t) params->tokens_per_cell*AW_ROUTES_PER_OWNER_TOKEN);
        if (params->route_pattern == 0) {
            for (size_t route = 0; route < route_expert.size(); ++route) {
                route_expert[route] = (int) (route % AW_PRIMARY_PER_GPU);
            }
        } else if (params->route_pattern == 1) {
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
        } else {
            constexpr std::array<int, 8> edge_counts = { 1, 16, 17, 31, 32, 48, 63, 64 };
            size_t route = 0;
            for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
                const int count = expert < (int) edge_counts.size() ? edge_counts[expert] :
                        68 + (expert < 24 ? 1 : 0);
                for (int i = 0; i < count; ++i) {
                    route_expert[route++] = expert;
                }
            }
            if (route != route_expert.size()) {
                throw std::runtime_error("internal edge-mix routing count mismatch");
            }
        }

        // Expert-contiguous route rows, but one owner request per token.  The
        // descriptor-local input_rows vector is the sparse expert gather.
        int route_cursor = 0;
        auto build_descriptor = [&](int desc, int expert, int cell_begin, int cell_end) {
                const int row0 = route_cursor;
                for (int cell = cell_begin; cell < cell_end; ++cell) {
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
                }
                const int rows = route_cursor - row0;
                desc_row0[desc] = row0;
                desc_rows[desc] = rows;
                int row = 0;
                if (config.q8_engine == "crestwave") {
                    while (rows - row >= 96) {
                        const int tile_rows = std::min(128, rows - row);
                        tiles_m128.push_back({ desc, row, tile_rows });
                        std::fill(route_groups.begin() + row0 + row,
                                route_groups.begin() + row0 + row + tile_rows,
                                config.m64_split);
                        row += tile_rows;
                    }
                }
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
                            route_groups.begin() + row0 + rows,
                            config.q8_engine == "tailwave" || config.q8_engine == "widewave" ||
                            config.q8_engine == "flexwave1" ? 1 : 2);
                    int tail_row = row;
                    int remaining = tile_rows;
                    while (remaining > 4) {
                        const int tail_rows = std::min(8, remaining);
                        tiles_m8.push_back({ desc, tail_row, tail_rows });
                        tail_row += tail_rows;
                        remaining -= tail_rows;
                    }
                    if (remaining != 0) {
                        tiles_m4.push_back({ desc, tail_row, remaining });
                    }
                }
                for (int warp_row = 0; warp_row < rows; warp_row += 16) {
                    tiles_warp.push_back({ desc, warp_row, std::min(16, rows - warp_row) });
                }
        };
        if (service_coalesce) {
            for (int group = 0; group < descriptor_groups; ++group) {
                const int cell_begin = group*service_group_cells;
                const int cell_end = cell_begin + service_group_cells;
                for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
                    const int desc = group*AW_PRIMARY_PER_GPU + expert;
                    build_descriptor(desc, expert, cell_begin, cell_end);
                }
            }
        } else {
            for (int cell = 0; cell < params->active_cells; ++cell) {
                for (int expert = 0; expert < AW_PRIMARY_PER_GPU; ++expert) {
                    const int desc = cell*AW_PRIMARY_PER_GPU + expert;
                    build_descriptor(desc, expert, cell, cell + 1);
                }
            }
        }
        if (route_cursor != route_rows) {
            throw std::runtime_error("internal route count mismatch");
        }
        if (config.q8_engine == "warpwave") {
            std::fill(route_groups.begin(), route_groups.end(), 1);
        } else if (config.q8_engine == "compact") {
            std::fill(route_groups.begin(), route_groups.end(), 2);
        }
        std::vector<int32_t> route_down_groups = route_groups;
        if (config.down_cache == 4) {
            std::fill(route_down_groups.begin(), route_down_groups.end(), config.m64_split);
        }

        size_t free_before = 0, total_bytes = 0;
        aw_cuda_throw(cudaMemGetInfo(&free_before, &total_bytes), "cudaMemGetInfo(before)");
        const size_t expected_alloc = source_bytes + wire_bytes +
                2*gate_up_panel_bytes + narrow_bytes +
                route_output_bytes + partial_bytes +
                (service_kpanels > 1 ? 4*narrow_bytes : 0) +
                relay_local_bytes + relay_gathered_bytes + rail_partial_bytes +
                npanel_recv_bytes +
                (size_t) descriptors*(2*q8_gate_bytes + q8_down_bytes) +
                down_f32_bytes +
                (config.q8_layout == "t64k32" ? (size_t) descriptors*q8_gate_bytes : 0) +
                route_input.size()*sizeof(int32_t) + token_routes.size()*sizeof(int32_t) +
                route_weights.size()*sizeof(float) +
                (tiles_m128.size() + tiles_m64.size() + tiles_m32.size() + tiles_m16.size() +
                        tiles_warp.size() + tiles_m8.size() + tiles_m4.size())*sizeof(aw_tile_desc) +
                (cohortrail ?
                        4*sizeof(int32_t) +
                        tiles_m16.size()*
                            (3*sizeof(aw_m16_cohort) +
                             sizeof(aw_tile_desc)) :
                        0) +
                (size_t) descriptors*(expandwave ? 4 : 3)*sizeof(aw_work_desc) +
                gate_panel_desc.size()*sizeof(aw_work_desc) +
                up_panel_desc.size()*sizeof(aw_work_desc) +
                down_panel_desc.size()*sizeof(aw_work_desc);
        const size_t required_headroom = 768ull*1024ull*1024ull;
        if (free_before < expected_alloc + required_headroom) {
            throw std::runtime_error("insufficient device memory for service prototype plus 768 MiB headroom");
        }

        aw_device_buffer source(source_bytes);
        aw_device_buffer wire(wire_bytes);
        aw_device_buffer gate(gate_up_panel_bytes);
        aw_device_buffer up(gate_up_panel_bytes);
        aw_device_buffer middle(narrow_bytes);
        aw_device_buffer route_output(route_output_bytes);
        aw_device_buffer partial(partial_bytes);
        aw_device_buffer npanel_recv(npanel_recv_bytes);
        aw_device_buffer gate_kpartial(service_kpanels > 1 ? 2*narrow_bytes : 0);
        aw_device_buffer up_kpartial(service_kpanels > 1 ? 2*narrow_bytes : 0);
        aw_device_buffer relay_local(relay_local_bytes);
        aw_device_buffer relay_gathered(relay_gathered_bytes);
        aw_device_buffer rail_partial(rail_partial_bytes);
        aw_device_buffer weights_gate((size_t) descriptors*q8_gate_bytes);
        aw_device_buffer weights_up((size_t) descriptors*q8_gate_bytes);
        aw_device_buffer weights_down((size_t) descriptors*q8_down_bytes);
        aw_device_buffer weights_down_f32(down_f32_bytes);
        aw_device_buffer layout_scratch(config.q8_layout == "t64k32" ?
                (size_t) descriptors*q8_gate_bytes : 0);
        aw_device_buffer route_input_dev(route_input.size()*sizeof(int32_t));
        aw_device_buffer token_routes_dev(token_routes.size()*sizeof(int32_t));
        aw_device_buffer route_weights_dev(route_weights.size()*sizeof(float));
        aw_device_buffer tiles_m128_dev(tiles_m128.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m64_dev(tiles_m64.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m32_dev(tiles_m32.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m16_dev(tiles_m16.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_warp_dev(tiles_warp.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m8_dev(tiles_m8.size()*sizeof(aw_tile_desc));
        aw_device_buffer tiles_m4_dev(tiles_m4.size()*sizeof(aw_tile_desc));
        aw_device_buffer cohort_counts_dev(
                cohortrail ? 4*sizeof(int32_t) : 0);
        aw_device_buffer cohorts_p2_dev(
                cohortrail ?
                tiles_m16.size()*sizeof(aw_m16_cohort) : 0);
        aw_device_buffer cohorts_p3_dev(
                cohortrail ?
                tiles_m16.size()*sizeof(aw_m16_cohort) : 0);
        aw_device_buffer cohorts_p4_dev(
                cohortrail ?
                tiles_m16.size()*sizeof(aw_m16_cohort) : 0);
        aw_device_buffer cohort_singles_dev(
                cohortrail ?
                tiles_m16.size()*sizeof(aw_tile_desc) : 0);
        aw_device_buffer gate_desc_dev((size_t) descriptors*sizeof(aw_work_desc));
        aw_device_buffer up_desc_dev((size_t) descriptors*sizeof(aw_work_desc));
        aw_device_buffer down_desc_dev((size_t) descriptors*sizeof(aw_work_desc));
        aw_device_buffer gate_panel_desc_dev(
                gate_panel_desc.size()*sizeof(aw_work_desc));
        aw_device_buffer up_panel_desc_dev(
                up_panel_desc.size()*sizeof(aw_work_desc));
        aw_device_buffer down_panel_desc_dev(
                down_panel_desc.size()*sizeof(aw_work_desc));
        aw_device_buffer down_desc_f32_dev(expandwave ?
                (size_t) descriptors*sizeof(aw_work_desc) : 0);

        const void * wire_input = f16_wire ?
                (const void *) ((uintptr_t) wire.get() | 1) : wire.get();
        for (int desc = 0; desc < descriptors; ++desc) {
            const int row0 = desc_row0[desc];
            const int rows = desc_rows[desc];
            const int layer = desc/AW_PRIMARY_PER_GPU;
            const int expert = desc % AW_PRIMARY_PER_GPU;
            const int weight_desc =
                    cohortrail ? expert : desc;
            gate_desc[desc] = {
                wire_input, (const char *) weights_gate.get() + (size_t) weight_desc*q8_gate_bytes,
                (float *) gate.get() + (size_t) row0*service_gu_panel_n,
                (const int32_t *) route_input_dev.get() + row0, row0, rows, layer, expert
            };
            up_desc[desc] = {
                wire_input, (const char *) weights_up.get() + (size_t) weight_desc*q8_gate_bytes,
                (float *) up.get() + (size_t) row0*service_gu_panel_n,
                (const int32_t *) route_input_dev.get() + row0, row0, rows, layer, expert
            };
            down_desc[desc] = {
                (const float *) middle.get() + (size_t) row0*AW_EXPERT_FF,
                config.down_cache == 4 ?
                        (const char *) weights_down_f32.get() +
                                (size_t) weight_desc*AW_EMBD*AW_EXPERT_FF*sizeof(float) :
                        (const char *) weights_down.get() + (size_t) weight_desc*q8_down_bytes,
                (float *) route_output.get() + (size_t) row0*AW_EMBD,
                nullptr, row0, rows, layer, expert
            };
            if (expandwave) {
                down_desc_f32[desc] = down_desc[desc];
                down_desc_f32[desc].weight = (const char *) weights_down_f32.get() +
                        (size_t) weight_desc*AW_EMBD*AW_EXPERT_FF*sizeof(float);
            }
        }
        if (service_gupanels > 1) {
            const size_t weight_panel_bytes =
                    (size_t) (service_gu_panel_n/AW_T64_ROWS)*
                    (AW_EMBD/AW_KSTAGE)*AW_T64_STAGE_BYTES;
            for (int panel = 0; panel < service_gupanels; ++panel) {
                for (int desc = 0; desc < descriptors; ++desc) {
                    const int weight_desc = cohortrail ?
                            desc % AW_PRIMARY_PER_GPU : desc;
                    aw_work_desc & gate_panel =
                            gate_panel_desc[(size_t) panel*descriptors + desc];
                    aw_work_desc & up_panel =
                            up_panel_desc[(size_t) panel*descriptors + desc];
                    gate_panel = gate_desc[desc];
                    up_panel = up_desc[desc];
                    gate_panel.weight =
                            (const char *) weights_gate.get() +
                            (size_t) weight_desc*q8_gate_bytes +
                            (size_t) panel*weight_panel_bytes;
                    up_panel.weight =
                            (const char *) weights_up.get() +
                            (size_t) weight_desc*q8_gate_bytes +
                            (size_t) panel*weight_panel_bytes;
                }
            }
        }
        if (service_npanels > 1) {
            const size_t weight_panel_bytes =
                    (size_t) (service_panel_n/AW_T64_ROWS)*
                    (AW_EXPERT_FF/AW_KSTAGE)*AW_T64_STAGE_BYTES;
            for (int panel = 0; panel < service_npanels; ++panel) {
                for (int desc = 0; desc < descriptors; ++desc) {
                    const int weight_desc = cohortrail ?
                            desc % AW_PRIMARY_PER_GPU : desc;
                    aw_work_desc & panel_desc =
                            down_panel_desc[(size_t) panel*descriptors + desc];
                    panel_desc = down_desc[desc];
                    panel_desc.weight =
                            (const char *) weights_down.get() +
                            (size_t) weight_desc*q8_down_bytes +
                            (size_t) panel*weight_panel_bytes;
                    panel_desc.output =
                            (float *) route_output.get() +
                            (size_t) desc_row0[desc]*service_panel_n;
                }
            }
        }

        cudaStream_t stream = nullptr;
        cudaEvent_t begin = nullptr, end = nullptr;
        std::array<cudaStream_t, AW_GPU_COUNT> relay_streams{};
        aw_cuda_throw(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        aw_cuda_throw(cudaEventCreate(&begin), "cudaEventCreate(begin)");
        aw_cuda_throw(cudaEventCreate(&end), "cudaEventCreate(end)");
        if (relay_copy || npanel_any_copy) {
            for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                aw_cuda_throw(cudaStreamCreateWithFlags(
                            &relay_streams[dst], cudaStreamNonBlocking),
                        "create service copy stream");
            }
        }

        aw_cuda_throw(cudaMemcpyAsync(route_input_dev.get(), route_input.data(), route_input_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy route input");
        aw_cuda_throw(cudaMemcpyAsync(token_routes_dev.get(), token_routes.data(), token_routes_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy token routes");
        aw_cuda_throw(cudaMemcpyAsync(route_weights_dev.get(), route_weights.data(), route_weights_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy route weights");
        if (!tiles_m128.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(
                        tiles_m128_dev.get(), tiles_m128.data(), tiles_m128_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M128 tiles");
        }
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
        if (!tiles_warp.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_warp_dev.get(), tiles_warp.data(), tiles_warp_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy WarpWave tiles");
        }
        if (!tiles_m8.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_m8_dev.get(), tiles_m8.data(), tiles_m8_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M8 tiles");
        }
        if (!tiles_m4.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(tiles_m4_dev.get(), tiles_m4.data(), tiles_m4_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy M4 tiles");
        }
        aw_cuda_throw(cudaMemcpyAsync(gate_desc_dev.get(), gate_desc.data(), gate_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy gate descriptors");
        aw_cuda_throw(cudaMemcpyAsync(up_desc_dev.get(), up_desc.data(), up_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy up descriptors");
        aw_cuda_throw(cudaMemcpyAsync(down_desc_dev.get(), down_desc.data(), down_desc_dev.size(),
                    cudaMemcpyHostToDevice, stream), "copy down descriptors");
        if (!gate_panel_desc.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(
                        gate_panel_desc_dev.get(),
                        gate_panel_desc.data(),
                        gate_panel_desc_dev.size(),
                        cudaMemcpyHostToDevice,
                        stream),
                    "copy gate panel descriptors");
            aw_cuda_throw(cudaMemcpyAsync(
                        up_panel_desc_dev.get(),
                        up_panel_desc.data(),
                        up_panel_desc_dev.size(),
                        cudaMemcpyHostToDevice,
                        stream),
                    "copy up panel descriptors");
        }
        if (!down_panel_desc.empty()) {
            aw_cuda_throw(cudaMemcpyAsync(
                        down_panel_desc_dev.get(),
                        down_panel_desc.data(),
                        down_panel_desc_dev.size(),
                        cudaMemcpyHostToDevice,
                        stream),
                    "copy down panel descriptors");
        }
        if (expandwave) {
            aw_cuda_throw(cudaMemcpyAsync(
                        down_desc_f32_dev.get(), down_desc_f32.data(), down_desc_f32_dev.size(),
                        cudaMemcpyHostToDevice, stream), "copy expanded down descriptors");
        }

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

        if (cohortrail) {
            aw_build_m16_cohorts<<<
                    1, AW_COHORTRAIL_MAX_TILES,
                    0, stream>>>(
                    (const aw_work_desc *)
                        gate_desc_dev.get(),
                    (const aw_tile_desc *)
                        tiles_m16_dev.get(),
                    (int) tiles_m16.size(), nullptr,
                    (int32_t *) cohort_counts_dev.get(),
                    (aw_m16_cohort *)
                        cohorts_p2_dev.get(),
                    (aw_m16_cohort *)
                        cohorts_p3_dev.get(),
                    (aw_m16_cohort *)
                        cohorts_p4_dev.get(),
                    (aw_tile_desc *)
                        cohort_singles_dev.get());
            int32_t cohort_counts[4] = {};
            aw_cuda_throw(cudaMemcpyAsync(
                        cohort_counts,
                        cohort_counts_dev.get(),
                        sizeof(cohort_counts),
                        cudaMemcpyDeviceToHost, stream),
                    "copy CohortRail queue counts");
            aw_cuda_throw(cudaStreamSynchronize(stream),
                    "build CohortRail queues");
            const int queued_tiles =
                    2*cohort_counts[0] +
                    3*cohort_counts[1] +
                    4*cohort_counts[2] +
                    cohort_counts[3];
            if (queued_tiles != (int) tiles_m16.size()) {
                throw std::runtime_error(
                        "CohortRail queue lost an M16 tile");
            }
            if (aw_env_on(getenv(
                            "GGML_CUDA_AW_COHORTRAIL_STATS"))) {
                fprintf(stderr,
                        "CohortRail queues: P4=%d P3=%d P2=%d singleton=%d\n",
                        cohort_counts[2], cohort_counts[1],
                        cohort_counts[0], cohort_counts[3]);
            }
        }

        if (relay_copy) {
            const size_t local_panel_bytes =
                    (size_t) params->tokens_per_cell*panel_k*sizeof(float);
            const size_t gathered_panel_bytes =
                    (size_t) tokens*panel_k*sizeof(float);
            for (int panel = 0; panel < service_kpanels; ++panel) {
                void * local_panel = (char *) relay_local.get() +
                        (size_t) panel*local_panel_bytes;
                void * gathered_panel = (char *) relay_gathered.get() +
                        (size_t) panel*gathered_panel_bytes;
                const float * source_panel = (const float *) source.get() +
                        (size_t) params->device*params->tokens_per_cell*AW_EMBD +
                        panel*panel_k;
                aw_cuda_throw(cudaMemcpy2DAsync(
                            local_panel,
                            panel_k*sizeof(float),
                            source_panel,
                            AW_EMBD*sizeof(float),
                            panel_k*sizeof(float),
                            params->tokens_per_cell,
                            cudaMemcpyDeviceToDevice,
                            stream),
                        "pack RelayWave local feature panel");
                aw_relay_coord.local[params->device][panel] = local_panel;
                aw_relay_coord.gathered[params->device][panel] = gathered_panel;
                aw_cuda_throw(cudaEventCreateWithFlags(
                            &aw_relay_coord.consumed[params->device][panel],
                            cudaEventDisableTiming),
                        "create RelayWave consumed event");
                aw_cuda_throw(cudaEventRecord(
                            aw_relay_coord.consumed[params->device][panel],
                            stream),
                        "initialize RelayWave consumed event");
                for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                    aw_cuda_throw(cudaEventCreateWithFlags(
                                &aw_relay_coord.sent[params->device][dst][panel],
                                cudaEventDisableTiming),
                            "create RelayWave sent event");
                }
            }
            aw_cuda_throw(cudaStreamSynchronize(stream), "initialize RelayWave panels");
            if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                throw std::runtime_error("RelayWave setup barrier was cancelled");
            }
        }
        if (npanel_any_copy) {
            aw_npanel_coord.recv[params->device] = npanel_recv.get();
            for (int panel = 0; panel < service_npanels; ++panel) {
                aw_cuda_throw(cudaEventCreateWithFlags(
                            &aw_npanel_coord.ready[params->device][panel],
                            cudaEventDisableTiming),
                        "create N-panel ready event");
                for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                    if (dst == params->device) {
                        continue;
                    }
                    aw_cuda_throw(cudaEventCreateWithFlags(
                                &aw_npanel_coord.copied[params->device][dst][panel],
                                cudaEventDisableTiming),
                            "create N-panel copied event");
                }
            }
            if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                throw std::runtime_error(
                        "N-panel copy setup barrier was cancelled");
            }
        }

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
        if (config.down_cache == 4 || expandwave) {
            aw_cuda_throw(cudaEventRecord(begin, stream), "record down expansion begin");
            const size_t tile_stages = (size_t) descriptors*(AW_EMBD/AW_T64_ROWS)*
                    (AW_EXPERT_FF/AW_KSTAGE);
            aw_expand_q8_t64_f32<<<(unsigned) std::min<size_t>(tile_stages, 65535), 256, 0, stream>>>(
                    (const char *) weights_down.get(), (float *) weights_down_f32.get(), tile_stages,
                    AW_EMBD, AW_EXPERT_FF);
            aw_cuda_throw(cudaEventRecord(end, stream), "record down expansion end");
            aw_cuda_throw(cudaEventSynchronize(end), "synchronize down expansion");
            aw_cuda_throw(cudaEventElapsedTime(&down_expand_ms, begin, end), "elapsed down expansion time");
            if (!expandwave) {
                down_expand_ms /= params->active_cells;
            }
        }

        const int n_mtiles128 = (int) tiles_m128.size();
        const int n_mtiles64 = (int) tiles_m64.size();
        const int n_mtiles32 = (int) tiles_m32.size();
        const int n_mtiles16 = (int) tiles_m16.size();
        const int n_warp_tiles = (int) tiles_warp.size();
        const int n_mtiles8 = (int) tiles_m8.size();
        const int n_mtiles4 = (int) tiles_m4.size();
        cudaDeviceProp prop;
        aw_cuda_throw(cudaGetDeviceProperties(&prop, params->device), "cudaGetDeviceProperties");
        const int persistent_blocks = prop.multiProcessorCount*2;
        const int warp_blocks = prop.multiProcessorCount*3;
        auto mark_stage = [&](cudaEvent_t * marks, int index) {
            if (marks != nullptr) {
                aw_cuda_throw(cudaEventRecord(marks[index], stream), "record stage timing");
            }
        };
        auto launch_q8 = [&](const aw_work_desc * desc_ptr, int n, int k,
                bool input_bf16, bool is_down = false) {
            if (cohortrail) {
                const auto * tiles64_ptr =
                        (const aw_tile_desc *)
                            tiles_m64_dev.get();
                const auto * tiles32_ptr =
                        (const aw_tile_desc *)
                            tiles_m32_dev.get();
                const auto * tiles16_ptr =
                        (const aw_tile_desc *)
                            tiles_m16_dev.get();
                auto * counts =
                        (int32_t *) cohort_counts_dev.get();
                auto * cohorts_p2 =
                        (aw_m16_cohort *)
                            cohorts_p2_dev.get();
                auto * cohorts_p3 =
                        (aw_m16_cohort *)
                            cohorts_p3_dev.get();
                auto * cohorts_p4 =
                        (aw_m16_cohort *)
                            cohorts_p4_dev.get();
                auto * singles =
                        (aw_tile_desc *)
                            cohort_singles_dev.get();
                aw_build_m16_cohorts<<<
                        1, AW_COHORTRAIL_MAX_TILES,
                        0, stream>>>(
                        desc_ptr, tiles16_ptr,
                        n_mtiles16, nullptr,
                        counts, cohorts_p2, cohorts_p3,
                        cohorts_p4, singles);
                if (input_bf16) {
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128_halfpipe_sync<
                                true>
                                <<<persistent_blocks, 256,
                                    0, stream>>>(
                                        desc_ptr, tiles64_ptr,
                                        n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_launch_cohortrail_m32<
                                true>(
                                    desc_ptr,
                                    tiles32_ptr,
                                    n_mtiles32,
                                    n, k, nullptr,
                                    stream,
                                    persistent_blocks);
                    }
                    aw_launch_cohortrail_m16<
                            true>(
                                desc_ptr, tiles16_ptr,
                                counts, cohorts_p2,
                                cohorts_p3, cohorts_p4,
                                singles, n, k, stream,
                                prop.multiProcessorCount);
                } else {
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128_halfpipe_sync<
                                false>
                                <<<persistent_blocks, 256,
                                    0, stream>>>(
                                        desc_ptr, tiles64_ptr,
                                        n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_launch_cohortrail_m32<
                                false>(
                                    desc_ptr,
                                    tiles32_ptr,
                                    n_mtiles32,
                                    n, k, nullptr,
                                    stream,
                                    persistent_blocks);
                    }
                    aw_launch_cohortrail_m16<
                            false>(
                                desc_ptr, tiles16_ptr,
                                counts, cohorts_p2,
                                cohorts_p3, cohorts_p4,
                                singles, n, k, stream,
                                prop.multiProcessorCount);
                }
                return;
            }
            if (config.q8_engine == "compact") {
                if (input_bf16) {
                    aw_q8_compact<true, 64><<<prop.multiProcessorCount*3, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m64_dev.get(),
                            n_mtiles64, n, k);
                    aw_q8_compact<true, 32><<<prop.multiProcessorCount*5, 128, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m32_dev.get(),
                            n_mtiles32, n, k);
                    aw_q8_compact<true, 16><<<prop.multiProcessorCount*6, 64, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m16_dev.get(),
                            n_mtiles16, n, k);
                } else {
                    aw_q8_compact<false, 64><<<prop.multiProcessorCount*3, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m64_dev.get(),
                            n_mtiles64, n, k);
                    aw_q8_compact<false, 32><<<prop.multiProcessorCount*5, 128, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m32_dev.get(),
                            n_mtiles32, n, k);
                    aw_q8_compact<false, 16><<<prop.multiProcessorCount*6, 64, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_m16_dev.get(),
                            n_mtiles16, n, k);
                }
                return;
            }
            if (config.q8_engine == "warpwave") {
                if (input_bf16) {
                    aw_q8_warpwave<true><<<warp_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_warp_dev.get(),
                            n_warp_tiles, n, k);
                } else {
                    aw_q8_warpwave<false><<<warp_blocks, 256, 0, stream>>>(
                            desc_ptr, (const aw_tile_desc *) tiles_warp_dev.get(),
                            n_warp_tiles, n, k);
                }
                return;
            }
            if (config.q8_engine == "railwave") {
                const auto * tiles64_ptr =
                        (const aw_tile_desc *) tiles_m64_dev.get();
                if (n_mtiles64 != 0) {
                    if (input_bf16) {
                        aw_q8_service_m64_n128_rail<true>
                                <<<prop.multiProcessorCount*3, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k,
                                        (float *) rail_partial.get());
                    } else {
                        aw_q8_service_m64_n128_rail<false>
                                <<<prop.multiProcessorCount*3, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k,
                                        (float *) rail_partial.get());
                    }
                    const size_t add_count =
                            (size_t) n_mtiles64*AW_T64_ROWS*n;
                    aw_q8_service_m64_n128_rail_add
                            <<<aw_grid(add_count), 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n,
                                    (const float *) rail_partial.get());
                }
                const auto * tiles32_ptr =
                        (const aw_tile_desc *) tiles_m32_dev.get();
                const auto * tiles16_ptr =
                        (const aw_tile_desc *) tiles_m16_dev.get();
                if (input_bf16) {
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<true, true, true>
                                <<<persistent_blocks, 128, 0, stream>>>(
                                        desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_service_m16_pair<true, true>
                                <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    }
                } else {
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<false, true, true>
                                <<<persistent_blocks, 128, 0, stream>>>(
                                        desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_service_m16_pair<false, true>
                                <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    }
                }
                return;
            }
            if (config.q8_engine == "crestwave") {
                const auto * tiles128_ptr =
                        (const aw_tile_desc *) tiles_m128_dev.get();
                const auto * tiles64_ptr =
                        (const aw_tile_desc *) tiles_m64_dev.get();
                const auto * tiles32_ptr =
                        (const aw_tile_desc *) tiles_m32_dev.get();
                const auto * tiles16_ptr =
                        (const aw_tile_desc *) tiles_m16_dev.get();
                if (input_bf16) {
                    if (n_mtiles128 != 0) {
                        aw_q8_service_m128_n128_512<true>
                                <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                        desc_ptr, tiles128_ptr,
                                        n_mtiles128, n, k);
                    }
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128_256<true>
                                <<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr,
                                        n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<true, true, true>
                                <<<persistent_blocks, 128, 0, stream>>>(
                                        desc_ptr, tiles32_ptr,
                                        n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_service_m16_pair<true, true>
                                <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                        desc_ptr, tiles16_ptr,
                                        n_mtiles16, n, k);
                    }
                } else {
                    if (n_mtiles128 != 0) {
                        aw_q8_service_m128_n128_512<false>
                                <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                        desc_ptr, tiles128_ptr,
                                        n_mtiles128, n, k);
                    }
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128_256<false>
                                <<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr,
                                        n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<false, true, true>
                                <<<persistent_blocks, 128, 0, stream>>>(
                                        desc_ptr, tiles32_ptr,
                                        n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_service_m16_pair<false, true>
                                <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                        desc_ptr, tiles16_ptr,
                                        n_mtiles16, n, k);
                    }
                }
                return;
            }
            if (config.q8_engine == "broadwave" ||
                    config.q8_engine == "vectorwave" ||
                    config.q8_engine == "halfpipe_sync" ||
                    config.q8_engine == "halfpipe_bar" ||
                    config.q8_engine == "doublewave" ||
                    config.q8_engine == "dualrailwave" ||
                    config.q8_engine == "regbwave" ||
                    config.q8_engine == "expandwave" ||
                    config.q8_engine == "flexwave" ||
                    config.q8_engine == "flexwave1" || config.q8_engine == "rendezvous" ||
                    config.q8_engine == "megawave" || config.q8_engine == "fabwave") {
                const auto * tiles64_ptr = (const aw_tile_desc *) tiles_m64_dev.get();
                const auto * tiles32_ptr = (const aw_tile_desc *) tiles_m32_dev.get();
                const auto * tiles16_ptr = (const aw_tile_desc *) tiles_m16_dev.get();
                const bool flexwave = config.q8_engine == "flexwave" ||
                        config.q8_engine == "flexwave1";
                const bool flexwave1 = config.q8_engine == "flexwave1";
                const bool doublewave =
                        config.q8_engine == "doublewave";
                const bool vectorwave =
                        config.q8_engine == "vectorwave";
                const bool dualrailwave =
                        config.q8_engine == "dualrailwave";
                const bool halfpipe_sync =
                        config.q8_engine == "halfpipe_sync";
                const bool halfpipe_bar =
                        config.q8_engine == "halfpipe_bar";
                const bool megawave = config.q8_engine == "megawave";
                const bool fabwave = config.q8_engine == "fabwave";
                const bool regbwave = config.q8_engine == "regbwave";
                const bool expanded_down = expandwave && is_down;
                const auto * expanded_desc_ptr =
                        (const aw_work_desc *) down_desc_f32_dev.get();
                const bool fabwave_k8 = aw_q8_fabwave_k8_enabled();
                const bool fabwave_qscale = aw_q8_fabwave_qscale_enabled();
                const bool fabwave_dual = aw_q8_fabwave_dual_enabled();
                const bool fabwave_exact = fabwave && aw_q8_fabwave_exact_projection(n, k);
                const int flex_blocks = prop.multiProcessorCount*(flexwave1 ? 6 : 4);
                if (input_bf16) {
                    if (n_mtiles64 != 0) {
                        if (doublewave) {
                            aw_q8_service_m64_n128_double<true>
                                    <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (vectorwave) {
                            aw_q8_service_m64_n128_vector<true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (dualrailwave) {
                            aw_q8_service_m64_n128_dualrail<true>
                                    <<<persistent_blocks, 512, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (halfpipe_sync) {
                            aw_q8_service_m64_n128_halfpipe_sync<true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (halfpipe_bar) {
                            aw_q8_service_m64_n128_halfpipe_bar<true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (expanded_down) {
                            aw_q8_service_m64_n128_256<true, true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            expanded_desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (megawave) {
                            aw_q8_service_m64_n128_1024<true>
                                    <<<prop.multiProcessorCount, 1024, 0, stream>>>(
                                            desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else if (fabwave && !fabwave_exact) {
                            if (fabwave_dual) {
                                aw_q8_fab_m64_n128<true, 4, true, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_k8 && fabwave_qscale) {
                                aw_q8_fab_m64_n128<true, 4, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_k8) {
                                aw_q8_fab_m64_n128<true, 4><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_qscale) {
                                aw_q8_fab_m64_n128<true, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else {
                                aw_q8_fab_m64_n128<true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            }
                        } else if (regbwave) {
                            aw_q8_service_m64_n128_regb<true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else {
                            aw_q8_service_m64_n128_256<true><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        }
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        if (flexwave) {
                            if (flexwave1) {
                                aw_q8_flexwave16<true, false><<<flex_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                            } else {
                                aw_q8_flexwave16<true, true><<<flex_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                            }
                        } else {
                            aw_q8_service_m16_pair<true, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                        }
                    }
                } else {
                    if (n_mtiles64 != 0) {
                        if (doublewave) {
                            aw_q8_service_m64_n128_double<false>
                                    <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (vectorwave) {
                            aw_q8_service_m64_n128_vector<false>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (dualrailwave) {
                            aw_q8_service_m64_n128_dualrail<false>
                                    <<<persistent_blocks, 512, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (halfpipe_sync) {
                            aw_q8_service_m64_n128_halfpipe_sync<false>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (halfpipe_bar) {
                            aw_q8_service_m64_n128_halfpipe_bar<false>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (expanded_down) {
                            aw_q8_service_m64_n128_256<false, true>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            expanded_desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else if (megawave) {
                            aw_q8_service_m64_n128_1024<false>
                                    <<<prop.multiProcessorCount, 1024, 0, stream>>>(
                                            desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else if (fabwave && !fabwave_exact) {
                            if (fabwave_dual) {
                                aw_q8_fab_m64_n128<false, 4, true, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_k8 && fabwave_qscale) {
                                aw_q8_fab_m64_n128<false, 4, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_k8) {
                                aw_q8_fab_m64_n128<false, 4><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else if (fabwave_qscale) {
                                aw_q8_fab_m64_n128<false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            } else {
                                aw_q8_fab_m64_n128<false><<<persistent_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                            }
                        } else if (regbwave) {
                            aw_q8_service_m64_n128_regb<false>
                                    <<<persistent_blocks, 256, 0, stream>>>(
                                            desc_ptr, tiles64_ptr,
                                            n_mtiles64, n, k);
                        } else {
                            aw_q8_service_m64_n128_256<false><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        }
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        if (flexwave) {
                            if (flexwave1) {
                                aw_q8_flexwave16<false, false><<<flex_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                            } else {
                                aw_q8_flexwave16<false, true><<<flex_blocks, 256, 0, stream>>>(
                                        desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                            }
                        } else {
                            aw_q8_service_m16_pair<false, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                        }
                    }
                }
                return;
            }
            if (config.q8_engine == "widewave") {
                const auto * tiles64_ptr = (const aw_tile_desc *) tiles_m64_dev.get();
                const auto * tiles32_ptr = (const aw_tile_desc *) tiles_m32_dev.get();
                const int tail_blocks = prop.multiProcessorCount*8;
                if (input_bf16) {
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128<true><<<prop.multiProcessorCount, 512, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles8 != 0) {
                        aw_q8_tailwave<true, 8><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m8_dev.get(),
                                n_mtiles8, n, k);
                    }
                    if (n_mtiles4 != 0) {
                        aw_q8_tailwave<true, 4><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m4_dev.get(),
                                n_mtiles4, n, k);
                    }
                } else {
                    if (n_mtiles64 != 0) {
                        aw_q8_service_m64_n128<false><<<prop.multiProcessorCount, 512, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    }
                    if (n_mtiles32 != 0) {
                        aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles8 != 0) {
                        aw_q8_tailwave<false, 8><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m8_dev.get(),
                                n_mtiles8, n, k);
                    }
                    if (n_mtiles4 != 0) {
                        aw_q8_tailwave<false, 4><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m4_dev.get(),
                                n_mtiles4, n, k);
                    }
                }
                return;
            }
            const auto * tiles64_ptr = (const aw_tile_desc *) tiles_m64_dev.get();
            const auto * tiles32_ptr = (const aw_tile_desc *) tiles_m32_dev.get();
            const bool t64 = config.q8_layout == "t64k32";
            const bool interleave = config.q8_kernel == "interleave";
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
                        if (interleave) {
                            aw_q8_service_m64<true, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else if (t64) {
                            aw_q8_service_m64<true, true><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        } else {
                            aw_q8_service_m64<true, false><<<persistent_blocks, 256, 0, stream>>>(
                                    desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                        }
                    } else if (interleave) {
                        aw_q8_service_m64<false, true, false, 2, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    } else if (t64) {
                        aw_q8_service_m64<false, true><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    } else {
                        aw_q8_service_m64<false, false><<<persistent_blocks, 256, 0, stream>>>(
                                desc_ptr, tiles64_ptr, n_mtiles64, n, k);
                    }
                }
            }
            if (config.q8_engine == "hybrid") {
                const auto * tiles16_ptr = (const aw_tile_desc *) tiles_m16_dev.get();
                if (input_bf16) {
                    if (n_mtiles32 != 0) {
                        aw_q8_compact<true, 32><<<prop.multiProcessorCount*5, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_compact<true, 16><<<prop.multiProcessorCount*6, 64, 0, stream>>>(
                                desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    }
                } else {
                    if (n_mtiles32 != 0) {
                        aw_q8_compact<false, 32><<<prop.multiProcessorCount*5, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                    if (n_mtiles16 != 0) {
                        aw_q8_compact<false, 16><<<prop.multiProcessorCount*6, 64, 0, stream>>>(
                                desc_ptr, tiles16_ptr, n_mtiles16, n, k);
                    }
                }
                return;
            }
            if (n_mtiles32 != 0) {
                if (input_bf16) {
                    if (interleave) {
                        aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    } else if (t64) {
                        aw_q8_service_m32<true, true><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    } else {
                        aw_q8_service_m32<true, false><<<persistent_blocks, 128, 0, stream>>>(
                                desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                    }
                } else if (interleave) {
                    aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                } else if (t64) {
                    aw_q8_service_m32<false, true><<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                } else {
                    aw_q8_service_m32<false, false><<<persistent_blocks, 128, 0, stream>>>(
                            desc_ptr, tiles32_ptr, n_mtiles32, n, k);
                }
            }
            if (config.q8_engine == "tailwave") {
                const int tail_blocks = prop.multiProcessorCount*8;
                if (input_bf16) {
                    if (n_mtiles8 != 0) {
                        aw_q8_tailwave<true, 8><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m8_dev.get(),
                                n_mtiles8, n, k);
                    }
                    if (n_mtiles4 != 0) {
                        aw_q8_tailwave<true, 4><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m4_dev.get(),
                                n_mtiles4, n, k);
                    }
                } else {
                    if (n_mtiles8 != 0) {
                        aw_q8_tailwave<false, 8><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m8_dev.get(),
                                n_mtiles8, n, k);
                    }
                    if (n_mtiles4 != 0) {
                        aw_q8_tailwave<false, 4><<<tail_blocks, 256, 0, stream>>>(
                                desc_ptr, (const aw_tile_desc *) tiles_m4_dev.get(),
                                n_mtiles4, n, k);
                    }
                }
                return;
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
            } else if (f16_wire) {
                aw_pack_requests_f16<<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                        (const float *) source.get(), wire.get(), (size_t) tokens*AW_EMBD);
            } else {
                aw_pack_requests<false><<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                        (const float *) source.get(), wire.get(), (size_t) tokens*AW_EMBD);
            }
            mark_stage(marks, 1);
            if (service_gupanels > 1) {
                const auto * gate_panel =
                        (const aw_work_desc *) gate_panel_desc_dev.get();
                const auto * up_panel =
                        (const aw_work_desc *) up_panel_desc_dev.get();
                for (int panel = 0; panel < service_gupanels; ++panel) {
                    launch_q8(
                            gate_panel + (size_t) panel*descriptors,
                            service_gu_panel_n,
                            AW_EMBD,
                            wire16);
                    launch_q8(
                            up_panel + (size_t) panel*descriptors,
                            service_gu_panel_n,
                            AW_EMBD,
                            wire16);
                    aw_swiglu_panel<<<
                            aw_grid((size_t) route_rows*service_gu_panel_n),
                            256, 0, stream>>>(
                                (const float *) gate.get(),
                                (const float *) up.get(),
                                (float *) middle.get(),
                                route_rows,
                                service_gu_panel_n,
                                AW_EXPERT_FF,
                                panel*service_gu_panel_n);
                }
                mark_stage(marks, 2);
                mark_stage(marks, 3);
            } else if (bench_fused_swiglu) {
                const auto * gate_desc_ptr =
                        (const aw_work_desc *) gate_desc_dev.get();
                const auto * up_desc_ptr =
                        (const aw_work_desc *) up_desc_dev.get();
                const auto * tiles64_ptr =
                        (const aw_tile_desc *) tiles_m64_dev.get();
                const auto * tiles32_ptr =
                        (const aw_tile_desc *) tiles_m32_dev.get();
                const auto * tiles16_ptr =
                        (const aw_tile_desc *) tiles_m16_dev.get();
                if (wire16) {
                    aw_q8_service_m64_gate_up<true, true>
                            <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                    gate_desc_ptr,
                                    up_desc_ptr,
                                    tiles64_ptr,
                                    n_mtiles64,
                                    AW_EXPERT_FF,
                                    AW_EMBD,
                                    nullptr,
                                    (float *) middle.get());
                    aw_q8_service_m32<true, true, true>
                            <<<persistent_blocks, 128, 0, stream>>>(
                                    gate_desc_ptr, tiles32_ptr,
                                    n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<true, true>
                            <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    gate_desc_ptr, tiles16_ptr,
                                    n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m32<true, true, true>
                            <<<persistent_blocks, 128, 0, stream>>>(
                                    up_desc_ptr, tiles32_ptr,
                                    n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<true, true>
                            <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    up_desc_ptr, tiles16_ptr,
                                    n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                } else {
                    aw_q8_service_m64_gate_up<false, true>
                            <<<prop.multiProcessorCount, 512, 0, stream>>>(
                                    gate_desc_ptr,
                                    up_desc_ptr,
                                    tiles64_ptr,
                                    n_mtiles64,
                                    AW_EXPERT_FF,
                                    AW_EMBD,
                                    nullptr,
                                    (float *) middle.get());
                    aw_q8_service_m32<false, true, true>
                            <<<persistent_blocks, 128, 0, stream>>>(
                                    gate_desc_ptr, tiles32_ptr,
                                    n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<false, true>
                            <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    gate_desc_ptr, tiles16_ptr,
                                    n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m32<false, true, true>
                            <<<persistent_blocks, 128, 0, stream>>>(
                                    up_desc_ptr, tiles32_ptr,
                                    n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<false, true>
                            <<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                                    up_desc_ptr, tiles16_ptr,
                                    n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                }
                if (n_mtiles32 != 0) {
                    aw_swiglu_tiles<<<n_mtiles32, 256, 0, stream>>>(
                            gate_desc_ptr, up_desc_ptr, tiles32_ptr,
                            n_mtiles32, (float *) middle.get(),
                            AW_EXPERT_FF);
                }
                if (n_mtiles16 != 0) {
                    aw_swiglu_tiles<<<n_mtiles16, 256, 0, stream>>>(
                            gate_desc_ptr, up_desc_ptr, tiles16_ptr,
                            n_mtiles16, (float *) middle.get(),
                            AW_EXPERT_FF);
                }
                mark_stage(marks, 2);
                mark_stage(marks, 3);
            } else if (service_kpanels == 1) {
                launch_q8(
                        (const aw_work_desc *) gate_desc_dev.get(),
                        AW_EXPERT_FF, AW_EMBD, wire16);
                mark_stage(marks, 2);
                launch_q8(
                        (const aw_work_desc *) up_desc_dev.get(),
                        AW_EXPERT_FF, AW_EMBD, wire16);
            } else {
                const auto * gate_desc_ptr = (const aw_work_desc *) gate_desc_dev.get();
                const auto * up_desc_ptr = (const aw_work_desc *) up_desc_dev.get();
                const auto * tiles64_ptr = (const aw_tile_desc *) tiles_m64_dev.get();
                const int stages_per_panel = (AW_EMBD/AW_KSTAGE)/service_kpanels;
                for (int panel = 0; panel < service_kpanels; ++panel) {
                    const int stage_begin = panel*stages_per_panel;
                    const void * panel_input = nullptr;
                    if (relay_copy) {
                        const size_t local_panel_ne =
                                (size_t) params->tokens_per_cell*panel_k;
                        const size_t local_panel_bytes =
                                local_panel_ne*sizeof(float);
                        for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                            cudaStream_t copy_stream = relay_streams[dst];
                            aw_cuda_throw(cudaStreamWaitEvent(
                                        copy_stream,
                                        aw_relay_coord.consumed[dst][panel],
                                        0),
                                    "wait for RelayWave panel consumer");
                            float * output = (float *) aw_relay_coord.gathered[dst][panel] +
                                    (size_t) params->device*local_panel_ne;
                            const void * input =
                                    aw_relay_coord.local[params->device][panel];
                            if (dst == params->device) {
                                aw_cuda_throw(cudaMemcpyAsync(
                                            output,
                                            input,
                                            local_panel_bytes,
                                            cudaMemcpyDeviceToDevice,
                                            copy_stream),
                                        "copy local RelayWave panel");
                            } else {
                                aw_cuda_throw(cudaMemcpyPeerAsync(
                                            output,
                                            dst,
                                            input,
                                            params->device,
                                            local_panel_bytes,
                                            copy_stream),
                                        "copy peer RelayWave panel");
                            }
                            aw_cuda_throw(cudaEventRecord(
                                        aw_relay_coord.sent[params->device][dst][panel],
                                        copy_stream),
                                    "record RelayWave panel send");
                        }
                        if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                            throw std::runtime_error(
                                    "RelayWave panel publication barrier was cancelled");
                        }
                        for (int source_device = 0;
                                source_device < AW_GPU_COUNT;
                                ++source_device) {
                            aw_cuda_throw(cudaStreamWaitEvent(
                                        stream,
                                        aw_relay_coord.sent[source_device][params->device][panel],
                                        0),
                                    "wait for RelayWave panel");
                        }
                        panel_input = aw_relay_coord.gathered[params->device][panel];
                    }
                    if (wire16) {
                        aw_q8_service_m64_n128_256_panel<true><<<persistent_blocks, 256, 0, stream>>>(
                                gate_desc_ptr, tiles64_ptr, n_mtiles64,
                                AW_EXPERT_FF, AW_EMBD, (float *) gate_kpartial.get(),
                                route_rows, stage_begin, stages_per_panel,
                                panel_input, relay_copy ? panel_k : AW_EMBD);
                        aw_q8_service_m64_n128_256_panel<true><<<persistent_blocks, 256, 0, stream>>>(
                                up_desc_ptr, tiles64_ptr, n_mtiles64,
                                AW_EXPERT_FF, AW_EMBD, (float *) up_kpartial.get(),
                                route_rows, stage_begin, stages_per_panel,
                                panel_input, relay_copy ? panel_k : AW_EMBD);
                    } else {
                        aw_q8_service_m64_n128_256_panel<false><<<persistent_blocks, 256, 0, stream>>>(
                                gate_desc_ptr, tiles64_ptr, n_mtiles64,
                                AW_EXPERT_FF, AW_EMBD, (float *) gate_kpartial.get(),
                                route_rows, stage_begin, stages_per_panel,
                                panel_input, relay_copy ? panel_k : AW_EMBD);
                        aw_q8_service_m64_n128_256_panel<false><<<persistent_blocks, 256, 0, stream>>>(
                                up_desc_ptr, tiles64_ptr, n_mtiles64,
                                AW_EXPERT_FF, AW_EMBD, (float *) up_kpartial.get(),
                                route_rows, stage_begin, stages_per_panel,
                                panel_input, relay_copy ? panel_k : AW_EMBD);
                    }
                    if (relay_copy) {
                        aw_cuda_throw(cudaEventRecord(
                                    aw_relay_coord.consumed[params->device][panel],
                                    stream),
                                "record RelayWave panel consumed");
                        if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                            throw std::runtime_error(
                                    "RelayWave panel consumption barrier was cancelled");
                        }
                    }
                }
                const auto * tiles32_ptr = (const aw_tile_desc *) tiles_m32_dev.get();
                const auto * tiles16_ptr = (const aw_tile_desc *) tiles_m16_dev.get();
                if (wire16) {
                    aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                            gate_desc_ptr, tiles32_ptr, n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<true, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                            gate_desc_ptr, tiles16_ptr, n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m32<true, true, true><<<persistent_blocks, 128, 0, stream>>>(
                            up_desc_ptr, tiles32_ptr, n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<true, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                            up_desc_ptr, tiles16_ptr, n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                } else {
                    aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                            gate_desc_ptr, tiles32_ptr, n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<false, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                            gate_desc_ptr, tiles16_ptr, n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m32<false, true, true><<<persistent_blocks, 128, 0, stream>>>(
                            up_desc_ptr, tiles32_ptr, n_mtiles32, AW_EXPERT_FF, AW_EMBD);
                    aw_q8_service_m16_pair<false, true><<<prop.multiProcessorCount*3, 128, 0, stream>>>(
                            up_desc_ptr, tiles16_ptr, n_mtiles16, AW_EXPERT_FF, AW_EMBD);
                }
                mark_stage(marks, 2);
            }
            if (service_gupanels == 1 && !bench_fused_swiglu) {
                mark_stage(marks, 3);
                aw_swiglu<<<aw_grid((size_t) route_rows*AW_EXPERT_FF), 256, 0, stream>>>(
                        (const float *) gate.get(), (const float *) up.get(), (float *) middle.get(),
                        (size_t) route_rows*AW_EXPERT_FF);
            }
            mark_stage(marks, 4);
            if (service_npanels > 1) {
                const auto * panel_desc =
                        (const aw_work_desc *) down_panel_desc_dev.get();
                for (int panel = 0; panel < service_npanels; ++panel) {
                    launch_q8(
                            panel_desc + (size_t) panel*descriptors,
                            service_panel_n,
                            AW_EXPERT_FF,
                            false);
                    aw_owner_reduce_bf16_panel<<<
                            aw_grid((size_t) tokens*service_panel_n),
                            256, 0, stream>>>(
                                (const float *) route_output.get(),
                                (const int32_t *) token_routes_dev.get(),
                                (const float *) route_weights_dev.get(),
                                (uint16_t *) partial.get() +
                                    (size_t) panel*tokens*service_panel_n,
                                tokens,
                                service_panel_n,
                                service_panel_n,
                                0);
                    if (npanel_any_copy) {
                        cudaEvent_t ready =
                                aw_npanel_coord.ready[params->device][panel];
                        aw_cuda_throw(cudaEventRecord(ready, stream),
                                "record N-panel ready");
                        const size_t cell_values =
                                (size_t) params->tokens_per_cell*service_panel_n;
                        cudaStream_t copy_stream = relay_streams[0];
                        const int step_end = npanel_pair_copy ?
                                2 : AW_GPU_COUNT;
                        for (int step = 1; step < step_end; ++step) {
                            const int dst = npanel_pair_copy ?
                                    (params->device ^ 1) :
                                    (params->device + step) % AW_GPU_COUNT;
                            aw_cuda_throw(cudaStreamWaitEvent(
                                        copy_stream, ready, 0),
                                    "wait for N-panel ready");
                            const int copies = npanel_pair_copy ? 2 : 1;
                            for (int copy = 0; copy < copies; ++copy) {
                                const int source_cell = npanel_pair_copy ?
                                        copy : dst;
                                const uint16_t * source =
                                        (const uint16_t *) partial.get() +
                                        ((size_t) panel*tokens +
                                         (size_t) source_cell*
                                            params->tokens_per_cell)*
                                        service_panel_n;
                                const size_t output_slot =
                                        npanel_pair_copy ?
                                        (size_t) params->device*2 + copy :
                                        (size_t) params->device;
                                uint16_t * output =
                                        (uint16_t *)
                                            aw_npanel_coord.recv[dst] +
                                        (output_slot*service_npanels +
                                         panel)*cell_values;
                                aw_cuda_throw(cudaMemcpyPeerAsync(
                                            output,
                                            dst,
                                            source,
                                            params->device,
                                            cell_values*sizeof(uint16_t),
                                            copy_stream),
                                        "copy N-panel owner partial");
                            }
                            aw_cuda_throw(cudaEventRecord(
                                        aw_npanel_coord.copied
                                            [params->device][dst][panel],
                                        copy_stream),
                                    "record N-panel copied");
                        }
                    }
                }
                if (npanel_any_copy) {
                    for (int panel = 0; panel < service_npanels; ++panel) {
                        const int step_end = npanel_pair_copy ?
                                2 : AW_GPU_COUNT;
                        for (int step = 1; step < step_end; ++step) {
                            const int dst = npanel_pair_copy ?
                                    (params->device ^ 1) :
                                    (params->device + step) % AW_GPU_COUNT;
                            aw_cuda_throw(cudaStreamWaitEvent(
                                        stream,
                                        aw_npanel_coord.copied
                                            [params->device][dst][panel],
                                        0),
                                    "wait for N-panel copy");
                        }
                    }
                }
            } else if (config.down_cache == 4) {
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
                launch_q8(
                        (const aw_work_desc *) down_desc_dev.get(),
                        AW_EMBD, AW_EXPERT_FF, false, true);
            }
            mark_stage(marks, 5);
            if (service_npanels == 1) {
                aw_owner_reduce_bf16<<<aw_grid((size_t) tokens*AW_EMBD), 256, 0, stream>>>(
                        (const float *) route_output.get(), (const int32_t *) token_routes_dev.get(),
                        (const float *) route_weights_dev.get(), (uint16_t *) partial.get(),
                        tokens, AW_EMBD);
            }
            mark_stage(marks, 6);
        };

        for (int warmup = 0; warmup < 2; ++warmup) {
            service_once();
        }
        aw_cuda_throw(cudaGetLastError(), "service warmup kernels");
        aw_cuda_throw(cudaStreamSynchronize(stream), "service warmup");
        if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
            throw std::runtime_error("benchmark barrier was cancelled");
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

        if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
            throw std::runtime_error("stage barrier was cancelled");
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
                                residue, wire_format, split_groups[gate_group], split_groups[down_group]);
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
                    size_t output_index = index;
                    if (service_npanels > 1) {
                        const int panel = col/service_panel_n;
                        const int panel_col = col - panel*service_panel_n;
                        output_index =
                                ((size_t) panel*tokens + token)*
                                service_panel_n + panel_col;
                    }
                    if (output[output_index] != token_expected) {
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
        const bool warpwave = config.q8_engine == "warpwave";
        const bool tailwave = config.q8_engine == "tailwave" || config.q8_engine == "widewave";
        const double tail_issued_rows = (double) tiles_m8.size()*8 + (double) tiles_m4.size()*4;
        const double crest_issued_rows = (double) tiles_m128.size()*128;
        const double gate_issued_rows = warpwave ? (double) tiles_warp.size()*16 : tailwave ?
                (double) tiles_m64.size()*64 + (double) tiles_m32.size()*32 + tail_issued_rows :
                crest_issued_rows + (double) tiles_m64.size()*64 + (double) tiles_m32.size()*32 +
                (double) tiles_m16.size()*16;
        const double down_issued_rows = warpwave ? (double) tiles_warp.size()*16 : tailwave ?
                (double) tiles_m64.size()*64 + (double) tiles_m32.size()*32 + tail_issued_rows :
                crest_issued_rows + (double) tiles_m64.size()*64 +
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
        result->tiles_m64 = 2*n_mtiles128 + n_mtiles64;
        result->tiles_m32 = n_mtiles32;
        result->tiles_m16 = n_mtiles16;
        result->tiles_warp = n_warp_tiles;
        result->tiles_m8 = n_mtiles8;
        result->tiles_m4 = n_mtiles4;
        result->q8_kernel = config.q8_kernel == "interleave" ? 1 : 0;
        result->q8_engine = config.q8_engine == "warpwave" ? 1 :
                config.q8_engine == "compact" ? 2 :
                config.q8_engine == "hybrid" ? 3 :
                config.q8_engine == "tailwave" ? 4 :
                config.q8_engine == "widewave" ? 5 :
                config.q8_engine == "broadwave" ? 6 :
                config.q8_engine == "flexwave" ? 7 :
                config.q8_engine == "flexwave1" ? 8 :
                config.q8_engine == "rendezvous" ? 9 :
                config.q8_engine == "megawave" ? 10 :
                config.q8_engine == "fabwave" ? 11 :
                config.q8_engine == "expandwave" ? 12 :
                config.q8_engine == "railwave" ? 13 :
                config.q8_engine == "regbwave" ? 14 :
                config.q8_engine == "crestwave" ? 15 :
                config.q8_engine == "dualrailwave" ? 16 :
                config.q8_engine == "doublewave" ? 17 :
                config.q8_engine == "vectorwave" ? 18 :
                config.q8_engine == "halfpipe_sync" ? 19 :
                config.q8_engine == "halfpipe_bar" ? 20 :
                config.q8_engine == "cohortrail" ? 21 : 0;
        result->check_ran = config.check ? 1 : 0;
        result->check_passed = check_passed;
        result->check_count = config.check ? check_count : 0;
        result->check_mismatches = check_mismatches;
        result->check_first_mismatch = check_first_mismatch;
        result->check_expected = expected;
        result->check_observed = observed;

        if (relay_copy) {
            if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                throw std::runtime_error("RelayWave cleanup barrier was cancelled");
            }
            for (int panel = 0; panel < service_kpanels; ++panel) {
                for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                    aw_cuda_throw(cudaEventDestroy(
                                aw_relay_coord.sent[params->device][dst][panel]),
                            "destroy RelayWave sent event");
                    aw_relay_coord.sent[params->device][dst][panel] = nullptr;
                }
                aw_cuda_throw(cudaEventDestroy(
                            aw_relay_coord.consumed[params->device][panel]),
                        "destroy RelayWave consumed event");
                aw_relay_coord.consumed[params->device][panel] = nullptr;
                aw_relay_coord.local[params->device][panel] = nullptr;
                aw_relay_coord.gathered[params->device][panel] = nullptr;
            }
        }
        if (npanel_any_copy) {
            if (!aw_bench_barrier.arrive_and_wait(params->participants)) {
                throw std::runtime_error(
                        "N-panel copy cleanup barrier was cancelled");
            }
            for (int panel = 0; panel < service_npanels; ++panel) {
                aw_cuda_throw(cudaEventDestroy(
                            aw_npanel_coord.ready[params->device][panel]),
                        "destroy N-panel ready event");
                aw_npanel_coord.ready[params->device][panel] = nullptr;
                for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                    if (dst == params->device) {
                        continue;
                    }
                    aw_cuda_throw(cudaEventDestroy(
                                aw_npanel_coord.copied
                                    [params->device][dst][panel]),
                            "destroy N-panel copied event");
                    aw_npanel_coord.copied[params->device][dst][panel] =
                            nullptr;
                }
            }
            aw_npanel_coord.recv[params->device] = nullptr;
        }
        if (relay_copy || npanel_any_copy) {
            for (int dst = 0; dst < AW_GPU_COUNT; ++dst) {
                aw_cuda_throw(cudaStreamDestroy(relay_streams[dst]),
                        "destroy service copy stream");
            }
        }
        aw_cuda_throw(cudaEventDestroy(begin), "cudaEventDestroy(begin)");
        aw_cuda_throw(cudaEventDestroy(end), "cudaEventDestroy(end)");
        for (cudaEvent_t event : stage_events) {
            aw_cuda_throw(cudaEventDestroy(event), "cudaEventDestroy(stage)");
        }
        aw_cuda_throw(cudaStreamDestroy(stream), "cudaStreamDestroy");
        if (!check_passed && !aw_env_on(getenv("GGML_CUDA_AW_CHECK_REPORT"))) {
            throw std::runtime_error("BF16 owner-partial accuracy check failed");
        }
        return 0;
    } catch (const std::exception & exception) {
        aw_bench_barrier.cancel();
        aw_set_error(error, error_capacity, exception.what());
        return 1;
    }
}
