#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

static thread_local int64_t ggml_backend_meta_aw_current_tokens = 0;
static std::atomic<uint64_t> ggml_backend_meta_aw_state_epoch{1};
static std::atomic<int64_t> ggml_backend_meta_aw_last_canonical_tokens{0};

static bool ggml_backend_meta_aw_serving() {
    const char * value = getenv("GGML_CUDA_AW_SERVE");
    return value != nullptr && strcmp(value, "1") == 0;
}

int64_t ggml_backend_meta_set_affinity_wave_tokens(int64_t n_tokens) {
    GGML_ASSERT(n_tokens >= 0);
    const int64_t previous = ggml_backend_meta_aw_current_tokens;
    ggml_backend_meta_aw_current_tokens = n_tokens;
    return previous;
}

void ggml_backend_meta_aw_invalidate_serving_state(void) {
    ggml_backend_meta_aw_last_canonical_tokens.store(0, std::memory_order_relaxed);
    ggml_backend_meta_aw_state_epoch.fetch_add(1, std::memory_order_relaxed);
}

void ggml_backend_meta_aw_invalidate_serving_state_from(const int64_t position) {
    const int64_t canonical_tokens =
            ggml_backend_meta_aw_last_canonical_tokens.load(std::memory_order_relaxed);
    if (position < 0) {
        ggml_backend_meta_aw_invalidate_serving_state();
    } else if (position < canonical_tokens) {
        // Sequence removal discards only the suffix. Keep the known-valid
        // prefix and make the next append copy start at the removal point.
        // Small prefixes are conservatively invalidated: a server request with
        // cache_prompt=false can reuse a slot while replacing that prefix, and
        // this backend cannot observe the request's token identity. Long cache
        // turns and the small padded boundary trim remain eligible.
        const bool likely_cached_prefix = position >= 4096 || canonical_tokens - position <= 4;
        if (likely_cached_prefix) {
            ggml_backend_meta_aw_last_canonical_tokens.store(position, std::memory_order_relaxed);
        } else {
            ggml_backend_meta_aw_invalidate_serving_state();
        }
    }
}

static int64_t ggml_backend_meta_aw_active_tokens() {
    const char * limit_env = getenv("GGML_CUDA_AW_WAVE_TOKENS");
    const int64_t limit = limit_env != nullptr ? atoll(limit_env) : 0;
    if (!ggml_backend_meta_aw_serving()) {
        return limit;
    }
    const char * tiny_mmvq = getenv("GGML_CUDA_AW_SERVE_TINY_MMVQ");
    if (ggml_backend_meta_aw_current_tokens <= 4 && tiny_mmvq != nullptr && strcmp(tiny_mmvq, "1") == 0) {
        return 0;
    }
    const char * split_env = getenv("GGML_CUDA_AW_WAVE_TOKEN_SPLIT");
    if (split_env == nullptr || strcmp(split_env, "0") == 0 ||
            ggml_backend_meta_aw_current_tokens < 4 ||
            ggml_backend_meta_aw_current_tokens % 4 != 0 ||
            ggml_backend_meta_aw_current_tokens > limit) {
        return 0;
    }
    return ggml_backend_meta_aw_current_tokens;
}

static bool ggml_backend_meta_aw_token_split() {
    const char * value = getenv("GGML_CUDA_AW_WAVE_TOKEN_SPLIT");
    return value != nullptr && strcmp(value, "0") != 0 &&
            (!ggml_backend_meta_aw_serving() || ggml_backend_meta_aw_active_tokens() != 0);
}

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    ggml_tensor lane_tensor;
    const ggml_tensor * alloc_tensor = tensor;
    const int64_t aw_tokens = ggml_backend_meta_aw_active_tokens();
    const bool aw_lane_balance = getenv("GGML_CUDA_AW_LANE_BALANCE") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_LANE_BALANCE"), "0") != 0;
    auto aw_max_lane = [&](int64_t extent) {
        return aw_lane_balance && n_simple_bufts == 4 ? (extent*300 + 999)/1000 :
                (ggml_backend_meta_aw_serving() ? (extent + 3)/4 : extent/4);
    };
    const bool aw_full_alloc = getenv("GGML_CUDA_AW_FULL_ALLOC") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_FULL_ALLOC"), "0") != 0;
    if (aw_tokens > 0 && !aw_full_alloc && ggml_is_contiguous(tensor) && tensor->view_src == nullptr) {
        lane_tensor = *tensor;
        bool changed = false;
        if (strstr(tensor->name, "attn_inp_kq_mask") != nullptr && lane_tensor.ne[1] == aw_tokens) {
            lane_tensor.ne[1] = aw_max_lane(lane_tensor.ne[1]);
            changed = true;
        } else {
            int split_dim = -1;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (lane_tensor.ne[d] == aw_tokens || lane_tensor.ne[d] == 4*aw_tokens) {
                    split_dim = d;
                } else if (lane_tensor.ne[d] == aw_tokens + 3) {
                    split_dim = d;
                } else if (lane_tensor.ne[d] == aw_tokens + 128) {
                    split_dim = d;
                }
            }
            if (split_dim >= 0) {
                if (lane_tensor.ne[split_dim] == aw_tokens + 3) {
                    lane_tensor.ne[split_dim] = aw_max_lane(aw_tokens) + 3;
                } else if (lane_tensor.ne[split_dim] == aw_tokens + 128) {
                    lane_tensor.ne[split_dim] = aw_max_lane(aw_tokens) + 128;
                } else if (ggml_backend_meta_aw_serving() && lane_tensor.ne[split_dim] == 4*aw_tokens) {
                    lane_tensor.ne[split_dim] = 4*aw_max_lane(aw_tokens);
                } else {
                    lane_tensor.ne[split_dim] = aw_max_lane(lane_tensor.ne[split_dim]);
                }
                changed = true;
            }
        }
        if (changed) {
            lane_tensor.nb[0] = ggml_type_size(lane_tensor.type);
            lane_tensor.nb[1] = lane_tensor.nb[0] * (lane_tensor.ne[0] / ggml_blck_size(lane_tensor.type));
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                lane_tensor.nb[d] = lane_tensor.nb[d - 1] * lane_tensor.ne[d - 1];
            }
            alloc_tensor = &lane_tensor;
        }
    }
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(
                ggml_backend_meta_buft_simple_buft(buft, i), alloc_tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[2];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute{std::move(stc_compute_0), std::move(stc_compute_1)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const bool aw_wave_token_split = ggml_backend_meta_aw_token_split();
    static const bool aw_lane_balance = getenv("GGML_CUDA_AW_LANE_BALANCE") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_LANE_BALANCE"), "0") != 0;

    auto aw_split_extent = [&](const ggml_backend_meta_split_state & ss, size_t j) -> int64_t {
        int64_t ret = 0;
        for (size_t s = 0; s < ss.n_segments; ++s) {
            ret += ss.ne[s*n_bufs + j] * ss.nr[s];
        }
        return ret;
    };

    auto aw_even_split = [&](ggml_backend_meta_split_axis axis, int64_t extent, uint32_t repeats = 1) {
        GGML_ASSERT(repeats > 0 && extent % repeats == 0);
        ggml_backend_meta_split_state ret;
        memset(&ret, 0, sizeof(ret));
        ret.axis = axis;
        ret.nr[0] = repeats;
        ret.n_segments = 1;
        const int64_t base = extent / repeats;
        static constexpr int64_t balanced_cumulative[5] = { 0, 270, 525, 768, 1000 };
        for (size_t j = 0; j < n_bufs; ++j) {
            auto balanced_boundary = [&](size_t boundary) {
                const int64_t raw = base*balanced_cumulative[boundary];
                return base >= 64 && base % 16 == 0 ? ((raw + 8000)/16000)*16 : raw/1000;
            };
            const int64_t lo = aw_lane_balance && n_bufs == 4 ?
                    balanced_boundary(j) : base*(int64_t) j/(int64_t) n_bufs;
            const int64_t hi = aw_lane_balance && n_bufs == 4 ?
                    balanced_boundary(j + 1) : base*(int64_t) (j + 1)/(int64_t) n_bufs;
            ret.ne[j] = hi - lo;
        }
        return ret;
    };

    auto aw_output_suffix_split = [&](int64_t n_outputs) {
        const int64_t n_tokens = ggml_backend_meta_aw_active_tokens();
        GGML_ASSERT(n_outputs > 0 && n_outputs <= n_tokens);

        ggml_backend_meta_split_state ret;
        memset(&ret, 0, sizeof(ret));
        ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
        ret.nr[0] = 1;
        ret.n_segments = 1;
        const int64_t output_begin = n_tokens - n_outputs;
        static constexpr int64_t balanced_cumulative[5] = { 0, 270, 525, 768, 1000 };
        auto token_boundary = [&](size_t boundary) {
            if (aw_lane_balance && n_bufs == 4) {
                const int64_t raw = n_tokens*balanced_cumulative[boundary];
                return n_tokens >= 64 && n_tokens % 16 == 0 ? ((raw + 8000)/16000)*16 : raw/1000;
            }
            return n_tokens*(int64_t) boundary/(int64_t) n_bufs;
        };
        for (size_t j = 0; j < n_bufs; ++j) {
            const int64_t token_lo = token_boundary(j);
            const int64_t token_hi = token_boundary(j + 1);
            ret.ne[j] = std::max<int64_t>(0, token_hi - std::max(token_lo, output_begin));
        }
        return ret;
    };

    auto aw_add_halo = [&](const ggml_backend_meta_split_state & src, ggml_backend_meta_split_axis axis, int64_t halo) {
        ggml_backend_meta_split_state ret;
        memset(&ret, 0, sizeof(ret));
        ret.axis = axis;
        ret.nr[0] = 1;
        ret.n_segments = 1;
        for (size_t j = 0; j < n_bufs; ++j) {
            ret.ne[j] = aw_split_extent(src, j) + halo;
        }
        return ret;
    };

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS &&
                !(aw_wave_token_split && ret.axis >= GGML_BACKEND_SPLIT_AXIS_2)) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (aw_wave_token_split && ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            fprintf(stderr, "AffinityWave: token split unsupported op=%s name=%s src-axes=%s,%s,%s,%s\n",
                    ggml_op_name(tensor->op), tensor->name,
                    ggml_backend_meta_split_axis_name(src_ss[0].axis),
                    ggml_backend_meta_split_axis_name(src_ss[1].axis),
                    ggml_backend_meta_split_axis_name(src_ss[2].axis),
                    ggml_backend_meta_split_axis_name(src_ss[3].axis));
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && strstr(tensor->name, "attn_gated-") != nullptr &&
                src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_0 && src_ss[0].axis <= GGML_BACKEND_SPLIT_AXIS_3 &&
                src_ss[1].axis >= GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis <= GGML_BACKEND_SPLIT_AXIS_3) {
            return src_ss[1];
        }
        // [TAG_MOE_EP] multiplicative broadcast preserves disjoint-support PARTIAL: scaling a
        // per-shard expert activation (PARTIAL) by mirrored routing weights, or gate*up. ADD/SUB
        // must NOT do this (they need the PARTIAL all-reduced first, handled elsewhere).
        if (tensor->op == GGML_OP_MUL &&
                ((src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                    (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)) ||
                 (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED))) {
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS &&
                tensor->src[0]->ne[src_ss[1].axis] == 1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (aw_wave_token_split && strstr(tensor->name, "conv_input-") != nullptr &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0 && concat_axis == GGML_BACKEND_SPLIT_AXIS_0) {
            return aw_add_halo(src_ss[1], GGML_BACKEND_SPLIT_AXIS_0, tensor->src[0]->ne[0]);
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // [TAG_MOE_EP] Stage C: expert-parallel MUL_MAT_ID. Expert weights (src0) are sharded
        // on the expert axis (2); tokens (src1) and ids (src2) are mirrored. Each device
        // computes only its local experts' contributions and zero-fills the (token,slot) rows
        // whose selected expert is non-local, so the element-wise SUM across devices equals the
        // true MoE output -> PARTIAL. The existing PARTIAL->MIRRORED all-reduce reconstructs it.
        // src1 is MIRRORED for gate/up (the FFN input) or PARTIAL for down (the GLU of gate*up,
        // which has disjoint per-device support = PARTIAL). Both yield a PARTIAL expert output.
        if (aw_wave_token_split && tensor->op == GGML_OP_MUL_MAT_ID &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[1].axis >= GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis <= GGML_BACKEND_SPLIT_AXIS_3) {
            return src_ss[1];
        }
        if (tensor->op == GGML_OP_MUL_MAT_ID && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                 src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)) {
            // assume_sync => the per-shard outputs have been all-reduced to the full (MIRRORED)
            // result; otherwise PARTIAL, which marks the all-reduce boundary (mirrors line ~603).
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis <= GGML_BACKEND_SPLIT_AXIS_3 &&
                src_ss[0].axis == src_ss[1].axis &&
                split_states_equal(src_ss[0], src_ss[1])) {
            return src_ss[0];
        }
        if (aw_wave_token_split) {
            fprintf(stderr, "AffinityWave: token split unsupported mul_mat name=%s src-axes=%s,%s"
                    " src0-ne=%lld,%lld,%lld,%lld src1-ne=%lld,%lld,%lld,%lld\\n",
                    tensor->name,
                    ggml_backend_meta_split_axis_name(src_ss[0].axis),
                    ggml_backend_meta_split_axis_name(src_ss[1].axis),
                    (long long) tensor->src[0]->ne[0], (long long) tensor->src[0]->ne[1],
                    (long long) tensor->src[0]->ne[2], (long long) tensor->src[0]->ne[3],
                    (long long) tensor->src[1]->ne[0], (long long) tensor->src[1]->ne[1],
                    (long long) tensor->src[1]->ne[2], (long long) tensor->src[1]->ne[3]);
        }
        GGML_ABORT("fatal error");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && strstr(tensor->name, "conv_state_last-") != nullptr) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (aw_wave_token_split && strstr(tensor->name, "new_state-") != nullptr) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (aw_wave_token_split && strstr(tensor->name, "attn_output-") != nullptr &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            ggml_backend_meta_split_state ret;
            memset(&ret, 0, sizeof(ret));
            ret.axis = GGML_BACKEND_SPLIT_AXIS_2;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            const int64_t halo = tensor->src[0]->ne[1] - tensor->ne[2];
            for (size_t j = 0; j < n_bufs; ++j) {
                ret.ne[j] = aw_split_extent(src_ss[0], j) - halo;
            }
            return ret;
        }
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && strcmp(tensor->name, "result_norm") == 0 &&
                src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (aw_wave_token_split && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_0 &&
                src_ss[0].axis <= GGML_BACKEND_SPLIT_AXIS_3 &&
                src_ss[1].axis >= GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis <= GGML_BACKEND_SPLIT_AXIS_3 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split &&
                (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2) &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
            return ret;
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                (tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED)) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (aw_wave_token_split) {
            fprintf(stderr, "AffinityWave: flash split unsupported name=%s axes=%s,%s,%s,%s,%s\n",
                    tensor->name,
                    ggml_backend_meta_split_axis_name(src_ss[0].axis),
                    ggml_backend_meta_split_axis_name(src_ss[1].axis),
                    ggml_backend_meta_split_axis_name(src_ss[2].axis),
                    ggml_backend_meta_split_axis_name(src_ss[3].axis),
                    ggml_backend_meta_split_axis_name(src_ss[4].axis));
        }
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret;
            memset(&ret, 0, sizeof(ret));
            ret.axis = GGML_BACKEND_SPLIT_AXIS_1;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            const int64_t halo = tensor->src[0]->ne[0] - tensor->ne[1];
            for (size_t j = 0; j < n_bufs; ++j) {
                ret.ne[j] = aw_split_extent(src_ss[0], j) - halo;
            }
            return ret;
        }
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (aw_wave_token_split && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 && src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_2 && src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return aw_add_halo(src_ss[0], GGML_BACKEND_SPLIT_AXIS_1, tensor->ne[1] - tensor->src[0]->ne[2]);
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (aw_wave_token_split && tensor->op == GGML_OP_NONE && strstr(tensor->name, "out_ids") != nullptr) {
            return aw_output_suffix_split(tensor->ne[0]);
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                if (aw_wave_token_split && strstr(tensor->name, "model.input_embed") != nullptr) {
                    split_state = aw_even_split(GGML_BACKEND_SPLIT_AXIS_1, tensor->ne[1]);
                } else if (aw_wave_token_split && strstr(tensor->name, "inp_pos") != nullptr) {
                    split_state = aw_even_split(GGML_BACKEND_SPLIT_AXIS_0, tensor->ne[0], 4);
                } else if (aw_wave_token_split && strstr(tensor->name, "out_ids") != nullptr) {
                    split_state = aw_output_suffix_split(tensor->ne[0]);
                } else if (aw_wave_token_split &&
                        (strstr(tensor->name, "self_k_idxs") != nullptr || strstr(tensor->name, "self_v_idxs") != nullptr)) {
                    split_state = aw_even_split(GGML_BACKEND_SPLIT_AXIS_0, tensor->ne[0]);
                } else if (aw_wave_token_split && strstr(tensor->name, "attn_inp_kq_mask") != nullptr) {
                    split_state = aw_even_split(GGML_BACKEND_SPLIT_AXIS_1, tensor->ne[1]);
                } else {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        const bool aw_custom_layout = aw_wave_token_split &&
                (tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_MUL_MAT_ID ||
                 tensor->op == GGML_OP_SSM_CONV || tensor->op == GGML_OP_GATED_DELTA_NET ||
                 strstr(tensor->name, "conv_input-") != nullptr ||
                 strstr(tensor->name, "attn_output-") != nullptr ||
                 strstr(tensor->name, "attn_gated-") != nullptr ||
                 strcmp(tensor->name, "result_norm") == 0);
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS && !aw_custom_layout) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
            auto scaled_lane_extent = [&](const ggml_backend_meta_split_state & src_split,
                                          const ggml_tensor * src, size_t lane) {
                const int64_t src_extent = src->ne[src_split.axis];
                GGML_ASSERT(src_extent > 0 && split_state.nr[0] > 0);
                GGML_ASSERT(tensor->ne[split_state.axis] % split_state.nr[0] == 0);
                const int64_t dst_extent = tensor->ne[split_state.axis]/split_state.nr[0];
                int64_t src_lo = 0;
                for (size_t j = 0; j < lane; ++j) {
                    src_lo += aw_split_extent(src_split, j);
                }
                const int64_t src_hi = src_lo + aw_split_extent(src_split, lane);
                return dst_extent*src_hi/src_extent - dst_extent*src_lo/src_extent;
            };

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        split_state.ne[j] = scaled_lane_extent(src_ss[i], tensor->src[i], j);
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        const int64_t expected = scaled_lane_extent(src_ss[i], tensor->src[i], j);
                        if (split_state.ne[j] != expected) {
                            GGML_ABORT("AffinityWave: inconsistent balanced split tensor=%s op=%s src=%s lane=%zu got=%lld expected=%lld",
                                    tensor->name, ggml_op_name(tensor->op), tensor->src[i]->name, j,
                                    (long long) split_state.ne[j], (long long) expected);
                        }
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        const bool aw_halo = aw_wave_token_split &&
                (strstr(tensor->name, "conv_input-") != nullptr || tensor->op == GGML_OP_GATED_DELTA_NET);
        assert(aw_halo || ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    const bool aw_wave_token_split = ggml_backend_meta_aw_token_split();
    static const bool aw_lane_balance = getenv("GGML_CUDA_AW_LANE_BALANCE") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_LANE_BALANCE"), "0") != 0;

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get();
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (aw_lane_balance && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_tensor * q = ggml_backend_meta_buffer_simple_tensor(tensor->src[0], j);
            GGML_ASSERT(q != nullptr && ne[1] > 0);
            const int64_t q_rows = q->ne[1]*q->ne[2];
            GGML_ASSERT(q_rows % ne[1] == 0);
            ne[2] = q_rows/ne[1];
            GGML_ASSERT(ggml_is_contiguous(tensor));
            nb[3] = nb[2]*ne[2];
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        // [TAG_MOE_EP] Stage C: stamp this device's expert-base offset into a reserved
        // op_params slot (index 15) for expert-parallel MUL_MAT_ID, so the CUDA kernel can map
        // global expert ids -> its local shard [base, base+ne02) and zero-fill non-local rows.
        // Encoding: 0 = no EP (default everywhere else); (base+1) = EP with that base, so device
        // 0's base 0 is still distinguishable from "not stamped".
        if (tensor->op == GGML_OP_MUL_MAT_ID && tensor->src[0] != nullptr &&
                ggml_backend_buffer_is_meta(tensor->src[0]->buffer)) {
            const ggml_backend_meta_split_state ss0 = ggml_backend_meta_get_split_state(tensor->src[0], /*assume_sync =*/ true);
            if (ss0.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                int32_t expert_base = 0;
                for (size_t k = 0; k < j; k++) {
                    for (size_t s = 0; s < ss0.n_segments; s++) {
                        expert_base += (int32_t)(ss0.ne[s*n_simple_bufs + k] * ss0.nr[s]);
                    }
                }
                t_ij->op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t) - 1] = expert_base + 1;
            }
        }
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (aw_wave_token_split && strstr(tensor->name, "conv_state_last-") != nullptr) {
                GGML_ASSERT(t_ij->view_src->ne[0] >= t_ij->ne[0]);
                t_ij->view_offs = (t_ij->view_src->ne[0] - t_ij->ne[0]) * t_ij->view_src->nb[0];
                t_ij->nb[1] = t_ij->view_src->nb[1];
                t_ij->nb[2] = t_ij->view_src->nb[2];
                t_ij->nb[3] = t_ij->view_src->nb[3];
            }
            if (aw_wave_token_split && strstr(tensor->name, "new_state-") != nullptr) {
                GGML_ASSERT(ggml_is_contiguous(t_ij));
                GGML_ASSERT(ggml_is_contiguous(t_ij->view_src));
                GGML_ASSERT(ggml_nbytes(t_ij->view_src) >= ggml_nbytes(t_ij));
                t_ij->view_offs = ggml_nbytes(t_ij->view_src) - ggml_nbytes(t_ij);
            }
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

// [TAG_MOE_EPLB] load-time expert-order permutation. The optional expert-only
// map decouples physical expert placement from router arithmetic.
static std::unordered_map<int, std::vector<int32_t>>
ggml_backend_meta_load_eplb_maps(const char * env_name) {
    std::unordered_map<int, std::vector<int32_t>> maps;
    const char * path = getenv(env_name);
    if (path == nullptr || path[0] == '\0') {
        return maps;
    }
    fprintf(stderr, "AffinityWave: loading %s %s\n",
            env_name, path);
    FILE * file = fopen(path, "r");
    if (file == nullptr) {
        GGML_LOG_WARN("%s: %s=%s not readable - IGNORED\n",
                __func__, env_name, path);
        return maps;
    }
    int layer;
    while (fscanf(file, "%d", &layer) == 1) {
        std::vector<int32_t> & permutation = maps[layer];
        int value;
        while (permutation.size() < 4096 &&
                fscanf(file, "%d", &value) == 1) {
            permutation.push_back(value);
            const int character = fgetc(file);
            if (character == '\n' || character == EOF) {
                break;
            }
            ungetc(character, file);
        }
    }
    fclose(file);
    fprintf(stderr, "AffinityWave: loaded %s layers=%zu\n",
            env_name, maps.size());
    return maps;
}

struct ggml_backend_meta_pairwave_layer {
    int pair = -1;
    std::array<int32_t, 4> owners{};
    std::array<int32_t, 2> flips{};
};

struct ggml_backend_meta_pairwave_manifest {
    bool enabled = false;
    std::array<ggml_backend_meta_pairwave_layer, 40> layers{};
};

static ggml_backend_meta_pairwave_manifest
ggml_backend_meta_load_pairwave_manifest() {
    ggml_backend_meta_pairwave_manifest manifest;
    const char * path =
            getenv("GGML_CUDA_AW_PAIRWAVE_MANIFEST");
    if (path == nullptr || path[0] == '\0') {
        return manifest;
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
            n_layers != 40 || n_devices != 4) {
        throw std::runtime_error(
                "invalid PairWave manifest header");
    }
    std::array<bool, 40> seen{};
    std::array<int, 2> layer_counts{};
    std::array<int, 2> attention_counts{};
    for (int row = 0; row < n_layers; ++row) {
        int layer = -1;
        ggml_backend_meta_pairwave_layer value;
        if (!(input >> layer >> value.pair >>
                    value.owners[0] >> value.owners[1] >>
                    value.owners[2] >> value.owners[3] >>
                    value.flips[0] >> value.flips[1])) {
            throw std::runtime_error(
                    "truncated PairWave manifest");
        }
        if (layer < 0 || layer >= n_layers || seen[layer]) {
            throw std::runtime_error(
                    "invalid or duplicate PairWave layer");
        }
        const int expected_pair =
                layer < 20 ? layer % 2 : (layer + 1) % 2;
        if (value.pair != expected_pair) {
            throw std::runtime_error(
                    "PairWave layer-pair schedule mismatch");
        }
        std::array<int, 4> owner_counts{};
        for (int group = 0; group < 4; ++group) {
            const int owner = value.owners[group];
            if (owner < value.pair*2 ||
                    owner >= value.pair*2 + 2) {
                throw std::runtime_error(
                        "PairWave group is outside its layer pair");
            }
            owner_counts[owner]++;
        }
        if (owner_counts[value.pair*2] != 2 ||
                owner_counts[value.pair*2 + 1] != 2) {
            throw std::runtime_error(
                    "PairWave requires two groups per active GPU");
        }
        if ((value.flips[0] != 0 && value.flips[0] != 1) ||
                (value.flips[1] != 0 &&
                 value.flips[1] != 1)) {
            throw std::runtime_error(
                    "invalid PairWave panel orientation");
        }
        manifest.layers[layer] = value;
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
    if (layer_counts[0] != 20 || layer_counts[1] != 20 ||
            attention_counts[0] != 5 ||
            attention_counts[1] != 5) {
        throw std::runtime_error(
                "unbalanced PairWave layer assignment");
    }
    manifest.enabled = true;
    fprintf(stderr,
            "AffinityWave: loaded PairWave manifest %s\n",
            path);
    return manifest;
}

static const std::vector<int32_t> * ggml_backend_meta_eplb_perm(
        const char * name, int64_t n_expert, int * axis_out) {
    static const std::unordered_map<int, std::vector<int32_t>>
            router_maps = ggml_backend_meta_load_eplb_maps(
                    "GGML_CUDA_MOE_EPLB_MAP");
    static const std::unordered_map<int, std::vector<int32_t>>
            expert_maps = ggml_backend_meta_load_eplb_maps(
                    "GGML_CUDA_MOE_EPLB_EXPERT_MAP");
    static const ggml_backend_meta_pairwave_manifest
            pairwave =
                    ggml_backend_meta_load_pairwave_manifest();
    static const std::unordered_map<int, std::vector<int32_t>>
            pairwave_expert_maps = [&]() {
                std::unordered_map<int,
                        std::vector<int32_t>> maps;
                if (!pairwave.enabled) {
                    return maps;
                }
                for (int layer = 0; layer < 40; ++layer) {
                    std::vector<int32_t> & permutation =
                            maps[layer];
                    permutation.reserve(256);
                    const auto router_it =
                            router_maps.find(layer);
                    if (router_it != router_maps.end() &&
                            router_it->second.size() != 256) {
                        throw std::runtime_error(
                                "PairWave router map has the wrong expert count");
                    }
                    for (int device = 0; device < 4;
                            ++device) {
                        for (int group = 0; group < 4;
                                ++group) {
                            if (pairwave.layers[layer].
                                    owners[group] != device) {
                                continue;
                            }
                            for (int local = 0; local < 64;
                                    ++local) {
                                const int physical =
                                        group*64 + local;
                                permutation.push_back(
                                        router_it ==
                                                router_maps.end() ?
                                            physical :
                                            router_it->second[
                                                physical]);
                            }
                        }
                    }
                    if (permutation.size() != 256) {
                        throw std::runtime_error(
                                "incomplete PairWave expert permutation");
                    }
                    std::array<bool, 256> source_seen{};
                    for (int32_t source : permutation) {
                        if (source < 0 || source >= 256 ||
                                source_seen[source]) {
                            throw std::runtime_error(
                                    "PairWave expert map is not a permutation");
                        }
                        source_seen[source] = true;
                    }
                }
                return maps;
            }();
    int layer = -1;
    if (sscanf(name, "blk.%d.", &layer) != 1) {
        return nullptr;
    }
    const bool is_exps = strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps");
    const bool is_inp  = strstr(name, "ffn_gate_inp") != nullptr && strstr(name, "shexp") == nullptr;
    if (!is_exps && !is_inp) {
        return nullptr;
    }
    const auto & maps =
            is_exps && pairwave.enabled ?
            pairwave_expert_maps :
            is_exps && !expert_maps.empty() ?
            expert_maps : router_maps;
    if (maps.empty()) {
        return nullptr;
    }
    const auto it = maps.find(layer);
    if (it == maps.end() || (int64_t) it->second.size() != n_expert) {
        return nullptr;
    }
    *axis_out = is_exps ? 2 : 1;
    return &it->second;
}

using ggml_backend_affinity_wave_repack_tensor_t = void (*)(const ggml_tensor * tensor);
using ggml_backend_affinity_wave_repack_tensor_async_t = void (*)(ggml_backend_t backend, const ggml_tensor * tensor);
using ggml_backend_affinity_wave_stream_fence_t = void (*)(ggml_backend_t backend, bool broadcast);

static void ggml_backend_meta_affinity_wave_repack_tensor(ggml_tensor * tensor) {
    if (getenv("GGML_CUDA_AFFINITY_WAVE") != nullptr && strstr(tensor->name, "exps") != nullptr) {
        static int probes = 0;
        if (probes++ < 8) {
            fprintf(stderr, "AffinityWave: meta repack probe %s ne=%lld,%lld,%lld\n", tensor->name,
                    (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2]);
        }
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    if (dev == nullptr) {
        static int null_dev_probes = 0;
        if (null_dev_probes++ < 8) {
            fprintf(stderr, "AffinityWave: no device for %s buffer type %s\n",
                    tensor->name, ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)));
        }
        return;
    }
    auto fn = (ggml_backend_affinity_wave_repack_tensor_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_affinity_wave_repack_tensor");
    static int callback_probes = 0;
    if (callback_probes++ < 8) {
        fprintf(stderr, "AffinityWave: repack callback %s for %s on %s\n",
                fn != nullptr ? "found" : "missing", tensor->name, ggml_backend_dev_name(dev));
    }
    if (fn != nullptr) {
        fn(tensor);
    }
}

static void ggml_backend_meta_affinity_wave_repack_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor) {
    auto fn = (ggml_backend_affinity_wave_repack_tensor_async_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),
            "ggml_backend_cuda_affinity_wave_repack_tensor_async");
    if (fn != nullptr) {
        fn(backend, tensor);
    }
}

static void ggml_backend_meta_affinity_wave_stream_fence(ggml_backend_t backend, bool broadcast) {
    auto fn = (ggml_backend_affinity_wave_stream_fence_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)),
            "ggml_backend_cuda_affinity_wave_stream_fence");
    GGML_ASSERT(fn != nullptr);
    fn(backend, broadcast);
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    // [TAG_MOE_EPLB] permute into a host scratch BEFORE the split dispatch, so the
    // existing slicing logic (any split mode) sees a plain reordered tensor.
    std::vector<char> eplb_tmp;
    {
        // scratch permute for the SMALL router tensor only (ffn_gate_inp, ~2MB);
        // the large *_exps tensors are gathered per expert in the AXIS_2 branch
        // below with NO staging (a full-tensor scratch OOM-killed the --no-mmap
        // server load: model already fills host RAM).
        int axis = -1;
        const std::vector<int32_t> * perm = nullptr;
        if (tensor->ne[2] == 1) {
            perm = ggml_backend_meta_eplb_perm(tensor->name, tensor->ne[1], &axis);
        }
        if (perm != nullptr) {
            GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
            const size_t chunk = tensor->nb[axis];
            const int64_t n    = tensor->ne[axis];
            GGML_ASSERT(chunk * n == size);
            eplb_tmp.resize(size);
            for (int64_t i = 0; i < n; ++i) {
                memcpy(eplb_tmp.data() + (size_t) i*chunk, (const char *) data + (size_t) (*perm)[i]*chunk, chunk);
            }
            data = eplb_tmp.data();
        }
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 && strstr(tensor->name, "out_ids") != nullptr) {
        GGML_ASSERT(tensor->type == GGML_TYPE_I32);
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        const int64_t n_tokens = ggml_backend_meta_aw_active_tokens();
        GGML_ASSERT(n_tokens > 0);

        const bool lane_balance = getenv("GGML_CUDA_AW_LANE_BALANCE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_LANE_BALANCE"), "0") != 0;
        static constexpr int64_t balanced_cumulative[5] = { 0, 270, 525, 768, 1000 };
        auto token_boundary = [&](size_t boundary) {
            if (lane_balance && n_bufs == 4) {
                const int64_t raw = n_tokens*balanced_cumulative[boundary];
                return n_tokens >= 64 && n_tokens % 16 == 0 ? ((raw + 8000)/16000)*16 : raw/1000;
            }
            return n_tokens*(int64_t) boundary/(int64_t) n_bufs;
        };
        size_t offset_data = 0;
        int64_t token_lo = token_boundary(0);
        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            const int64_t n_ids = simple_tensor->ne[0];
            const int64_t token_hi = token_boundary(j + 1);
            std::vector<int32_t> local_ids(n_ids);
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = ((const int32_t *) data)[offset_data/sizeof(int32_t) + i];
                GGML_ASSERT(id >= token_lo && id < token_hi);
                local_ids[i] = id - token_lo;
            }
            if (n_ids != 0) {
                ggml_backend_tensor_set(simple_tensor, local_ids.data(), 0, local_ids.size()*sizeof(int32_t));
            }
            offset_data += local_ids.size()*sizeof(int32_t);
            token_lo = token_hi;
        }
        GGML_ASSERT(offset_data == size);
        return;
    }
    if (getenv("GGML_CUDA_AFFINITY_WAVE") != nullptr && strstr(tensor->name, "exps") != nullptr) {
        static int probes = 0;
        if (probes++ < 8) {
            fprintf(stderr, "AffinityWave: meta set probe %s axis=%d segments=%u nr0=%u ne=%lld,%lld,%lld\n",
                    tensor->name, (int) split_state.axis, split_state.n_segments, split_state.nr[0],
                    (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2]);
        }
    }

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            // [TAG_MOE_EPLB] expert-axis permutation: gather source experts per
            // device slot straight from the loader buffer (no host staging).
            if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_2 && tensor->ne[2] > 1) {
                int eplb_axis = -1;
                const std::vector<int32_t> * perm = ggml_backend_meta_eplb_perm(tensor->name, tensor->ne[2], &eplb_axis);
                if (perm != nullptr) {
                    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
                    const size_t exp_bytes = tensor->nb[2];
                    int64_t slot0 = 0;
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        const int64_t ne2j = simple_tensor->ne[2];
                        for (int64_t sl = 0; sl < ne2j; sl++) {
                            ggml_backend_tensor_set(simple_tensor,
                                (const char *) data + (size_t) (*perm)[slot0 + sl]*exp_bytes,
                                (size_t) sl*exp_bytes, exp_bytes);
                        }
                        ggml_backend_meta_affinity_wave_repack_tensor(simple_tensor);
                        slot0 += ne2j;
                    }
                    GGML_ASSERT(slot0 == tensor->ne[2]);
                    break;
                }
            }
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                    ggml_backend_meta_affinity_wave_repack_tensor(simple_tensor);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            const bool aw_shared_weight = getenv("GGML_CUDA_AW_SHARED_SERVICE") != nullptr &&
                    strcmp(getenv("GGML_CUDA_AW_SHARED_SERVICE"), "0") != 0 && tensor->type == GGML_TYPE_Q8_0 &&
                    offset == 0 && size == ggml_nbytes(tensor) &&
                    (strstr(tensor->name, "ffn_gate_shexp.weight") != nullptr ||
                     strstr(tensor->name, "ffn_up_shexp.weight") != nullptr ||
                     strstr(tensor->name, "ffn_down_shexp.weight") != nullptr);
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
                if (aw_shared_weight) {
                    ggml_backend_meta_affinity_wave_repack_tensor(simple_tensor);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t compute_headroom = 16; // Maximum number of views per statically allocated tensor that can be created between evals.
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

struct ggml_backend_meta_aw_live_cell {
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

struct ggml_backend_meta_aw_headfold_gdn_lane {
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

using ggml_backend_meta_aw_live_service_t = int (*)(
        ggml_backend_t const * backends,
        const ggml_backend_meta_aw_live_cell * cells,
        int32_t n_cells,
        char * error,
        size_t error_capacity);

using ggml_backend_meta_aw_r44_prefetch_t = int (*)(
        int32_t layer,
        char * error,
        size_t error_capacity);

using ggml_backend_meta_aw_headfold_gdn_t = int (*)(
        ggml_backend_t const * backends,
        const ggml_backend_meta_aw_headfold_gdn_lane * lanes,
        int32_t n_lanes,
        char * error,
        size_t error_capacity);

using ggml_backend_meta_aw_corridor_copy_t = int (*)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst);

using ggml_backend_meta_aw_corridor_wait_t = int (*)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst);

enum ggml_backend_meta_aw_trace_action {
    GGML_BACKEND_META_AW_TRACE_PUSH = 0,
    GGML_BACKEND_META_AW_TRACE_POP = 1,
    GGML_BACKEND_META_AW_TRACE_MARK = 2,
    GGML_BACKEND_META_AW_TRACE_THREAD = 3,
};

using ggml_backend_meta_aw_trace_t = void (*)(
        int32_t action, const char * name, uint32_t category, uint64_t payload);

struct ggml_backend_meta_aw_trace_scope {
    ggml_backend_meta_aw_trace_t trace;

    ggml_backend_meta_aw_trace_scope(
            ggml_backend_meta_aw_trace_t trace,
            const char * name,
            uint32_t category,
            uint64_t payload) : trace(trace) {
        if (trace != nullptr) {
            trace(GGML_BACKEND_META_AW_TRACE_PUSH, name, category, payload);
        }
    }

    ~ggml_backend_meta_aw_trace_scope() {
        if (trace != nullptr) {
            trace(GGML_BACKEND_META_AW_TRACE_POP, nullptr, 0, 0);
        }
    }
};

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    // [TAG_META_SUBMIT] persistent per-device submission workers for prefill-sized MoE graphs
    // (env GGML_META_SUBMIT_THREADS). The per-backend graph_compute_async calls are not truly
    // async at prefill: the mul_mat_id fallback does host-side work (D2H ids copy + sort + H2D,
    // two stream syncs, one cuBLAS launch per expert), so issuing the devices from one thread
    // serializes them. Worker w services backend w+1 exclusively (backend 0 runs on the main
    // thread), keeping thread_local CUDA-side caches consistently thread-affine.
    struct submit_pool {
        struct slot {
            ggml_backend_t backend = nullptr;
            ggml_cgraph *  cgraph  = nullptr;
            ggml_status    status  = GGML_STATUS_SUCCESS;
        };
        std::mutex               m;
        std::condition_variable  cv_go;
        std::condition_variable  cv_done;
        uint64_t                 seq       = 0;
        size_t                   n_pending = 0;
        bool                     stop      = false;
        std::vector<slot>        slots;
        std::vector<std::thread> threads;
        // [TAG_META_SUBMIT] mode-2 (SPMD): when set, worker w runs (*spmd_job)(w+1) — the whole
        // per-device subgraph loop including that device's allreduces — instead of one cgraph.
        const std::function<ggml_status(size_t)> * spmd_job = nullptr;

        explicit submit_pool(size_t n_workers) : slots(n_workers) {
            threads.reserve(n_workers);
            for (size_t w = 0; w < n_workers; w++) {
                threads.emplace_back([this, w]() {
                    uint64_t seen = 0;
                    for (;;) {
                        std::unique_lock<std::mutex> lock(m);
                        cv_go.wait(lock, [&]() { return stop || seq != seen; });
                        if (stop) {
                            return;
                        }
                        seen = seq;
                        slot & s = slots[w];
                        const auto * job = spmd_job;
                        lock.unlock();
                        s.status = job != nullptr ? (*job)(w + 1)
                                                  : ggml_backend_graph_compute_async(s.backend, s.cgraph);
                        lock.lock();
                        n_pending--;
                        if (n_pending == 0) {
                            cv_done.notify_one();
                        }
                    }
                });
            }
        }

        ~submit_pool() {
            {
                std::lock_guard<std::mutex> lock(m);
                stop = true;
            }
            cv_go.notify_all();
            for (auto & t : threads) {
                t.join();
            }
        }
    };

    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    int64_t                     aw_precaptured_tokens = -1;
    std::unordered_set<uint64_t> aw_precaptured_signatures;
    bool                        aw_canonical_state_valid = false;
    int64_t                     aw_canonical_tokens = 0;
    uint64_t                    aw_canonical_epoch = 0;
    std::vector<ggml_backend_buffer_ptr> aw_input_snapshot_buffers;
    std::unique_ptr<submit_pool> submit;

    void *                                      comm_ctx              = nullptr;
    ggml_backend_comm_allreduce_tensor_t        comm_allreduce        = nullptr;
    ggml_backend_comm_allreduce_tensor_single_t comm_allreduce_single = nullptr; // [TAG_META_SUBMIT] optional, SPMD mode
    ggml_backend_moe_asyncep_register_t         asyncep_register      = nullptr; // [TAG_MOE_ASYNCEP] optional
    ggml_backend_moe_asyncep_set_enabled_t      asyncep_set_enabled   = nullptr; // [TAG_MOE_ASYNCEP] optional
    ggml_backend_tbo_begin_eval_t               tbo_begin_eval        = nullptr; // [TAG_META_TBO] optional
    ggml_backend_tbo_end_eval_t                 tbo_end_eval          = nullptr; // [TAG_META_TBO] optional
    ggml_backend_meta_aw_live_service_t         aw_live_service       = nullptr;
    ggml_backend_meta_aw_r44_prefetch_t         aw_r44_prefetch       = nullptr;
    ggml_backend_meta_aw_headfold_gdn_t         aw_headfold_gdn       = nullptr;
    ggml_backend_meta_aw_corridor_copy_t        aw_corridor_copy      = nullptr;
    ggml_backend_meta_aw_corridor_wait_t        aw_corridor_wait      = nullptr;
    ggml_backend_meta_aw_trace_t                aw_trace              = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
            comm_allreduce_single = (ggml_backend_comm_allreduce_tensor_single_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor_single");
            asyncep_register = (ggml_backend_moe_asyncep_register_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_moe_asyncep_register");
            asyncep_set_enabled = (ggml_backend_moe_asyncep_set_enabled_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_moe_asyncep_set_enabled");
            tbo_begin_eval = (ggml_backend_tbo_begin_eval_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_tbo_begin_eval");
            tbo_end_eval = (ggml_backend_tbo_end_eval_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_tbo_end_eval");
        }
        aw_live_service = (ggml_backend_meta_aw_live_service_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_live_service");
        aw_r44_prefetch = (ggml_backend_meta_aw_r44_prefetch_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_r44_prefetch");
        aw_headfold_gdn = (ggml_backend_meta_aw_headfold_gdn_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_headfold_gdn");
        aw_corridor_copy = (ggml_backend_meta_aw_corridor_copy_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_corridor_copy");
        aw_corridor_wait = (ggml_backend_meta_aw_corridor_wait_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_corridor_wait");
        aw_trace = (ggml_backend_meta_aw_trace_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                ggml_backend_get_device(simple_backends[0])), "ggml_backend_cuda_affinity_wave_trace");
    }

    ~ggml_backend_meta_context() {
        // [TAG_META_SUBMIT] join workers before the backends they reference are freed below
        submit.reset();
        // Drain in-flight work on every device before tearing down the comm: an asynchronously
        // replayed decode CUDA graph may still hold NCCL kernels in flight on secondary devices,
        // and destroying the comm under them poisons their context ("unspecified launch failure"
        // at the subsequent cudaStreamDestroy).
        for (auto & bc : backend_configs) {
            ggml_backend_synchronize(bc.backend);
        }
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 && strstr(tensor->name, "out_ids") != nullptr) {
        GGML_ASSERT(tensor->type == GGML_TYPE_I32);
        GGML_ASSERT(size == ggml_nbytes(tensor));
        const int64_t n_tokens = ggml_backend_meta_aw_active_tokens();
        GGML_ASSERT(n_tokens > 0);

        const bool lane_balance = getenv("GGML_CUDA_AW_LANE_BALANCE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_LANE_BALANCE"), "0") != 0;
        static constexpr int64_t balanced_cumulative[5] = { 0, 270, 525, 768, 1000 };
        auto token_boundary = [&](size_t boundary) {
            if (lane_balance && n_backends == 4) {
                const int64_t raw = n_tokens*balanced_cumulative[boundary];
                return n_tokens >= 64 && n_tokens % 16 == 0 ? ((raw + 8000)/16000)*16 : raw/1000;
            }
            return n_tokens*(int64_t) boundary/(int64_t) n_backends;
        };
        size_t offset_data = 0;
        int64_t token_lo = token_boundary(0);
        for (size_t j = 0; j < n_backends; ++j) {
            ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
            const int64_t n_ids = simple_tensor->ne[0];
            const int64_t token_hi = token_boundary(j + 1);
            std::vector<int32_t> local_ids(n_ids);
            for (int64_t i = 0; i < n_ids; ++i) {
                const int32_t id = ((const int32_t *) data)[offset_data/sizeof(int32_t) + i];
                GGML_ASSERT(id >= token_lo && id < token_hi);
                local_ids[i] = id - token_lo;
            }
            if (n_ids != 0) {
                ggml_backend_tensor_set(simple_tensor, local_ids.data(), 0, local_ids.size()*sizeof(int32_t));
            }
            offset_data += local_ids.size()*sizeof(int32_t);
            token_lo = token_hi;
        }
        GGML_ASSERT(offset_data == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                    ggml_backend_meta_affinity_wave_repack_tensor_async(simple_backend, simple_tensor);
                }
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static ggml_tensor * ggml_backend_meta_aw_view_root(ggml_tensor * tensor) {
    while (tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

// Limit copies to the token-major KV view actually consumed by attention.
// Unsupported layouts retain the original full-root copy behavior.
static void ggml_backend_meta_aw_kv_prefix(ggml_tensor & root, const ggml_tensor * view) {
    const char * enabled = getenv("GGML_CUDA_AW_SERVE_KV_PREFIX");
    if (!ggml_backend_meta_aw_serving() || enabled == nullptr || strcmp(enabled, "1") != 0 ||
            root.ne[2] != 1 || root.ne[3] != 1 || view->ne[3] != 1 ||
            view->type != root.type || view->ne[1] <= 0 || view->ne[1] > root.ne[1] ||
            view->ne[0]*view->ne[2] != root.ne[0] ||
            view->nb[1] != root.nb[1] || !ggml_is_contiguous(&root)) {
        return;
    }
    if (view->nb[2] != ggml_row_size(root.type, view->ne[0])) {
        return;
    }
    for (const ggml_tensor * tensor = view; tensor->view_src != nullptr; tensor = tensor->view_src) {
        if (tensor->view_offs != 0) {
            return;
        }
    }
    root.ne[1] = view->ne[1];
    root.nb[2] = root.nb[1]*root.ne[1];
    root.nb[3] = root.nb[2];
}

// Restrict a canonical KV copy to newly appended rows when the destination was
// fully canonicalized by the preceding serving request. Unsupported layouts
// deliberately fall back to the existing full-copy path.
static bool ggml_backend_meta_aw_kv_suffix(
        ggml_tensor & src, ggml_tensor & dst, const int64_t previous_tokens) {
    if (previous_tokens <= 0 || src.type != dst.type ||
            src.ne[2] != 1 || src.ne[3] != 1 ||
            src.ne[1] != dst.ne[1] || src.ne[1] <= previous_tokens ||
            src.ne[0] != dst.ne[0] || src.nb[0] != dst.nb[0] || src.nb[1] != dst.nb[1] ||
            src.nb[2] != dst.nb[2] || src.nb[3] != dst.nb[3] ||
            src.nb[1] == 0 || !ggml_is_contiguous(&src) || !ggml_is_contiguous(&dst)) {
        return false;
    }
    const int64_t suffix_tokens = src.ne[1] - previous_tokens;
    src.data = (char *) src.data + (size_t) previous_tokens * src.nb[1];
    dst.data = (char *) dst.data + (size_t) previous_tokens * dst.nb[1];
    src.ne[1] = suffix_tokens;
    dst.ne[1] = suffix_tokens;
    src.nb[2] = src.nb[1] * src.ne[1];
    dst.nb[2] = dst.nb[1] * dst.ne[1];
    src.nb[3] = src.nb[2];
    dst.nb[3] = dst.nb[2];
    return ggml_are_same_layout(&src, &dst);
}

struct ggml_backend_meta_aw_cell {
    int layer;
    int begin;
    int expert_begin;
    int expert_end;
    int shared_begin;
    int shared_raw;
    int shared_end;
    int post_begin;
    int end;
    int conv_input;
    int conv_state_last;
    int conv_state_update;
    int conv_state_clear;
    int conv;
    int q_predelta;
    int k_predelta;
    int v_predelta;
    int gate;
    int beta;
    int recurrent;
    int new_state;
    int state_update;
    int state_clear;
    int flash_attn;
    int attention_state_end;
};

static bool ggml_backend_meta_aw_name_is(const ggml_tensor * tensor, const char * base, int layer) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "%s-%d", base, layer);
    return strcmp(tensor->name, name) == 0;
}

static bool ggml_backend_meta_aw_name_contains(const ggml_tensor * tensor, const char * base, int layer) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "%s-%d", base, layer);
    return strstr(tensor->name, name) != nullptr;
}

static std::vector<ggml_backend_meta_aw_cell> ggml_backend_meta_aw_partition(const ggml_cgraph * cgraph) {
    std::vector<ggml_backend_meta_aw_cell> cells;
    int begin = 0;

    for (int layer = 0; ; ++layer) {
        ggml_backend_meta_aw_cell cell = {
            layer, begin, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1
        };
        for (int i = begin; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            char conv_cache_name[GGML_MAX_NAME];
            char state_cache_name[GGML_MAX_NAME];
            snprintf(conv_cache_name, sizeof(conv_cache_name), "cache_r_l%d", layer);
            snprintf(state_cache_name, sizeof(state_cache_name), "cache_s_l%d", layer);
            ggml_tensor * root = node->view_src != nullptr ?
                    ggml_backend_meta_aw_view_root(const_cast<ggml_tensor *>(node)) : nullptr;
            if (node->op == GGML_OP_SCALE && root != nullptr &&
                    strcmp(root->name, conv_cache_name) == 0) {
                cell.conv_state_clear = i;
            } else if (node->op == GGML_OP_SCALE && root != nullptr &&
                    strcmp(root->name, state_cache_name) == 0) {
                cell.state_clear = i;
            } else if (ggml_backend_meta_aw_name_is(node, "conv_input", layer)) {
                cell.conv_input = i;
            } else if (ggml_backend_meta_aw_name_is(node, "conv_state_last", layer)) {
                cell.conv_state_last = i;
            } else if (ggml_backend_meta_aw_name_is(node, "conv_state_update", layer)) {
                cell.conv_state_update = i;
            } else if (node->op == GGML_OP_SSM_CONV) {
                cell.conv = i;
            } else if (ggml_backend_meta_aw_name_is(node, "q_conv_predelta", layer)) {
                cell.q_predelta = i;
            } else if (ggml_backend_meta_aw_name_is(node, "k_conv_predelta", layer)) {
                cell.k_predelta = i;
            } else if (ggml_backend_meta_aw_name_is(node, "v_conv_predelta", layer)) {
                cell.v_predelta = i;
            } else if (node->op == GGML_OP_RESHAPE &&
                    node->src[0] != nullptr &&
                    ggml_backend_meta_aw_name_is(
                        node->src[0], "gate", layer)) {
                cell.gate = i;
            } else if (ggml_backend_meta_aw_name_is(node, "beta_sigmoid", layer)) {
                cell.beta = i;
            } else if (node->op == GGML_OP_GATED_DELTA_NET) {
                cell.recurrent = i;
            } else if (ggml_backend_meta_aw_name_is(node, "new_state", layer)) {
                cell.new_state = i;
            } else if (node->op == GGML_OP_CPY && node->src[0] != nullptr &&
                    ggml_backend_meta_aw_name_is(node->src[0], "new_state", layer)) {
                cell.state_update = i;
            } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                cell.flash_attn = i;
            } else if (node->op == GGML_OP_SET_ROWS) {
                char state_name[GGML_MAX_NAME];
                snprintf(state_name, sizeof(state_name), "cache_v_l%d", layer);
                if (strncmp(node->name, state_name, strlen(state_name)) == 0) {
                    cell.attention_state_end = i;
                }
            } else if ((ggml_backend_meta_aw_name_is(node, "ffn_moe_gate", layer) ||
                        ggml_backend_meta_aw_name_is(node, "ffn_moe_up", layer)) &&
                    node->op == GGML_OP_MUL_MAT_ID && cell.expert_begin < 0) {
                cell.expert_begin = i;
            } else if (ggml_backend_meta_aw_name_is(node, "ffn_moe_out", layer)) {
                cell.expert_end = i;
            } else if (ggml_backend_meta_aw_name_is(
                        node, "aw_dense_f16_shared", layer)) {
                cell.shared_begin = i;
            } else if (ggml_backend_meta_aw_name_is(node, "ffn_gate", layer) &&
                    node->op == GGML_OP_MUL_MAT &&
                    cell.shared_begin < 0) {
                cell.shared_begin = i;
            } else if (ggml_backend_meta_aw_name_is(node, "ffn_shexp", layer)) {
                cell.shared_raw = i;
            } else if (ggml_backend_meta_aw_name_is(node, "ffn_shexp_gated", layer)) {
                cell.shared_end = i;
            } else if (ggml_backend_meta_aw_name_is(node, "ffn_out", layer)) {
                cell.post_begin = i;
            } else if (ggml_backend_meta_aw_name_is(node, "l_out", layer)) {
                cell.end = i;
                break;
            }
        }

        if (cell.end < 0) {
            break;
        }
        GGML_ASSERT(cell.expert_begin >= cell.begin);
        GGML_ASSERT(cell.expert_end   >= cell.expert_begin);
        GGML_ASSERT(cell.shared_begin == cell.expert_end + 1);
        GGML_ASSERT(cell.shared_raw   >= cell.shared_begin);
        GGML_ASSERT(cell.shared_raw   <  cell.shared_end);
        GGML_ASSERT(cell.shared_end   >= cell.shared_begin);
        GGML_ASSERT(cell.post_begin   == cell.shared_end + 1);
        GGML_ASSERT(cell.end          == cell.post_begin + 1);
        GGML_ASSERT((cell.recurrent >= 0) != (cell.flash_attn >= 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.conv_input < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.conv_state_last < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.conv_state_update < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.conv_state_clear < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.conv < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.q_predelta < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.k_predelta < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.v_predelta < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.gate < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.beta < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.new_state < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.state_update < 0));
        GGML_ASSERT((cell.recurrent < 0) == (cell.state_clear < 0));
        GGML_ASSERT((cell.flash_attn < 0) == (cell.attention_state_end < 0));
        GGML_ASSERT(cell.attention_state_end < 0 ||
                (cell.attention_state_end >= cell.begin && cell.attention_state_end < cell.flash_attn));

        const ggml_tensor * post = cgraph->nodes[cell.post_begin];
        GGML_ASSERT(post->op == GGML_OP_ADD);
        GGML_ASSERT(post->src[0] == cgraph->nodes[cell.expert_end]);
        GGML_ASSERT(post->src[1] == cgraph->nodes[cell.shared_end]);

        const ggml_tensor * out = cgraph->nodes[cell.end];
        GGML_ASSERT(out->op == GGML_OP_ADD);
        GGML_ASSERT(out->src[0] == post);
        GGML_ASSERT(out->src[1] != nullptr && ggml_backend_meta_aw_name_is(out->src[1], "attn_residual", layer));

        std::unordered_set<const ggml_tensor *> expert_nodes;
        for (int i = cell.expert_begin; i <= cell.expert_end; ++i) {
            expert_nodes.insert(cgraph->nodes[i]);
        }
        for (int i = cell.shared_begin; i <= cell.shared_end; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                GGML_ASSERT(expert_nodes.count(node->src[s]) == 0);
            }
        }

        cells.push_back(cell);
        begin = cell.end + 1;
    }

    return cells;
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    const uint64_t state_epoch = ggml_backend_meta_aw_state_epoch.load(std::memory_order_relaxed);
    if (backend_ctx->aw_canonical_epoch != state_epoch) {
        backend_ctx->aw_canonical_state_valid = false;
        backend_ctx->aw_canonical_tokens = 0;
        backend_ctx->aw_canonical_epoch = state_epoch;
    } else if (backend_ctx->aw_canonical_state_valid) {
        const int64_t trimmed_tokens =
                ggml_backend_meta_aw_last_canonical_tokens.load(std::memory_order_relaxed);
        if (trimmed_tokens != backend_ctx->aw_canonical_tokens) {
            backend_ctx->aw_canonical_tokens = trimmed_tokens;
        }
    }

    // [TAG_META_SUBMIT] engage per-device submission threads only for prefill-sized MoE graphs:
    // decode-sized graphs measured NEGATIVE under a threaded driver (scheduler jitter on tiny
    // launches), while at prefill the host-side mul_mat_id sort/launch work dominates and
    // overlaps across devices. Gate = graph contains a MUL_MAT_ID with >= min_tokens tokens.
    // Mode 1 (any value): per-subgraph worker dispatch, allreduce on the main thread.
    // Mode 2 ("2", SPMD): each worker runs its device's WHOLE subgraph loop incl. allreduce.
    static const int submit_threads_mode = []() {
        const char * s = getenv("GGML_META_SUBMIT_THREADS");
        if (s == nullptr) {
            return 0;
        }
        return atoi(s) >= 2 ? 2 : 1;
    }();
    static const bool submit_threads = submit_threads_mode != 0;
    static const int64_t submit_min_tokens = []() {
        const char * s = getenv("GGML_META_SUBMIT_MIN_TOKENS");
        return s != nullptr ? atoll(s) : 64;
    }();
    bool use_submit_threads = false;
    if (submit_threads && n_backends > 1) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] >= submit_min_tokens) {
                use_submit_threads = true;
                break;
            }
        }
    }
    if (use_submit_threads && !backend_ctx->submit) {
        backend_ctx->submit.reset(new ggml_backend_meta_context::submit_pool(n_backends - 1));
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    if (needs_rebuild && getenv("GGML_CUDA_AW_WAVE_DRY") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_WAVE_DRY"), "0") != 0) {
        static std::atomic<bool> planned{false};
        if (!planned.load()) {
            const std::vector<ggml_backend_meta_aw_cell> cells = ggml_backend_meta_aw_partition(cgraph);
            if (cells.size() == 40 && !planned.exchange(true)) {
                GGML_ASSERT(n_backends == 4);
                fprintf(stderr, "AffinityWave: validated %zu layer cells, %zu token lanes, %zu wave diagonals\n",
                    cells.size(), n_backends, cells.size() + n_backends - 1);
                for (const ggml_backend_meta_aw_cell & cell : cells) {
                    fprintf(stderr, "AffinityWave: cell l=%d pre=[%d,%d) expert=[%d,%d] shared=[%d,%d] post=[%d,%d]\n",
                        cell.layer, cell.begin, cell.expert_begin, cell.expert_begin, cell.expert_end,
                        cell.shared_begin, cell.shared_end, cell.post_begin, cell.end);
                }
            }
        }
    }

    if (needs_rebuild && getenv("GGML_CUDA_AW_WAVE_INSPECT") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_WAVE_INSPECT"), "0") != 0) {
        static std::atomic<bool> inspected{false};
        if (!inspected.exchange(true)) {
            fprintf(stderr, "AffinityWave: graph census nodes=%d leafs=%d\n", cgraph->n_nodes, cgraph->n_leafs);
            const bool full = strcmp(getenv("GGML_CUDA_AW_WAVE_INSPECT"), "full") == 0;
            int full_begin = 0;
            int full_end = std::min(cgraph->n_nodes, 110);
            const char * inspect_layer_env = getenv("GGML_CUDA_AW_WAVE_INSPECT_LAYER");
            if (full && inspect_layer_env != nullptr && inspect_layer_env[0] != '\0') {
                const int inspect_layer = atoi(inspect_layer_env);
                const std::vector<ggml_backend_meta_aw_cell> inspect_cells =
                        ggml_backend_meta_aw_partition(cgraph);
                if (inspect_layer >= 0 && inspect_layer < (int) inspect_cells.size()) {
                    full_begin = inspect_cells[inspect_layer].begin;
                    full_end = inspect_cells[inspect_layer].end + 1;
                }
            }
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                const char * name = node->name;
                const bool boundary = node->op == GGML_OP_NONE || strstr(name, "attn_post_norm") != nullptr ||
                        strstr(name, "ffn_moe_out") != nullptr || strstr(name, "post_moe") != nullptr ||
                        strstr(name, "l_out") != nullptr || strstr(name, "cache_") != nullptr;
                if (!boundary && !(full && i >= full_begin && i < full_end)) {
                    continue;
                }
                const bool is_meta = node->buffer != nullptr && ggml_backend_buffer_is_meta(node->buffer);
                const ggml_backend_meta_split_state ss = is_meta ? ggml_backend_meta_get_split_state(node, false) :
                        ggml_backend_meta_split_state{GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                fprintf(stderr, "AffinityWave: graph node=%d op=%s name=%s ne=%lld,%lld,%lld,%lld split=%s src=%s|%s|%s\n",
                        i, ggml_op_name(node->op), name,
                        (long long) node->ne[0], (long long) node->ne[1],
                        (long long) node->ne[2], (long long) node->ne[3],
                        ggml_backend_meta_split_axis_name(ss.axis),
                        node->src[0] != nullptr ? node->src[0]->name : "",
                        node->src[1] != nullptr ? node->src[1]->name : "",
                        node->src[2] != nullptr ? node->src[2]->name : "");
            }
            if (full) {
                std::unordered_set<const ggml_tensor *> graph_nodes;
                std::set<const ggml_tensor *> external;
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    graph_nodes.insert(cgraph->nodes[i]);
                }
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        const ggml_tensor * src = cgraph->nodes[i]->src[s];
                        if (src != nullptr && graph_nodes.count(src) == 0 && src->buffer != nullptr &&
                                ggml_backend_buffer_is_meta(src->buffer) &&
                                ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                            external.insert(src);
                        }
                    }
                }
                for (const ggml_tensor * src : external) {
                    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(src, false);
                    fprintf(stderr, "AffinityWave: graph external name=%s ne=%lld,%lld,%lld,%lld split=%s\n",
                            src->name, (long long) src->ne[0], (long long) src->ne[1],
                            (long long) src->ne[2], (long long) src->ne[3],
                            ggml_backend_meta_split_axis_name(ss.axis));
                }
            }
        }
    }

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                // [TAG_MOE_EP] Expert-parallel MoE chain: gate/up MUL_MAT_ID outputs are
                // disjoint-support partials (each (token,slot) row is wholly owned by the device
                // holding its expert, zeros elsewhere) and every op between them and the down
                // MUL_MAT_ID maps zero rows to zero rows (GLU/SILU/GELU/RELU/CLAMP/MUL). The
                // all-reduce can therefore be delayed through the whole expert FFN to the
                // weighted slot-sum handled below, removing 2 of the 3 FFN all-reduces per
                // layer. Guarded on the expert weights being sharded on the expert axis
                // (SPLIT_AXIS_2) so general row/col TP behavior is unchanged.
                auto is_ep_mmid = [&](const ggml_tensor * t) {
                    return t->op == GGML_OP_MUL_MAT_ID && t->src[0] != nullptr &&
                        ggml_backend_meta_get_split_state(t->src[0], false).axis == GGML_BACKEND_SPLIT_AXIS_2;
                };
                if (is_ep_mmid(node)) {
                    ggml_tensor * part[6]     = { node };
                    int           part_id[6]  = { id, -1, -1, -1, -1, -1 };
                    bool          consumed[6] = { false };
                    int n_part = 1;
                    int id_ep  = id;
                    while (id_ep + 1 < cgraph->n_nodes && n_part < 6) {
                        ggml_tensor * next = cgraph->nodes[id_ep+1];
                        auto part_idx = [&](const ggml_tensor * t) {
                            for (int k = 0; k < n_part; k++) {
                                if (part[k] == t) {
                                    return k;
                                }
                            }
                            return -1;
                        };
                        // The down MUL_MAT_ID consuming the chain as its input ends the walk.
                        if (is_ep_mmid(next) && next->src[1] != nullptr && part_idx(next->src[1]) >= 0) {
                            consumed[part_idx(next->src[1])] = true;
                            // Safe only if every chain member was consumed exactly once, inside
                            // the chain: a member with another (later) consumer would be read
                            // before its all-reduce.
                            bool safe = true;
                            for (int q = 0; q < n_part; q++) {
                                safe = safe && consumed[q] && ggml_node_get_use_count(cgraph, part_id[q]) == 1;
                            }
                            if (safe) {
                                node   = next;
                                id     = id_ep + 1;
                                idr    = id;
                                n_used = ggml_node_get_use_count(cgraph, id);
                            }
                            break;
                        }
                        // The sibling gate/up MUL_MAT_ID over the mirrored FFN input.
                        if (is_ep_mmid(next) && next->src[1] != nullptr && part_idx(next->src[1]) < 0 &&
                                ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            part_id[n_part] = id_ep + 1;
                            part[n_part++]  = next;
                            id_ep++;
                            continue;
                        }
                        // Zero-preserving elementwise op over chain members (+ mirrored operands).
                        const bool zero_preserving =
                            next->op == GGML_OP_GLU || next->op == GGML_OP_MUL || next->op == GGML_OP_CLAMP ||
                            (next->op == GGML_OP_UNARY && (ggml_get_unary_op(next) == GGML_UNARY_OP_SILU ||
                                                           ggml_get_unary_op(next) == GGML_UNARY_OP_GELU ||
                                                           ggml_get_unary_op(next) == GGML_UNARY_OP_RELU));
                        bool consumes = false;
                        bool srcs_ok  = true;
                        for (int s = 0; s < GGML_MAX_SRC && srcs_ok; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            const int k = part_idx(next->src[s]);
                            if (k >= 0) {
                                // each chain member must have exactly one consumer (this one)
                                consumes = true;
                                srcs_ok  = srcs_ok && !consumed[k];
                                consumed[k] = true;
                            } else if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                srcs_ok = false;
                            }
                        }
                        if (!consumes && srcs_ok) { // unrelated mirrored node interleaved in the chain
                            id_ep++;
                            continue;
                        }
                        if (!srcs_ok || !zero_preserving) {
                            break; // unsafe -> keep the immediate all-reduce at node i
                        }
                        part_id[n_part] = id_ep + 1;
                        part[n_part++]  = next;
                        id_ep++;
                    }
                }

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // The tmp buffers only ever hold the tensor that is actually all-reduced, which
                // with a delayed all-reduce is nodes[i_delayed] (e.g. the weighted expert SUM,
                // [ne0, n_tokens]) — NOT the (much larger) boundary node itself (e.g. the
                // [ne0, n_expert_used, n_tokens] MUL_MAT_ID output). Sizing by the boundary node
                // over-allocated the tmp buffers up to 4x and OOMed large-ubatch configs.
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(cgraph->nodes[i_delayed]));
                }

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                // Resetting ctx invalidates every old graph, including slots
                // beyond this graph's current subgraph count. Recreate the
                // full recorded capacity so later shape changes cannot reuse
                // dangling pointers or graphs with undersized node arrays.
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    if (getenv("GGML_CUDA_AW_WAVE_TOKEN_PLAN") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_WAVE_TOKEN_PLAN"), "0") != 0) {
        const std::vector<ggml_backend_meta_aw_cell> cells = ggml_backend_meta_aw_partition(cgraph);
        if (cells.size() == 40) {
            int recurrent = 0;
            for (size_t j = 0; j < n_backends; ++j) {
                const auto & bcj = backend_ctx->backend_configs[j];
                for (const ggml_backend_meta_aw_cell & cell : cells) {
                    const ggml_tensor * out = bcj.nodes[cell.end];
                    GGML_ASSERT(out->ne[1] * (int64_t) n_backends == cgraph->nodes[cell.end]->ne[1]);
                    const ggml_tensor * expert = bcj.nodes[cell.expert_begin];
                    GGML_ASSERT(expert->ne[2] * (int64_t) n_backends == cgraph->nodes[cell.expert_begin]->ne[2]);
                    for (int i = cell.begin; i < cell.expert_begin; ++i) {
                        const ggml_tensor * original = cgraph->nodes[i];
                        const ggml_tensor * simple = bcj.nodes[i];
                        if (strstr(original->name, "conv_input-") != nullptr) {
                            const int64_t halo = original->src[0]->ne[0];
                            GGML_ASSERT(simple->ne[0] == (original->ne[0] - halo) / (int64_t) n_backends + halo);
                            recurrent++;
                        }
                        if (original->op == GGML_OP_GATED_DELTA_NET) {
                            const int64_t halo = original->ne[1] - original->src[0]->ne[2];
                            GGML_ASSERT(simple->ne[1] == (original->ne[1] - halo) / (int64_t) n_backends + halo);
                        }
                    }
                }
            }
            fprintf(stderr, "AffinityWave: token plan validated lanes=%zu layers=%zu recurrent-cells=%d; execution withheld\n",
                    n_backends, cells.size(), recurrent / (int) n_backends);
            return GGML_STATUS_ABORTED;
        }
    }

    if ((!ggml_backend_meta_aw_serving() || ggml_backend_meta_aw_active_tokens() != 0) &&
            getenv("GGML_CUDA_AW_WAVE_DENSE_BENCH") != nullptr &&
            strcmp(getenv("GGML_CUDA_AW_WAVE_DENSE_BENCH"), "0") != 0) {
        const int64_t dense_begin_us = ggml_time_us();
        const bool fixed_only = strcmp(getenv("GGML_CUDA_AW_WAVE_DENSE_BENCH"), "fixed") == 0;
        const bool live_service = strcmp(getenv("GGML_CUDA_AW_WAVE_DENSE_BENCH"), "service") == 0;
        const bool headfold = live_service && getenv("GGML_CUDA_AW_HEADFOLD") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_HEADFOLD"), "0") != 0;
        const bool headfold_gdn = headfold &&
                (getenv("GGML_CUDA_AW_HEADFOLD_GDN") == nullptr ||
                 strcmp(getenv("GGML_CUDA_AW_HEADFOLD_GDN"), "0") != 0);
        const bool headfold_attention = headfold &&
                (getenv("GGML_CUDA_AW_HEADFOLD_ATTENTION") == nullptr ||
                 strcmp(getenv("GGML_CUDA_AW_HEADFOLD_ATTENTION"), "0") != 0);
        const bool headfold_split_pre = headfold &&
                getenv("GGML_CUDA_AW_HEADFOLD_SPLIT_PRE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_HEADFOLD_SPLIT_PRE"), "0") != 0;
        const bool headfold_r44 = headfold &&
                getenv("GGML_CUDA_AW_R44") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_R44"), "0") != 0;
        const bool return_output = live_service && getenv("GGML_CUDA_AW_WAVE_OUTPUT") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_WAVE_OUTPUT"), "0") != 0;
        const bool shared_service = live_service && getenv("GGML_CUDA_AW_SHARED_SERVICE") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_SHARED_SERVICE"), "0") != 0;
        const bool verify_local = live_service && getenv("GGML_CUDA_AW_VERIFY_LOCAL") != nullptr &&
                strcmp(getenv("GGML_CUDA_AW_VERIFY_LOCAL"), "0") != 0;
        const std::vector<ggml_backend_meta_aw_cell> cells = ggml_backend_meta_aw_partition(cgraph);
        if (cells.size() == 40 && n_backends == 4) {
            GGML_ASSERT(!headfold || (backend_ctx->aw_headfold_gdn != nullptr &&
                    backend_ctx->aw_corridor_copy != nullptr &&
                    backend_ctx->aw_corridor_wait != nullptr));
            GGML_ASSERT(!headfold_r44 ||
                    backend_ctx->aw_r44_prefetch != nullptr);
            GGML_ASSERT(!headfold || !verify_local);
            const char * inspect_simple_layer_env = getenv("GGML_CUDA_AW_WAVE_INSPECT_SIMPLE_LAYER");
            if (inspect_simple_layer_env != nullptr && inspect_simple_layer_env[0] != '\0') {
                static std::atomic<bool> inspected_simple{false};
                const int inspect_layer = atoi(inspect_simple_layer_env);
                if (inspect_layer >= 0 && inspect_layer < (int) cells.size() &&
                        !inspected_simple.exchange(true)) {
                    const ggml_backend_meta_aw_cell & inspect_cell = cells[inspect_layer];
                    for (size_t j = 0; j < n_backends; ++j) {
                        const auto & bcj = backend_ctx->backend_configs[j];
                        for (int i = inspect_cell.begin; i <= inspect_cell.end; ++i) {
                            const ggml_tensor * node = bcj.nodes[i];
                            fprintf(stderr,
                                    "AffinityWave: simple lane=%zu layer=%d node=%d op=%s name=%s "
                                    "ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu view=%zu\n",
                                    j, inspect_layer, i, ggml_op_name(node->op), node->name,
                                    (long long) node->ne[0], (long long) node->ne[1],
                                    (long long) node->ne[2], (long long) node->ne[3],
                                    node->nb[0], node->nb[1], node->nb[2], node->nb[3],
                                    node->view_offs);
                            for (int s = 0; s < GGML_MAX_SRC && node->src[s] != nullptr; ++s) {
                                const ggml_tensor * src = node->src[s];
                                fprintf(stderr,
                                        "AffinityWave: simple-src lane=%zu node=%d src=%d name=%s "
                                        "ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu view=%zu\n",
                                        j, i, s, src->name,
                                        (long long) src->ne[0], (long long) src->ne[1],
                                        (long long) src->ne[2], (long long) src->ne[3],
                                        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                                        src->view_offs);
                            }
                        }
                    }
                }
            }
            if (!backend_ctx->submit) {
                backend_ctx->submit.reset(new ggml_backend_meta_context::submit_pool(n_backends - 1));
            }
            struct input_snapshot {
                ggml_tensor * tensor;
                void * original_data;
                size_t offset;
                ggml_tensor persistent;
            };
            std::array<std::vector<input_snapshot>, 4> input_snapshots;
            if (backend_ctx->aw_input_snapshot_buffers.size() != n_backends) {
                backend_ctx->aw_input_snapshot_buffers.resize(n_backends);
            }
            struct input_snapshot_restore {
                std::array<std::vector<input_snapshot>, 4> * snapshots;
                ~input_snapshot_restore() {
                    for (auto & lane : *snapshots) {
                        for (input_snapshot & snapshot : lane) {
                            snapshot.tensor->data = snapshot.original_data;
                        }
                    }
                }
            } restore_inputs = { &input_snapshots };
            for (size_t j = 0; j < n_backends; ++j) {
                size_t snapshot_bytes = 0;
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t layer = 0; layer < cells.size(); ++layer) {
                    for (int i = cells[layer].begin; i < cells[layer].expert_begin; ++i) {
                        ggml_tensor * node = bcj.nodes[i];
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            ggml_tensor * src = node->src[s];
                            if (src == nullptr || src->op != GGML_OP_NONE || strncmp(src->name, "Meta(", 5) != 0) {
                                continue;
                            }
                            auto it = std::find_if(input_snapshots[j].begin(), input_snapshots[j].end(),
                                    [&](const input_snapshot & snapshot) { return snapshot.tensor == src; });
                            if (it == input_snapshots[j].end()) {
                                snapshot_bytes = GGML_PAD(snapshot_bytes, 256);
                                input_snapshot snapshot = {};
                                snapshot.tensor = src;
                                snapshot.original_data = src->data;
                                snapshot.offset = snapshot_bytes;
                                input_snapshots[j].push_back(snapshot);
                                snapshot_bytes += ggml_nbytes(src);
                            }
                        }
                    }
                }
                ggml_backend_buffer_ptr & snapshot_buffer = backend_ctx->aw_input_snapshot_buffers[j];
                if (snapshot_buffer == nullptr ||
                        ggml_backend_buffer_get_size(snapshot_buffer.get()) < snapshot_bytes) {
                    snapshot_buffer.reset(ggml_backend_alloc_buffer(bcj.backend, snapshot_bytes));
                }
                GGML_ASSERT(snapshot_buffer != nullptr);
                char * base = (char *) ggml_backend_buffer_get_base(snapshot_buffer.get());
                for (input_snapshot & snapshot : input_snapshots[j]) {
                    snapshot.persistent = *snapshot.tensor;
                    snapshot.persistent.buffer = snapshot_buffer.get();
                    snapshot.persistent.view_src = nullptr;
                    snapshot.persistent.view_offs = 0;
                    snapshot.persistent.data = base + snapshot.offset;
                    ggml_backend_tensor_copy_async(bcj.backend, bcj.backend,
                            snapshot.tensor, &snapshot.persistent);
                }
                ggml_backend_synchronize(bcj.backend);
                for (input_snapshot & snapshot : input_snapshots[j]) {
                    snapshot.tensor->data = snapshot.persistent.data;
                }
                fprintf(stderr, "AffinityWave: pinned lane %zu inputs=%zu bytes=%.1f MiB\n",
                        j, input_snapshots[j].size(), snapshot_bytes/(1024.0*1024.0));
            }
            const char * lane_stagger_env = getenv("GGML_CUDA_AW_LANE_STAGGER");
            const int lane_stagger = lane_stagger_env != nullptr ? atoi(lane_stagger_env) : 1;
            GGML_ASSERT(lane_stagger > 0 && lane_stagger <= (int) cells.size());
            const char * corridor_early_env = getenv("GGML_CUDA_AW_CORRIDOR_EARLY");
            const bool corridor_early =
                    corridor_early_env != nullptr && strcmp(corridor_early_env, "0") != 0;
            GGML_ASSERT(!corridor_early ||
                    strcmp(corridor_early_env, "1") == 0 ||
                    strcmp(corridor_early_env, "all") == 0 ||
                    strcmp(corridor_early_env, "recurrent") == 0 ||
                    strcmp(corridor_early_env, "attention") == 0);
            GGML_ASSERT(!corridor_early ||
                    (backend_ctx->aw_corridor_copy != nullptr &&
                     backend_ctx->aw_corridor_wait != nullptr));
            auto corridor_early_for = [&](const ggml_backend_meta_aw_cell & cell) {
                return corridor_early &&
                        (strcmp(corridor_early_env, "1") == 0 ||
                         strcmp(corridor_early_env, "all") == 0 ||
                         (strcmp(corridor_early_env, "recurrent") == 0 && cell.recurrent >= 0) ||
                         (strcmp(corridor_early_env, "attention") == 0 && cell.recurrent < 0));
            };
            const bool corridor_state_split =
                    getenv("GGML_CUDA_AW_CORRIDOR_STATE_SPLIT") != nullptr &&
                    strcmp(getenv("GGML_CUDA_AW_CORRIDOR_STATE_SPLIT"), "0") != 0;
            GGML_ASSERT(!corridor_state_split || corridor_early);
            auto corridor_state_split_for = [&](const ggml_backend_meta_aw_cell & cell, size_t lane) {
                return corridor_state_split && lane + 1 < n_backends &&
                        cell.attention_state_end >= 0 && corridor_early_for(cell);
            };
            const char * precapture_env = getenv("GGML_CUDA_AW_PRECAPTURE");
            const bool precapture_requested =
                    precapture_env != nullptr && strcmp(precapture_env, "0") != 0;
            const char * precapture_output_env = getenv("GGML_CUDA_AW_PRECAPTURE_OUTPUT");
            const bool precapture_output =
                    precapture_output_env != nullptr && strcmp(precapture_output_env, "0") != 0;
            const int64_t wave_tokens =
                    backend_ctx->backend_configs[0].nodes[cells.front().expert_end]->ne[1];
            const bool serving = ggml_backend_meta_aw_serving();
            bool resets_recurrent_state = true;
            uint64_t capture_signature = 1469598103934665603ULL;
            if (serving) {
                auto hash_bytes = [&](const void * data, size_t size) {
                    const auto * bytes = static_cast<const uint8_t *>(data);
                    for (size_t i = 0; i < size; ++i) {
                        capture_signature = (capture_signature ^ bytes[i])*1099511628211ULL;
                    }
                };
                auto hash_tensor = [&](const ggml_tensor * tensor) {
                    hash_bytes(&tensor->op, sizeof(tensor->op));
                    hash_bytes(&tensor->type, sizeof(tensor->type));
                    hash_bytes(tensor->ne, sizeof(tensor->ne));
                    hash_bytes(tensor->nb, sizeof(tensor->nb));
                    hash_bytes(tensor->op_params, sizeof(tensor->op_params));
                    hash_bytes(&tensor->data, sizeof(tensor->data));
                    hash_bytes(&tensor->view_offs, sizeof(tensor->view_offs));
                };
                hash_bytes(&cgraph->n_nodes, sizeof(cgraph->n_nodes));
                for (const auto & bc : backend_ctx->backend_configs) {
                    for (int i = 0; i < cgraph->n_nodes; ++i) {
                        const ggml_tensor * node = bc.nodes[i];
                        hash_tensor(node);
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            const bool present = node->src[s] != nullptr;
                            hash_bytes(&present, sizeof(present));
                            if (present) {
                                hash_tensor(node->src[s]);
                            }
                        }
                    }
                }
                const auto & bc = backend_ctx->backend_configs[0];
                for (const auto & cell : cells) {
                    if (cell.recurrent >= 0) {
                        resets_recurrent_state &= ggml_nelements(bc.nodes[cell.conv_state_clear]) > 0 &&
                                ggml_nelements(bc.nodes[cell.state_clear]) > 0;
                    }
                }
            }
            const bool precaptured = serving ?
                    backend_ctx->aw_precaptured_signatures.count(capture_signature) != 0 :
                    backend_ctx->aw_precaptured_tokens == wave_tokens;
            const char * precapture_min_env = getenv("GGML_CUDA_AW_SERVE_PRECAPTURE_MIN_TOKENS");
            const int64_t precapture_min = precapture_min_env != nullptr ?
                    std::max<int64_t>(0, atoll(precapture_min_env)) : 0;
            const int warmup_passes =
                    precapture_requested && (!return_output || precapture_output) &&
                    (!serving || ggml_backend_meta_aw_active_tokens() >= precapture_min) &&
                    resets_recurrent_state && !precaptured ? 2 : 0;
            if (serving && !resets_recurrent_state) {
                fprintf(stderr, "AffinityWave: continued state, skipping pre-capture passes\n");
            }
            if (return_output && precapture_requested && !precapture_output) {
                fprintf(stderr, "AffinityWave: pre-capture disabled while producing output\n");
            }
            double elapsed_ms = 0.0;
            uint64_t corridor_bytes = 0;
            auto submit_diagonal = [&](
                    int diagonal,
                    const char * stage,
                    ggml_backend_meta_aw_trace_t trace) -> ggml_status {
                if (getenv("GGML_CUDA_AW_DEBUG_NODE_SYNC") != nullptr &&
                        strcmp(getenv("GGML_CUDA_AW_DEBUG_NODE_SYNC"), "0") != 0) {
                    for (size_t j = 0; j < n_backends; ++j) {
                        const int layer = diagonal - lane_stagger*(int) j;
                        if (layer < 0 || layer >= (int) cells.size()) {
                            continue;
                        }
                        char range_name[96];
                        snprintf(range_name, sizeof(range_name),
                                "diag=%02d/lane=%zu/layer=%02d/stage=%s",
                                diagonal, j, layer, stage);
                        ggml_backend_meta_aw_trace_scope trace_scope(
                                trace, range_name, 3,
                                ((uint64_t) (uint32_t) diagonal << 32) |
                                ((uint64_t) (uint32_t) j << 24) |
                                (uint32_t) layer);
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_cgraph * graph = bcj.cgraphs[0].cgraph_main;
                        const int n_nodes = graph->n_nodes;
                        std::vector<ggml_tensor *> nodes(graph->nodes, graph->nodes + n_nodes);
                        for (int i = 0; i < n_nodes; ++i) {
                            graph->n_nodes = 1;
                            graph->nodes[0] = nodes[i];
                            graph->uid = ggml_graph_next_uid();
                            fprintf(stderr, "AffinityWave: debug d=%d lane=%zu layer=%d node=%d/%d name=%s op=%s ne=%lld,%lld,%lld,%lld\n",
                                    diagonal, j, layer, i, n_nodes, nodes[i]->name, ggml_op_name(nodes[i]->op),
                                    (long long) nodes[i]->ne[0], (long long) nodes[i]->ne[1],
                                    (long long) nodes[i]->ne[2], (long long) nodes[i]->ne[3]);
                            if (nodes[i]->op == GGML_OP_FLASH_ATTN_EXT) {
                                for (int s = 0; s < GGML_MAX_SRC && nodes[i]->src[s] != nullptr; ++s) {
                                    const ggml_tensor * src = nodes[i]->src[s];
                                    fprintf(stderr, "AffinityWave: debug fattn src%d name=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n",
                                            s, src->name, (long long) src->ne[0], (long long) src->ne[1],
                                            (long long) src->ne[2], (long long) src->ne[3],
                                            src->nb[0], src->nb[1], src->nb[2], src->nb[3]);
                                }
                            }
                            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, graph);
                            if (status != GGML_STATUS_SUCCESS) {
                                return status;
                            }
                            ggml_backend_synchronize(bcj.backend);
                        }
                        graph->n_nodes = n_nodes;
                        std::copy(nodes.begin(), nodes.end(), graph->nodes);
                    }
                    return GGML_STATUS_SUCCESS;
                }
                const std::function<ggml_status(size_t)> diagonal_job = [&](size_t j) -> ggml_status {
                    const int layer = diagonal - lane_stagger*(int) j;
                    if (layer < 0 || layer >= (int) cells.size()) {
                        return GGML_STATUS_SUCCESS;
                    }
                    char thread_name[32];
                    snprintf(thread_name, sizeof(thread_name), "aw-submit/d%zu", j);
                    if (trace != nullptr) {
                        trace(GGML_BACKEND_META_AW_TRACE_THREAD, thread_name, 0, 0);
                    }
                    char range_name[96];
                    snprintf(range_name, sizeof(range_name),
                            "diag=%02d/lane=%zu/layer=%02d/stage=%s",
                            diagonal, j, layer, stage);
                    ggml_backend_meta_aw_trace_scope trace_scope(
                            trace, range_name, 3,
                            ((uint64_t) (uint32_t) diagonal << 32) |
                            ((uint64_t) (uint32_t) j << 24) |
                            (uint32_t) layer);
                    auto & bcj = backend_ctx->backend_configs[j];
                    return ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[0].cgraph_main);
                };
                auto & pool = *backend_ctx->submit;
                {
                    std::lock_guard<std::mutex> lock(pool.m);
                    pool.spmd_job = &diagonal_job;
                    pool.n_pending = n_backends - 1;
                    pool.seq++;
                }
                pool.cv_go.notify_all();
                const ggml_status status0 = diagonal_job(0);
                {
                    std::unique_lock<std::mutex> lock(pool.m);
                    pool.cv_done.wait(lock, [&]() { return pool.n_pending == 0; });
                    pool.spmd_job = nullptr;
                }
                if (status0 != GGML_STATUS_SUCCESS) {
                    return status0;
                }
                for (size_t w = 0; w < n_backends - 1; ++w) {
                    if (pool.slots[w].status != GGML_STATUS_SUCCESS) {
                        return pool.slots[w].status;
                    }
                }
                return GGML_STATUS_SUCCESS;
            };
            auto submit_headfold = [&](
                    int layer,
                    const char * stage,
                    ggml_backend_meta_aw_trace_t trace) -> ggml_status {
                const std::function<ggml_status(size_t)> headfold_job =
                        [&](size_t j) -> ggml_status {
                    char thread_name[32];
                    snprintf(thread_name, sizeof(thread_name),
                            "aw-submit/d%zu", j);
                    if (trace != nullptr) {
                        trace(GGML_BACKEND_META_AW_TRACE_THREAD,
                                thread_name, 0, 0);
                    }
                    char range_name[96];
                    snprintf(range_name, sizeof(range_name),
                            "headfold/lane=%zu/layer=%02d/stage=%s",
                            j, layer, stage);
                    ggml_backend_meta_aw_trace_scope trace_scope(
                            trace, range_name, 11,
                            ((uint64_t) (uint32_t) layer << 32) |
                            (uint32_t) j);
                    auto & bcj = backend_ctx->backend_configs[j];
                    return ggml_backend_graph_compute_async(
                            bcj.backend, bcj.cgraphs[0].cgraph_main);
                };
                auto & pool = *backend_ctx->submit;
                {
                    std::lock_guard<std::mutex> lock(pool.m);
                    pool.spmd_job = &headfold_job;
                    pool.n_pending = n_backends - 1;
                    pool.seq++;
                }
                pool.cv_go.notify_all();
                const ggml_status status0 = headfold_job(0);
                {
                    std::unique_lock<std::mutex> lock(pool.m);
                    pool.cv_done.wait(lock,
                            [&]() { return pool.n_pending == 0; });
                    pool.spmd_job = nullptr;
                }
                if (status0 != GGML_STATUS_SUCCESS) {
                    return status0;
                }
                for (size_t w = 0; w < n_backends - 1; ++w) {
                    if (pool.slots[w].status !=
                            GGML_STATUS_SUCCESS) {
                        return pool.slots[w].status;
                    }
                }
                return GGML_STATUS_SUCCESS;
            };
            if (warmup_passes != 0) {
                fprintf(stderr, "AffinityWave: pre-capturing exact cell graphs with %d untimed wave passes\n",
                        warmup_passes);
            } else if (precapture_requested && (!return_output || precapture_output) &&
                    precaptured) {
                fprintf(stderr, "AffinityWave: reusing pre-captured exact cell graphs\n");
            }
            const double setup_ms = (ggml_time_us() - dense_begin_us) / 1000.0;
            const char * trace_env = getenv("GGML_CUDA_AW_TRACE");
            const bool trace_enabled = trace_env != nullptr && strcmp(trace_env, "0") != 0;
            ggml_backend_meta_aw_trace_t timed_trace =
                    trace_enabled ? backend_ctx->aw_trace : nullptr;
            if (timed_trace != nullptr) {
                timed_trace(GGML_BACKEND_META_AW_TRACE_THREAD, "aw-meta", 0, 0);
            }
            for (int pass = 0; pass <= warmup_passes; ++pass) {
                ggml_backend_meta_aw_trace_t trace = timed_trace;
                const bool final_pass = pass == warmup_passes;
                ggml_backend_meta_aw_trace_scope pass_trace(
                        trace, final_pass ? "timed-pass" : "mapping-pass",
                        1, (uint64_t) wave_tokens);
                const int64_t pass_begin_us = ggml_time_us();
                uint64_t pass_corridor_bytes = 0;
                if (headfold) {
                    std::vector<ggml_backend_t> simple_backends(
                            n_backends);
                    for (size_t j = 0; j < n_backends; ++j) {
                        simple_backends[j] =
                                backend_ctx->backend_configs[j].backend;
                    }
                    auto append_range = [](ggml_cgraph * graph,
                            const ggml_backend_meta_context::backend_config & bc,
                            int begin, int end) {
                        for (int i = begin; i < end; ++i) {
                            graph->nodes[graph->n_nodes++] = bc.nodes[i];
                        }
                    };
                    auto route_stride = [](const ggml_tensor * route,
                            const ggml_tensor * output) {
                        if (route->ne[0] == 8 &&
                                route->ne[1] == output->ne[1]) {
                            return route->nb[1];
                        }
                        if (route->ne[0] == 1 &&
                                route->ne[1] == 8 &&
                                route->ne[2] == output->ne[1]) {
                            return route->nb[2];
                        }
                        GGML_ABORT(
                                "unexpected AffinityWave route layout");
                    };
                    auto mapped_view = [&](size_t lane,
                            ggml_tensor * tensor, bool pinned) {
                        size_t view_offs = 0;
                        ggml_tensor * root = tensor;
                        while (root->view_src != nullptr) {
                            view_offs += root->view_offs;
                            root = root->view_src;
                        }
                        auto it = std::find_if(
                                input_snapshots[lane].begin(),
                                input_snapshots[lane].end(),
                                [&](const input_snapshot & snapshot) {
                            return snapshot.tensor == root;
                        });
                        ggml_tensor mapped = *tensor;
                        if (it == input_snapshots[lane].end()) {
                            return mapped;
                        }
                        mapped.buffer = pinned ?
                                it->persistent.buffer : root->buffer;
                        mapped.data = (char *) (pinned ?
                                it->persistent.data : it->original_data) +
                                view_offs;
                        mapped.view_src = nullptr;
                        mapped.view_offs = 0;
                        return mapped;
                    };
                    for (int layer = 0;
                            layer < (int) cells.size(); ++layer) {
                        char layer_name[40];
                        snprintf(layer_name, sizeof(layer_name),
                                "headfold-layer=%02d", layer);
                        ggml_backend_meta_aw_trace_scope layer_trace(
                                trace, layer_name, 12,
                                (uint64_t) layer);
                        const ggml_backend_meta_aw_cell & cell =
                                cells[layer];
                        ggml_status status = GGML_STATUS_SUCCESS;
                        if (headfold_r44) {
                            char r44_error[256] = {};
                            if (backend_ctx->aw_r44_prefetch(
                                        layer, r44_error,
                                        sizeof(r44_error)) != 0) {
                                fprintf(stderr,
                                        "AffinityWave: R44 prefetch failed: %s\n",
                                        r44_error);
                                return GGML_STATUS_FAILED;
                            }
                        }
                        const bool legacy_placement =
                                (cell.recurrent >= 0 &&
                                 !headfold_gdn) ||
                                (cell.recurrent < 0 &&
                                 !headfold_attention);
                        if (legacy_placement) {
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                if (j > 0) {
                                    auto & src =
                                            backend_ctx->backend_configs[
                                                    j - 1];
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            src.backend, false);
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            bcj.backend, false);
                                    if (cell.recurrent >= 0) {
                                        ggml_tensor src_conv =
                                                mapped_view(
                                                    j - 1,
                                                    src.nodes[
                                                        cell.conv_state_update],
                                                    false);
                                        ggml_tensor dst_conv =
                                                mapped_view(
                                                    j,
                                                    bcj.nodes[
                                                        cell.conv_state_clear],
                                                    true);
                                        ggml_tensor src_state =
                                                mapped_view(
                                                    j - 1,
                                                    src.nodes[
                                                        cell.state_update]->
                                                        src[1],
                                                    false);
                                        ggml_tensor dst_state =
                                                mapped_view(
                                                    j,
                                                    bcj.nodes[
                                                        cell.state_clear],
                                                    true);
                                        GGML_ASSERT(
                                                ggml_are_same_layout(
                                                    &src_conv,
                                                    &dst_conv));
                                        GGML_ASSERT(
                                                ggml_are_same_layout(
                                                    &src_state,
                                                    &dst_state));
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_copy(
                                                        src.backend,
                                                        bcj.backend,
                                                        &src_conv,
                                                        &dst_conv) == 0);
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_copy(
                                                        src.backend,
                                                        bcj.backend,
                                                        &src_state,
                                                        &dst_state) == 0);
                                        pass_corridor_bytes +=
                                                ggml_nbytes(&dst_conv) +
                                                ggml_nbytes(&dst_state);
                                    } else {
                                        ggml_tensor * src_flash =
                                                src.nodes[
                                                    cell.flash_attn];
                                        ggml_tensor * dst_flash =
                                                bcj.nodes[
                                                    cell.flash_attn];
                                        ggml_tensor src_k =
                                                mapped_view(
                                                    j - 1,
                                                    ggml_backend_meta_aw_view_root(
                                                        src_flash->src[1]),
                                                    false);
                                        ggml_tensor dst_k =
                                                mapped_view(
                                                    j,
                                                    ggml_backend_meta_aw_view_root(
                                                        dst_flash->src[1]),
                                                    true);
                                        ggml_tensor src_v =
                                                mapped_view(
                                                    j - 1,
                                                    ggml_backend_meta_aw_view_root(
                                                        src_flash->src[2]),
                                                    false);
                                        ggml_tensor dst_v =
                                                mapped_view(
                                                    j,
                                                    ggml_backend_meta_aw_view_root(
                                                        dst_flash->src[2]),
                                                    true);
                                        GGML_ASSERT(
                                                ggml_are_same_layout(
                                                    &src_k, &dst_k));
                                        GGML_ASSERT(
                                                ggml_are_same_layout(
                                                    &src_v, &dst_v));
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_copy(
                                                        src.backend,
                                                        bcj.backend,
                                                        &src_k,
                                                        &dst_k) == 0);
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_copy(
                                                        src.backend,
                                                        bcj.backend,
                                                        &src_v,
                                                        &dst_v) == 0);
                                        pass_corridor_bytes +=
                                                ggml_nbytes(&dst_k) +
                                                ggml_nbytes(&dst_v);
                                    }
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_wait(
                                                src.backend,
                                                bcj.backend) == 0);
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            bcj.backend, true);
                                }
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                for (int i = cell.begin;
                                        i < cell.expert_begin; ++i) {
                                    if (j > 0 &&
                                            (i ==
                                                cell.conv_state_clear ||
                                             i ==
                                                cell.state_clear)) {
                                        continue;
                                    }
                                    graph->nodes[
                                        graph->n_nodes++] =
                                            bcj.nodes[i];
                                }
                                graph->uid = ggml_graph_next_uid();
                                status =
                                        ggml_backend_graph_compute_async(
                                                bcj.backend, graph);
                                if (status != GGML_STATUS_SUCCESS) {
                                    return status;
                                }
                            }
                        } else if (cell.recurrent >= 0) {
                            if (!headfold_split_pre) {
                                for (size_t j = 0;
                                        j < n_backends; ++j) {
                                    auto & bcj =
                                            backend_ctx->
                                                backend_configs[j];
                                    if (j > 0) {
                                        auto & src =
                                                backend_ctx->
                                                    backend_configs[j - 1];
                                        ggml_backend_meta_affinity_wave_stream_fence(
                                                src.backend, false);
                                        ggml_backend_meta_affinity_wave_stream_fence(
                                                bcj.backend, false);
                                        ggml_tensor src_conv = mapped_view(
                                                j - 1,
                                                src.nodes[
                                                    cell.conv_state_update],
                                                false);
                                        ggml_tensor dst_conv = mapped_view(
                                                j,
                                                bcj.nodes[
                                                    cell.conv_state_clear],
                                                true);
                                        GGML_ASSERT(
                                                ggml_are_same_layout(
                                                    &src_conv,
                                                    &dst_conv));
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_copy(
                                                        src.backend,
                                                        bcj.backend,
                                                        &src_conv,
                                                        &dst_conv) == 0);
                                        GGML_ASSERT(
                                                backend_ctx->
                                                    aw_corridor_wait(
                                                        src.backend,
                                                        bcj.backend) == 0);
                                        ggml_backend_meta_affinity_wave_stream_fence(
                                                bcj.backend, true);
                                        pass_corridor_bytes +=
                                                ggml_nbytes(&dst_conv);
                                        const char * dump_conv =
                                                getenv(
                                                    "GGML_CUDA_AW_HEADFOLD_DUMP_CONV");
                                        const char * dump_layer_env =
                                                getenv(
                                                    "GGML_CUDA_AW_HEADFOLD_DUMP_LAYER");
                                        const int dump_layer =
                                                dump_layer_env != nullptr ?
                                                atoi(dump_layer_env) : 0;
                                        if (final_pass &&
                                                dump_conv != nullptr &&
                                                dump_conv[0] != '\0' &&
                                                layer == dump_layer) {
                                            ggml_backend_synchronize(
                                                    src.backend);
                                            ggml_backend_synchronize(
                                                    bcj.backend);
                                            auto dump_halo = [&](
                                                    const char * label,
                                                    const ggml_tensor * tensor) {
                                                std::vector<uint8_t> data(
                                                        ggml_nbytes(tensor));
                                                ggml_backend_tensor_get(
                                                        tensor,
                                                        data.data(), 0,
                                                        data.size());
                                                char path[4096];
                                                snprintf(path,
                                                        sizeof(path),
                                                        "%s-layer%02d-link%zu-%s.bin",
                                                        dump_conv,
                                                        layer, j - 1,
                                                        label);
                                                std::ofstream output(
                                                        path,
                                                        std::ios::out |
                                                        std::ios::binary |
                                                        std::ios::trunc);
                                                GGML_ASSERT(output.good());
                                                output.write(
                                                        (const char *)
                                                            data.data(),
                                                        data.size());
                                                GGML_ASSERT(output.good());
                                            };
                                            dump_halo("source",
                                                    &src_conv);
                                            dump_halo("destination",
                                                    &dst_conv);
                                        }
                                    }
                                    ggml_cgraph * graph =
                                            bcj.cgraphs[0].cgraph_main;
                                    graph->n_nodes = 0;
                                    for (int i = cell.begin;
                                            i < cell.recurrent; ++i) {
                                        if (j > 0 &&
                                                i ==
                                                    cell.conv_state_clear) {
                                            continue;
                                        }
                                        graph->nodes[
                                            graph->n_nodes++] =
                                                bcj.nodes[i];
                                    }
                                    graph->uid =
                                            ggml_graph_next_uid();
                                    status =
                                            ggml_backend_graph_compute_async(
                                                    bcj.backend, graph);
                                    if (status != GGML_STATUS_SUCCESS) {
                                        return status;
                                    }
                                }
                            } else {
                                for (size_t j = 0;
                                        j < n_backends; ++j) {
                                    auto & bcj =
                                            backend_ctx->
                                                backend_configs[j];
                                    ggml_cgraph * graph =
                                            bcj.cgraphs[0].cgraph_main;
                                    graph->n_nodes = 0;
                                    append_range(graph, bcj,
                                            cell.begin,
                                            cell.conv_state_clear - 2);
                                    append_range(graph, bcj,
                                            cell.conv_input - 3,
                                            cell.conv_input);
                                    graph->uid =
                                            ggml_graph_next_uid();
                                }
                                status = submit_headfold(
                                        layer, "gdn-local-pre", trace);
                                if (status != GGML_STATUS_SUCCESS) {
                                    return status;
                                }

                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                if (j > 0) {
                                    auto & src =
                                            backend_ctx->backend_configs[
                                                    j - 1];
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            src.backend, false);
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            bcj.backend, false);
                                    ggml_tensor src_conv = mapped_view(
                                            j - 1,
                                            src.nodes[
                                                cell.conv_state_update],
                                            false);
                                    ggml_tensor dst_conv = mapped_view(
                                            j,
                                            bcj.nodes[
                                                cell.conv_state_clear],
                                            true);
                                    GGML_ASSERT(ggml_are_same_layout(
                                                &src_conv, &dst_conv));
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_copy(
                                                src.backend,
                                                bcj.backend,
                                                &src_conv,
                                                &dst_conv) == 0);
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_wait(
                                                src.backend,
                                                bcj.backend) == 0);
                                    ggml_backend_meta_affinity_wave_stream_fence(
                                            bcj.backend, true);
                                    pass_corridor_bytes +=
                                            ggml_nbytes(&dst_conv);
                                    const char * dump_conv =
                                            getenv(
                                                "GGML_CUDA_AW_HEADFOLD_DUMP_CONV");
                                    const char * dump_layer_env =
                                            getenv(
                                                "GGML_CUDA_AW_HEADFOLD_DUMP_LAYER");
                                    const int dump_layer =
                                            dump_layer_env != nullptr ?
                                            atoi(dump_layer_env) : 0;
                                    if (final_pass &&
                                            dump_conv != nullptr &&
                                            dump_conv[0] != '\0' &&
                                            layer == dump_layer) {
                                        ggml_backend_synchronize(
                                                src.backend);
                                        ggml_backend_synchronize(
                                                bcj.backend);
                                        auto dump_halo = [&](
                                                const char * label,
                                                const ggml_tensor * tensor) {
                                            std::vector<uint8_t> data(
                                                    ggml_nbytes(tensor));
                                            ggml_backend_tensor_get(
                                                    tensor,
                                                    data.data(), 0,
                                                    data.size());
                                            char path[4096];
                                            snprintf(path,
                                                    sizeof(path),
                                                    "%s-layer%02d-link%zu-%s.bin",
                                                    dump_conv,
                                                    layer, j - 1,
                                                    label);
                                            std::ofstream output(
                                                    path,
                                                    std::ios::out |
                                                    std::ios::binary |
                                                    std::ios::trunc);
                                            GGML_ASSERT(output.good());
                                            output.write(
                                                    (const char *)
                                                        data.data(),
                                                    data.size());
                                            GGML_ASSERT(output.good());
                                        };
                                        dump_halo("source", &src_conv);
                                        dump_halo("destination",
                                                &dst_conv);
                                    }
                                }
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                append_range(graph, bcj,
                                        cell.conv_state_clear - 2,
                                        cell.conv_state_clear);
                                if (j == 0) {
                                    graph->nodes[
                                        graph->n_nodes++] =
                                            bcj.nodes[
                                                cell.conv_state_clear];
                                }
                                append_range(graph, bcj,
                                        cell.conv_state_clear + 1,
                                        cell.conv_input - 3);
                                append_range(graph, bcj,
                                        cell.conv_input,
                                        cell.conv_state_update + 2);
                                graph->uid = ggml_graph_next_uid();
                                status =
                                        ggml_backend_graph_compute_async(
                                                bcj.backend, graph);
                                if (status != GGML_STATUS_SUCCESS) {
                                    return status;
                                }
                            }

                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                append_range(graph, bcj,
                                        cell.conv_state_update + 2,
                                        cell.recurrent);
                                graph->uid = ggml_graph_next_uid();
                            }
                            status = submit_headfold(
                                    layer, "gdn-conv-scalars", trace);
                            if (status != GGML_STATUS_SUCCESS) {
                                return status;
                            }
                            }

                            std::array<
                                    ggml_backend_meta_aw_headfold_gdn_lane,
                                    4> gdn_lanes{};
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_tensor * recurrent =
                                        bcj.nodes[cell.recurrent];
                                ggml_tensor * state_update =
                                        bcj.nodes[cell.state_update];
                                ggml_tensor * q =
                                        bcj.nodes[cell.q_predelta];
                                ggml_tensor * k =
                                        bcj.nodes[cell.k_predelta];
                                ggml_tensor * v =
                                        bcj.nodes[cell.v_predelta];
                                ggml_tensor * gate =
                                        bcj.nodes[cell.gate];
                                ggml_tensor * beta =
                                        bcj.nodes[cell.beta];
                                GGML_ASSERT(
                                        recurrent->src[5] != nullptr &&
                                        state_update->src[1] != nullptr &&
                                        q->nb[2] == k->nb[2] &&
                                        gate->nb[2] == beta->nb[2] &&
                                        q->nb[2] % sizeof(float) == 0 &&
                                        v->nb[2] % sizeof(float) == 0 &&
                                        gate->nb[2] %
                                            sizeof(float) == 0);
                                gdn_lanes[j] = {
                                    (int32_t)
                                        v->ne[2],
                                    (int32_t)
                                        (q->nb[2]/sizeof(float)),
                                    (int32_t)
                                        (v->nb[2]/sizeof(float)),
                                    (int32_t)
                                        (gate->nb[2]/sizeof(float)),
                                    (const float *) q->data,
                                    (const float *) k->data,
                                    (const float *) v->data,
                                    (const float *) gate->data,
                                    (const float *) beta->data,
                                    (const float *)
                                        recurrent->src[5]->data,
                                    (float *) recurrent->data,
                                    (float *)
                                        state_update->src[1]->data,
                                };
                            }
                            const char * gdn_dump =
                                    getenv(
                                        "GGML_CUDA_AW_HEADFOLD_DUMP");
                            const char * gdn_dump_layer_env =
                                    getenv(
                                        "GGML_CUDA_AW_HEADFOLD_DUMP_LAYER");
                            const int gdn_dump_layer =
                                    gdn_dump_layer_env != nullptr ?
                                    atoi(gdn_dump_layer_env) : 0;
                            if (final_pass &&
                                    gdn_dump != nullptr &&
                                    gdn_dump[0] != '\0' &&
                                    layer == gdn_dump_layer) {
                                for (size_t j = 0;
                                        j < n_backends; ++j) {
                                    ggml_backend_synchronize(
                                            backend_ctx->
                                                backend_configs[j].backend);
                                }
                                for (size_t j = 0;
                                        j < n_backends; ++j) {
                                    auto dump_pre = [&](const char * label,
                                            const ggml_tensor * tensor) {
                                        std::vector<uint8_t> data(
                                                ggml_nbytes(tensor));
                                        ggml_backend_tensor_get(
                                                tensor, data.data(), 0,
                                                data.size());
                                        char path[4096];
                                        snprintf(path, sizeof(path),
                                                "%s-layer%02d-lane%zu-pre-%s.bin",
                                                gdn_dump, layer, j,
                                                label);
                                        std::ofstream output(
                                                path,
                                                std::ios::out |
                                                std::ios::binary |
                                                std::ios::trunc);
                                        GGML_ASSERT(output.good());
                                        output.write(
                                                (const char *)
                                                    data.data(),
                                                data.size());
                                        GGML_ASSERT(output.good());
                                    };
                                    auto & bcj =
                                            backend_ctx->
                                                backend_configs[j];
                                    dump_pre("q",
                                            bcj.nodes[
                                                cell.q_predelta]);
                                    dump_pre("k",
                                            bcj.nodes[
                                                cell.k_predelta]);
                                    dump_pre("v",
                                            bcj.nodes[
                                                cell.v_predelta]);
                                    dump_pre("gate",
                                            bcj.nodes[cell.gate]);
                                    dump_pre("beta",
                                            bcj.nodes[cell.beta]);
                                    dump_pre("state",
                                            bcj.nodes[
                                                cell.recurrent]->
                                                src[5]);
                                }
                            }
                            char gdn_error[256] = {};
                            if (backend_ctx->aw_headfold_gdn(
                                        simple_backends.data(),
                                        gdn_lanes.data(),
                                        (int32_t) gdn_lanes.size(),
                                        gdn_error,
                                        sizeof(gdn_error)) != 0) {
                                fprintf(stderr,
                                        "AffinityWave: HeadFold GDN failed: %s\n",
                                        gdn_error);
                                return GGML_STATUS_FAILED;
                            }

                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                append_range(graph, bcj,
                                        cell.state_update + 1,
                                        cell.expert_begin);
                                graph->uid = ggml_graph_next_uid();
                            }
                            status = submit_headfold(
                                    layer, "gdn-local-post", trace);
                            if (status != GGML_STATUS_SUCCESS) {
                                return status;
                            }
                        } else {
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                append_range(graph, bcj,
                                        cell.begin,
                                        cell.attention_state_end + 1);
                                graph->uid = ggml_graph_next_uid();
                            }
                            status = submit_headfold(
                                    layer, "attention-state", trace);
                            if (status != GGML_STATUS_SUCCESS) {
                                return status;
                            }

                            std::array<ggml_tensor *, 4> cache_k{};
                            std::array<ggml_tensor *, 4> cache_v{};
                            int64_t token_begin = 0;
                            std::array<int64_t, 5> token_offsets{};
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_tensor * flash =
                                        bcj.nodes[cell.flash_attn];
                                cache_k[j] =
                                        ggml_backend_meta_aw_view_root(
                                                flash->src[1]);
                                cache_v[j] =
                                        ggml_backend_meta_aw_view_root(
                                                flash->src[2]);
                                token_offsets[j] = token_begin;
                                token_begin +=
                                        flash->src[0]->ne[1]*
                                        flash->src[0]->ne[2]/
                                        cgraph->nodes[
                                            cell.flash_attn]->src[0]->ne[2];
                            }
                            token_offsets[n_backends] = token_begin;
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                GGML_ASSERT(
                                        cache_k[j]->ne[1] >=
                                            token_begin &&
                                        cache_v[j]->ne[1] >=
                                            token_begin);
                                ggml_backend_meta_affinity_wave_stream_fence(
                                        backend_ctx->
                                            backend_configs[j].backend,
                                        false);
                            }
                            for (size_t source = 0;
                                    source < n_backends; ++source) {
                                const int64_t source_tokens =
                                        token_offsets[source + 1] -
                                        token_offsets[source];
                                for (size_t destination = 0;
                                        destination < n_backends;
                                        ++destination) {
                                    if (source == destination) {
                                        continue;
                                    }
                                    ggml_tensor src_k =
                                            *cache_k[source];
                                    ggml_tensor dst_k =
                                            *cache_k[destination];
                                    ggml_tensor src_v =
                                            *cache_v[source];
                                    ggml_tensor dst_v =
                                            *cache_v[destination];
                                    src_k.ne[1] = source_tokens;
                                    dst_k.ne[1] = source_tokens;
                                    src_v.ne[1] = source_tokens;
                                    dst_v.ne[1] = source_tokens;
                                    const size_t k_offset =
                                            token_offsets[source]*
                                            src_k.nb[1];
                                    const size_t v_offset =
                                            token_offsets[source]*
                                            src_v.nb[1];
                                    src_k.data =
                                            (char *) src_k.data +
                                            k_offset;
                                    dst_k.data =
                                            (char *) dst_k.data +
                                            k_offset;
                                    src_v.data =
                                            (char *) src_v.data +
                                            v_offset;
                                    dst_v.data =
                                            (char *) dst_v.data +
                                            v_offset;
                                    src_k.view_src = nullptr;
                                    dst_k.view_src = nullptr;
                                    src_v.view_src = nullptr;
                                    dst_v.view_src = nullptr;
                                    src_k.view_offs = 0;
                                    dst_k.view_offs = 0;
                                    src_v.view_offs = 0;
                                    dst_v.view_offs = 0;
                                    GGML_ASSERT(ggml_are_same_layout(
                                                &src_k, &dst_k));
                                    GGML_ASSERT(ggml_are_same_layout(
                                                &src_v, &dst_v));
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_copy(
                                                backend_ctx->
                                                    backend_configs[
                                                        source].backend,
                                                backend_ctx->
                                                    backend_configs[
                                                        destination].backend,
                                                &src_k, &dst_k) == 0);
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_copy(
                                                backend_ctx->
                                                    backend_configs[
                                                        source].backend,
                                                backend_ctx->
                                                    backend_configs[
                                                        destination].backend,
                                                &src_v, &dst_v) == 0);
                                    pass_corridor_bytes +=
                                            ggml_nbytes(&dst_k) +
                                            ggml_nbytes(&dst_v);
                                }
                            }
                            for (size_t destination = 0;
                                    destination < n_backends;
                                    ++destination) {
                                for (size_t source = 0;
                                        source < n_backends;
                                        ++source) {
                                    if (source == destination) {
                                        continue;
                                    }
                                    GGML_ASSERT(
                                            backend_ctx->aw_corridor_wait(
                                                backend_ctx->
                                                    backend_configs[
                                                        source].backend,
                                                backend_ctx->
                                                    backend_configs[
                                                        destination].backend)
                                                == 0);
                                }
                                ggml_backend_meta_affinity_wave_stream_fence(
                                        backend_ctx->
                                            backend_configs[
                                                destination].backend,
                                        true);
                            }

                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                ggml_cgraph * graph =
                                        bcj.cgraphs[0].cgraph_main;
                                graph->n_nodes = 0;
                                append_range(graph, bcj,
                                        cell.attention_state_end + 1,
                                        cell.expert_begin);
                                graph->uid = ggml_graph_next_uid();
                            }
                            status = submit_headfold(
                                    layer, "attention-compute", trace);
                            if (status != GGML_STATUS_SUCCESS) {
                                return status;
                            }
                        }

                        const char * headfold_dump =
                                getenv("GGML_CUDA_AW_HEADFOLD_DUMP");
                        const char * headfold_dump_layer_env =
                                getenv(
                                    "GGML_CUDA_AW_HEADFOLD_DUMP_LAYER");
                        const int headfold_dump_layer =
                                headfold_dump_layer_env != nullptr ?
                                atoi(headfold_dump_layer_env) : 0;
                        if (final_pass &&
                                headfold_dump != nullptr &&
                                headfold_dump[0] != '\0' &&
                                layer == headfold_dump_layer &&
                                cell.recurrent >= 0) {
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                ggml_backend_synchronize(
                                        backend_ctx->
                                            backend_configs[j].backend);
                            }
                            auto dump_tensor = [&](size_t lane,
                                    const char * label,
                                    const ggml_tensor * tensor) {
                                std::vector<uint8_t> data(
                                        ggml_nbytes(tensor));
                                ggml_backend_tensor_get(
                                        tensor, data.data(), 0,
                                        data.size());
                                char path[4096];
                                snprintf(path, sizeof(path),
                                        "%s-layer%02d-lane%zu-%s.bin",
                                        headfold_dump, layer,
                                        lane, label);
                                std::ofstream output(path,
                                        std::ios::out |
                                        std::ios::binary |
                                        std::ios::trunc);
                                GGML_ASSERT(output.good());
                                output.write(
                                        (const char *) data.data(),
                                        data.size());
                                GGML_ASSERT(output.good());
                            };
                            for (size_t j = 0;
                                    j < n_backends; ++j) {
                                auto & bcj =
                                        backend_ctx->backend_configs[j];
                                dump_tensor(j, "q",
                                        bcj.nodes[
                                            cell.q_predelta]);
                                dump_tensor(j, "k",
                                        bcj.nodes[
                                            cell.k_predelta]);
                                dump_tensor(j, "v",
                                        bcj.nodes[
                                            cell.v_predelta]);
                                dump_tensor(j, "gate",
                                        bcj.nodes[cell.gate]);
                                dump_tensor(j, "beta",
                                        bcj.nodes[cell.beta]);
                                dump_tensor(j, "recurrent",
                                        bcj.nodes[
                                            cell.recurrent]);
                                dump_tensor(j, "state",
                                        bcj.nodes[
                                            cell.state_update]->
                                            src[1]);
                                dump_tensor(j, "conv-state",
                                        bcj.nodes[
                                            cell.conv_state_update]);
                            }
                        }

                        std::array<
                                ggml_backend_meta_aw_live_cell,
                                4> live_cells{};
                        for (size_t j = 0;
                                j < n_backends; ++j) {
                            auto & bcj =
                                    backend_ctx->backend_configs[j];
                            ggml_tensor * gate = nullptr;
                            ggml_tensor * weighted = nullptr;
                            for (int i = cell.expert_begin;
                                    i <= cell.expert_end; ++i) {
                                ggml_tensor * node = bcj.nodes[i];
                                if (ggml_backend_meta_aw_name_contains(
                                            node, "ffn_moe_gate",
                                            layer) &&
                                        node->op ==
                                            GGML_OP_MUL_MAT_ID) {
                                    gate = node;
                                } else if (
                                        ggml_backend_meta_aw_name_contains(
                                            node,
                                            "ffn_moe_weighted",
                                            layer) &&
                                        node->op == GGML_OP_MUL) {
                                    weighted = node;
                                }
                            }
                            ggml_tensor * output =
                                    bcj.nodes[cell.expert_end];
                            GGML_ASSERT(gate != nullptr &&
                                    weighted != nullptr);
                            live_cells[j] = {
                                layer,
                                (int32_t) j,
                                (int32_t) output->ne[1],
                                0,
                                route_stride(
                                    gate->src[2], output),
                                route_stride(
                                    weighted->src[1], output),
                                (const float *)
                                    gate->src[1]->data,
                                (const int32_t *)
                                    gate->src[2]->data,
                                (const float *)
                                    weighted->src[1]->data,
                                (float *) output->data,
                                shared_service ?
                                    (float *)
                                        bcj.nodes[
                                            cell.shared_raw]->data :
                                    nullptr,
                                nullptr,
                            };
                        }
                        char service_error[256] = {};
                        const char * pairwave_service_env =
                                getenv(
                                    "GGML_CUDA_AW_PAIRWAVE_SERVICE");
                        const bool pairwave_service =
                                pairwave_service_env != nullptr &&
                                strcmp(pairwave_service_env, "0") != 0;
                        const int service_calls =
                                pairwave_service ? 2 : 1;
                        for (int service_call = 0;
                                service_call < service_calls;
                                ++service_call) {
                            const int cell_offset =
                                    pairwave_service ?
                                    service_call*2 : 0;
                            const int cell_count =
                                    pairwave_service ?
                                    2 : (int) live_cells.size();
                            if (backend_ctx->aw_live_service(
                                        simple_backends.data(),
                                        live_cells.data() +
                                            cell_offset,
                                        cell_count,
                                        service_error,
                                        sizeof(service_error)) != 0) {
                                fprintf(stderr,
                                        "AffinityWave: HeadFold service failed: %s\n",
                                        service_error);
                                return GGML_STATUS_FAILED;
                            }
                        }
                        for (size_t j = 0;
                                j < n_backends; ++j) {
                            auto & bcj =
                                    backend_ctx->backend_configs[j];
                            ggml_cgraph * graph =
                                    bcj.cgraphs[0].cgraph_main;
                            graph->n_nodes = 0;
                            if (!shared_service) {
                                append_range(graph, bcj,
                                        cell.shared_begin,
                                        cell.shared_end + 1);
                            } else {
                                append_range(graph, bcj,
                                        cell.shared_raw + 1,
                                        cell.shared_end + 1);
                            }
                            append_range(graph, bcj,
                                    cell.post_begin,
                                    cell.end + 1);
                            graph->uid = ggml_graph_next_uid();
                        }
                        status = submit_headfold(
                                layer, "post", trace);
                        if (status != GGML_STATUS_SUCCESS) {
                            return status;
                        }
                    }
                } else {
                for (int diagonal = 0;
                        diagonal < (int) cells.size() + lane_stagger*((int) n_backends - 1); ++diagonal) {
                char diagonal_name[32];
                snprintf(diagonal_name, sizeof(diagonal_name), "diagonal=%02d", diagonal);
                ggml_backend_meta_aw_trace_scope diagonal_trace(
                        trace, diagonal_name, 2, (uint64_t) diagonal);
                for (size_t j = 0; j < n_backends; ++j) {
                    const int layer = diagonal - lane_stagger*(int) j;
                    if (layer < 0 || layer >= (int) cells.size()) {
                        continue;
                    }
                    const ggml_backend_meta_aw_cell & cell = cells[layer];
                    if (j > 0) {
                        auto & src_config = backend_ctx->backend_configs[j - 1];
                        auto & dst_config = backend_ctx->backend_configs[j];
                        ggml_backend_meta_affinity_wave_stream_fence(dst_config.backend, false);
                        if (corridor_early_for(cell)) {
                            GGML_ASSERT(backend_ctx->aw_corridor_wait(
                                    src_config.backend, dst_config.backend) == 0);
                            ggml_backend_meta_affinity_wave_stream_fence(dst_config.backend, true);
                        } else {
                        ggml_backend_meta_affinity_wave_stream_fence(src_config.backend, false);
                        auto corridor_view = [&](size_t lane, ggml_tensor * tensor, bool pinned) {
                            size_t view_offs = 0;
                            ggml_tensor * root = tensor;
                            while (root->view_src != nullptr) {
                                view_offs += root->view_offs;
                                root = root->view_src;
                            }
                            auto it = std::find_if(input_snapshots[lane].begin(), input_snapshots[lane].end(),
                                    [&](const input_snapshot & snapshot) { return snapshot.tensor == root; });
                            ggml_tensor mapped = *tensor;
                            if (it == input_snapshots[lane].end()) {
                                if (cell.layer == 0 && j == 1 &&
                                        getenv("GGML_CUDA_AW_DEBUG_CORRIDOR") != nullptr &&
                                        strcmp(getenv("GGML_CUDA_AW_DEBUG_CORRIDOR"), "0") != 0) {
                                    fprintf(stderr,
                                            "AffinityWave: corridor direct lane=%zu tensor=%s root=%s op=%s bytes=%zu\n",
                                            lane, tensor->name, root->name, ggml_op_name(root->op), ggml_nbytes(tensor));
                                }
                                return mapped;
                            }
                            mapped.buffer = pinned ? it->persistent.buffer : root->buffer;
                            mapped.data = (char *) (pinned ? it->persistent.data : it->original_data) + view_offs;
                            mapped.view_src = nullptr;
                            mapped.view_offs = 0;
                            return mapped;
                        };
                        auto copy_corridor = [&](ggml_tensor * src, ggml_tensor * dst, bool root) {
                            if (root) {
                                src = ggml_backend_meta_aw_view_root(src);
                                dst = ggml_backend_meta_aw_view_root(dst);
                            }
                            if (!ggml_are_same_layout(src, dst)) {
                                fprintf(stderr,
                                        "AffinityWave: corridor layout mismatch src=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu dst=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu\n",
                                        src->name,
                                        (long long) src->ne[0], (long long) src->ne[1],
                                        (long long) src->ne[2], (long long) src->ne[3],
                                        src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                                        dst->name,
                                        (long long) dst->ne[0], (long long) dst->ne[1],
                                        (long long) dst->ne[2], (long long) dst->ne[3],
                                        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
                            }
                            GGML_ASSERT(ggml_are_same_layout(src, dst));
                            ggml_backend_tensor_copy_async(src_config.backend, dst_config.backend, src, dst);
                            pass_corridor_bytes += ggml_nbytes(dst);
                        };
                        if (cell.recurrent >= 0) {
                            ggml_tensor src_conv = corridor_view(j - 1,
                                    src_config.nodes[cell.conv_state_update], false);
                            ggml_tensor dst_conv = corridor_view(j,
                                    dst_config.nodes[serving ? cell.conv_state_update : cell.conv_state_clear], true);
                            ggml_tensor src_state = corridor_view(j - 1,
                                    src_config.nodes[cell.state_update]->src[1], false);
                            ggml_tensor dst_state = corridor_view(j,
                                    serving ? dst_config.nodes[cell.state_update]->src[1] :
                                        dst_config.nodes[cell.state_clear], true);
                            const bool debug_corridor =
                                    cell.layer == 0 && j == 1 &&
                                    getenv("GGML_CUDA_AW_DEBUG_CORRIDOR") != nullptr &&
                                    strcmp(getenv("GGML_CUDA_AW_DEBUG_CORRIDOR"), "0") != 0;
                            auto tensor_hash = [](ggml_backend_t backend, const ggml_tensor * tensor) {
                                ggml_backend_synchronize(backend);
                                std::vector<uint8_t> data(ggml_nbytes(tensor));
                                ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
                                uint64_t hash = 1469598103934665603ULL;
                                for (uint8_t value : data) {
                                    hash = (hash ^ value) * 1099511628211ULL;
                                }
                                return hash;
                            };
                            if (debug_corridor) {
                                fprintf(stderr,
                                        "AffinityWave: corridor before conv=%016llx/%016llx state=%016llx/%016llx bytes=%zu/%zu\n",
                                        (unsigned long long) tensor_hash(src_config.backend, &src_conv),
                                        (unsigned long long) tensor_hash(dst_config.backend, &dst_conv),
                                        (unsigned long long) tensor_hash(src_config.backend, &src_state),
                                        (unsigned long long) tensor_hash(dst_config.backend, &dst_state),
                                        ggml_nbytes(&src_conv), ggml_nbytes(&src_state));
                            }
                            copy_corridor(&src_conv, &dst_conv, false);
                            copy_corridor(&src_state, &dst_state, false);
                            if (debug_corridor) {
                                fprintf(stderr,
                                        "AffinityWave: corridor after conv=%016llx/%016llx state=%016llx/%016llx\n",
                                        (unsigned long long) tensor_hash(src_config.backend, &src_conv),
                                        (unsigned long long) tensor_hash(dst_config.backend, &dst_conv),
                                        (unsigned long long) tensor_hash(src_config.backend, &src_state),
                                        (unsigned long long) tensor_hash(dst_config.backend, &dst_state));
                            }
                        } else {
                            ggml_tensor * src_attn = src_config.nodes[cell.flash_attn];
                            ggml_tensor * dst_attn = dst_config.nodes[cell.flash_attn];
                            ggml_tensor src_k = corridor_view(j - 1,
                                    ggml_backend_meta_aw_view_root(src_attn->src[1]), false);
                            ggml_tensor dst_k = corridor_view(j,
                                    ggml_backend_meta_aw_view_root(dst_attn->src[1]), true);
                            ggml_tensor src_v = corridor_view(j - 1,
                                    ggml_backend_meta_aw_view_root(src_attn->src[2]), false);
                            ggml_tensor dst_v = corridor_view(j,
                                    ggml_backend_meta_aw_view_root(dst_attn->src[2]), true);
                            ggml_backend_meta_aw_kv_prefix(src_k, src_attn->src[1]);
                            ggml_backend_meta_aw_kv_prefix(dst_k, dst_attn->src[1]);
                            ggml_backend_meta_aw_kv_prefix(src_v, src_attn->src[2]);
                            ggml_backend_meta_aw_kv_prefix(dst_v, dst_attn->src[2]);
                            copy_corridor(&src_k, &dst_k, false);
                            copy_corridor(&src_v, &dst_v, false);
                        }
                        ggml_backend_meta_affinity_wave_stream_fence(dst_config.backend, true);
                        }
                    }
                    ggml_cgraph * graph = backend_ctx->backend_configs[j].cgraphs[0].cgraph_main;
                    graph->n_nodes = 0;
                    const int pre_end = corridor_state_split_for(cell, j) ?
                            cell.attention_state_end + 1 : cell.expert_begin;
                    for (int i = cell.begin; i < pre_end; ++i) {
                        ggml_tensor * node = backend_ctx->backend_configs[j].nodes[i];
                        if (j > 0 && (i == cell.conv_state_clear || i == cell.state_clear)) {
                            continue;
                        }
                        graph->nodes[graph->n_nodes++] = node;
                    }
                    if (!fixed_only && !live_service) {
                        for (int i = cell.expert_begin; i <= cell.expert_end; ++i) {
                            graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                        }
                    }
                    if (!live_service) {
                        for (int i = cell.shared_begin; i <= cell.shared_end; ++i) {
                            graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                        }
                        for (int i = cell.post_begin; i <= cell.end; ++i) {
                            graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                        }
                    }
                    graph->uid = ggml_graph_next_uid();
                }
                ggml_status status = submit_diagonal(diagonal, "pre", trace);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                if (corridor_early) {
                    for (size_t j = 0; j + 1 < n_backends; ++j) {
                        const int layer = diagonal - lane_stagger*(int) j;
                        if (layer < 0 || layer >= (int) cells.size()) {
                            continue;
                        }
                        const ggml_backend_meta_aw_cell & cell = cells[layer];
                        if (!corridor_early_for(cell)) {
                            continue;
                        }
                        auto & src_config = backend_ctx->backend_configs[j];
                        auto & dst_config = backend_ctx->backend_configs[j + 1];
                        auto corridor_view = [&](size_t lane, ggml_tensor * tensor, bool pinned) {
                            size_t view_offs = 0;
                            ggml_tensor * root = tensor;
                            while (root->view_src != nullptr) {
                                view_offs += root->view_offs;
                                root = root->view_src;
                            }
                            auto it = std::find_if(input_snapshots[lane].begin(), input_snapshots[lane].end(),
                                    [&](const input_snapshot & snapshot) { return snapshot.tensor == root; });
                            ggml_tensor mapped = *tensor;
                            if (it == input_snapshots[lane].end()) {
                                return mapped;
                            }
                            mapped.buffer = pinned ? it->persistent.buffer : root->buffer;
                            mapped.data = (char *) (pinned ? it->persistent.data : it->original_data) + view_offs;
                            mapped.view_src = nullptr;
                            mapped.view_offs = 0;
                            return mapped;
                        };
                        auto copy_corridor = [&](ggml_tensor * src, ggml_tensor * dst) {
                            GGML_ASSERT(ggml_are_same_layout(src, dst));
                            GGML_ASSERT(backend_ctx->aw_corridor_copy(
                                    src_config.backend, dst_config.backend, src, dst) == 0);
                            pass_corridor_bytes += ggml_nbytes(dst);
                        };
                        ggml_backend_meta_affinity_wave_stream_fence(src_config.backend, false);
                        if (cell.recurrent >= 0) {
                            ggml_tensor src_conv = corridor_view(j,
                                    src_config.nodes[cell.conv_state_update], false);
                            ggml_tensor dst_conv = corridor_view(j + 1,
                                    dst_config.nodes[serving ? cell.conv_state_update : cell.conv_state_clear], true);
                            ggml_tensor src_state = corridor_view(j,
                                    src_config.nodes[cell.state_update]->src[1], false);
                            ggml_tensor dst_state = corridor_view(j + 1,
                                    serving ? dst_config.nodes[cell.state_update]->src[1] :
                                        dst_config.nodes[cell.state_clear], true);
                            copy_corridor(&src_conv, &dst_conv);
                            copy_corridor(&src_state, &dst_state);
                        } else {
                            ggml_tensor * src_attn = src_config.nodes[cell.flash_attn];
                            ggml_tensor * dst_attn = dst_config.nodes[cell.flash_attn];
                            ggml_tensor src_k = corridor_view(j,
                                    ggml_backend_meta_aw_view_root(src_attn->src[1]), false);
                            ggml_tensor dst_k = corridor_view(j + 1,
                                    ggml_backend_meta_aw_view_root(dst_attn->src[1]), true);
                            ggml_tensor src_v = corridor_view(j,
                                    ggml_backend_meta_aw_view_root(src_attn->src[2]), false);
                            ggml_tensor dst_v = corridor_view(j + 1,
                                    ggml_backend_meta_aw_view_root(dst_attn->src[2]), true);
                            ggml_backend_meta_aw_kv_prefix(src_k, src_attn->src[1]);
                            ggml_backend_meta_aw_kv_prefix(dst_k, dst_attn->src[1]);
                            ggml_backend_meta_aw_kv_prefix(src_v, src_attn->src[2]);
                            ggml_backend_meta_aw_kv_prefix(dst_v, dst_attn->src[2]);
                            copy_corridor(&src_k, &dst_k);
                            copy_corridor(&src_v, &dst_v);
                        }
                    }
                }
                if (corridor_state_split) {
                    for (size_t j = 0; j < n_backends; ++j) {
                        const int layer = diagonal - lane_stagger*(int) j;
                        if (layer < 0 || layer >= (int) cells.size()) {
                            continue;
                        }
                        const ggml_backend_meta_aw_cell & cell = cells[layer];
                        if (!corridor_state_split_for(cell, j)) {
                            continue;
                        }
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_cgraph * graph = bcj.cgraphs[0].cgraph_main;
                        graph->n_nodes = 0;
                        for (int i = cell.attention_state_end + 1; i < cell.expert_begin; ++i) {
                            graph->nodes[graph->n_nodes++] = bcj.nodes[i];
                        }
                        graph->uid = ggml_graph_next_uid();
                        status = ggml_backend_graph_compute_async(bcj.backend, graph);
                        if (status != GGML_STATUS_SUCCESS) {
                            return status;
                        }
                    }
                }
                if (live_service) {
                    GGML_ASSERT(backend_ctx->aw_live_service != nullptr);
                    std::vector<ggml_backend_meta_aw_live_cell> live_cells;
                    std::vector<ggml_backend_t> simple_backends(n_backends);
                    struct live_source {
                        size_t device;
                        ggml_tensor * input;
                        ggml_tensor * ids;
                        ggml_tensor * weights;
                        ggml_tensor * output;
                    };
                    std::vector<live_source> live_sources;
                    for (size_t j = 0; j < n_backends; ++j) {
                        simple_backends[j] = backend_ctx->backend_configs[j].backend;
                        const int layer = diagonal - lane_stagger*(int) j;
                        if (layer < 0 || layer >= (int) cells.size()) {
                            continue;
                        }
                        const ggml_backend_meta_aw_cell & cell = cells[layer];
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_tensor * gate = nullptr;
                        ggml_tensor * weighted = nullptr;
                        for (int i = cell.expert_begin; i <= cell.expert_end; ++i) {
                            ggml_tensor * node = bcj.nodes[i];
                            if (ggml_backend_meta_aw_name_contains(node, "ffn_moe_gate", layer) &&
                                    node->op == GGML_OP_MUL_MAT_ID) {
                                gate = node;
                            } else if (ggml_backend_meta_aw_name_contains(node, "ffn_moe_weighted", layer) &&
                                    node->op == GGML_OP_MUL) {
                                weighted = node;
                            }
                        }
                        GGML_ASSERT(gate != nullptr && weighted != nullptr);
                        ggml_tensor * output = bcj.nodes[cell.expert_end];
                        GGML_ASSERT(gate->op == GGML_OP_MUL_MAT_ID && weighted->op == GGML_OP_MUL);
                        GGML_ASSERT(gate->src[1]->type == GGML_TYPE_F32 && gate->src[2]->type == GGML_TYPE_I32 &&
                                weighted->src[1]->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32);
                        auto route_stride = [&](const ggml_tensor * route) {
                            if (route->ne[0] == 8 && route->ne[1] == output->ne[1]) {
                                return route->nb[1];
                            }
                            if (route->ne[0] == 1 && route->ne[1] == 8 &&
                                    route->ne[2] == output->ne[1]) {
                                return route->nb[2];
                            }
                            GGML_ABORT("unexpected AffinityWave route layout");
                        };
                        live_cells.push_back({
                                layer, (int32_t) j, (int32_t) output->ne[1], 0,
                                route_stride(gate->src[2]), route_stride(weighted->src[1]),
                                (const float *) gate->src[1]->data,
                                (const int32_t *) gate->src[2]->data,
                                (const float *) weighted->src[1]->data,
                                (float *) output->data,
                                shared_service ? (float *) bcj.nodes[cell.shared_raw]->data : nullptr,
                                nullptr
                        });
                        live_sources.push_back({j, gate->src[1], gate->src[2], weighted->src[1], output});
                    }
                    struct verify_snapshot {
                        ggml_backend_buffer_ptr buffer;
                        ggml_tensor tensor;
                    };
                    std::vector<verify_snapshot> verify_snapshots;
                    if (verify_local) {
                        verify_snapshots.reserve(live_cells.size()*4);
                        auto snapshot_tensor = [&](size_t device, const ggml_tensor * src) -> void * {
                            ggml_backend_t backend = simple_backends[device];
                            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
                            verify_snapshots.emplace_back();
                            verify_snapshot & snapshot = verify_snapshots.back();
                            snapshot.buffer.reset(ggml_backend_buft_alloc_buffer(
                                        buft, ggml_backend_buft_get_alloc_size(buft, src)));
                            GGML_ASSERT(snapshot.buffer != nullptr);
                            snapshot.tensor = *src;
                            snapshot.tensor.buffer = snapshot.buffer.get();
                            snapshot.tensor.data = ggml_backend_buffer_get_base(snapshot.buffer.get());
                            snapshot.tensor.view_src = nullptr;
                            snapshot.tensor.view_offs = 0;
                            ggml_backend_tensor_copy_async(backend, backend, src, &snapshot.tensor);
                            return snapshot.tensor.data;
                        };
                        for (size_t i = 0; i < live_cells.size(); ++i) {
                            const live_source & source = live_sources[i];
                            live_cells[i].input = (const float *) snapshot_tensor(source.device, source.input);
                            live_cells[i].ids = (const int32_t *) snapshot_tensor(source.device, source.ids);
                            live_cells[i].weights = (const float *) snapshot_tensor(source.device, source.weights);
                        }
                        for (size_t j = 0; j < n_backends; ++j) {
                            ggml_backend_synchronize(simple_backends[j]);
                        }
                        for (size_t j = 0; j < n_backends; ++j) {
                            const int layer = diagonal - lane_stagger*(int) j;
                            if (layer < 0 || layer >= (int) cells.size()) {
                                continue;
                            }
                            const ggml_backend_meta_aw_cell & cell = cells[layer];
                            ggml_cgraph * graph = backend_ctx->backend_configs[j].cgraphs[0].cgraph_main;
                            graph->n_nodes = 0;
                            for (int i = cell.expert_begin; i <= cell.expert_end; ++i) {
                                graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                            }
                            graph->uid = ggml_graph_next_uid();
                        }
                        status = submit_diagonal(diagonal, "expert-reference", trace);
                        if (status != GGML_STATUS_SUCCESS) {
                            return status;
                        }
                        for (size_t j = 0; j < n_backends; ++j) {
                            ggml_backend_synchronize(simple_backends[j]);
                        }
                        for (size_t i = 0; i < live_cells.size(); ++i) {
                            const live_source & source = live_sources[i];
                            live_cells[i].reference = (const float *) snapshot_tensor(source.device, source.output);
                        }
                        for (size_t j = 0; j < n_backends; ++j) {
                            ggml_backend_synchronize(simple_backends[j]);
                        }
                    }
                    char service_error[256] = {};
                    if (backend_ctx->aw_live_service(simple_backends.data(), live_cells.data(),
                                (int32_t) live_cells.size(), service_error, sizeof(service_error)) != 0) {
                        fprintf(stderr, "AffinityWave: live service failed: %s\n", service_error);
                        return GGML_STATUS_FAILED;
                    }
                    for (size_t j = 0; j < n_backends; ++j) {
                        const int layer = diagonal - lane_stagger*(int) j;
                        if (layer < 0 || layer >= (int) cells.size()) {
                            continue;
                        }
                        const ggml_backend_meta_aw_cell & cell = cells[layer];
                        ggml_cgraph * graph = backend_ctx->backend_configs[j].cgraphs[0].cgraph_main;
                        graph->n_nodes = 0;
                        if (!shared_service) {
                            for (int i = cell.shared_begin; i <= cell.shared_end; ++i) {
                                graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                            }
                        } else {
                            for (int i = cell.shared_raw + 1; i <= cell.shared_end; ++i) {
                                graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                            }
                        }
                        for (int i = cell.post_begin; i <= cell.end; ++i) {
                            graph->nodes[graph->n_nodes++] = backend_ctx->backend_configs[j].nodes[i];
                        }
                        graph->uid = ggml_graph_next_uid();
                    }
                    status = submit_diagonal(diagonal, "post", trace);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
                if (getenv("GGML_CUDA_AW_SYNC_DIAGONAL") != nullptr &&
                        strcmp(getenv("GGML_CUDA_AW_SYNC_DIAGONAL"), "0") != 0) {
                    for (size_t j = 0; j < n_backends; ++j) {
                        ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    }
                }
                }
                }
                const char * publish_overlap_env = getenv("GGML_CUDA_AW_SERVE_PUBLISH_OVERLAP");
                const bool publish_overlap = serving && publish_overlap_env != nullptr &&
                        strcmp(publish_overlap_env, "1") == 0;
                if (publish_overlap) {
                    // The owner stream is also the source stream for the
                    // canonical copies.  CUDA's async copy path orders the
                    // copy after the owner's queued work, while destination
                    // synchronization prevents overwriting a lane that is
                    // still reading its prior state.  This removes only the
                    // redundant owner-side barrier; non-CUDA backends retain
                    // their internal synchronous fallback.
                    for (size_t j = 0; j + 1 < n_backends; ++j) {
                        ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    }
                } else {
                    for (size_t j = 0; j < n_backends; ++j) {
                        ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    }
                }
                if (pass == warmup_passes) {
                    if (serving) {
                        const char * suffix_env = getenv("GGML_CUDA_AW_SERVE_KV_SUFFIX");
                        const bool suffix_requested = suffix_env != nullptr &&
                                strcmp(suffix_env, "1") == 0 &&
                                backend_ctx->aw_canonical_state_valid &&
                                !resets_recurrent_state && backend_ctx->aw_canonical_tokens > 0;
                        int64_t canonical_tokens = 0;
                        auto canonical_view = [&](size_t lane, ggml_tensor * tensor) {
                            size_t view_offs = 0;
                            ggml_tensor * root = tensor;
                            while (root->view_src != nullptr) {
                                view_offs += root->view_offs;
                                root = root->view_src;
                            }
                            auto it = std::find_if(input_snapshots[lane].begin(), input_snapshots[lane].end(),
                                    [&](const input_snapshot & snapshot) { return snapshot.tensor == root; });
                            ggml_tensor mapped = *tensor;
                            if (it != input_snapshots[lane].end()) {
                                mapped.buffer = root->buffer;
                                mapped.data = static_cast<char *>(it->original_data) + view_offs;
                                mapped.view_src = nullptr;
                                mapped.view_offs = 0;
                            }
                            return mapped;
                        };
                        const size_t owner = n_backends - 1;
                        auto & src_config = backend_ctx->backend_configs[owner];
                        for (const auto & cell : cells) {
                            for (size_t j = 0; j < owner; ++j) {
                                auto & dst_config = backend_ctx->backend_configs[j];
                                auto copy_canonical = [&](ggml_tensor * src, ggml_tensor * dst,
                                        bool allow_suffix) {
                                    if (allow_suffix && suffix_requested &&
                                            ggml_backend_meta_aw_kv_suffix(
                                                *src, *dst, backend_ctx->aw_canonical_tokens)) {
                                        ggml_backend_tensor_copy_async(src_config.backend, dst_config.backend, src, dst);
                                        pass_corridor_bytes += ggml_nbytes(dst);
                                        return;
                                    }
                                    GGML_ASSERT(ggml_are_same_layout(src, dst));
                                    ggml_backend_tensor_copy_async(src_config.backend, dst_config.backend, src, dst);
                                    pass_corridor_bytes += ggml_nbytes(dst);
                                };
                                if (cell.recurrent >= 0) {
                                    ggml_tensor src_conv = canonical_view(owner, src_config.nodes[cell.conv_state_update]);
                                    ggml_tensor dst_conv = canonical_view(j, dst_config.nodes[cell.conv_state_update]);
                                    ggml_tensor src_state = canonical_view(owner, src_config.nodes[cell.state_update]->src[1]);
                                    ggml_tensor dst_state = canonical_view(j, dst_config.nodes[cell.state_update]->src[1]);
                                    copy_canonical(&src_conv, &dst_conv, false);
                                    copy_canonical(&src_state, &dst_state, false);
                                } else {
                                    ggml_tensor * src_attn = src_config.nodes[cell.flash_attn];
                                    ggml_tensor * dst_attn = dst_config.nodes[cell.flash_attn];
                                    ggml_tensor src_k = canonical_view(owner, ggml_backend_meta_aw_view_root(src_attn->src[1]));
                                    ggml_tensor dst_k = canonical_view(j, ggml_backend_meta_aw_view_root(dst_attn->src[1]));
                                    ggml_tensor src_v = canonical_view(owner, ggml_backend_meta_aw_view_root(src_attn->src[2]));
                                    ggml_tensor dst_v = canonical_view(j, ggml_backend_meta_aw_view_root(dst_attn->src[2]));
                                    ggml_backend_meta_aw_kv_prefix(src_k, src_attn->src[1]);
                                    ggml_backend_meta_aw_kv_prefix(dst_k, dst_attn->src[1]);
                                    ggml_backend_meta_aw_kv_prefix(src_v, src_attn->src[2]);
                                    ggml_backend_meta_aw_kv_prefix(dst_v, dst_attn->src[2]);
                                    canonical_tokens = std::max(canonical_tokens, src_k.ne[1]);
                                    copy_canonical(&src_k, &dst_k, true);
                                    copy_canonical(&src_v, &dst_v, true);
                                }
                            }
                        }
                        for (size_t j = 0; j < owner; ++j) {
                            ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                        }
                        backend_ctx->aw_canonical_state_valid = canonical_tokens > 0;
                        backend_ctx->aw_canonical_tokens = canonical_tokens;
                        backend_ctx->aw_canonical_epoch =
                                ggml_backend_meta_aw_state_epoch.load(std::memory_order_relaxed);
                        ggml_backend_meta_aw_last_canonical_tokens.store(
                                canonical_tokens, std::memory_order_relaxed);
                    }
                    elapsed_ms = (ggml_time_us() - pass_begin_us) / 1000.0;
                    corridor_bytes = pass_corridor_bytes;
                }
            }
            if (warmup_passes != 0) {
                backend_ctx->aw_precaptured_tokens = wave_tokens;
                if (serving) {
                    if (backend_ctx->aw_precaptured_signatures.size() >= 64) {
                        backend_ctx->aw_precaptured_signatures.clear();
                    }
                    backend_ctx->aw_precaptured_signatures.insert(capture_signature);
                }
            }
            double tail_ms = 0.0;
            if (return_output) {
                ggml_backend_meta_aw_trace_scope tail_trace(
                        timed_trace, "output-tail", 10, (uint64_t) wave_tokens);
                const int64_t tail_begin_us = ggml_time_us();
                int result_norm = -1;
                for (int i = cells.back().end + 1; i < cgraph->n_nodes; ++i) {
                    if (strcmp(cgraph->nodes[i]->name, "result_norm") == 0) {
                        result_norm = i;
                        break;
                    }
                }
                GGML_ASSERT(result_norm > cells.back().end + 1 && result_norm + 1 < cgraph->n_nodes);

                auto submit_tail_range = [&](int begin, int end) -> ggml_status {
                    for (size_t j = 0; j < n_backends; ++j) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        ggml_cgraph * graph = bcj.cgraphs[0].cgraph_main;
                        graph->n_nodes = 0;
                        for (int i = begin; i < end; ++i) {
                            graph->nodes[graph->n_nodes++] = bcj.nodes[i];
                        }
                        graph->uid = ggml_graph_next_uid();
                    }
                    const std::function<ggml_status(size_t)> tail_job = [&](size_t j) -> ggml_status {
                        auto & bcj = backend_ctx->backend_configs[j];
                        return ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[0].cgraph_main);
                    };
                    auto & pool = *backend_ctx->submit;
                    {
                        std::lock_guard<std::mutex> lock(pool.m);
                        pool.spmd_job = &tail_job;
                        pool.n_pending = n_backends - 1;
                        pool.seq++;
                    }
                    pool.cv_go.notify_all();
                    const ggml_status status0 = tail_job(0);
                    {
                        std::unique_lock<std::mutex> lock(pool.m);
                        pool.cv_done.wait(lock, [&]() { return pool.n_pending == 0; });
                        pool.spmd_job = nullptr;
                    }
                    if (status0 != GGML_STATUS_SUCCESS) {
                        return status0;
                    }
                    for (size_t w = 0; w < n_backends - 1; ++w) {
                        if (pool.slots[w].status != GGML_STATUS_SUCCESS) {
                            return pool.slots[w].status;
                        }
                    }
                    for (size_t j = 0; j < n_backends; ++j) {
                        ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                    }
                    return GGML_STATUS_SUCCESS;
                };

                ggml_status status = submit_tail_range(cells.back().end + 1, result_norm);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }

                int64_t output_row = 0;
                for (size_t src_device = 0; src_device < n_backends; ++src_device) {
                    auto & src_config = backend_ctx->backend_configs[src_device];
                    ggml_tensor * src_result = src_config.nodes[result_norm];
                    ggml_tensor * src_hidden = src_result->src[0];
                    ggml_tensor * src_ids = src_result->src[1];
                    const int64_t rows = src_ids->ne[0];
                    if (rows == 0) {
                        continue;
                    }
                    GGML_ASSERT(rows <= src_hidden->ne[1]);
                    ggml_tensor src_view = *src_hidden;
                    src_view.ne[1] = rows;
                    src_view.data = (char *) src_hidden->data + (src_hidden->ne[1] - rows)*src_hidden->nb[1];
                    for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
                        src_view.nb[dim] = src_view.nb[dim - 1]*src_view.ne[dim - 1];
                    }
                    for (size_t dst_device = 0; dst_device < n_backends; ++dst_device) {
                        auto & dst_config = backend_ctx->backend_configs[dst_device];
                        ggml_tensor * dst_result = dst_config.nodes[result_norm];
                        GGML_ASSERT(dst_result->ne[1] == cgraph->nodes[result_norm]->ne[1]);
                        ggml_tensor dst_view = *dst_result;
                        dst_view.ne[1] = rows;
                        dst_view.data = (char *) dst_result->data + output_row*dst_result->nb[1];
                        for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
                            dst_view.nb[dim] = dst_view.nb[dim - 1]*dst_view.ne[dim - 1];
                        }
                        ggml_backend_tensor_copy_async(
                                src_config.backend, dst_config.backend, &src_view, &dst_view);
                    }
                    output_row += rows;
                }
                GGML_ASSERT(output_row == cgraph->nodes[result_norm]->ne[1]);
                for (size_t j = 0; j < n_backends; ++j) {
                    ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
                }

                status = submit_tail_range(result_norm + 1, cgraph->n_nodes);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                tail_ms = (ggml_time_us() - tail_begin_us) / 1000.0;
            }
            fprintf(stderr, "AffinityWave: dense lane mode=%s tokens=%lld cells=%zu corridors=%.1f MiB setup=%.3f ms wave=%.3f ms tail=%.3f ms total=%.3f ms throughput=%.1f tok/s; output=%s\n",
                    fixed_only ? "fixed" : (live_service ? "service" : "local-expert"),
                    (long long) cgraph->nodes[cells[0].end]->ne[1], cells.size()*n_backends,
                    corridor_bytes/(1024.0*1024.0), setup_ms, elapsed_ms, tail_ms,
                    setup_ms + elapsed_ms + tail_ms,
                    cgraph->nodes[cells[0].end]->ne[1] * 1000.0 / (setup_ms + elapsed_ms + tail_ms),
                    return_output ? "logits" : "withheld");
            return return_output ? GGML_STATUS_SUCCESS : GGML_STATUS_ABORTED;
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    // [TAG_MOE_ASYNCEP] token-slice EP with background weight gather. Register the expert
    // shards once (graph node order = the prefetch order) and raise the CUDA-side enable
    // flag for graphs carrying a prefill-sized EP MUL_MAT_ID; the CUDA fallback engages per
    // call (its own >= min-tokens check), everything else keeps the expert-filtered path.
    // The combine stays the PARTIAL allreduce below - token-sliced outputs are
    // disjoint-support partials (foreign-token rows are zero), so summing them is exact and
    // mixed expert-mode/token-slice evals need no boundary coordination.
    static const bool asyncep_env = getenv("GGML_CUDA_MOE_ASYNCEP") != nullptr;
    if (asyncep_env && n_backends > 1 && backend_ctx->asyncep_register != nullptr &&
            backend_ctx->asyncep_set_enabled != nullptr) {
        static const int64_t asyncep_min_tokens = []() {
            const char * s = getenv("GGML_CUDA_MOE_ASYNCEP_MIN_TOKENS");
            return s != nullptr ? atoll(s) : 3072;
        }();
        bool engage = false;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] >= asyncep_min_tokens && node->src[0] != nullptr &&
                    ggml_backend_meta_get_split_state(node->src[0], false).axis == GGML_BACKEND_SPLIT_AXIS_2) {
                engage = true;
                break;
            }
        }
        if (engage) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr ||
                        ggml_backend_meta_get_split_state(node->src[0], false).axis != GGML_BACKEND_SPLIT_AXIS_2) {
                    continue;
                }
                std::vector<const void *> shard_data(n_backends);
                std::vector<size_t>       shard_offs(n_backends);
                size_t shard_bytes = 0;
                bool ok = true;
                for (size_t j = 0; j < n_backends && ok; j++) {
                    const ggml_tensor * w = ggml_backend_meta_buffer_simple_tensor(node->src[0], j);
                    const ggml_tensor * d = backend_ctx->backend_configs[j].nodes[i];
                    const int32_t stamp = d != nullptr ? d->op_params[GGML_MAX_OP_PARAMS/sizeof(int32_t) - 1] : 0;
                    ok = w != nullptr && w->data != nullptr && stamp != 0;
                    if (!ok) {
                        break;
                    }
                    shard_data[j] = w->data;
                    shard_offs[j] = (size_t) (stamp - 1) * node->src[0]->nb[2];
                    if (j == 0) {
                        shard_bytes = ggml_nbytes(w);
                    }
                    ok = ggml_nbytes(w) == shard_bytes; // equal shards required (n_expert % n_ranks == 0)
                }
                if (ok) {
                    backend_ctx->asyncep_register(shard_data.data(), shard_offs.data(), (int) n_backends, shard_bytes);
                }
            }
        }
        backend_ctx->asyncep_set_enabled(engage ? 1 : 0);
    }

    // [TAG_META_TBO] Stage D two-batch overlap: alternate consecutive evals between two CUDA
    // streams per device so eval k+1's host phases overlap eval k's queued kernels and eval
    // k's allreduces overlap eval k+1's compute. Engage ONLY on rebuild evals: the meta
    // compute container flip (stc_compute[2], toggled per rebuild) is what double-buffers the
    // activations between in-flight evals -- a REUSED graph writes the same container and
    // must not overlap (run with LLAMA_GRAPH_REUSE_DISABLE=1 to make every prefill eval a
    // rebuild). The begin_eval hook must run every eval so TBO->decode transitions drain.
    static const bool tbo_env = getenv("GGML_META_TBO") != nullptr;
    if (tbo_env && n_backends > 1 && backend_ctx->tbo_begin_eval != nullptr &&
            backend_ctx->comm_ctx != nullptr && backend_ctx->comm_allreduce_single != nullptr &&
            backend_ctx->comm_allreduce_single(backend_ctx->comm_ctx, nullptr, 0)) {
        // NCCL-only (probe above): the per-slot comm sets serialize collectives per stream;
        // the butterfly/internal allreduce paths are not slot-aware.
        static const int64_t tbo_min_tokens = []() {
            const char * s = getenv("GGML_META_TBO_MIN_TOKENS");
            return s != nullptr ? atoll(s) : 64;
        }();
        bool tbo_engage = false;
        if (needs_rebuild) {
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] >= tbo_min_tokens) {
                    tbo_engage = true;
                    break;
                }
            }
        }
        for (size_t j = 0; j < n_backends; j++) {
            backend_ctx->tbo_begin_eval(backend_ctx->backend_configs[j].backend, tbo_engage ? 1 : 0);
        }
    }

    // [TAG_META_SUBMIT] mode 2 (SPMD): each device's worker runs the ENTIRE subgraph loop,
    // including its own convert->allreduce->convert on its own comm. This removes the
    // per-subgraph join and the main-thread serial allreduce issue that left the GPUs idle
    // between bursts (P7 nsys: ~6.4ms gap before every allreduce kernel, devices ~48% busy).
    // Per-comm collective order is unchanged (every device walks subgraphs in order), and the
    // NCCL ring order is fixed by comm topology, so results are byte-identical to mode 1.
    const bool use_spmd = use_submit_threads && submit_threads_mode >= 2 &&
        backend_ctx->comm_ctx != nullptr && backend_ctx->comm_allreduce_single != nullptr &&
        backend_ctx->comm_allreduce_single(backend_ctx->comm_ctx, nullptr, 0); // availability probe
    if (use_spmd) {
        static std::atomic<bool> spmd_logged{false};
        if (!spmd_logged.exchange(true)) {
            // WARN not INFO: the completion/server frontends filter ggml INFO from stderr,
            // and gate scripts key on this line to prove the mode actually engaged.
            GGML_LOG_WARN("%s: [TAG_META_SUBMIT] SPMD mode engaged (per-device subgraph loops + per-comm allreduce)\n", __func__);
        }
        static const bool dbg_ar = getenv("GGML_META_DEBUG_ALLREDUCE") != nullptr;
        const std::function<ggml_status(size_t)> spmd_job = [&](size_t j) -> ggml_status {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                if (i < backend_ctx->n_subgraphs - 1) {
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    ggml_tensor * node = cgraph_ij->nodes[cgraph_ij->n_nodes - 1];
                    if (dbg_ar && j == 0) {
                        static std::atomic<long> ar_count{0};
                        const long c = ar_count++;
                        if (c < 400) {
                            GGML_LOG_WARN("meta allreduce #%ld: %-24s %zu bytes\n", c, node->name, ggml_nbytes(node));
                        }
                    }
                    if (!backend_ctx->comm_allreduce_single(backend_ctx->comm_ctx, node, (int) j)) {
                        return GGML_STATUS_FAILED;
                    }
                }
            }
            return GGML_STATUS_SUCCESS;
        };
        auto & pool = *backend_ctx->submit;
        {
            std::lock_guard<std::mutex> lock(pool.m);
            pool.spmd_job  = &spmd_job;
            pool.n_pending = n_backends - 1;
            pool.seq++;
        }
        pool.cv_go.notify_all();
        const ggml_status status0 = spmd_job(0);
        {
            std::unique_lock<std::mutex> lock(pool.m);
            pool.cv_done.wait(lock, [&]() { return pool.n_pending == 0; });
            pool.spmd_job = nullptr;
        }
        if (status0 != GGML_STATUS_SUCCESS) {
            return status0;
        }
        for (size_t w = 0; w < n_backends - 1; w++) {
            if (pool.slots[w].status != GGML_STATUS_SUCCESS) {
                return pool.slots[w].status;
            }
        }
        static const bool dbg_sync_spmd = getenv("GGML_META_DEBUG_SYNC") != nullptr;
        if (dbg_sync_spmd) {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
            }
        }
        // [TAG_META_TBO] depth-2 bound: wait for the PREVIOUS eval before the caller reuses
        // its compute container / stream pool (no-op when TBO is not active).
        if (backend_ctx->tbo_end_eval != nullptr) {
            for (size_t j = 0; j < n_backends; j++) {
                backend_ctx->tbo_end_eval(backend_ctx->backend_configs[j].backend);
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        if (use_submit_threads) {
            // [TAG_META_SUBMIT] backends 1..n-1 on workers, backend 0 inline, then join.
            // The allreduce below stays on this thread so collective ordering is unchanged.
            auto & pool = *backend_ctx->submit;
            {
                std::lock_guard<std::mutex> lock(pool.m);
                for (size_t j = 1; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    pool.slots[j - 1].backend = bcj.backend;
                    pool.slots[j - 1].cgraph  = bcj.cgraphs[i].cgraph_main;
                }
                pool.n_pending = n_backends - 1;
                pool.seq++;
            }
            pool.cv_go.notify_all();
            auto & bc0 = backend_ctx->backend_configs[0];
            const ggml_status status0 = ggml_backend_graph_compute_async(bc0.backend, bc0.cgraphs[i].cgraph_main);
            {
                std::unique_lock<std::mutex> lock(pool.m);
                pool.cv_done.wait(lock, [&]() { return pool.n_pending == 0; });
            }
            if (status0 != GGML_STATUS_SUCCESS) {
                return status0;
            }
            for (size_t w = 0; w < n_backends - 1; w++) {
                if (pool.slots[w].status != GGML_STATUS_SUCCESS) {
                    return pool.slots[w].status;
                }
            }
        } else {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                // [TAG_MOE_EP] debug census of allreduce boundaries (name + bytes), first 400 only
                static const bool dbg_ar = getenv("GGML_META_DEBUG_ALLREDUCE") != nullptr;
                if (dbg_ar) {
                    static std::atomic<long> ar_count{0};
                    const long c = ar_count++;
                    if (c < 400) {
                        GGML_LOG_WARN("meta allreduce #%ld: %-24s %zu bytes\n", c, nodes[0]->name, ggml_nbytes(nodes[0]));
                    }
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    // [TAG_MOE_EP] debug: force a full-device sync after every graph so a latent async fault
    // aborts at the step it occurs instead of surfacing at teardown.
    static const bool dbg_sync = getenv("GGML_META_DEBUG_SYNC") != nullptr;
    if (dbg_sync) {
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_synchronize(backend_ctx->backend_configs[j].backend);
        }
    }
    // [TAG_META_TBO] depth-2 bound: wait for the PREVIOUS eval before the caller reuses its
    // compute container / stream pool (no-op when TBO is not active).
    if (backend_ctx->tbo_end_eval != nullptr) {
        for (size_t j = 0; j < n_backends; j++) {
            backend_ctx->tbo_end_eval(backend_ctx->backend_configs[j].backend);
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
