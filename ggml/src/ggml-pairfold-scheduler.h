#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml_pairfold {

constexpr int n_layers      = 40;
constexpr int n_panels      = 2;
constexpr int n_pairs       = 2;
constexpr int n_generations = 2;
constexpr int n_tasks       = n_layers*n_panels;
constexpr int max_deps      = 3;

constexpr std::size_t q8_0_block_bytes  = 34;
constexpr int t64_rows                  = 64;
constexpr int t64_kstage                = 32;
constexpr std::size_t t64_stage_bytes   =
        t64_rows*2 + t64_rows*t64_kstage;
static_assert(
        t64_stage_bytes == t64_rows*q8_0_block_bytes,
        "T64 packing must preserve the Q8_0 byte count");

inline bool pack_q8_0_t64(
        const void * source, void * destination, std::size_t bytes,
        int n, int k) {
    if (source == nullptr || destination == nullptr || bytes == 0 ||
            n <= 0 || k <= 0 || n % t64_rows != 0 ||
            k % t64_kstage != 0) {
        return false;
    }

    const std::size_t rows = static_cast<std::size_t>(n);
    const std::size_t kblocks =
            static_cast<std::size_t>(k/t64_kstage);
    const std::size_t size_max =
            std::numeric_limits<std::size_t>::max();
    if (rows > size_max/kblocks) {
        return false;
    }
    const std::size_t blocks_per_matrix = rows*kblocks;
    if (blocks_per_matrix > size_max/q8_0_block_bytes) {
        return false;
    }
    const std::size_t matrix_bytes =
            blocks_per_matrix*q8_0_block_bytes;
    if (bytes % matrix_bytes != 0) {
        return false;
    }

    const std::uintptr_t source_begin =
            reinterpret_cast<std::uintptr_t>(source);
    const std::uintptr_t destination_begin =
            reinterpret_cast<std::uintptr_t>(destination);
    const std::uintptr_t pointer_max =
            std::numeric_limits<std::uintptr_t>::max();
    if (bytes > pointer_max - source_begin ||
            bytes > pointer_max - destination_begin) {
        return false;
    }
    const std::uintptr_t source_end = source_begin + bytes;
    const std::uintptr_t destination_end =
            destination_begin + bytes;
    if (source_begin < destination_end &&
            destination_begin < source_end) {
        return false;
    }

    const auto * input =
            static_cast<const std::uint8_t *>(source);
    auto * output = static_cast<std::uint8_t *>(destination);
    const std::size_t blocks = bytes/q8_0_block_bytes;
    const std::size_t ntiles = rows/t64_rows;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t matrix = block/blocks_per_matrix;
        const std::size_t within =
                block - matrix*blocks_per_matrix;
        const std::size_t row = within/kblocks;
        const std::size_t kblock = within - row*kblocks;
        const std::size_t tile =
                (matrix*ntiles + row/t64_rows)*kblocks +
                kblock;
        std::uint8_t * tile_stage =
                output + tile*t64_stage_bytes;
        const std::uint8_t * q8 =
                input + block*q8_0_block_bytes;
        std::memcpy(
                tile_stage + row%t64_rows*2,
                q8,
                2);
        std::memcpy(
                tile_stage + t64_rows*2 +
                    row%t64_rows*t64_kstage,
                q8 + 2,
                t64_kstage);
    }
    return true;
}

struct task {
    int layer      = -1;
    int panel      = -1;
    int pair       = -1;
    int generation = -1;
    int event      = -1;
    int handoff    = -1;
    std::array<int, max_deps> deps{{-1, -1, -1}};
    int n_deps = 0;
};

struct schedule {
    std::array<task, n_tasks> tasks{};
    std::array<int, n_tasks> order{};
};

inline int task_index(int layer, int panel) {
    return layer*n_panels + panel;
}

inline int pair_for_layer(int layer) {
    return layer < n_layers/2 ? layer % n_pairs : (layer + 1) % n_pairs;
}

inline int pair_local_ordinal(int layer) {
    const int pair = pair_for_layer(layer);
    int ordinal = 0;
    for (int previous = 0; previous < layer; ++previous) {
        if (pair_for_layer(previous) == pair) {
            ++ordinal;
        }
    }
    return ordinal;
}

inline int weight_slot_for_layer(int layer) {
    return pair_local_ordinal(layer) % 2;
}

inline void add_dependency(task & value, int dependency) {
    for (int i = 0; i < value.n_deps; ++i) {
        if (value.deps[i] == dependency) {
            return;
        }
    }
    value.deps[value.n_deps++] = dependency;
}

inline schedule make_schedule(int generation) {
    schedule result;
    const int bank = generation % n_generations;
    for (int layer = 0; layer < n_layers; ++layer) {
        for (int panel = 0; panel < n_panels; ++panel) {
            const int index = task_index(layer, panel);
            task & value = result.tasks[index];
            value.layer = layer;
            value.panel = panel;
            value.pair = pair_for_layer(layer);
            value.generation = generation;
            value.event = bank*n_tasks + index;
            value.handoff = panel;
            if (panel != 0) {
                add_dependency(value, task_index(layer, panel - 1));
            }
            if (layer != 0) {
                add_dependency(value, task_index(layer - 1, panel));
            }
        }
    }

    std::array<int, n_pairs> previous_on_pair{{-1, -1}};
    int next = 0;
    for (int wave = 0; wave <= n_layers; ++wave) {
        for (int panel = n_panels - 1; panel >= 0; --panel) {
            const int layer = wave - panel;
            if (layer < 0 || layer >= n_layers) {
                continue;
            }
            const int index = task_index(layer, panel);
            task & value = result.tasks[index];
            if (previous_on_pair[value.pair] >= 0) {
                add_dependency(value, previous_on_pair[value.pair]);
            }
            previous_on_pair[value.pair] = index;
            result.order[next++] = index;
        }
    }
    return result;
}

} // namespace ggml_pairfold
