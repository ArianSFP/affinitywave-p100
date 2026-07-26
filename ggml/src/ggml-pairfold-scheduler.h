#pragma once

#include <array>
#include <cstdint>

namespace ggml_pairfold {

constexpr int n_layers      = 40;
constexpr int n_panels      = 2;
constexpr int n_pairs       = 2;
constexpr int n_generations = 2;
constexpr int n_tasks       = n_layers*n_panels;
constexpr int max_deps      = 3;

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
