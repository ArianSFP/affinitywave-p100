#include "../ggml/src/ggml-pairfold-scheduler.h"

#include <algorithm>
#include <array>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace ggml_pairfold;

static bool has_dependency(const task & value, int dependency) {
    for (int i = 0; i < value.n_deps; ++i) {
        if (value.deps[i] == dependency) {
            return true;
        }
    }
    return false;
}

static void test_static_dag() {
    const schedule value = make_schedule(7);
    std::array<int, n_tasks> position{};
    std::set<int> emitted;
    for (int i = 0; i < n_tasks; ++i) {
        const int index = value.order[i];
        assert(index >= 0 && index < n_tasks);
        assert(emitted.insert(index).second);
        position[index] = i;
        const task & current = value.tasks[index];
        assert(current.layer == index/n_panels);
        assert(current.panel == index%n_panels);
        assert(current.pair == pair_for_layer(current.layer));
        assert(current.generation == 7);
        assert(current.event == n_tasks + index);
        assert(current.handoff == current.panel);
    }
    assert(emitted.size() == n_tasks);

    for (const task & current : value.tasks) {
        if (current.panel == 1) {
            assert(has_dependency(current, task_index(current.layer, 0)));
        }
        if (current.layer != 0) {
            assert(has_dependency(current,
                        task_index(current.layer - 1, current.panel)));
        }
        for (int i = 0; i < current.n_deps; ++i) {
            assert(position[current.deps[i]] <
                    position[task_index(current.layer, current.panel)]);
        }
    }

    const task & layer20_panel0 = value.tasks[task_index(20, 0)];
    assert(layer20_panel0.pair == 1);
    assert(value.tasks[task_index(19, 1)].pair == 1);
    assert(has_dependency(layer20_panel0, task_index(19, 1)));

    for (int pair = 0; pair < n_pairs; ++pair) {
        int previous = -1;
        for (int ordered : value.order) {
            const task & current = value.tasks[ordered];
            if (current.pair != pair) {
                continue;
            }
            if (previous >= 0) {
                assert(has_dependency(current, previous));
            }
            previous = ordered;
        }
    }
}

struct simulated_call {
    schedule value;
    std::array<uint64_t, n_tasks> stage_end{};
    std::array<uint64_t, n_tasks> begin{};
    std::array<uint64_t, n_tasks> end{};
    std::array<uint64_t, n_tasks> hidden_ready{};
};

static simulated_call simulate(int generation, uint32_t seed) {
    simulated_call result{make_schedule(generation), {}, {}, {}, {}};
    std::array<uint64_t, n_pairs> pair_ready{};
    std::array<std::array<uint64_t, n_panels>, n_pairs>
            slot_consumed{};
    std::array<std::array<int, n_panels>, n_pairs>
            slot_owner{};
    for (auto & owner : slot_owner) {
        owner.fill(-1);
    }
    std::mt19937 random(seed);
    std::uniform_int_distribution<uint64_t> duration(1, 17);
    std::uniform_int_distribution<uint64_t> copy_duration(1, 5);
    for (int index : result.value.order) {
        const task & current = result.value.tasks[index];
        if (current.layer != 0) {
            const int previous =
                    task_index(current.layer - 1,
                            current.panel);
            const int previous_pair =
                    result.value.tasks[previous].pair;
            const int read_pair =
                    previous_pair == current.pair ?
                        previous_pair : current.pair;
            result.stage_end[index] =
                    result.hidden_ready[previous] +
                    copy_duration(random);
            slot_consumed[read_pair][current.panel] =
                    result.stage_end[index];
        }

        uint64_t ready = pair_ready[current.pair];
        ready = std::max(ready, result.stage_end[index]);
        for (int i = 0; i < current.n_deps; ++i) {
            ready = std::max(ready, result.end[current.deps[i]]);
        }
        result.begin[index] = ready;
        result.end[index] = ready + duration(random);
        pair_ready[current.pair] = result.end[index];

        const int next_pair =
                current.layer + 1 < n_layers ?
                    pair_for_layer(current.layer + 1) :
                    current.panel;
        uint64_t publish =
                std::max(result.end[index],
                        slot_consumed[
                            current.pair][current.panel]);
        const int previous_local_owner =
                slot_owner[current.pair][current.panel];
        if (previous_local_owner >= 0) {
            assert(publish >=
                    slot_consumed[
                        current.pair][current.panel]);
        }
        publish += copy_duration(random);
        slot_owner[current.pair][current.panel] = index;

        if (next_pair != current.pair) {
            publish = std::max(publish,
                    slot_consumed[next_pair][current.panel]);
            const int previous_remote_owner =
                    slot_owner[next_pair][current.panel];
            if (previous_remote_owner >= 0) {
                assert(publish >=
                        slot_consumed[
                            next_pair][current.panel]);
            }
            publish += copy_duration(random);
            slot_owner[next_pair][current.panel] = index;
        }
        result.hidden_ready[index] = publish;
    }
    return result;
}

static void test_delayed_completions() {
    for (uint32_t seed = 0; seed < 128; ++seed) {
        const simulated_call call = simulate(seed, seed*17 + 3);
        for (const task & current : call.value.tasks) {
            const int index = task_index(current.layer, current.panel);
            assert(call.begin[index] >= call.stage_end[index]);
            for (int i = 0; i < current.n_deps; ++i) {
                assert(call.begin[index] >= call.end[current.deps[i]]);
            }
            if (current.layer != 0) {
                const int previous = task_index(
                        current.layer - 1,
                        current.panel);
                assert(call.stage_end[index] >
                        call.hidden_ready[previous]);
            }
        }

        std::array<int, n_pairs> previous_on_pair{{-1, -1}};
        for (int index : call.value.order) {
            const task & current = call.value.tasks[index];
            const int previous =
                    previous_on_pair[current.pair];
            if (previous >= 0) {
                assert(call.begin[index] >= call.end[previous]);
            }
            previous_on_pair[current.pair] = index;
        }
    }
}

static void test_generation_wraparound() {
    std::array<uint64_t, n_generations> terminal{};
    uint64_t now = 0;
    for (int generation = 0; generation < 11; ++generation) {
        const int bank = generation % n_generations;
        now = std::max(now, terminal[bank]);
        const simulated_call call = simulate(generation,
                (uint32_t) generation + 91);
        uint64_t duration = 0;
        for (int task = 0; task < n_tasks; ++task) {
            duration = std::max(duration,
                    std::max(call.end[task],
                        call.hidden_ready[task]));
        }
        terminal[bank] = now + duration;
        now += (generation % 3 == 0) ? 0 : 1;
        for (const task & current : call.value.tasks) {
            assert(current.event ==
                    bank*n_tasks +
                    task_index(current.layer, current.panel));
        }
    }
    assert(terminal[0] != 0 && terminal[1] != 0);
}

int main() {
    test_static_dag();
    test_delayed_completions();
    test_generation_wraparound();
    std::puts("pairfold scheduler tests passed");
    return 0;
}
