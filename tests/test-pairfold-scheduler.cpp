#include "../ggml/src/ggml-pairfold-scheduler.h"

#include <algorithm>
#include <array>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

static void test_q8_0_t64_case(
        int n, int k, std::size_t matrices,
        std::uint32_t salt) {
    const std::size_t kblocks =
            static_cast<std::size_t>(k/t64_kstage);
    const std::size_t blocks_per_matrix =
            static_cast<std::size_t>(n)*kblocks;
    const std::size_t bytes =
            matrices*blocks_per_matrix*q8_0_block_bytes;
    std::vector<std::uint8_t> source(bytes);
    std::vector<std::uint8_t> packed(bytes, 0xa5);
    std::vector<std::uint8_t> expected(bytes, 0x5a);
    std::vector<std::uint8_t> recovered(bytes, 0);
    std::vector<std::uint8_t> written(bytes, 0);

    for (std::size_t i = 0; i < bytes; ++i) {
        const std::uint32_t mixed =
                static_cast<std::uint32_t>(i)*131u +
                static_cast<std::uint32_t>(i >> 7) +
                salt*17u;
        source[i] = static_cast<std::uint8_t>(
                mixed ^ (mixed >> 11));
    }

    const std::size_t ntiles =
            static_cast<std::size_t>(n/t64_rows);
    for (std::size_t matrix = 0;
            matrix < matrices; ++matrix) {
        for (int row = 0; row < n; ++row) {
            for (std::size_t kblock = 0;
                    kblock < kblocks; ++kblock) {
                const std::size_t block =
                        matrix*blocks_per_matrix +
                        static_cast<std::size_t>(row)*kblocks +
                        kblock;
                const std::size_t tile =
                        (matrix*ntiles +
                            static_cast<std::size_t>(
                                row/t64_rows))*kblocks +
                        kblock;
                const std::size_t input_offset =
                        block*q8_0_block_bytes;
                const std::size_t stage_offset =
                        tile*t64_stage_bytes;
                const std::size_t scale_offset =
                        stage_offset +
                        static_cast<std::size_t>(
                                row%t64_rows)*2;
                const std::size_t values_offset =
                        stage_offset + t64_rows*2 +
                        static_cast<std::size_t>(
                                row%t64_rows)*t64_kstage;
                std::memcpy(
                        expected.data() + scale_offset,
                        source.data() + input_offset,
                        2);
                std::memcpy(
                        expected.data() + values_offset,
                        source.data() + input_offset + 2,
                        t64_kstage);
                for (std::size_t byte = 0; byte < 2;
                        ++byte) {
                    ++written[scale_offset + byte];
                }
                for (std::size_t byte = 0;
                        byte <
                            static_cast<std::size_t>(
                                t64_kstage);
                        ++byte) {
                    ++written[values_offset + byte];
                }
            }
        }
    }
    assert(std::all_of(
            written.begin(), written.end(),
            [](std::uint8_t count) {
                return count == 1;
            }));

    assert(pack_q8_0_t64(
            source.data(), packed.data(), bytes, n, k));
    assert(packed == expected);

    for (std::size_t matrix = 0;
            matrix < matrices; ++matrix) {
        for (int row = 0; row < n; ++row) {
            for (std::size_t kblock = 0;
                    kblock < kblocks; ++kblock) {
                const std::size_t block =
                        matrix*blocks_per_matrix +
                        static_cast<std::size_t>(row)*kblocks +
                        kblock;
                const std::size_t tile =
                        (matrix*ntiles +
                            static_cast<std::size_t>(
                                row/t64_rows))*kblocks +
                        kblock;
                const std::size_t output_offset =
                        block*q8_0_block_bytes;
                const std::size_t stage_offset =
                        tile*t64_stage_bytes;
                const std::size_t scale_offset =
                        stage_offset +
                        static_cast<std::size_t>(
                                row%t64_rows)*2;
                const std::size_t values_offset =
                        stage_offset + t64_rows*2 +
                        static_cast<std::size_t>(
                                row%t64_rows)*t64_kstage;
                std::memcpy(
                        recovered.data() + output_offset,
                        packed.data() + scale_offset,
                        2);
                std::memcpy(
                        recovered.data() + output_offset + 2,
                        packed.data() + values_offset,
                        t64_kstage);
            }
        }
    }
    assert(recovered == source);
}

static void test_q8_0_t64_pack() {
    test_q8_0_t64_case(512, 2048, 1, 1);
    test_q8_0_t64_case(512, 2048, 1, 2);
    test_q8_0_t64_case(2048, 512, 1, 3);
    test_q8_0_t64_case(128, 96, 3, 4);

    const std::size_t bytes =
            static_cast<std::size_t>(t64_rows)*
            q8_0_block_bytes;
    std::vector<std::uint8_t> storage(2*bytes + 1, 0x31);
    std::vector<std::uint8_t> destination(bytes);
    assert(pack_q8_0_t64(
            storage.data(), destination.data(), bytes,
            t64_rows, t64_kstage));
    assert(pack_q8_0_t64(
            storage.data(), storage.data() + bytes, bytes,
            t64_rows, t64_kstage));

    assert(!pack_q8_0_t64(
            nullptr, destination.data(), bytes,
            t64_rows, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), nullptr, bytes,
            t64_rows, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), destination.data(), 0,
            t64_rows, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), destination.data(), bytes,
            t64_rows - 1, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), destination.data(), bytes,
            t64_rows, t64_kstage - 1));
    assert(!pack_q8_0_t64(
            storage.data(), destination.data(), bytes - 1,
            t64_rows, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), storage.data(), bytes,
            t64_rows, t64_kstage));
    assert(!pack_q8_0_t64(
            storage.data(), storage.data() + 1, bytes,
            t64_rows, t64_kstage));
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

static void test_weight_slots() {
    std::array<int, n_pairs> expected_ordinal{};
    std::array<std::array<int, 2>, n_pairs> last_layer{{
        {{-1, -1}},
        {{-1, -1}},
    }};
    for (int layer = 0; layer < n_layers; ++layer) {
        const int pair = pair_for_layer(layer);
        const int ordinal = pair_local_ordinal(layer);
        const int slot = weight_slot_for_layer(layer);
        assert(ordinal == expected_ordinal[pair]++);
        assert(slot == ordinal % 2);
        if (last_layer[pair][slot] >= 0) {
            assert(pair_local_ordinal(last_layer[pair][slot]) ==
                    ordinal - 2);
        }
        last_layer[pair][slot] = layer;
    }
    assert(expected_ordinal[0] == n_layers/2);
    assert(expected_ordinal[1] == n_layers/2);
    assert(pair_for_layer(19) == 1);
    assert(pair_for_layer(20) == 1);
    assert(pair_local_ordinal(20) == pair_local_ordinal(19) + 1);
    assert(weight_slot_for_layer(20) != weight_slot_for_layer(19));
}

struct weight_lease {
    int generation = -1;
    int layer = -1;
    int released_generation = -1;
    int released_layer = -1;
    int released_panel = -1;
    uint64_t ready = 0;
    uint64_t released = 0;
};

static int next_selected_layer(
        int layer, int pair, int first, int last) {
    for (int next = std::max(layer + 1, first);
            next <= last; ++next) {
        if (pair_for_layer(next) == pair) {
            return next;
        }
    }
    return -1;
}

static void test_weight_lookahead() {
    const std::array<std::array<int, 2>, 4> ranges{{
        {{0, 0}},
        {{19, 20}},
        {{16, 23}},
        {{0, 39}},
    }};
    for (const auto & range : ranges) {
        std::array<std::array<weight_lease, 2>, n_pairs>
                slots{};
        std::array<uint64_t, n_pairs> copy_ready{};
        std::array<uint64_t, n_pairs> compute_ready{};
        std::mt19937 random(
                (uint32_t) (range[0]*41 + range[1]));
        std::uniform_int_distribution<uint64_t>
                copy_duration(1, 23);
        std::uniform_int_distribution<uint64_t>
                compute_duration(1, 31);
        auto acquire = [&](int generation, int layer) {
            const int pair = pair_for_layer(layer);
            const int slot = weight_slot_for_layer(layer);
            weight_lease & lease = slots[pair][slot];
            if (lease.layer >= 0) {
                assert(lease.released_generation ==
                        lease.generation);
                assert(lease.released_layer == lease.layer);
                assert(lease.released_panel == 1);
            }
            const uint64_t copy_begin =
                    std::max(copy_ready[pair],
                            lease.released);
            assert(copy_begin >= lease.released);
            lease.ready =
                    copy_begin + copy_duration(random);
            copy_ready[pair] = lease.ready;
            lease.generation = generation;
            lease.layer = layer;
            lease.released_generation = -1;
            lease.released_layer = -1;
            lease.released_panel = -1;
        };

        for (int generation = 0; generation < 5;
                ++generation) {
            for (int pair = 0; pair < n_pairs; ++pair) {
                const int first = next_selected_layer(
                        range[0] - 1, pair,
                        range[0], range[1]);
                if (first >= 0) {
                    acquire(generation, first);
                }
            }
            const schedule value =
                    make_schedule(generation);
            for (int index : value.order) {
                const task & current =
                        value.tasks[index];
                if (current.layer < range[0] ||
                        current.layer > range[1]) {
                    continue;
                }
                const int slot =
                        weight_slot_for_layer(
                            current.layer);
                weight_lease & lease =
                        slots[current.pair][slot];
                assert(lease.generation == generation);
                assert(lease.layer == current.layer);
                if (current.panel == 0) {
                    const int next = next_selected_layer(
                            current.layer,
                            current.pair,
                            range[0], range[1]);
                    if (next >= 0) {
                        assert(weight_slot_for_layer(
                                    next) != slot);
                        acquire(generation, next);
                    }
                }
                const uint64_t begin =
                        std::max(compute_ready[
                                current.pair],
                                lease.ready);
                const uint64_t end =
                        begin + compute_duration(random);
                compute_ready[current.pair] = end;
                lease.released_generation = generation;
                lease.released_layer = current.layer;
                lease.released_panel = current.panel;
                lease.released = end;
            }
        }
    }
}

int main() {
    test_q8_0_t64_pack();
    test_static_dag();
    test_delayed_completions();
    test_generation_wraparound();
    test_weight_slots();
    test_weight_lookahead();
    std::puts("pairfold scheduler tests passed");
    return 0;
}
