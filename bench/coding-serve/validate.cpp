#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Teacher-forced fresh/cached transitions. Save the entire vocabulary at
// each boundary, not sampled text or rounded HTTP probabilities.
int main(int argc, char ** argv) {
    try {
        common_init();
        common_params params;
        params.n_ctx = 16384;
        params.n_parallel = 1;
        params.warmup = false;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
            return 1;
        }
        const char * output = getenv("GGML_CUDA_AW_VALIDATION_OUTPUT");
        if (output == nullptr) {
            throw std::runtime_error("validation output path required");
        }
        std::ofstream out(output, std::ios::binary);
        if (!out) {
            throw std::runtime_error("cannot open validation output");
        }
        llama_backend_init();
        auto initialized = common_init_from_params(params);
        auto * ctx = initialized->context();
        auto * model = initialized->model();
        if (ctx == nullptr || model == nullptr || llama_n_ctx(ctx) < 10000) {
            throw std::runtime_error("validation needs a loaded model and at least 10000 context tokens");
        }
        const std::string code =
            "// Review this patch and suggest a minimal correction with tests.\n"
            "template<class T> class Buffer { std::vector<T> values; public:\n"
            "void append(const T& value) { values.push_back(value); }\n"
            "const T& at(size_t index) const { return values.at(index); } };\n"
            "// Tool result: build completed; two tests passed. Check edge cases.\n";
        std::string text;
        for (int i = 0; i < 200; ++i) {
            text += code;
        }
        const auto tokens = common_tokenize(ctx, params.prompt.empty() ? text : params.prompt, true);
        if (tokens.size() < 10000) {
            throw std::runtime_error("validation fixture too short");
        }
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const char * quality_tokens = getenv("GGML_CUDA_AW_VALIDATION_TOKENS");
        if (quality_tokens != nullptr) {
            const uint32_t total = std::stoul(quality_tokens);
            const uint32_t rows = 512;
            if (total < rows || total + 1 > tokens.size() || total > llama_n_ctx(ctx)) {
                throw std::runtime_error("invalid quality token count");
            }
            auto batch = llama_batch_init(total, 0, 1);
            for (uint32_t i = 0; i < total; ++i) {
                common_batch_add(batch, tokens[i], i, {0}, i >= total - rows);
            }
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) {
                throw std::runtime_error("quality decode failed");
            }
            out.write("AWQLOG01", 8);
            out.write(reinterpret_cast<const char *>(&total), sizeof(total));
            out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char *>(&n_vocab), sizeof(n_vocab));
            double loss = 0;
            for (uint32_t row = 0; row < rows; ++row) {
                const float * logits = llama_get_logits_ith(ctx, int(row) - int(rows));
                const uint32_t target = tokens[total - rows + row + 1];
                if (logits == nullptr) {
                    throw std::runtime_error("missing quality logits");
                }
                const float maximum = *std::max_element(logits, logits + n_vocab);
                double sum = 0;
                for (uint32_t i = 0; i < n_vocab; ++i) {
                    if (!std::isfinite(logits[i])) {
                        throw std::runtime_error("non-finite quality logits");
                    }
                    sum += std::exp(double(logits[i]) - maximum);
                }
                loss += std::log(sum) + maximum - logits[target];
                out.write(reinterpret_cast<const char *>(&target), sizeof(target));
                out.write(reinterpret_cast<const char *>(logits), n_vocab*sizeof(float));
            }
            if (!out) {
                throw std::runtime_error("quality logit write failed");
            }
            fprintf(stdout, "quality tokens=%u evaluated_suffix=%u ppl=%.9f\n", total, rows, std::exp(loss/rows));
            return 0;
        }
        int position = 0;
        auto step = [&](const std::string & label, int count) {
            auto batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; ++i) {
                common_batch_add(batch, tokens.at(position + i), position + i, {0}, i == count - 1);
            }
            const int status = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (status != 0) {
                throw std::runtime_error("decode failed at " + label);
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                throw std::runtime_error("missing logits at " + label);
            }
            for (uint32_t i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) {
                    throw std::runtime_error("non-finite logits at " + label);
                }
            }
            const uint32_t length = label.size();
            out.write(reinterpret_cast<const char *>(&length), sizeof(length));
            out.write(label.data(), length);
            out.write(reinterpret_cast<const char *>(&n_vocab), sizeof(n_vocab));
            out.write(reinterpret_cast<const char *>(logits), n_vocab*sizeof(float));
            out.flush();
            if (!out) {
                throw std::runtime_error("logit write failed");
            }
            position += count;
            fprintf(stdout, "validated boundary %s position=%d vocab=%u\n", label.c_str(), position, n_vocab);
            fflush(stdout);
        };
        auto fresh = [&](const std::string & label, const std::vector<int> & chunks) {
            llama_memory_clear(llama_get_memory(ctx), true);
            position = 0;
            for (size_t i = 0; i < chunks.size(); ++i) {
                step(label + "/" + std::to_string(i), chunks[i]);
            }
        };
        fresh("fresh128", {124, 4});
        fresh("fresh513", {508, 4, 1});
        fresh("fresh1025", {1020, 4, 1});
        fresh("repeat128", {124, 4});
        fresh("seed2048", {2044, 4});
        step("cached65/0", 60);
        step("cached65/1", 4);
        step("cached65/2", 1);
        step("cached130/0", 124);
        step("cached130/1", 4);
        step("cached130/2", 2);
        fresh("long8128", {8124, 4});
        for (int count : {64, 128, 256, 512}) {
            step("long-cached" + std::to_string(count), count);
        }
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "validation failed: %s\n", error.what());
        return 1;
    }
}
