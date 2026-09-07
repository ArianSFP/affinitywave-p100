#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

struct ggml_cuda_concat_gather {
    const char * base = nullptr;
    const int32_t * rows = nullptr;
    int64_t row_stride = 0;
    char * gdst = nullptr;
};

bool ggml_cuda_concat_can_fuse_gather(const ggml_tensor * concat, const ggml_tensor * gr);

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
