#ifndef ROPE_KERNEL_CUDA_CUH
#define ROPE_KERNEL_CUDA_CUH
#include "tensor.h"

namespace kernel{
    void sin_cos_calc_cuda(size_t head_size, size_t max_len_size, const Tensor& sin_tensor, const Tensor& cos_tensor, cudaStream_t steam);
    void rope_kernel_cuda(size_t dim_size, size_t k_dim, size_t head_size, const Tensor& pos_input, const Tensor& k_input, const Tensor& q_input, const Tensor& sin_cache, const Tensor& cos_cache, void* stream);
}
#endif
// ROPE_KERNEL_CUDA_CUH